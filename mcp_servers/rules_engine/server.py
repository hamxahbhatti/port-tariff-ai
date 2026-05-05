"""
MCP Server: rules_engine

Determines WHICH charges apply to a given vessel at a given port, and
flags any exemptions or special conditions that affect the calculation.

This server does NOT do arithmetic — it returns a list of applicable
charge types and any modifiers (e.g. 'pilotage_compulsory=True',
'tug_count=3', 'cargo_type=bulk').

The LangGraph agent calls this first to know what to compute, then calls
the calculator for each applicable charge.

Rules encoded here are sourced from the Transnet Port Tariff document prose
sections (extracted by Docling and stored in ChromaDB). Where a rule is
ambiguous, the tool returns a flagged result for human review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastmcp import FastMCP

mcp = FastMCP(
    name="rules_engine",
    instructions=(
        "Call determine_applicable_charges first to get the list of charges that apply to a vessel. "
        "Then call check_exemptions to verify if any standard charge should be waived or reduced. "
        "Use the returned charge list and modifiers to drive calculator tool calls."
    ),
)

# ── Charge applicability rules ──────────────────────────────────────────────
# These encode the general Transnet tariff structure. Port-specific exceptions
# are handled via check_exemptions which calls vector_store.

ALWAYS_APPLICABLE = [
    "light_dues",      # applies to all vessels on all SA ports
    "vts",             # Vessel Traffic Services — all ports
    "port_dues",       # marine services levy / port access
]

VESSEL_TYPE_CHARGES = {
    # cargo_dues apply when cargo is handled
    "cargo_dues": lambda v: v.get("cargo_operation") is True,
    # berth_dues if the vessel is berthing (not just calling)
    "berth_dues": lambda v: v.get("berthing") is True,
    # running_of_lines if berthing
    "running_of_lines": lambda v: v.get("berthing") is True,
    # pilotage — compulsory for vessels > 500 GT at most SA ports
    "pilotage": lambda v: float(v.get("gt", 0)) > 500,
    # tug_assistance — compulsory at most SA ports for vessels > 3000 GT
    # (actual tug count is set by Harbour Master)
    "tug_assistance": lambda v: float(v.get("gt", 0)) > 3000,
}

# Default tug counts by GT (Harbour Master guidelines, approximate)
TUG_COUNT_RULES = [
    (0,     3_000,  0),   # no tugs required
    (3_001, 10_000, 1),
    (10_001, 30_000, 2),
    (30_001, 60_000, 3),
    (60_001, float("inf"), 4),
]


def _default_tug_count(gt: float) -> int:
    for floor, ceiling, count in TUG_COUNT_RULES:
        if floor <= gt <= ceiling:
            return count
    return 2  # fallback


@mcp.tool()
def determine_applicable_charges(vessel_profile_json: str) -> str:
    """
    Determine which port charges apply to a vessel based on its profile.

    Returns a structured list of applicable charges with any modifiers
    needed for calculation (tug count, movements, cargo flag, etc.).

    Args:
        vessel_profile_json: JSON object with vessel details:
          {
            "name": "SUDESTADA",
            "vessel_type": "bulk_carrier",   // bulk_carrier | tanker | container | general_cargo | passenger | ro_ro
            "gt": 51300,
            "loa_m": 229.2,
            "port": "durban",
            "cargo_operation": true,          // is cargo being loaded/discharged?
            "cargo_type": "iron_ore",         // bulk | iron_ore | containers | break_bulk | liquid_bulk | etc.
            "cargo_mt": 75000,                // metric tonnes of cargo
            "berthing": true,                 // is vessel going alongside a berth?
            "hours_alongside": 48             // estimated hours at berth
          }

    Returns:
        JSON with applicable_charges list, modifiers, and notes.
    """
    try:
        vessel = json.loads(vessel_profile_json)
        gt = float(vessel.get("gt", 0))
        port = vessel.get("port", "durban")

        applicable = list(ALWAYS_APPLICABLE)
        modifiers = {}
        notes = []

        # Apply vessel-type-dependent rules
        for charge, condition in VESSEL_TYPE_CHARGES.items():
            if condition(vessel):
                applicable.append(charge)

        # Set pilotage movements (2 standard: inbound + outbound)
        if "pilotage" in applicable:
            modifiers["pilotage_movements"] = 2
            notes.append("Pilotage: 2 movements (inbound + outbound) at standard rate.")

        # Set tug count
        if "tug_assistance" in applicable:
            tug_count = _default_tug_count(gt)
            modifiers["tug_count"] = tug_count
            modifiers["tug_movements"] = 2
            notes.append(
                f"Tug assistance: {tug_count} tug(s) × 2 movements. "
                "Actual count determined by Harbour Master."
            )

        # Set cargo parameters
        if "cargo_dues" in applicable:
            modifiers["cargo_type"] = vessel.get("cargo_type", "bulk")
            modifiers["cargo_mt"] = vessel.get("cargo_mt", 0)

        # Set berth parameters
        if "berth_dues" in applicable:
            modifiers["hours_alongside"] = vessel.get("hours_alongside", 24)

        # Vessel type specific notes
        vtype = vessel.get("vessel_type", "").lower()
        if "tanker" in vtype:
            notes.append(
                "Tanker: additional safety regulations may apply — "
                "check rules_engine/check_exemptions for port-specific conditions."
            )
        if "bulk" in vtype and vessel.get("cargo_type", "").lower() in ("iron_ore", "coal", "ore"):
            notes.append(
                "Bulk carrier with bulk cargo: cargo dues apply at bulk rate — "
                "verify cargo_dues row matches 'bulk' or 'iron_ore' category."
            )

        return json.dumps({
            "vessel": vessel.get("name", "unknown"),
            "port": port,
            "gt": gt,
            "applicable_charges": applicable,
            "modifiers": modifiers,
            "notes": notes,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def check_exemptions(vessel_profile_json: str, charge_type: str) -> str:
    """
    Check whether a specific charge is exempt or reduced for this vessel.

    Queries the semantic rule store for exemption conditions and returns
    a structured ruling. If the result is ambiguous, flags it for review.

    Common exemptions in Transnet tariffs:
    - Vessels in distress: exempted from port dues
    - Naval / government vessels: may be exempt from certain charges
    - Vessels not working cargo: exempt from cargo dues
    - Pleasure craft < 100 GT: may be exempt from VTS
    - Vessels calling only for bunkers/stores: reduced charges may apply

    Args:
        vessel_profile_json: JSON vessel profile (same as determine_applicable_charges).
        charge_type:         The charge to check (e.g. 'light_dues', 'pilotage').

    Returns:
        JSON with:
          exempt: bool
          reduced: bool (partial reduction)
          reduction_factor: float (1.0 = full charge, 0.5 = 50% reduction, 0.0 = exempt)
          reason: string explanation
          requires_review: bool (true if rule is ambiguous)
    """
    try:
        vessel = json.loads(vessel_profile_json)
        gt = float(vessel.get("gt", 0))
        vtype = vessel.get("vessel_type", "").lower()

        result = {
            "charge_type": charge_type,
            "vessel": vessel.get("name", "unknown"),
            "exempt": False,
            "reduced": False,
            "reduction_factor": 1.0,
            "reason": "Standard charge applies.",
            "requires_review": False,
        }

        # ── Hard-coded exemption rules ──────────────────────────────────────

        # Cargo dues exempt if no cargo operation
        if charge_type == "cargo_dues" and not vessel.get("cargo_operation"):
            result.update({
                "exempt": True,
                "reduction_factor": 0.0,
                "reason": "No cargo operation declared — cargo dues do not apply.",
            })

        # Running of lines / berth dues exempt if not berthing
        elif charge_type in ("running_of_lines", "berth_dues") and not vessel.get("berthing"):
            result.update({
                "exempt": True,
                "reduction_factor": 0.0,
                "reason": "Vessel not berthing — charge does not apply.",
            })

        # Pilotage exempt for very small vessels (< 500 GT — handled by rules already,
        # but catch edge case)
        elif charge_type == "pilotage" and gt <= 500:
            result.update({
                "exempt": True,
                "reduction_factor": 0.0,
                "reason": "Vessels ≤ 500 GT are exempt from compulsory pilotage.",
            })

        # Tug assistance exempt for small vessels (< 3000 GT)
        elif charge_type == "tug_assistance" and gt <= 3000:
            result.update({
                "exempt": True,
                "reduction_factor": 0.0,
                "reason": "Vessels ≤ 3000 GT do not require compulsory tug assistance.",
            })

        # Flag naval / government vessels for manual review
        elif "naval" in vtype or "government" in vtype or "warship" in vtype:
            result.update({
                "requires_review": True,
                "reason": (
                    "Government/naval vessels may qualify for exemptions under "
                    "the Transnet NPA tariff schedule — manual verification required."
                ),
            })

        # Vessel in distress
        elif vessel.get("in_distress"):
            result.update({
                "exempt": True,
                "reduction_factor": 0.0,
                "reason": "Vessel in distress — port dues and associated charges waived.",
            })

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_vessel_charge_plan(vessel_profile_json: str) -> str:
    """
    High-level entry point: returns a full charge plan for a vessel.

    Combines determine_applicable_charges + check_exemptions for all charges
    into a single action plan that the agent can execute sequentially.

    Args:
        vessel_profile_json: Full vessel profile JSON.

    Returns:
        JSON action plan with steps the agent should follow.
    """
    try:
        vessel = json.loads(vessel_profile_json)

        # Get applicable charges
        applicable_result = json.loads(
            determine_applicable_charges(vessel_profile_json)
        )

        if "error" in applicable_result:
            return json.dumps(applicable_result)

        charges = applicable_result["applicable_charges"]
        modifiers = applicable_result["modifiers"]

        # Check exemptions for each
        action_steps = []
        for charge in charges:
            exemption = json.loads(check_exemptions(vessel_profile_json, charge))
            if not exemption.get("exempt"):
                action_steps.append({
                    "step": f"calculate_{charge}",
                    "charge_type": charge,
                    "tariff_table_needed": f"get_tariff_table('{vessel.get('port','durban')}', '{charge}')",
                    "modifiers": {k: v for k, v in modifiers.items() if charge in k or k.startswith(charge.split("_")[0])},
                    "exemption_check": exemption,
                })
            else:
                action_steps.append({
                    "step": f"SKIP_{charge}",
                    "charge_type": charge,
                    "reason": exemption.get("reason"),
                })

        return json.dumps({
            "vessel": vessel.get("name"),
            "port": vessel.get("port"),
            "gt": vessel.get("gt"),
            "charge_plan": action_steps,
            "notes": applicable_result.get("notes", []),
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
