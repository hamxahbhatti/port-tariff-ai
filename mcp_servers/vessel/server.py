"""
MCP Server: vessel

Manages vessel profiles in memory and provides vessel-specific lookups.

In a production system this would connect to a vessel registry (AIS, Lloyd's).
For this assessment it accepts a vessel profile JSON and stores it for the
duration of a calculation session.

Also provides helpers for converting vessel measurements and classifying
vessel types for tariff purposes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastmcp import FastMCP

mcp = FastMCP(
    name="vessel",
    instructions=(
        "Use register_vessel to store a vessel profile at the start of a calculation. "
        "Use get_vessel to retrieve it by name. "
        "Use classify_vessel_for_tariff to get the vessel type category used in tariff tables."
    ),
)

# In-memory store (single-session; replace with Redis/DB in production)
_vessel_registry: dict[str, dict] = {}


@mcp.tool()
def register_vessel(vessel_profile_json: str) -> str:
    """
    Register a vessel profile for a tariff calculation session.

    Validates the profile and stores it for retrieval by name.

    Required fields:
      name         : vessel name (string)
      gt           : Gross Tonnage (number)
      loa_m        : Length Overall in metres (number)
      vessel_type  : one of bulk_carrier | tanker | container | general_cargo | passenger | ro_ro
      port         : port of call (string, lowercase)

    Optional fields:
      cargo_operation : bool (default false)
      cargo_type      : string (e.g. 'iron_ore', 'containers', 'crude_oil')
      cargo_mt        : float (metric tonnes of cargo)
      berthing        : bool (default true — assume berthing unless otherwise stated)
      hours_alongside : float (estimated hours at berth, default 24)
      in_distress     : bool (default false)
      nrt             : Net Register Tonnage (for some historical charges)
      flag_state      : vessel flag (string)

    Args:
        vessel_profile_json: JSON object with vessel details.

    Returns:
        JSON with validation result and registered profile.
    """
    try:
        profile = json.loads(vessel_profile_json)

        # Validate required fields
        required = ["name", "gt", "loa_m", "vessel_type", "port"]
        missing = [f for f in required if f not in profile]
        if missing:
            return json.dumps({
                "error": f"Missing required fields: {missing}",
                "required": required,
            })

        # Normalise
        profile["name"] = str(profile["name"]).upper()
        profile["gt"] = float(profile["gt"])
        profile["loa_m"] = float(profile["loa_m"])
        profile["port"] = profile["port"].lower().strip().replace(" ", "_")
        profile["vessel_type"] = profile["vessel_type"].lower().strip()

        # Set defaults
        profile.setdefault("cargo_operation", False)
        profile.setdefault("berthing", True)
        profile.setdefault("hours_alongside", 24.0)
        profile.setdefault("in_distress", False)
        profile.setdefault("cargo_type", "")
        profile.setdefault("cargo_mt", 0.0)

        # Store
        _vessel_registry[profile["name"]] = profile

        return json.dumps({
            "status": "registered",
            "vessel": profile["name"],
            "profile": profile,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_vessel(vessel_name: str) -> str:
    """
    Retrieve a previously registered vessel profile.

    Args:
        vessel_name: Vessel name (case-insensitive).

    Returns:
        JSON with vessel profile, or error if not found.
    """
    name = vessel_name.upper()
    profile = _vessel_registry.get(name)
    if profile is None:
        return json.dumps({
            "error": f"Vessel '{name}' not registered. Call register_vessel first.",
            "registered_vessels": list(_vessel_registry.keys()),
        })
    return json.dumps(profile, indent=2)


@mcp.tool()
def classify_vessel_for_tariff(vessel_profile_json: str) -> str:
    """
    Classify a vessel into the categories used by Transnet tariff tables.

    Transnet uses different rate rows for:
    - Vessel size (GT-based tonnage bands)
    - Vessel type (for some charges like cargo dues)
    - Cargo type (bulk vs break-bulk vs containerised)

    Args:
        vessel_profile_json: JSON vessel profile.

    Returns:
        JSON with tariff classification and any relevant notes.
    """
    try:
        v = json.loads(vessel_profile_json)
        gt = float(v.get("gt", 0))
        vtype = v.get("vessel_type", "").lower()
        cargo = v.get("cargo_type", "").lower()

        # GT size category
        if gt < 500:
            gt_category = "small"
        elif gt < 10_000:
            gt_category = "medium"
        elif gt < 50_000:
            gt_category = "large"
        else:
            gt_category = "very_large"

        # Cargo classification for cargo dues table lookup
        cargo_class = "general"
        if any(word in cargo for word in ("iron", "ore", "coal", "grain", "bulk")):
            cargo_class = "bulk"
        elif any(word in cargo for word in ("container", "teu")):
            cargo_class = "containerised"
        elif any(word in cargo for word in ("crude", "liquid", "oil", "chemical", "lpg", "lng")):
            cargo_class = "liquid_bulk"
        elif any(word in cargo for word in ("break", "general", "project")):
            cargo_class = "break_bulk"
        elif any(word in cargo for word in ("ro_ro", "roro", "vehicle", "car")):
            cargo_class = "ro_ro"

        # Pilotage/tug classification
        requires_compulsory_pilotage = gt > 500
        required_tugs = 0
        if gt > 3_000:
            if gt <= 10_000:
                required_tugs = 1
            elif gt <= 30_000:
                required_tugs = 2
            elif gt <= 60_000:
                required_tugs = 3
            else:
                required_tugs = 4

        return json.dumps({
            "vessel_name": v.get("name", "unknown"),
            "gt": gt,
            "gt_category": gt_category,
            "vessel_type": vtype,
            "cargo_class": cargo_class,
            "tariff_notes": {
                "requires_compulsory_pilotage": requires_compulsory_pilotage,
                "default_tug_count": required_tugs,
                "note": "Actual tug count set by Harbour Master — this is a guide only.",
            },
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def list_registered_vessels() -> str:
    """
    List all vessel profiles registered in this session.

    Returns:
        JSON with list of vessel names and their GT.
    """
    return json.dumps({
        "registered": [
            {"name": name, "gt": p.get("gt"), "port": p.get("port")}
            for name, p in _vessel_registry.items()
        ]
    })


if __name__ == "__main__":
    mcp.run()
