"""
MCP Server: calculator

Pure Python arithmetic for each Transnet port charge type.
No LLM calls — deterministic, auditable, testable.

Each tool takes the tariff table rows (from get_tariff_table) plus vessel
parameters and returns a structured breakdown with the math shown step-by-step.

Charge types handled:
  - light_dues         : GT × rate/100GT
  - vts                : GT × rate/GT per call
  - pilotage           : base band fee + incremental per 100GT above band floor
  - tug_assistance     : base band fee + incremental per tug per 100GT
  - port_dues          : GT × rate/GT
  - cargo_dues         : metric_tonnes × rate/MT
  - berth_dues         : GT × rate/24h × duration_hours / 24
  - running_of_lines   : flat rate per service
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastmcp import FastMCP

mcp = FastMCP(
    name="calculator",
    instructions=(
        "Use these tools to compute exact port charges from tariff table rows and vessel data. "
        "Always pass the full rows list from get_tariff_table. "
        "Return structured breakdowns — never round intermediate values."
    ),
)


# ── helpers ────────────────────────────────────────────────────────────────

def _clean_number_str(s: str) -> str:
    """Remove 'tons', 'GT', spaces-within-numbers, commas so '2 000 tons' → '2000'."""
    s = re.sub(r'\btons?\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bGT\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bplus\b', '', s, flags=re.IGNORECASE)
    # Remove spaces that are surrounded by digits (thousand separators)
    s = re.sub(r'(\d)\s+(\d)', r'\1\2', s)
    s = s.replace(',', '').strip()
    return s


def _parse_band(band_str: str) -> tuple[float, float | None]:
    """
    Parse a tonnage band string into (floor, ceiling) GT values.

    Handles formats like:
      'Up to 2000'            → (0, 2000)
      'Up to 2 000 tons'      → (0, 2000)
      '2 000 - 10 000 tons'   → (2000, 10000)
      '10 001 - 50 000 tons Plus' → (10001, 50000)
      '100001 and above'      → (100001, None)
      'All vessels'           → (0, None)
    Descriptive strings (e.g. 'Basic Fee Per Service') return (0, None) as
    a catch-all so the row matches any vessel.
    """
    s = _clean_number_str(band_str).strip().lower()

    if s in ("all vessels", "all", ""):
        return (0.0, None)

    if s.startswith("up to"):
        try:
            ceiling = float(_clean_number_str(s.replace("up to", "").strip()))
            return (0.0, ceiling)
        except ValueError:
            pass

    if "and above" in s or s.endswith("above"):
        try:
            floor = float(_clean_number_str(
                s.replace("and above", "").replace("above", "").strip()
            ))
            return (floor, None)
        except ValueError:
            pass

    if "-" in s:
        # Check original band_str for "plus" BEFORE cleaning strips it
        open_ended = bool(re.search(r'\bplus\b', band_str, re.IGNORECASE))
        parts = s.split("-")
        try:
            lo = float(_clean_number_str(parts[0]))
            if open_ended:
                return (lo, None)
            hi = float(_clean_number_str(parts[1]))
            return (lo, hi)
        except (ValueError, IndexError):
            pass

    # Try parsing as a plain number
    try:
        return (float(s), None)
    except ValueError:
        pass

    # Descriptive string (e.g. "Basic Fee Per Service") — treat as catch-all
    return (0.0, None)


def _find_band_row(rows: list[dict], gt: float, port_key: str) -> dict | None:
    """Return the non-incremental row whose tonnage band contains gt."""
    for row in rows:
        if row.get("is_incremental"):
            continue
        band = row.get("tonnage_band", "All vessels")
        floor, ceiling = _parse_band(band)
        if gt >= floor and (ceiling is None or gt <= ceiling):
            if port_key in (row.get("values") or {}):
                return row
    return None


def _find_incremental_row(rows: list[dict], parent_band: str, port_key: str) -> dict | None:
    """Return the incremental sub-row for a given parent tonnage band."""
    for row in rows:
        if not row.get("is_incremental"):
            continue
        if row.get("parent_band", "").lower() == parent_band.lower():
            if port_key in (row.get("values") or {}):
                return row
    return None


def _port_key(port_name: str) -> str:
    return port_name.lower().strip().replace(" ", "_").replace("-", "_")


# ── tools ──────────────────────────────────────────────────────────────────

@mcp.tool()
def calculate_light_dues(
    rows_json: str,
    gt: float,
    port_name: str,
) -> str:
    """
    Calculate Light Dues.

    Formula: (GT / 100) × rate_per_100GT

    Light dues apply once per port call regardless of cargo type.

    Args:
        rows_json: JSON string of the 'rows' array from get_tariff_table('durban','light_dues').
        gt:        Vessel Gross Tonnage.
        port_name: Port name (e.g. 'durban').

    Returns:
        JSON with charge_zar, formula, and breakdown.
    """
    try:
        rows = json.loads(rows_json)
        pk = _port_key(port_name)

        row = _find_band_row(rows, gt, pk)
        if row is None:
            # light dues often has a single 'All vessels' row
            row = next((r for r in rows if not r.get("is_incremental")), None)

        if row is None:
            return json.dumps({"error": "No matching row found for light dues."})

        rate = row["values"].get(pk)
        if rate is None:
            return json.dumps({"error": f"No rate for port '{port_name}' in light_dues table."})

        unit = row.get("unit", "per_100GT")
        if "100" in unit:
            units_of_100gt = gt / 100.0
            charge = units_of_100gt * rate
        else:
            charge = gt * rate

        return json.dumps({
            "charge_type": "light_dues",
            "port": port_name,
            "gt": gt,
            "rate": rate,
            "unit": unit,
            "formula": f"({gt} GT / 100) × R{rate} = R{charge:.2f}",
            "charge_zar": round(charge, 2),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def calculate_vts(
    rows_json: str,
    gt: float,
    port_name: str,
) -> str:
    """
    Calculate Vessel Traffic Services (VTS) dues.

    Formula: GT × rate_per_GT

    VTS is charged per port call.

    Args:
        rows_json: JSON string of the 'rows' array from get_tariff_table(port,'vts').
        gt:        Vessel Gross Tonnage.
        port_name: Port name (e.g. 'durban').

    Returns:
        JSON with charge_zar and formula.
    """
    try:
        rows = json.loads(rows_json)
        pk = _port_key(port_name)

        # VTS table is ports_as_rows — each row has a 'port' field
        row = None
        for r in rows:
            row_port = _port_key(r.get("port") or "")
            if row_port == pk:
                row = r
                break

        # Fallback: try values dict
        if row is None:
            row = next(
                (r for r in rows if pk in (r.get("values") or {})),
                None
            )

        if row is None:
            return json.dumps({"error": f"No VTS row found for port '{port_name}'."})

        values = row.get("values") or {}
        rate = values.get(pk)
        if rate is None:
            return json.dumps({"error": f"No VTS rate for '{port_name}'."})

        charge = gt * rate

        return json.dumps({
            "charge_type": "vts",
            "port": port_name,
            "gt": gt,
            "rate": rate,
            "unit": row.get("unit", "per_GT"),
            "formula": f"{gt} GT × R{rate}/GT = R{charge:.2f}",
            "charge_zar": round(charge, 2),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def calculate_pilotage(
    rows_json: str,
    gt: float,
    port_name: str,
    movements: int = 2,
) -> str:
    """
    Calculate Pilotage dues.

    Formula (per movement):
      base_fee (from tonnage band row)
      + incremental_rate × ceil((GT - band_floor) / 100)  [if incremental row exists]

    Total = per_movement_fee × movements

    Standard: 2 movements per port call (inbound + outbound).

    Args:
        rows_json: JSON string of the 'rows' array from get_tariff_table(port,'pilotage').
        gt:        Vessel Gross Tonnage.
        port_name: Port name.
        movements: Number of pilotage movements (default 2 = inbound + outbound).

    Returns:
        JSON with charge_zar, per_movement breakdown, and formula.
    """
    try:
        rows = json.loads(rows_json)
        pk = _port_key(port_name)

        base_row = _find_band_row(rows, gt, pk)

        # Pilotage often uses "Basic Fee Per Service" + "Per 100 tons" rows
        # instead of GT bands. Detect and handle both formats.
        if base_row is None:
            # Look for a "basic fee" or "flat fee" row
            base_row = next(
                (r for r in rows
                 if not r.get("is_incremental")
                 and pk in (r.get("values") or {})
                 and any(kw in (r.get("tonnage_band") or "").lower()
                         for kw in ("basic", "flat", "per service", "fee per"))),
                None,
            )
            # Fallback: first non-incremental row with this port
            if base_row is None:
                base_row = next(
                    (r for r in rows
                     if not r.get("is_incremental") and pk in (r.get("values") or {})),
                    None,
                )

        if base_row is None:
            return json.dumps({"error": f"No pilotage base row found for '{port_name}'."})

        base_fee = base_row["values"][pk]
        band = base_row.get("tonnage_band", "")
        band_floor, _ = _parse_band(band)

        # Find incremental row — either via parent_band match or "per 100 tons" keyword
        inc_row = _find_incremental_row(rows, band, pk)
        if inc_row is None:
            inc_row = next(
                (r for r in rows
                 if pk in (r.get("values") or {})
                 and "100" in (r.get("tonnage_band") or "").lower()),
                None,
            )

        incremental_charge = 0.0
        incremental_detail = "no incremental row"

        if inc_row is not None:
            inc_rate = inc_row["values"][pk]
            # If band_floor is 0 (descriptive band), incremental applies to full GT
            excess_gt = gt if band_floor == 0.0 else max(0.0, gt - band_floor)
            units = math.ceil(excess_gt / 100.0)
            incremental_charge = units * inc_rate
            incremental_detail = (
                f"ceil({excess_gt} / 100) × R{inc_rate} "
                f"= {units} × R{inc_rate} = R{incremental_charge:.2f}"
            )

        per_movement = base_fee + incremental_charge
        total = per_movement * movements

        return json.dumps({
            "charge_type": "pilotage",
            "port": port_name,
            "gt": gt,
            "tonnage_band": band,
            "base_fee_per_movement": base_fee,
            "incremental_detail": incremental_detail,
            "incremental_charge_per_movement": round(incremental_charge, 2),
            "per_movement_total": round(per_movement, 2),
            "movements": movements,
            "formula": (
                f"({base_fee} + {round(incremental_charge,2)}) × {movements} movements"
                f" = R{round(total, 2)}"
            ),
            "charge_zar": round(total, 2),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def calculate_tug_assistance(
    rows_json: str,
    gt: float,
    port_name: str,
    num_tugs: int = 2,
    movements: int = 2,
) -> str:
    """
    Calculate Tug Assistance dues.

    Formula (per tug per movement):
      base_fee (from tonnage band) + incremental per 100GT above band floor

    Total = per_tug_per_movement × num_tugs × movements

    Args:
        rows_json: JSON string of the 'rows' array from get_tariff_table(port,'tug_assistance').
        gt:        Vessel Gross Tonnage.
        port_name: Port name.
        num_tugs:  Number of tugs used (Harbour Master determines this; default 2).
        movements: Number of tug movements (default 2).

    Returns:
        JSON with charge_zar and full breakdown.
    """
    try:
        rows = json.loads(rows_json)
        pk = _port_key(port_name)

        base_row = _find_band_row(rows, gt, pk)
        if base_row is None:
            return json.dumps({"error": f"No tonnage band found for GT={gt} at '{port_name}'."})

        base_fee = base_row["values"][pk]
        band = base_row.get("tonnage_band", "")
        band_floor, _ = _parse_band(band)

        inc_row = _find_incremental_row(rows, band, pk)
        incremental_charge = 0.0
        incremental_detail = "no incremental row"

        if inc_row is not None:
            inc_rate = inc_row["values"][pk]
            excess_gt = max(0.0, gt - band_floor)
            units = math.ceil(excess_gt / 100.0)
            incremental_charge = units * inc_rate
            incremental_detail = (
                f"ceil(({gt} - {band_floor}) / 100) × R{inc_rate} "
                f"= {units} × R{inc_rate} = R{incremental_charge:.2f}"
            )

        per_tug_per_movement = base_fee + incremental_charge
        total = per_tug_per_movement * num_tugs * movements

        return json.dumps({
            "charge_type": "tug_assistance",
            "port": port_name,
            "gt": gt,
            "tonnage_band": band,
            "base_fee_per_tug_per_movement": base_fee,
            "incremental_detail": incremental_detail,
            "incremental_per_tug_per_movement": round(incremental_charge, 2),
            "per_tug_per_movement_total": round(per_tug_per_movement, 2),
            "num_tugs": num_tugs,
            "movements": movements,
            "formula": (
                f"({base_fee} + {round(incremental_charge,2)}) "
                f"× {num_tugs} tugs × {movements} movements = R{round(total,2)}"
            ),
            "charge_zar": round(total, 2),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def calculate_port_dues(
    rows_json: str,
    gt: float,
    port_name: str,
) -> str:
    """
    Calculate Port Dues (also called Marine Services Levy or Port Access Levy).

    Formula: GT × rate_per_GT

    Args:
        rows_json: JSON string of the 'rows' array from get_tariff_table(port,'port_dues').
        gt:        Vessel Gross Tonnage.
        port_name: Port name.

    Returns:
        JSON with charge_zar and formula.
    """
    try:
        rows = json.loads(rows_json)
        pk = _port_key(port_name)

        row = _find_band_row(rows, gt, pk)
        if row is None:
            row = next((r for r in rows if not r.get("is_incremental") and pk in (r.get("values") or {})), None)

        if row is None:
            return json.dumps({"error": f"No port dues row for GT={gt} at '{port_name}'."})

        rate = row["values"][pk]
        unit = row.get("unit", "per_GT")
        band = (row.get("tonnage_band") or "").lower()

        # Port dues can be per GT or per 100 GT — detect from unit/band string
        if "100" in band or "100" in unit:
            charge = (gt / 100.0) * rate
            formula = f"({gt} GT / 100) × R{rate}/100GT = R{charge:.2f}"
        else:
            charge = gt * rate
            formula = f"{gt} GT × R{rate}/GT = R{charge:.2f}"

        return json.dumps({
            "charge_type": "port_dues",
            "port": port_name,
            "gt": gt,
            "rate": rate,
            "unit": unit,
            "formula": formula,
            "charge_zar": round(charge, 2),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def calculate_cargo_dues(
    rows_json: str,
    metric_tonnes: float,
    cargo_type: str,
    port_name: str,
) -> str:
    """
    Calculate Cargo Dues.

    Formula: metric_tonnes × rate_per_MT

    The rate varies by cargo type (bulk, break-bulk, containerised, etc.).

    Args:
        rows_json:     JSON string of the 'rows' array from get_tariff_table(port,'cargo_dues').
        metric_tonnes: Cargo mass in metric tonnes.
        cargo_type:    Cargo description (e.g. 'bulk', 'iron_ore', 'containers', 'break_bulk').
        port_name:     Port name.

    Returns:
        JSON with charge_zar, matched cargo row, and formula.
    """
    try:
        rows = json.loads(rows_json)
        pk = _port_key(port_name)
        ct_lower = cargo_type.lower()

        # Find row whose tonnage_band / description matches cargo type
        matched_row = None
        for row in rows:
            desc = (row.get("tonnage_band") or row.get("description") or "").lower()
            if ct_lower in desc or any(word in desc for word in ct_lower.split("_")):
                if pk in (row.get("values") or {}):
                    matched_row = row
                    break

        # Fallback: first non-incremental row with a value for this port
        if matched_row is None:
            matched_row = next(
                (r for r in rows if not r.get("is_incremental") and pk in (r.get("values") or {})),
                None,
            )

        if matched_row is None:
            return json.dumps({"error": f"No cargo dues row for '{cargo_type}' at '{port_name}'."})

        rate = matched_row["values"][pk]
        unit = matched_row.get("unit", "per_MT")
        charge = metric_tonnes * rate

        return json.dumps({
            "charge_type": "cargo_dues",
            "port": port_name,
            "cargo_type": cargo_type,
            "metric_tonnes": metric_tonnes,
            "rate": rate,
            "unit": unit,
            "matched_row_description": matched_row.get("tonnage_band") or matched_row.get("description"),
            "formula": f"{metric_tonnes} MT × R{rate}/MT = R{charge:.2f}",
            "charge_zar": round(charge, 2),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def calculate_berth_dues(
    rows_json: str,
    gt: float,
    port_name: str,
    hours_alongside: float,
) -> str:
    """
    Calculate Berth Dues (also called Wharfage or Time in Berth charges).

    Formula: GT × rate_per_24h × (hours_alongside / 24)
    Minimum charge: 24 hours is typically the minimum billing period.

    Args:
        rows_json:       JSON string of the 'rows' array from get_tariff_table(port,'berth_dues').
        gt:              Vessel Gross Tonnage.
        port_name:       Port name.
        hours_alongside: Total hours the vessel is alongside the berth.

    Returns:
        JSON with charge_zar and formula.
    """
    try:
        rows = json.loads(rows_json)
        pk = _port_key(port_name)

        row = _find_band_row(rows, gt, pk)
        if row is None:
            # Berth dues may use "Per 100 tons" flat rate rows
            row = next((r for r in rows if not r.get("is_incremental") and pk in (r.get("values") or {})), None)

        if row is None:
            return json.dumps({"error": f"No berth dues row for GT={gt} at '{port_name}'."})

        rate = row["values"][pk]
        unit = row.get("unit", "per_24h")
        band = (row.get("tonnage_band") or "").lower()

        # Minimum 24 hours
        billable_hours = max(24.0, hours_alongside)
        periods = billable_hours / 24.0

        # Berth dues can be per GT or per 100 GT — detect from band/unit
        if "100" in band or "100" in unit:
            units_of_100gt = gt / 100.0
            charge = units_of_100gt * rate * periods
            formula = (
                f"({gt} GT / 100) × R{rate}/100GT × {round(periods,4)} periods"
                f" = R{charge:.2f}"
            )
        else:
            charge = gt * rate * periods
            formula = f"{gt} GT × R{rate}/24h × {round(periods,4)} periods = R{charge:.2f}"

        return json.dumps({
            "charge_type": "berth_dues",
            "port": port_name,
            "gt": gt,
            "rate": rate,
            "unit": unit,
            "hours_alongside": hours_alongside,
            "billable_hours": billable_hours,
            "periods_24h": round(periods, 4),
            "formula": formula,
            "charge_zar": round(charge, 2),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def calculate_running_of_lines(
    rows_json: str,
    port_name: str,
    num_services: int = 2,
) -> str:
    """
    Calculate Running of Lines (mooring/unmooring) charges.

    Formula: flat_rate_per_service × num_services

    Standard: 2 services per port call (mooring on arrival + unmooring on departure).

    Args:
        rows_json:    JSON string of the 'rows' array from get_tariff_table(port,'running_of_lines').
        port_name:    Port name.
        num_services: Number of mooring/unmooring services (default 2).

    Returns:
        JSON with charge_zar and formula.
    """
    try:
        rows = json.loads(rows_json)
        pk = _port_key(port_name)

        row = next(
            (r for r in rows if not r.get("is_incremental") and pk in (r.get("values") or {})),
            None,
        )

        # Fall back to "other_ports" catch-all if port not explicitly listed
        if row is None:
            row = next(
                (r for r in rows
                 if not r.get("is_incremental")
                 and "other" in (r.get("port") or "").lower()
                 and "other_ports" in (r.get("values") or {})),
                None,
            )
            if row:
                pk = "other_ports"  # use the other_ports rate

        if row is None:
            return json.dumps({"error": f"No running of lines row for '{port_name}'."})

        rate = row["values"][pk]
        charge = rate * num_services

        return json.dumps({
            "charge_type": "running_of_lines",
            "port": port_name,
            "rate_per_service": rate,
            "num_services": num_services,
            "formula": f"R{rate} × {num_services} services = R{charge:.2f}",
            "charge_zar": round(charge, 2),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def aggregate_charges(charge_results_json: str) -> str:
    """
    Sum all individual charge results into a final invoice total.

    Args:
        charge_results_json: JSON array of charge result objects (each with 'charge_zar' field).

    Returns:
        JSON with total_zar, line_items summary, and any errors encountered.
    """
    try:
        charges = json.loads(charge_results_json)
        if not isinstance(charges, list):
            charges = [charges]

        total = 0.0
        line_items = []
        errors = []

        for c in charges:
            if "error" in c:
                errors.append(c)
                continue
            amount = c.get("charge_zar", 0.0)
            total += amount
            line_items.append({
                "charge_type": c.get("charge_type", "unknown"),
                "port": c.get("port", ""),
                "charge_zar": amount,
                "formula": c.get("formula", ""),
            })

        return json.dumps({
            "total_zar": round(total, 2),
            "line_items": line_items,
            "errors": errors,
            "currency": "ZAR",
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
