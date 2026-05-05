"""
Test Layer 2: MCP Servers + Calculator correctness

Tests all 4 MCP servers with synthetic tariff data (no API key required).
Validates:
  1. rules_engine  — correct charge list for SUDESTADA
  2. vessel        — registration and classification
  3. calculator    — correct arithmetic for each charge type
  4. tariff_rag    — store read/write roundtrip

Run with:
    python -m tests.test_mcp_servers

No GEMINI_API_KEY needed for these tests.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── SUDESTADA test profile ──────────────────────────────────────────────────

SUDESTADA = {
    "name": "SUDESTADA",
    "vessel_type": "bulk_carrier",
    "gt": 51300,
    "loa_m": 229.2,
    "port": "durban",
    "cargo_operation": True,
    "cargo_type": "iron_ore",
    "cargo_mt": 75000,
    "berthing": True,
    "hours_alongside": 48,
}

# Synthetic tariff rows matching known Durban rates from the PDF
LIGHT_DUES_ROWS = [
    {"tonnage_band": "All vessels", "is_incremental": False,
     "values": {"durban": 117.08}, "unit": "per_100GT", "notes": ""}
]

VTS_ROWS = [
    {"port": "Port of Durban", "tonnage_band": None, "is_incremental": False,
     "values": {"durban": 0.65}, "unit": "per_GT", "notes": ""}
]

# Pilotage: 50000-60000 GT band
PILOTAGE_ROWS = [
    {"tonnage_band": "50000-60000", "is_incremental": False,
     "values": {"durban": 18608.61}, "unit": "per_movement", "notes": ""},
    {"tonnage_band": "50000-60000", "is_incremental": True, "parent_band": "50000-60000",
     "values": {"durban": 9.72}, "unit": "per_100GT",
     "notes": "Plus per 100GT above lower limit of band"},
]

# Tug assistance: 50000-60000 GT band
TUG_ROWS = [
    {"tonnage_band": "50000-60000", "is_incremental": False,
     "values": {"durban": 39999.88}, "unit": "per_tug_per_movement", "notes": ""},
    {"tonnage_band": "50000-60000", "is_incremental": True, "parent_band": "50000-60000",
     "values": {"durban": 19.94}, "unit": "per_100GT",
     "notes": "Plus per 100GT above lower limit"},
]

PORT_DUES_ROWS = [
    {"tonnage_band": "All vessels", "is_incremental": False,
     "values": {"durban": 3.50}, "unit": "per_GT", "notes": ""}
]

CARGO_DUES_ROWS = [
    {"tonnage_band": "Bulk cargo", "is_incremental": False,
     "values": {"durban": 21.50}, "unit": "per_MT", "notes": "Iron ore, coal, grain, etc."}
]

BERTH_DUES_ROWS = [
    {"tonnage_band": "All vessels", "is_incremental": False,
     "values": {"durban": 2.10}, "unit": "per_24h", "notes": ""}
]

RUNNING_ROWS = [
    {"tonnage_band": "All vessels", "is_incremental": False,
     "values": {"durban": 5800.00}, "unit": "per_service", "notes": ""}
]


def _j(rows) -> str:
    return json.dumps(rows)


# ── Test helpers ─────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n── {title} {'─' * (50 - len(title))}")


def ok(msg: str):
    print(f"  ✅ {msg}")


def fail(msg: str):
    print(f"  ❌ {msg}")
    raise AssertionError(msg)


def check(condition: bool, msg: str):
    if condition:
        ok(msg)
    else:
        fail(msg)


# ── Test 1: Rules Engine ─────────────────────────────────────────────────────

def test_rules_engine():
    from mcp_servers.rules_engine.server import (
        check_exemptions,
        determine_applicable_charges,
        get_vessel_charge_plan,
    )

    section("Test 1: Rules Engine")

    result = json.loads(determine_applicable_charges(json.dumps(SUDESTADA)))
    charges = result["applicable_charges"]
    modifiers = result["modifiers"]

    check("light_dues" in charges, "light_dues in applicable charges")
    check("vts" in charges, "vts in applicable charges")
    check("pilotage" in charges, "pilotage in applicable charges (GT=51300 > 500)")
    check("tug_assistance" in charges, "tug_assistance in applicable charges (GT=51300 > 3000)")
    check("cargo_dues" in charges, "cargo_dues in applicable charges (cargo_operation=True)")
    check("berth_dues" in charges, "berth_dues in applicable charges (berthing=True)")
    check("running_of_lines" in charges, "running_of_lines in applicable charges")
    check(modifiers.get("tug_count") == 3, f"tug_count=3 for GT=51300 (got {modifiers.get('tug_count')})")
    check(modifiers.get("pilotage_movements") == 2, "pilotage_movements=2")

    # Exemption checks
    no_cargo = {**SUDESTADA, "cargo_operation": False}
    exempt = json.loads(check_exemptions(json.dumps(no_cargo), "cargo_dues"))
    check(exempt["exempt"] is True, "cargo_dues exempt when cargo_operation=False")

    small = {**SUDESTADA, "gt": 400}
    exempt2 = json.loads(check_exemptions(json.dumps(small), "pilotage"))
    check(exempt2["exempt"] is True, "pilotage exempt for GT=400 (≤500)")

    plan = json.loads(get_vessel_charge_plan(json.dumps(SUDESTADA)))
    non_skipped = [s for s in plan["charge_plan"] if not s["step"].startswith("SKIP")]
    check(len(non_skipped) >= 7, f"≥7 non-exempt charges in plan (got {len(non_skipped)})")

    print(f"  Applicable charges: {charges}")
    print(f"  Tug count: {modifiers.get('tug_count')}")


# ── Test 2: Vessel Server ────────────────────────────────────────────────────

async def test_vessel_server():
    from mcp_servers.vessel.server import mcp as vessel_mcp

    section("Test 2: Vessel Server")

    reg = await vessel_mcp.call_tool("register_vessel", {"vessel_profile_json": json.dumps(SUDESTADA)})
    reg_data = json.loads(reg.content[0].text)
    check(reg_data["status"] == "registered", "vessel registered")
    check(reg_data["profile"]["name"] == "SUDESTADA", "vessel name normalised to SUDESTADA")
    check(reg_data["profile"]["vessel_type"] == "bulk_carrier", "vessel_type preserved")

    got = await vessel_mcp.call_tool("get_vessel", {"vessel_name": "sudestada"})
    got_data = json.loads(got.content[0].text)
    check(got_data["gt"] == 51300.0, "GT retrieved correctly")

    cls = await vessel_mcp.call_tool("classify_vessel_for_tariff", {"vessel_profile_json": json.dumps(SUDESTADA)})
    cls_data = json.loads(cls.content[0].text)
    check(cls_data["gt_category"] == "very_large", "GT category = very_large (51300)")
    check(cls_data["cargo_class"] == "bulk", "cargo_class = bulk for iron_ore")
    check(cls_data["tariff_notes"]["default_tug_count"] == 3, "default tug count = 3")


# ── Test 3: Calculator ───────────────────────────────────────────────────────

async def test_calculator():
    from mcp_servers.calculator.server import mcp as calc_mcp

    section("Test 3: Calculator — arithmetic verification")

    # Light dues: (51300 / 100) × 117.08 = 60062.04
    r = await calc_mcp.call_tool("calculate_light_dues", {
        "rows_json": _j(LIGHT_DUES_ROWS), "gt": 51300.0, "port_name": "durban"
    })
    d = json.loads(r.content[0].text)
    check(abs(d["charge_zar"] - 60062.04) < 0.01,
          f"Light dues R60,062.04 (got R{d['charge_zar']})")

    # VTS: 51300 × 0.65 = 33345.00
    r = await calc_mcp.call_tool("calculate_vts", {
        "rows_json": _j(VTS_ROWS), "gt": 51300.0, "port_name": "durban"
    })
    d = json.loads(r.content[0].text)
    check(abs(d["charge_zar"] - 33345.00) < 0.01,
          f"VTS R33,345.00 (got R{d['charge_zar']})")

    # Pilotage: base=18608.61 + ceil((51300-50000)/100)×9.72 = 18608.61 + 13×9.72 = 18734.97
    #           × 2 movements = 37469.94
    r = await calc_mcp.call_tool("calculate_pilotage", {
        "rows_json": _j(PILOTAGE_ROWS), "gt": 51300.0, "port_name": "durban", "movements": 2
    })
    d = json.loads(r.content[0].text)
    expected_pilot = (18608.61 + 13 * 9.72) * 2
    check(abs(d["charge_zar"] - expected_pilot) < 0.01,
          f"Pilotage R{expected_pilot:.2f} (got R{d['charge_zar']})")
    check(d["incremental_charge_per_movement"] > 0, "incremental charge detected")

    # Tug: base=39999.88 + ceil((51300-50000)/100)×19.94 = 39999.88 + 13×19.94 = 40259.10
    #      × 3 tugs × 2 movements = 241,554.60
    r = await calc_mcp.call_tool("calculate_tug_assistance", {
        "rows_json": _j(TUG_ROWS), "gt": 51300.0, "port_name": "durban",
        "num_tugs": 3, "movements": 2
    })
    d = json.loads(r.content[0].text)
    expected_tug = (39999.88 + 13 * 19.94) * 3 * 2
    check(abs(d["charge_zar"] - expected_tug) < 0.01,
          f"Tug assistance R{expected_tug:.2f} (got R{d['charge_zar']})")

    # Port dues: 51300 × 3.50 = 179,550.00
    r = await calc_mcp.call_tool("calculate_port_dues", {
        "rows_json": _j(PORT_DUES_ROWS), "gt": 51300.0, "port_name": "durban"
    })
    d = json.loads(r.content[0].text)
    check(abs(d["charge_zar"] - 179550.00) < 0.01,
          f"Port dues R179,550.00 (got R{d['charge_zar']})")

    # Cargo dues: 75000 × 21.50 = 1,612,500.00
    r = await calc_mcp.call_tool("calculate_cargo_dues", {
        "rows_json": _j(CARGO_DUES_ROWS), "metric_tonnes": 75000,
        "cargo_type": "iron_ore", "port_name": "durban"
    })
    d = json.loads(r.content[0].text)
    check(abs(d["charge_zar"] - 1_612_500.00) < 0.01,
          f"Cargo dues R1,612,500.00 (got R{d['charge_zar']})")

    # Berth dues: 51300 × 2.10 × (48/24) = 51300 × 2.10 × 2 = 215,460.00
    r = await calc_mcp.call_tool("calculate_berth_dues", {
        "rows_json": _j(BERTH_DUES_ROWS), "gt": 51300.0,
        "port_name": "durban", "hours_alongside": 48
    })
    d = json.loads(r.content[0].text)
    check(abs(d["charge_zar"] - 215460.00) < 0.01,
          f"Berth dues R215,460.00 (got R{d['charge_zar']})")

    # Running of lines: 5800 × 2 = 11,600.00
    r = await calc_mcp.call_tool("calculate_running_of_lines", {
        "rows_json": _j(RUNNING_ROWS), "port_name": "durban", "num_services": 2
    })
    d = json.loads(r.content[0].text)
    check(abs(d["charge_zar"] - 11600.00) < 0.01,
          f"Running of lines R11,600.00 (got R{d['charge_zar']})")

    # Aggregate test
    charges_list = [
        {"charge_type": "light_dues", "port": "durban", "charge_zar": 60062.04, "formula": "..."},
        {"charge_type": "vts", "port": "durban", "charge_zar": 33345.00, "formula": "..."},
    ]
    r = await calc_mcp.call_tool("aggregate_charges", {
        "charge_results_json": json.dumps(charges_list)
    })
    d = json.loads(r.content[0].text)
    check(abs(d["total_zar"] - 93407.04) < 0.01,
          f"Aggregate R93,407.04 (got R{d['total_zar']})")

    print(f"\n  Summary of charges (using synthetic rates):")
    summary = {
        "light_dues": 60062.04,
        "vts": 33345.00,
        "pilotage": round(expected_pilot, 2),
        "tug_assistance": round(expected_tug, 2),
        "port_dues": 179550.00,
        "cargo_dues": 1_612_500.00,
        "berth_dues": 215460.00,
        "running_of_lines": 11600.00,
    }
    total = sum(summary.values())
    for k, v in summary.items():
        print(f"    {k:<22} R{v:>14,.2f}")
    print(f"    {'TOTAL':<22} R{total:>14,.2f}")


# ── Test 4: Tariff RAG roundtrip ─────────────────────────────────────────────

def test_tariff_rag_store():
    from knowledge_store import tariff_store

    section("Test 4: Tariff Store roundtrip")

    # Write a synthetic table
    from ingestion.vision_extractor import VisionTableResult

    synthetic_result = VisionTableResult(
        page_number=99,
        tables=[{
            "charge_type": "test_charge",
            "section": "0.0",
            "description": "Test charge for unit testing",
            "ports_covered": ["durban"],
            "rows": [{"tonnage_band": "All vessels", "is_incremental": False,
                      "values": {"durban": 999.99}, "unit": "per_GT", "notes": ""}],
            "general_conditions": "",
        }],
        confident=True,
        flagged=False,
    )

    saved = tariff_store.save_tables("_test_port", [synthetic_result], {})
    check(saved == 1, "1 table saved to tariff store")

    loaded = tariff_store.load_table("_test_port", "test_charge")
    check(loaded is not None, "table loaded from tariff store")
    check(loaded["rows"][0]["values"]["durban"] == 999.99, "value roundtrips correctly")

    charges = tariff_store.list_charge_types("_test_port")
    check("test_charge" in charges, "list_charge_types includes test_charge")

    # Cleanup
    import shutil
    from config import TARIFF_STORE_DIR
    test_dir = TARIFF_STORE_DIR / "_test_port"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    ok("test data cleaned up")


# ── Runner ───────────────────────────────────────────────────────────────────

async def run_async_tests():
    await test_vessel_server()
    await test_calculator()


if __name__ == "__main__":
    print("=" * 60)
    print("  PORT TARIFF AI — Layer 2 MCP Server Tests")
    print("=" * 60)

    try:
        test_rules_engine()
        asyncio.run(run_async_tests())
        test_tariff_rag_store()

        print("\n" + "=" * 60)
        print("  All Layer 2 tests passed ✅")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
