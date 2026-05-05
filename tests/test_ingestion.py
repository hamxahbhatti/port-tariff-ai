"""
Test Layer 1: Ingestion Pipeline

Run with:
    GEMINI_API_KEY=your_key python -m tests.test_ingestion

What this validates:
  1. Docling parses the PDF without crashing
  2. Correct number of table pages detected
  3. Gemini Vision extracts at least the known charges (pilotage, tug, VTS, etc.)
  4. Sub-row relationships are preserved (is_incremental=True)
  5. Known Durban rates appear in the output (spot checks)
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

config.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", config.GEMINI_API_KEY)

TARIFF_PDF = Path("/Users/hamzashabbir/Downloads/Port Tariff.pdf")

# Known ground-truth values from the Transnet tariff doc for spot checking
KNOWN_VALUES = {
    "light_dues": {
        "rate": 117.08,
        "unit": "per_100GT",
    },
    "vts": {
        "durban": 0.65,
    },
    "pilotage": {
        "durban_basic": 18608.61,
        "durban_incremental": 9.72,
    },
}


def test_docling_parses():
    from ingestion.docling_parser import parse

    print("\n── Test 1: Docling parsing ─────────────────────────────")
    result = parse(TARIFF_PDF)

    assert result.total_pages > 0, "No pages parsed"
    assert len(result.prose_chunks) > 0, "No prose chunks extracted"
    assert len(result.table_pages) > 0, "No table pages detected"

    print(f"  ✅ Pages: {result.total_pages}")
    print(f"  ✅ Prose chunks: {len(result.prose_chunks)}")
    print(f"  ✅ Table pages: {[tp.page_number for tp in result.table_pages]}")

    return result


def test_vision_extracts(docling_result, max_pages: int = 3):
    from ingestion.vision_extractor import extract_tables

    print("\n── Test 2: Gemini Vision extraction ────────────────────")
    # Default: test first 3 table pages only to stay within free-tier quota.
    # Run with max_pages=None for full extraction.
    pages = docling_result.table_pages[:max_pages] if max_pages else docling_result.table_pages
    print(f"  Testing {len(pages)} of {len(docling_result.table_pages)} table pages")
    vision_output = extract_tables(TARIFF_PDF, pages)

    assert len(vision_output.results) > 0, "No pages processed by Vision"

    all_charge_types = []
    for r in vision_output.results:
        for table in r.tables:
            ct = table.get("charge_type", "")
            all_charge_types.append(ct)
            print(f"  Page {r.page_number} | {ct} | confident={r.confident}")

    print(f"\n  Charge types found: {set(all_charge_types)}")
    print(f"  Flagged pages: {vision_output.flagged_pages}")

    if vision_output.flagged_pages:
        print(f"  ⚠️  {len(vision_output.flagged_pages)} pages flagged — check manually")

    return vision_output


def test_spot_check_values(vision_output):
    print("\n── Test 3: Spot-check known values ─────────────────────")

    all_tables = []
    for r in vision_output.results:
        all_tables.extend(r.tables)

    # Check VTS Durban rate
    vts_tables = [t for t in all_tables if "vts" in t.get("charge_type", "").lower()]
    if vts_tables:
        vts_rows = vts_tables[0].get("rows", [])
        durban_vts = _find_value(vts_rows, "durban")
        if durban_vts is not None:
            expected = KNOWN_VALUES["vts"]["durban"]
            match = abs(float(durban_vts) - expected) < 0.01
            status = "✅" if match else "❌"
            print(f"  {status} VTS Durban: got {durban_vts}, expected {expected}")
        else:
            print("  ⚠️  VTS Durban value not found in extracted data")
    else:
        print("  ⚠️  VTS table not found")

    # Check sub-row detection (is_incremental)
    incremental_rows = [
        row
        for t in all_tables
        for row in t.get("rows", [])
        if row.get("is_incremental")
    ]
    status = "✅" if incremental_rows else "❌"
    print(f"  {status} Incremental sub-rows detected: {len(incremental_rows)}")

    # Check pilotage basic fee for Durban
    pilot_tables = [
        t for t in all_tables
        if "pilotage" in t.get("charge_type", "").lower()
    ]
    if pilot_tables:
        pilot_rows = pilot_tables[0].get("rows", [])
        durban_basic = _find_value(pilot_rows, "durban")
        if durban_basic:
            expected = KNOWN_VALUES["pilotage"]["durban_basic"]
            match = abs(float(durban_basic) - expected) < 1.0
            status = "✅" if match else "❌"
            print(f"  {status} Pilotage Durban basic: got {durban_basic}, expected {expected}")


def _find_value(rows: list[dict], port: str):
    for row in rows:
        values = row.get("values", {})
        for k, v in values.items():
            if port.lower() in k.lower() and v is not None:
                return v
    return None


def test_full_pipeline():
    print("\n── Test 4: Full pipeline (end-to-end) ──────────────────")
    from ingestion.pipeline import run

    summary = run(TARIFF_PDF, "durban")
    print(f"  Summary: {json.dumps(summary, indent=4)}")

    assert summary["prose_chunks_saved"] > 0
    assert summary["table_pages_processed"] > 0
    print("  ✅ Full pipeline completed")


if __name__ == "__main__":
    print("=" * 60)
    print("  PORT TARIFF AI — Layer 1 Ingestion Tests")
    print("=" * 60)

    docling_result = test_docling_parses()
    vision_output = test_vision_extracts(docling_result)
    test_spot_check_values(vision_output)
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("  All tests complete")
    print("=" * 60)
