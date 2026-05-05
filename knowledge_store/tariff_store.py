"""
Structured JSON tariff store.

Stores exact numeric tariff data per port in:
  data/tariffs/{port_name}/{charge_type}.json

This is the authoritative source for exact rate lookups — not ChromaDB.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import TARIFF_STORE_DIR
from ingestion.vision_extractor import VisionTableResult

logger = logging.getLogger(__name__)


def save_tables(
    port_name: str,
    vision_results: list[VisionTableResult],
    mineru_overrides: dict[int, str],
) -> int:
    """
    Persist extracted tables to JSON.

    For flagged pages where MinerU recovered content, that content is
    stored alongside the VLM output so both are available for review.

    Returns count of tables saved.
    """
    port_dir = TARIFF_STORE_DIR / port_name
    port_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for result in vision_results:
        page_no = result.page_number

        if not result.tables:
            logger.warning(f"Tariff store: no tables on page {page_no}")
            continue

        for table in result.tables:
            charge_type = _normalise_charge_type(
                table.get("charge_type", f"unknown_p{page_no}")
            )
            out_path = port_dir / f"{charge_type}.json"

            payload = {
                "port": port_name,
                "charge_type": charge_type,
                "section": table.get("section", ""),
                "description": table.get("description", ""),
                "ports_covered": table.get("ports_covered", []),
                "rows": table.get("rows", []),
                "general_conditions": table.get("general_conditions", ""),
                "extraction_meta": {
                    "page_number": page_no,
                    "confident": result.confident,
                    "flagged": result.flagged,
                    "mineru_backup_available": page_no in mineru_overrides,
                },
            }

            # If page was flagged and MinerU has content, attach it
            if page_no in mineru_overrides:
                payload["mineru_raw"] = mineru_overrides[page_no]

            # Merge if file already exists (multi-page tables)
            if out_path.exists():
                existing = json.loads(out_path.read_text())
                existing["rows"].extend(payload["rows"])
                out_path.write_text(json.dumps(existing, indent=2))
            else:
                out_path.write_text(json.dumps(payload, indent=2))

            logger.info(f"Tariff store: saved {charge_type} → {out_path}")
            saved += 1

    return saved


def load_table(port_name: str, charge_type: str) -> dict | None:
    path = TARIFF_STORE_DIR / port_name / f"{_normalise_charge_type(charge_type)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_charge_types(port_name: str) -> list[str]:
    port_dir = TARIFF_STORE_DIR / port_name
    if not port_dir.exists():
        return []
    return [p.stem for p in port_dir.glob("*.json")]


def list_ports() -> list[str]:
    if not TARIFF_STORE_DIR.exists():
        return []
    return [d.name for d in TARIFF_STORE_DIR.iterdir() if d.is_dir()]


def _normalise_charge_type(raw: str) -> str:
    return raw.lower().strip().replace(" ", "_").replace("-", "_")
