"""
Main ingestion pipeline orchestrator.

Flow:
  1. Docling parses full PDF → prose chunks + table page list (local, ~60s)
  2. Smart routing per table page:
       a. Docling markdown looks clean → batch Gemini TEXT call (1 API call)
       b. Docling markdown looks sparse/complex → Gemini Vision double-pass
     Typically only 3-8 of 23 pages need Vision.
  3. MinerU handles any pages Vision still flags
  4. Results written progressively (per page) to:
       knowledge_store/tariff_store/  (JSON per charge type)
       knowledge_store/vector_store/  (ChromaDB for prose + descriptions)

Typical runtime: ~2-4 minutes (vs hours with Vision-on-every-page).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from ingestion import docling_parser, vision_extractor, mineru_backup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def run(pdf_path: str | Path, port_name: str) -> dict:
    """
    Full ingestion pipeline for one port tariff PDF.

    Args:
        pdf_path:  Path to the tariff PDF.
        port_name: Canonical port name (e.g. 'durban').

    Returns:
        Summary dict with counts of what was extracted.
    """
    pdf_path = Path(pdf_path)
    port_name = port_name.lower().strip()

    logger.info(f"Pipeline: starting ingestion for '{port_name}' — {pdf_path.name}")

    from knowledge_store.tariff_store import save_tables
    from knowledge_store.vector_store import save_prose_chunks

    # ── Step 1: Docling ────────────────────────────────────────────────────
    logger.info("Step 1/3: Docling — layout + table structure parsing (local)")
    try:
        docling_result = docling_parser.parse(pdf_path)
    except Exception as e:
        logger.error(f"Docling failed: {e}")
        raise

    logger.info(
        f"Docling done: {len(docling_result.prose_chunks)} prose chunks, "
        f"{len(docling_result.table_pages)} table pages"
    )

    # ── Step 2: Smart extraction (text batch + Vision only where needed) ───
    logger.info("Step 2/3: Smart extraction — text batch + Vision on complex pages")

    if not docling_result.table_pages:
        logger.warning("No table pages found by Docling — check the PDF")
        vision_output = vision_extractor.VisionExtractionOutput()
    else:
        vision_output = vision_extractor.extract_tables(
            pdf_path,
            docling_result.table_pages,
        )

    logger.info(
        f"Extraction done: {len(vision_output.results)} pages, "
        f"{len(vision_output.flagged_pages)} flagged"
    )

    # ── Step 3: MinerU backup for flagged pages ────────────────────────────
    logger.info("Step 3/3: MinerU backup (flagged pages only)")

    mineru_results: dict[int, str] = {}

    if vision_output.flagged_pages:
        if mineru_backup.is_available():
            for page_no in vision_output.flagged_pages:
                logger.info(f"MinerU: re-processing page {page_no}")
                content = mineru_backup.parse_page(pdf_path, page_no)
                if content:
                    mineru_results[page_no] = content
        else:
            logger.warning(
                "MinerU not installed — flagged pages will remain unverified. "
                "Install with: pip install mineru"
            )

    # ── Step 4: Save progressively (per page) ─────────────────────────────
    logger.info("Saving to knowledge store (progressive)")

    tables_saved = 0
    for result in vision_output.results:
        saved = save_tables(port_name, [result], mineru_results)
        tables_saved += saved
        if saved:
            logger.info(f"Saved {saved} table(s) from page {result.page_number}")

    chunks_saved = save_prose_chunks(port_name, docling_result.prose_chunks)

    summary = {
        "port": port_name,
        "pdf": pdf_path.name,
        "total_pages": docling_result.total_pages,
        "prose_chunks_saved": chunks_saved,
        "table_pages_processed": len(vision_output.results),
        "tables_extracted": tables_saved,
        "flagged_pages": vision_output.flagged_pages,
        "mineru_recovered": list(mineru_results.keys()),
    }

    logger.info(f"Pipeline complete: {summary}")
    return summary


if __name__ == "__main__":
    import json
    import os

    pdf = Path.home() / "Downloads" / "Port Tariff.pdf"
    result = run(pdf, "durban")
    print(json.dumps(result, indent=2))
