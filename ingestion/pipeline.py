"""
Main ingestion pipeline orchestrator.

Flow:
  1. Docling parses full PDF → prose chunks + table page list
  2. Gemini Vision (double-pass) extracts tables from flagged pages
  3. MinerU handles any pages Docling failed on
  4. Results written to:
     - knowledge_store/tariff_store.py  (structured JSON per charge/port)
     - knowledge_store/vector_store.py  (ChromaDB for prose + descriptions)
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
        port_name: Canonical port name (e.g. 'durban'). Used as the
                   storage key in the tariff JSON store.

    Returns:
        Summary dict with counts of what was extracted.
    """
    pdf_path = Path(pdf_path)
    port_name = port_name.lower().strip()

    logger.info(f"Pipeline: starting ingestion for '{port_name}' — {pdf_path.name}")

    # ── Step 1: Docling ────────────────────────────────────────────────────
    logger.info("Step 1/3: Docling — layout parsing")
    try:
        docling_result = docling_parser.parse(pdf_path)
    except Exception as e:
        logger.error(f"Docling failed entirely: {e}")
        raise

    logger.info(
        f"Docling done: {len(docling_result.prose_chunks)} prose chunks, "
        f"{len(docling_result.table_pages)} table pages"
    )

    # ── Step 2: Gemini Vision on table pages ──────────────────────────────
    logger.info("Step 2/3: Gemini Vision — table extraction (double-pass)")

    if not docling_result.table_pages:
        logger.warning("No table pages found by Docling — check the PDF")
        vision_output = vision_extractor.VisionExtractionOutput()
    else:
        vision_output = vision_extractor.extract_tables(
            pdf_path,
            docling_result.table_pages,
        )

    logger.info(
        f"Vision done: {len(vision_output.results)} pages processed, "
        f"{len(vision_output.flagged_pages)} flagged"
    )

    # ── Step 3: MinerU backup for failed pages ─────────────────────────────
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

    # ── Step 4: Save to knowledge store ───────────────────────────────────
    logger.info("Saving to knowledge store")

    from knowledge_store.tariff_store import save_tables
    from knowledge_store.vector_store import save_prose_chunks

    tables_saved = save_tables(port_name, vision_output.results, mineru_results)
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
