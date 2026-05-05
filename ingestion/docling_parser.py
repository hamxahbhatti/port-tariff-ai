"""
Primary PDF parser using Docling.

Responsibilities:
- Parse the full PDF with DocLayNet (layout) + TableFormer (table structure)
- Separate prose pages from table-containing pages
- Extract prose text directly (authoritative for rules/conditions)
- Return table page numbers + Docling's partial table text as VLM context
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

logger = logging.getLogger(__name__)


@dataclass
class ProseChunk:
    page_number: int
    section_heading: str
    text: str
    source: str = "docling"


@dataclass
class TablePage:
    page_number: int
    docling_context: str      # Docling's extracted text — used as VLM prompt context
    table_count: int


@dataclass
class DoclingResult:
    prose_chunks: list[ProseChunk] = field(default_factory=list)
    table_pages: list[TablePage] = field(default_factory=list)
    total_pages: int = 0
    failed_pages: list[int] = field(default_factory=list)


def parse(pdf_path: str | Path) -> DoclingResult:
    """
    Run Docling on the full PDF.

    Returns prose chunks (authoritative) and table page metadata
    (page numbers + context text for passing to Gemini Vision).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Docling: parsing {pdf_path.name}")

    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,      # TableFormer on
        do_ocr=False,                 # vector PDF — no OCR needed
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(str(pdf_path))
    doc = result.document

    output = DoclingResult(total_pages=_count_pages(doc))

    # ── Separate prose from table pages ────────────────────────────────────
    pages_with_tables: set[int] = set()

    for table in doc.tables:
        prov = table.prov
        if prov:
            pages_with_tables.update(p.page_no for p in prov)

    logger.info(
        f"Docling: {output.total_pages} pages total, "
        f"{len(pages_with_tables)} contain tables"
    )

    # ── Extract prose chunks from ALL pages ───────────────────────────────
    # Pages with tables also contain prose rules (conditions, surcharges).
    # We extract prose from everywhere — VLM handles the numeric table data.
    current_heading = "General"
    current_text_parts: list[str] = []
    current_page = 0

    for text_item in doc.texts:
        prov = text_item.prov
        page_no = prov[0].page_no if prov else 0

        # Skip items that are inside a table cell — VLM handles those
        if _is_table_cell_text(text_item, doc):
            continue

        label = str(getattr(text_item, "label", "")).lower()

        if "heading" in label or "title" in label or "section" in label:
            if current_text_parts:
                output.prose_chunks.append(
                    ProseChunk(
                        page_number=current_page,
                        section_heading=current_heading,
                        text=" ".join(current_text_parts).strip(),
                    )
                )
                current_text_parts = []
            current_heading = text_item.text.strip()
            current_page = page_no
        else:
            text = text_item.text.strip()
            if text:
                current_text_parts.append(text)
                current_page = page_no

    if current_text_parts:
        output.prose_chunks.append(
            ProseChunk(
                page_number=current_page,
                section_heading=current_heading,
                text=" ".join(current_text_parts).strip(),
            )
        )

    # ── Build TablePage entries with Docling context ───────────────────────
    for page_no in sorted(pages_with_tables):
        context_parts = []

        for text_item in doc.texts:
            prov = text_item.prov
            if prov and prov[0].page_no == page_no:
                context_parts.append(text_item.text.strip())

        for table in doc.tables:
            prov = table.prov
            if prov and any(p.page_no == page_no for p in prov):
                try:
                    # Pass doc to avoid deprecation warning
                    context_parts.append(table.export_to_markdown(doc=doc))
                except Exception:
                    pass

        table_count = sum(
            1 for t in doc.tables
            if t.prov and any(p.page_no == page_no for p in t.prov)
        )

        output.table_pages.append(
            TablePage(
                page_number=page_no,
                docling_context="\n".join(context_parts),
                table_count=table_count,
            )
        )

    logger.info(
        f"Docling: extracted {len(output.prose_chunks)} prose chunks, "
        f"{len(output.table_pages)} table pages"
    )

    return output


def _count_pages(doc) -> int:
    try:
        return len(doc.pages)
    except Exception:
        return 0


def _is_table_cell_text(text_item, doc) -> bool:
    """Return True if this text item lives inside a table cell."""
    try:
        label = str(getattr(text_item, "label", "")).lower()
        return "table" in label or "cell" in label
    except Exception:
        return False
