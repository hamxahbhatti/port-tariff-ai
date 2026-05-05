"""
Gemini Vision extractor for table pages.

Responsibilities:
- Render PDF pages to images (pymupdf)
- Run Gemini Vision twice per table page (double-pass)
- Compare outputs — if they match, store as confident result
- If mismatch, flag the page for review
- Docling's context text is included in the prompt to ground the VLM
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf
import google.generativeai as genai
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from ingestion.docling_parser import TablePage

logger = logging.getLogger(__name__)

genai.configure(api_key=config.GEMINI_API_KEY)

TABLE_EXTRACTION_PROMPT = """You are a specialist in extracting structured data from port tariff documents.

Below is context text that another parser extracted from this page (may be incomplete or have ordering errors):
---
{docling_context}
---

Now look carefully at the page image above.

Extract ALL tables from this page as a JSON array. For each table follow this schema exactly:

{{
  "charge_type": "string (e.g. pilotage, tug_assistance, port_dues, vts, light_dues, cargo_dues)",
  "section": "string (e.g. '3.3', '4.1')",
  "description": "string (natural language description of what this table covers)",
  "ports_covered": ["list of port names as column headers, e.g. Durban, Cape Town"],
  "rows": [
    {{
      "tonnage_band": "string (e.g. '10000-50000', 'All vessels', 'Up to 2000')",
      "is_incremental": false,
      "parent_band": null,
      "values": {{
        "durban": 38494.51,
        "cape_town": 26938.00
      }},
      "unit": "string (e.g. 'per_100GT', 'per_GT', 'per_service', 'per_MT')",
      "notes": "string (any conditions or footnotes on this row)"
    }}
  ],
  "general_conditions": "string (any conditions that apply to the whole table)"
}}

CRITICAL RULES:
1. If a row starts with "Plus" or "plus per" or is visually indented — it is an INCREMENTAL sub-row.
   Set is_incremental=true and parent_band to the tonnage band of the row directly above it.
2. Preserve exact numeric values — do not round or estimate.
3. Match each value to its correct port column — check column headers carefully.
4. If a cell says "on application" or "-", use null for that value.
5. Return ONLY valid JSON. No markdown fences, no explanation text.
"""


@dataclass
class VisionTableResult:
    page_number: int
    tables: list[dict]
    confident: bool          # True if both passes matched
    flagged: bool = False    # True if passes disagreed — needs review
    pass1_raw: str = ""
    pass2_raw: str = ""


@dataclass
class VisionExtractionOutput:
    results: list[VisionTableResult] = field(default_factory=list)
    flagged_pages: list[int] = field(default_factory=list)


def extract_tables(
    pdf_path: str | Path,
    table_pages: list[TablePage],
) -> VisionExtractionOutput:
    """
    Run Gemini Vision (double-pass) on all flagged table pages.
    Returns structured table JSON per page.
    """
    pdf_path = Path(pdf_path)
    output = VisionExtractionOutput()

    pdf_doc = fitz.open(str(pdf_path))

    for table_page in table_pages:
        page_no = table_page.page_number
        logger.info(f"Vision: processing page {page_no}")

        try:
            image = _render_page(pdf_doc, page_no)
            result = _double_pass_extract(image, table_page)
            output.results.append(result)

            if result.flagged:
                output.flagged_pages.append(page_no)
                logger.warning(
                    f"Vision: page {page_no} flagged — passes disagreed. "
                    "Manual review recommended."
                )
        except Exception as e:
            logger.error(f"Vision: failed on page {page_no}: {e}")
            output.results.append(
                VisionTableResult(
                    page_number=page_no,
                    tables=[],
                    confident=False,
                    flagged=True,
                )
            )
            output.flagged_pages.append(page_no)

    pdf_doc.close()

    logger.info(
        f"Vision: processed {len(output.results)} pages, "
        f"{len(output.flagged_pages)} flagged"
    )

    return output


def _render_page(pdf_doc: fitz.Document, page_no: int) -> Image.Image:
    # Docling uses 1-based page numbers; pymupdf uses 0-based
    page = pdf_doc[page_no - 1]
    mat = fitz.Matrix(config.PDF_RENDER_DPI / 72, config.PDF_RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _double_pass_extract(
    image: Image.Image,
    table_page: TablePage,
) -> VisionTableResult:
    prompt = TABLE_EXTRACTION_PROMPT.format(
        docling_context=table_page.docling_context[:3000]
    )

    pass1_raw = _call_vision(image, prompt)
    pass2_raw = _call_vision(image, prompt)

    tables1 = _parse_json(pass1_raw)
    tables2 = _parse_json(pass2_raw)

    confident, tables = _compare_passes(tables1, tables2)

    return VisionTableResult(
        page_number=table_page.page_number,
        tables=tables,
        confident=confident,
        flagged=not confident,
        pass1_raw=pass1_raw,
        pass2_raw=pass2_raw,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_vision(image: Image.Image, prompt: str) -> str:
    model = genai.GenerativeModel(config.GEMINI_VISION_MODEL)
    response = model.generate_content([image, prompt])
    return response.text.strip()


def _parse_json(raw: str) -> list[dict]:
    # Strip markdown fences if model added them
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"Vision: JSON parse failed: {e}")
        return []


def _compare_passes(
    tables1: list[dict],
    tables2: list[dict],
) -> tuple[bool, list[dict]]:
    """
    Compare two VLM extraction passes.

    Strategy: extract all numeric values from both passes and compute
    match ratio. If >= threshold, use pass1 as the result.
    """
    nums1 = _extract_numbers(tables1)
    nums2 = _extract_numbers(tables2)

    if not nums1 and not nums2:
        # No numbers — likely a text-only page flagged by mistake
        return True, tables1

    if not nums1 or not nums2:
        return False, tables1

    matches = sum(1 for a, b in zip(nums1, nums2) if abs(a - b) < 0.01)
    ratio = matches / max(len(nums1), len(nums2))

    confident = ratio >= config.VLM_MATCH_THRESHOLD
    return confident, tables1


def _extract_numbers(tables: list[dict]) -> list[float]:
    numbers = []
    raw = json.dumps(tables)
    for match in re.findall(r"\d+(?:\.\d+)?", raw):
        try:
            numbers.append(float(match))
        except ValueError:
            pass
    return numbers
