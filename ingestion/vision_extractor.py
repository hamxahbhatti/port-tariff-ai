"""
Gemini Vision extractor for table pages.

Responsibilities:
- Render PDF pages to images (pymupdf)
- Run Gemini Vision twice per table page (double-pass)
- Compare outputs — if they match, store as confident result
- If mismatch, flag the page for review
- Docling's context text is included in the prompt to ground the VLM

Rate limiting: free tier = 15 requests/min for gemini-2.0-flash.
With double-pass over N table pages = 2N requests.
A 4-second inter-request delay keeps us safely within limits.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf
from google import genai
from google.genai import types
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from ingestion.docling_parser import TablePage

logger = logging.getLogger(__name__)

_client: genai.Client | None = None

TABLE_EXTRACTION_PROMPT = """You are a specialist in extracting structured data from South African port tariff documents.

Below is context text that another parser extracted from this page (may be incomplete):
---
{docling_context}
---

Now look carefully at the page image.

Extract ALL tables from this page as a JSON array. For each table use this exact schema:

{{
  "charge_type": "string (e.g. pilotage, tug_assistance, port_dues, vts, light_dues, cargo_dues, berthing, running_of_lines, berth_dues)",
  "section": "string (e.g. '3.3', '4.1')",
  "description": "string (natural language description of what this table covers)",
  "table_orientation": "string — 'ports_as_columns' if port names are column headers, 'ports_as_rows' if each row is a different port",
  "ports_covered": ["list of port names found in this table"],
  "rows": [
    {{
      "port": "string — port name IF table_orientation is ports_as_rows, else null",
      "tonnage_band": "string (e.g. '10000-50000', 'All vessels', 'Up to 2000') — null if row is a port-level rate",
      "is_incremental": false,
      "parent_band": null,
      "values": {{
        "durban": 38494.51,
        "richards_bay": 39999.88,
        "east_london": 27956.91,
        "port_elizabeth": 35861.45,
        "cape_town": 26938.00,
        "saldanha": 44977.00
      }},
      "unit": "string (e.g. 'per_100GT', 'per_GT', 'per_service', 'per_MT', 'per_24h', 'per_port_call')",
      "notes": "string (any conditions or footnotes)"
    }}
  ],
  "general_conditions": "string (conditions applying to the whole table)"
}}

CRITICAL RULES:
1. INCREMENTAL ROWS: If a row starts with "Plus" or "plus per" or is visually indented below
   a tonnage band row — set is_incremental=true and parent_band to the tonnage band directly above.
2. PORTS AS ROWS (e.g. VTS table): If each row represents one port (e.g. "Port of Durban — R0.65/GT"),
   set table_orientation="ports_as_rows", set the "port" field on each row, and put the numeric
   value under its port key in "values".
3. PORTS AS COLUMNS (e.g. pilotage, tug tables): If ports are column headers and tonnage bands
   are rows, set table_orientation="ports_as_columns", leave "port" as null on each row, and
   map each column value to its port key in "values".
4. Preserve exact numeric values — do not round or estimate.
5. Port name keys in "values" must be lowercase with underscores (e.g. "richards_bay").
6. If a cell says "on application" or "-", use null.
7. Return ONLY valid JSON array. No markdown fences, no explanation.
"""


@dataclass
class VisionTableResult:
    page_number: int
    tables: list[dict]
    confident: bool
    flagged: bool = False
    pass1_raw: str = ""
    pass2_raw: str = ""


@dataclass
class VisionExtractionOutput:
    results: list[VisionTableResult] = field(default_factory=list)
    flagged_pages: list[int] = field(default_factory=list)


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def extract_tables(
    pdf_path: str | Path,
    table_pages: list[TablePage],
) -> VisionExtractionOutput:
    """
    Run Gemini Vision (double-pass) on all flagged table pages.
    Applies rate-limit delay between API calls.
    """
    pdf_path = Path(pdf_path)
    output = VisionExtractionOutput()

    pdf_doc = fitz.open(str(pdf_path))
    total = len(table_pages)

    for idx, table_page in enumerate(table_pages):
        page_no = table_page.page_number
        logger.info(f"Vision: page {page_no} ({idx+1}/{total})")

        try:
            image_bytes = _render_page_bytes(pdf_doc, page_no)
            result = _double_pass_extract(image_bytes, table_page)
            output.results.append(result)

            if result.flagged:
                output.flagged_pages.append(page_no)
                logger.warning(f"Vision: page {page_no} flagged — passes disagreed")

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

        # Rate limit: free tier = 15 req/min. Double-pass = 2 calls per page.
        # 4s gap keeps us at ~15 req/min across two passes.
        if idx < total - 1:
            time.sleep(4)

    pdf_doc.close()

    logger.info(
        f"Vision: processed {len(output.results)} pages, "
        f"{len(output.flagged_pages)} flagged"
    )
    return output


def _render_page_bytes(pdf_doc: fitz.Document, page_no: int) -> bytes:
    # pymupdf is 0-based; Docling page numbers are 1-based
    page = pdf_doc[page_no - 1]
    mat = fitz.Matrix(config.PDF_RENDER_DPI / 72, config.PDF_RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _double_pass_extract(
    image_bytes: bytes,
    table_page: TablePage,
) -> VisionTableResult:
    prompt = TABLE_EXTRACTION_PROMPT.format(
        docling_context=table_page.docling_context[:3000]
    )

    pass1_raw = _call_vision(image_bytes, prompt)
    time.sleep(4)  # gap between the two passes
    pass2_raw = _call_vision(image_bytes, prompt)

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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=10, max=60))
def _call_vision(image_bytes: bytes, prompt: str) -> str:
    client = _get_client()
    response = client.models.generate_content(
        model=config.GEMINI_VISION_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
    )
    return response.text.strip()


def _parse_json(raw: str) -> list[dict]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"Vision: JSON parse failed: {e}\nRaw: {raw[:300]}")
        return []


def _compare_passes(
    tables1: list[dict],
    tables2: list[dict],
) -> tuple[bool, list[dict]]:
    """
    Compare numeric values from both VLM passes.
    Match ratio >= VLM_MATCH_THRESHOLD → confident, use pass1.
    """
    nums1 = _extract_numbers(tables1)
    nums2 = _extract_numbers(tables2)

    if not nums1 and not nums2:
        return True, tables1

    if not nums1 or not nums2:
        return False, tables1

    # Compare element-wise up to the shorter list length
    min_len = min(len(nums1), len(nums2))
    matches = sum(
        1 for a, b in zip(nums1[:min_len], nums2[:min_len])
        if abs(a - b) < 0.01
    )
    ratio = matches / max(len(nums1), len(nums2))
    confident = ratio >= config.VLM_MATCH_THRESHOLD

    return confident, tables1


def _extract_numbers(tables: list[dict]) -> list[float]:
    numbers = []
    raw = json.dumps(tables)
    for match in re.findall(r"\b\d+(?:\.\d+)?\b", raw):
        try:
            val = float(match)
            # Skip numbers that are clearly metadata (page numbers, counts)
            if val > 0.001:
                numbers.append(val)
        except ValueError:
            pass
    return numbers
