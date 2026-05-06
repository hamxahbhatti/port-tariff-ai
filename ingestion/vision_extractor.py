"""
Hybrid table extractor: Docling-first, Vision-only-when-needed.

Design:
  1. Assess each table page's Docling output quality.
  2. If Docling extracted clean, number-rich markdown → batch-convert via a
     single Gemini TEXT call (fast, cheap, ~5 seconds total).
  3. Only pages where Docling's output looks sparse or structurally confused
     (merged cells, rotated headers, etc.) go to Gemini Vision (double-pass).

This means Vision is called on ~3-8 pages out of 23, not all 23.
Typical runtime: Docling text batch (~5s) + Vision on flagged pages (~1-2 min).

Rate limiting: still handled on the Vision path.
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
from google.genai.errors import ClientError, ServerError
from PIL import Image

import config
from ingestion.docling_parser import TablePage

logger = logging.getLogger(__name__)

_client: genai.Client | None = None
_active_model: str | None = None  # sticky fallback once Vision 503s persistently
_FALLBACK_MODEL = "models/gemini-2.5-flash-lite"


# ── Shared extraction schema (used by both text and Vision prompts) ────────

_SCHEMA_BLOCK = """
Extract ALL tables as a JSON array. For each table use this exact schema:

{
  "charge_type": "string (e.g. pilotage, tug_assistance, port_dues, vts, light_dues, cargo_dues, berthing, running_of_lines, berth_dues)",
  "section": "string (e.g. '3.3', '4.1')",
  "description": "string (natural language description of what this table covers)",
  "table_orientation": "string — 'ports_as_columns' if port names are column headers, 'ports_as_rows' if each row is a different port",
  "ports_covered": ["list of port names found in this table"],
  "rows": [
    {
      "port": "string — port name IF table_orientation is ports_as_rows, else null",
      "tonnage_band": "string (e.g. '10000-50000', 'All vessels', 'Up to 2000') — null if row is a port-level rate",
      "is_incremental": false,
      "parent_band": null,
      "values": {
        "durban": 38494.51,
        "richards_bay": 39999.88,
        "east_london": 27956.91,
        "port_elizabeth": 35861.45,
        "cape_town": 26938.00,
        "saldanha": 44977.00
      },
      "unit": "string (e.g. 'per_100GT', 'per_GT', 'per_service', 'per_MT', 'per_24h', 'per_port_call')",
      "notes": "string (any conditions or footnotes)"
    }
  ],
  "general_conditions": "string (conditions applying to the whole table)"
}

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

TEXT_EXTRACTION_PROMPT = """You are a specialist in South African port tariff documents.

Below is the text and table content extracted from several pages of a port tariff PDF.
Each page is separated by a page marker.

{pages_context}

""" + _SCHEMA_BLOCK

VISION_EXTRACTION_PROMPT = """You are a specialist in extracting structured data from South African port tariff documents.

Below is context text that another parser extracted from this page (may be incomplete):
---
{docling_context}
---

Now look carefully at the page image.

""" + _SCHEMA_BLOCK


# ── Data classes ──────────────────────────────────────────────────────────

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


# ── Client ────────────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


# ── Main entry point ──────────────────────────────────────────────────────

def extract_tables(
    pdf_path: str | Path,
    table_pages: list[TablePage],
) -> VisionExtractionOutput:
    """
    Hybrid extraction:
    - Pages with clean Docling markdown → single batched Gemini TEXT call
    - Pages where Docling looks incomplete → Gemini Vision (double-pass)
    """
    pdf_path = Path(pdf_path)
    output = VisionExtractionOutput()

    # ── Split pages by quality ────────────────────────────────────────────
    text_pages: list[TablePage] = []
    vision_pages: list[TablePage] = []

    for tp in table_pages:
        if _is_docling_sufficient(tp):
            text_pages.append(tp)
        else:
            vision_pages.append(tp)

    logger.info(
        f"Routing: {len(text_pages)} pages → Gemini TEXT, "
        f"{len(vision_pages)} pages → Gemini Vision"
    )

    # ── Path A: batch text extraction ─────────────────────────────────────
    if text_pages:
        logger.info("Text extraction: sending batch to Gemini text API")
        text_results = _batch_text_extract(text_pages)
        output.results.extend(text_results)
        logger.info(f"Text extraction done: {len(text_results)} results")

    # ── Path B: Vision for complex pages ──────────────────────────────────
    if vision_pages:
        logger.info(f"Vision extraction: {len(vision_pages)} complex pages")
        pdf_doc = fitz.open(str(pdf_path))

        for idx, table_page in enumerate(vision_pages):
            page_no = table_page.page_number
            logger.info(f"Vision: page {page_no} ({idx+1}/{len(vision_pages)})")

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

            if idx < len(vision_pages) - 1:
                time.sleep(config.INTER_REQUEST_DELAY)

        pdf_doc.close()

    # Sort results by page number
    output.results.sort(key=lambda r: r.page_number)

    logger.info(
        f"Extraction complete: {len(output.results)} pages processed, "
        f"{len(output.flagged_pages)} flagged"
    )
    return output


# ── Quality assessment ────────────────────────────────────────────────────

def _is_docling_sufficient(table_page: TablePage) -> bool:
    """
    Returns True if Docling's extracted markdown is clean enough to send
    directly to a text model.

    Returns False (→ route to Vision) when:
    - No markdown table structure at all (Docling missed it entirely)
    - Very few numeric values (merged-cell confusion → numbers lost)
    - High ratio of empty cells (sign of merged column/row headers)
    """
    context = table_page.docling_context

    # Must have markdown table delimiters
    if "|" not in context:
        logger.debug(f"Page {table_page.page_number}: no markdown table → Vision")
        return False

    # Count real numeric values (ignore year/section numbers like 2024, 3.3)
    numbers = [
        float(m)
        for m in re.findall(r"\b\d+(?:\.\d+)?\b", context)
        if float(m) > 10  # tariff rates are always > 10 (cents excluded intentionally)
    ]
    if len(numbers) < 4:
        logger.debug(
            f"Page {table_page.page_number}: only {len(numbers)} numbers → Vision"
        )
        return False

    # Check for excessive empty cells (merged-header confusion)
    pipe_count = context.count("|")
    empty_cell_count = len(re.findall(r"\|\s*\|", context))
    if pipe_count > 0 and (empty_cell_count / pipe_count) > 0.35:
        logger.debug(
            f"Page {table_page.page_number}: {empty_cell_count}/{pipe_count} "
            f"empty cells → Vision"
        )
        return False

    return True


# ── Text extraction (batch) ───────────────────────────────────────────────

def _batch_text_extract(table_pages: list[TablePage]) -> list[VisionTableResult]:
    """
    Send all clean Docling pages to Gemini in one text API call.
    Returns one VisionTableResult per page.
    """
    # Build a single context block with page markers
    pages_context = "\n\n".join(
        f"=== PAGE {tp.page_number} ===\n{tp.docling_context[:4000]}"
        for tp in table_pages
    )

    prompt = TEXT_EXTRACTION_PROMPT.replace("{pages_context}", pages_context)

    raw = _call_text(prompt)
    all_tables = _parse_json(raw)

    # Map extracted tables back to their source pages
    # Each table should have a charge_type; we do best-effort page attribution
    # by matching section/content keywords to page context
    results: list[VisionTableResult] = []
    used_table_indices: set[int] = set()

    for tp in table_pages:
        page_tables = []
        page_keywords = tp.docling_context.lower()

        for i, table in enumerate(all_tables):
            if i in used_table_indices:
                continue
            charge = table.get("charge_type", "").lower()
            section = table.get("section", "")
            desc = table.get("description", "").lower()

            # Match table to page by checking if charge/section appears in page context
            if (
                charge in page_keywords
                or section in page_keywords
                or any(w in page_keywords for w in desc.split() if len(w) > 5)
            ):
                page_tables.append(table)
                used_table_indices.add(i)

        results.append(
            VisionTableResult(
                page_number=tp.page_number,
                tables=page_tables,
                confident=True,
                flagged=False,
            )
        )

    # Any tables not matched to a page → attach to the first page as fallback
    unmatched = [t for i, t in enumerate(all_tables) if i not in used_table_indices]
    if unmatched and results:
        results[0].tables.extend(unmatched)
        logger.warning(
            f"Text extract: {len(unmatched)} tables could not be matched to a "
            f"specific page — attached to page {results[0].page_number}"
        )

    return results


def _call_text(prompt: str, max_attempts: int = 4) -> str:
    """
    Single Gemini text call. Falls back to flash-lite on persistent 503s
    (same sticky-fallback logic as Vision).
    """
    client = _get_client()
    last_err: Exception | None = None
    model = config.GEMINI_TEXT_MODEL
    consecutive_503 = 0

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt],
            )
            return response.text.strip()
        except ClientError as e:
            last_err = e
            consecutive_503 = 0
            retry_delay = _parse_retry_delay(e) + 5
            logger.warning(f"Text API: ClientError attempt {attempt+1} — sleeping {retry_delay:.0f}s")
            time.sleep(retry_delay)
        except ServerError as e:
            last_err = e
            consecutive_503 += 1
            if consecutive_503 >= 1 and model != _FALLBACK_MODEL:
                model = _FALLBACK_MODEL
                consecutive_503 = 0
                logger.warning(f"Text API: 503 — switching to {_FALLBACK_MODEL}")
                time.sleep(3)
            else:
                wait = 15 * (attempt + 1)
                logger.warning(f"Text API: ServerError attempt {attempt+1} using {model} — sleeping {wait}s")
                time.sleep(wait)
        except Exception as e:
            last_err = e
            consecutive_503 = 0
            logger.warning(f"Text API: error attempt {attempt+1}: {e}")
            time.sleep(10)

    raise RuntimeError(f"Text API failed after {max_attempts} attempts: {last_err}")


# ── Vision extraction (complex pages only) ───────────────────────────────

def _render_page_bytes(pdf_doc: fitz.Document, page_no: int) -> bytes:
    """Render a PDF page to PNG bytes. pymupdf is 0-based; we use 1-based page_no."""
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
    """Two Vision calls on the same page; agree → confident, disagree → flagged."""
    prompt = VISION_EXTRACTION_PROMPT.replace(
        "{docling_context}", table_page.docling_context[:3000]
    )

    pass1_raw = _call_vision(image_bytes, prompt)
    time.sleep(config.INTER_REQUEST_DELAY)
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


def _call_vision(image_bytes: bytes, prompt: str, max_attempts: int = 4) -> str:
    """
    Vision API call with 429-aware retry and sticky 503 fallback to flash-lite.
    """
    global _active_model
    client = _get_client()
    last_err: Exception | None = None

    if _active_model is None:
        _active_model = config.GEMINI_VISION_MODEL
    model = _active_model
    consecutive_503 = 0

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            text = response.text
            if text is None:
                raise ValueError("Model returned empty response (safety filter or overload)")
            return text.strip()

        except ClientError as e:
            last_err = e
            consecutive_503 = 0
            retry_delay = _parse_retry_delay(e) + 5
            logger.warning(
                f"Vision: ClientError attempt {attempt+1}/{max_attempts} — "
                f"status {e.args[0] if e.args else 'unknown'}, sleeping {retry_delay:.0f}s"
            )
            time.sleep(retry_delay)

        except ServerError as e:
            last_err = e
            consecutive_503 += 1
            if consecutive_503 >= 1 and model != _FALLBACK_MODEL:
                _active_model = _FALLBACK_MODEL
                model = _FALLBACK_MODEL
                consecutive_503 = 0
                logger.warning(
                    f"Vision: 503 — permanently switching to {_FALLBACK_MODEL}"
                )
                time.sleep(3)
            else:
                wait = 15 * (attempt + 1)
                logger.warning(
                    f"Vision: ServerError attempt {attempt+1}/{max_attempts} "
                    f"using {model} — sleeping {wait}s"
                )
                time.sleep(wait)

        except Exception as e:
            last_err = e
            consecutive_503 = 0
            logger.warning(f"Vision: unexpected error attempt {attempt+1}: {e}")
            time.sleep(8)

    raise RuntimeError(f"Vision failed after {max_attempts} attempts: {last_err}")


# ── Helpers ───────────────────────────────────────────────────────────────

def _parse_retry_delay(err: ClientError) -> float:
    try:
        details = err.args[1] if len(err.args) > 1 else {}
        for detail in details.get("error", {}).get("details", []):
            if "retryDelay" in detail:
                return float(detail["retryDelay"].rstrip("s"))
    except Exception:
        pass
    return 30.0


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
        logger.warning(f"JSON parse failed: {e}\nRaw: {raw[:300]}")
        return []


def _compare_passes(
    tables1: list[dict],
    tables2: list[dict],
) -> tuple[bool, list[dict]]:
    """
    Compare numeric values from both Vision passes.
    Match ratio >= VLM_MATCH_THRESHOLD → confident, return pass1.
    """
    nums1 = _extract_numbers(tables1)
    nums2 = _extract_numbers(tables2)

    if not nums1 and not nums2:
        return True, tables1
    if not nums1 or not nums2:
        return False, tables1

    min_len = min(len(nums1), len(nums2))
    matches = sum(
        1 for a, b in zip(nums1[:min_len], nums2[:min_len])
        if abs(a - b) < 0.01
    )
    ratio = matches / max(len(nums1), len(nums2))
    return ratio >= config.VLM_MATCH_THRESHOLD, tables1


def _extract_numbers(tables: list[dict]) -> list[float]:
    numbers = []
    for match in re.findall(r"\b\d+(?:\.\d+)?\b", json.dumps(tables)):
        try:
            val = float(match)
            if val > 0.001:
                numbers.append(val)
        except ValueError:
            pass
    return numbers
