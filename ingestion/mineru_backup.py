"""
MinerU backup parser.

Invoked only when Docling fails to parse a page (exception or empty output).
MinerU outputs tables as HTML which preserves structure better than plain text.
"""

from __future__ import annotations

import logging
import subprocess
import json
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        result = subprocess.run(
            ["mineru", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def parse_page(pdf_path: str | Path, page_number: int) -> str:
    """
    Run MinerU on a single page and return extracted content as string.
    page_number is 1-based.
    """
    pdf_path = Path(pdf_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        cmd = [
            "mineru",
            "-p", str(pdf_path),
            "-o", str(output_dir),
            "--page-start", str(page_number),
            "--page-end", str(page_number),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                logger.error(f"MinerU failed on page {page_number}: {result.stderr}")
                return ""

            md_files = list(output_dir.rglob("*.md"))
            if not md_files:
                logger.warning(f"MinerU: no output for page {page_number}")
                return ""

            return md_files[0].read_text(encoding="utf-8")

        except subprocess.TimeoutExpired:
            logger.error(f"MinerU: timeout on page {page_number}")
            return ""
        except Exception as e:
            logger.error(f"MinerU: unexpected error on page {page_number}: {e}")
            return ""
