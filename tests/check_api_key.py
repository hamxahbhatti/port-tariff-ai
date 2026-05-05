"""
Quick API key validator. Run this first after getting a new key.
Uses one text call + one vision call on a single page — minimal quota usage.

Usage:
    GEMINI_API_KEY=your_new_key python -m tests.check_api_key
"""

import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from google import genai
from google.genai import types
from PIL import Image

TARIFF_PDF = Path("/Users/hamzashabbir/Downloads/Port Tariff.pdf")
API_KEY = os.getenv("GEMINI_API_KEY", "")


def check_text():
    print("── Text call (1 request) ───────────────────────────────")
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=["Reply with exactly: KEY_OK"],
    )
    result = response.text.strip()
    ok = "KEY_OK" in result
    print(f"  {'✅' if ok else '❌'} Response: {result}")
    return ok


def check_vision():
    print("── Vision call on page 5 (1 request) ──────────────────")
    client = genai.Client(api_key=API_KEY)

    pdf_doc = fitz.open(str(TARIFF_PDF))
    page = pdf_doc[4]  # page 5, 0-based
    mat = fitz.Matrix(150 / 72, 150 / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pdf_doc.close()

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
            "What is the first charge type mentioned in this table? Reply in 5 words or less.",
        ],
    )
    result = response.text.strip()
    print(f"  ✅ Vision response: {result}")
    return True


if __name__ == "__main__":
    if not API_KEY:
        print("❌ GEMINI_API_KEY not set")
        sys.exit(1)

    print(f"Testing key: {API_KEY[:12]}...{API_KEY[-4:]}\n")

    text_ok = check_text()
    if text_ok:
        vision_ok = check_vision()
        if vision_ok:
            print("\n✅ Key is valid and vision works — safe to run full test")
            print("   Run: GEMINI_API_KEY=your_key python -m tests.test_ingestion")
    else:
        print("\n❌ Key failed — get a new one from https://aistudio.google.com/apikey")
        sys.exit(1)
