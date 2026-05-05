import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
TARIFF_STORE_DIR = DATA_DIR / "tariffs"
CHROMA_DIR = DATA_DIR / "chroma_db"

TARIFF_STORE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── Gemini ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_TEXT_MODEL = "gemini-2.0-flash"
GEMINI_VISION_MODEL = "gemini-2.0-flash"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

# ── Ingestion ──────────────────────────────────────────────────────────────
# DPI for PDF→image rendering (higher = more detail for VLM)
PDF_RENDER_DPI = 200

# How similar two VLM passes must be to be considered matching (0-1)
VLM_MATCH_THRESHOLD = 0.95

# Max tokens per prose chunk stored in ChromaDB
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "port_tariff_rules"
