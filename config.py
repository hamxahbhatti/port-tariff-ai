import os
from pathlib import Path

# Load .env from repo root (if present) so GEMINI_API_KEY works without exporting
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on shell env

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
TARIFF_STORE_DIR = DATA_DIR / "tariffs"
CHROMA_DIR = DATA_DIR / "chroma_db"

TARIFF_STORE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── Gemini ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_TEXT_MODEL = "models/gemini-2.5-flash"
GEMINI_VISION_MODEL = "models/gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

# ── Ingestion ──────────────────────────────────────────────────────────────
# DPI for PDF→image rendering (higher = more detail for VLM)
PDF_RENDER_DPI = 200

# How similar two VLM passes must be to be considered matching (0-1)
VLM_MATCH_THRESHOLD = 0.95

# Delay (seconds) between Vision API calls.
# Free tier (20 req/min): set to 7. Paid tier: set to 1 or 0.
# Override via env: VISION_DELAY=1
INTER_REQUEST_DELAY = float(os.getenv("VISION_DELAY", "2"))

VISION_CACHE_DIR = DATA_DIR / "vision_cache"
VISION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Max tokens per prose chunk stored in ChromaDB
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "port_tariff_rules"
