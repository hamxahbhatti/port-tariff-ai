# Port Tariff AI

AI-powered South African port tariff calculator. Accepts a vessel profile and automatically computes all applicable port dues (light dues, VTS, pilotage, tug assistance, port dues, cargo dues, berth dues, running of lines) for any Transnet-managed port.

**Test case:** MV SUDESTADA — Bulk Carrier, GT 51,300, LOA 229.2m, exporting Iron Ore at Port of Durban.

---

## Architecture

```
PDF (Port Tariff)
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  Layer 1: Ingestion Pipeline                          │
│  Docling (layout + tables) → Gemini Vision (double-   │
│  pass VLM) → JSON Tariff Store + ChromaDB vector store│
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  Layer 2: MCP Servers (FastMCP)                       │
│  tariff_rag  │ calculator │ rules_engine │ vessel      │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  Layer 3: LangGraph ReAct Agent                       │
│  Gemini orchestrates: identify → retrieve → compute   │
│  → validate → aggregate                               │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  Layer 4: FastAPI REST Endpoint                       │
│  POST /calculate (agent) │ POST /calculate/quick      │
└───────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| PDF parsing | Docling (DocLayNet + TableFormer) | Best layout + table structure on complex PDFs |
| Table extraction | Gemini Vision double-pass | Tables are complex; VLM outperforms pdfplumber on rate tables with merged cells |
| Rate storage | JSON files per port/charge | Exact numerics, auditable, diff-friendly |
| Semantic search | ChromaDB + Gemini embeddings | Prose rules and exemptions need semantic retrieval |
| Calculation | Pure Python (no LLM math) | Deterministic, testable, no hallucination risk |
| Orchestration | LangGraph ReAct | Stateful agent loop; tools are isolated and swappable |
| LLM | Gemini 2.5 Flash | Free tier available; good vision + reasoning |

---

## Setup

### Prerequisites

- Python 3.11+
- A Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- The Port Tariff PDF at `~/Downloads/Port Tariff.pdf`

### Install

```bash
git clone https://github.com/hamxahbhatti/port-tariff-ai.git
cd port-tariff-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

```bash
export GEMINI_API_KEY=your_key_here
```

Or create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

---

## Usage

### Step 1 — Validate your API key (2 requests)

```bash
GEMINI_API_KEY=your_key python -m tests.check_api_key
```

### Step 2 — Run ingestion (processes the PDF, ~23 Gemini Vision calls)

```bash
GEMINI_API_KEY=your_key python -m ingestion.pipeline
```

This populates:
- `data/tariffs/durban/*.json` — exact numeric rate tables per charge type
- `data/chroma_db/` — prose rules and conditions for semantic search

> **Rate limit note:** Gemini 2.5 Flash free tier = 20 requests/day per project.
> The pipeline sleeps 7s between calls (~8 req/min). For 23 table pages (46 double-pass calls)
> you need ~3 days of daily quota, or a paid tier key (500 RPD).

### Step 3 — Run tests

**Layer 2 tests (no API key needed — pure arithmetic):**
```bash
python -m tests.test_mcp_servers
```

**Layer 1 tests (requires API key + PDF):**
```bash
GEMINI_API_KEY=your_key python -m tests.test_ingestion
```

### Step 4 — Run the API

```bash
GEMINI_API_KEY=your_key python -m api.main
# → http://localhost:8000/docs
```

### Step 5 — Calculate dues for SUDESTADA

**Quick (deterministic, no LLM, requires ingested data):**
```bash
curl -s -X POST http://localhost:8000/calculate/quick \
  -H "Content-Type: application/json" \
  -d '{
    "name": "SUDESTADA",
    "vessel_type": "bulk_carrier",
    "gt": 51300,
    "loa_m": 229.2,
    "port": "durban",
    "cargo_operation": true,
    "cargo_type": "iron_ore",
    "cargo_mt": 75000,
    "berthing": true,
    "hours_alongside": 48
  }' | python -m json.tool
```

**Full agent (LangGraph + Gemini):**
```bash
curl -s -X POST http://localhost:8000/calculate \
  -H "Content-Type: application/json" \
  -d '{ ... same body ... }' | python -m json.tool
```

---

## Project structure

```
port-tariff-ai/
├── config.py                      # Central config (paths, model names, thresholds)
├── requirements.txt
│
├── ingestion/
│   ├── docling_parser.py          # Layer 1a: Docling PDF → prose chunks + table pages
│   ├── vision_extractor.py        # Layer 1b: Gemini Vision double-pass table extractor
│   ├── mineru_backup.py           # Layer 1c: MinerU fallback for flagged pages
│   └── pipeline.py                # Orchestrator: Docling → Vision → store
│
├── knowledge_store/
│   ├── tariff_store.py            # JSON store: exact numeric rates per port/charge
│   └── vector_store.py            # ChromaDB: prose rules + table descriptions
│
├── mcp_servers/
│   ├── tariff_rag/server.py       # MCP: search_rules, get_tariff_table
│   ├── calculator/server.py       # MCP: calculate_* for all 8 charge types
│   ├── rules_engine/server.py     # MCP: determine_applicable_charges, check_exemptions
│   └── vessel/server.py           # MCP: register_vessel, classify_vessel_for_tariff
│
├── agent/
│   └── tariff_agent.py            # LangGraph ReAct agent (17 tools, stateful loop)
│
├── api/
│   └── main.py                    # FastAPI: /calculate, /calculate/quick, /health
│
├── tests/
│   ├── check_api_key.py           # 2-request API key validator
│   ├── test_ingestion.py          # Layer 1 tests (Docling + Vision spot checks)
│   └── test_mcp_servers.py        # Layer 2 tests (MCP server + calculator arithmetic)
│
└── data/
    ├── tariffs/                   # JSON rate tables (populated by ingestion)
    └── chroma_db/                 # Vector store (populated by ingestion)
```

---

## Charge types computed

| Charge | Formula |
|--------|---------|
| Light Dues | (GT / 100) × rate_per_100GT |
| VTS | GT × rate_per_GT |
| Pilotage | (base_band_fee + incremental_per_100GT) × movements |
| Tug Assistance | (base_fee + incremental) × num_tugs × movements |
| Port Dues | GT × rate_per_GT |
| Cargo Dues | metric_tonnes × rate_per_MT |
| Berth Dues | GT × rate_per_24h × (hours / 24) |
| Running of Lines | flat_rate × num_services |

Incremental rows (e.g. "Plus per 100GT above 50,000 GT") are detected during Vision extraction via `is_incremental=True` / `parent_band` fields and applied with `ceil()` arithmetic.

---

## Known limitations

1. **Free tier quota:** Gemini 2.5 Flash free tier = 20 req/day. Full 23-page ingestion needs ~46 Vision calls. Workaround: paid API key (500 RPD) or ingest over multiple days.
2. **MinerU fallback:** Requires `magic-pdf` CLI installed separately — only triggered on flagged pages.
3. **Tug count:** Determined by Harbour Master at time of arrival. The rules engine uses GT-band guidelines as a default; actual count should be confirmed with port authority.
4. **Port scope:** Ingestion is currently configured for Durban. Other ports (Cape Town, Richards Bay, etc.) require re-running ingestion with their tariff sections.
