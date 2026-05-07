# Port Tariff AI

**Conversational port dues calculator for Transnet National Ports Authority vessels**

A production AI system that accepts plain-language queries from shipping agents and vessel operators, determines which port charges apply to a given vessel, retrieves the correct rate tables, and returns a fully itemised breakdown with the formula shown at every line — powered by Gemini 2.5 Flash Lite, LangChain, FastMCP, and FastAPI.

> **Live demo:** [port-tariff-ai.vercel.app](https://port-tariff-ai.vercel.app)  
> **Backend API:** [port-tariff-ai-production.up.railway.app/health](https://port-tariff-ai-production.up.railway.app/health)

---

## What It Does

Type a vessel description in plain English. The agent asks one follow-up question at a time if any detail is missing, then calculates all applicable charges against the official TNPA 2024/25 tariff schedule and returns a structured breakdown.

```
User:   Calculate dues for a 51,300 GT bulk carrier at Durban carrying
        75,000 MT iron ore for 48 hours

Agent:  Total port dues: R 6,334,124.31

        Light Dues       R    60,062.04   (51,300 GT / 100) × R117.08
        VTS              R    33,345.00   51,300 GT × R0.65/GT
        Port Dues        R    98,870.49   (51,300 GT / 100) × R192.73
        Cargo Dues       R 5,598,000.00  75,000 MT × R74.64/MT (iron ore)
        Berth Dues       R    51,874.56   (51,300 GT / 100) × R50.56 × 2 periods
        Running of Lines R     3,309.12   R1,654.56 × 2 services
        Pilotage         R    47,189.94   (18,608.61 + 4,986.36) × 2 movements
        Tug Assistance   R   441,473.16  (38,494.51 + 35,084.35) × 3 tugs × 2 movements

User:   What if we stay 72 hours instead?

Agent:  [recalculates berth dues only — remembers the rest]
```

---

## Architecture

The system separates data extraction from runtime inference. Extraction runs once on a developer machine and its output travels to production as committed files. The production system has no extraction dependencies.

### Development Pipeline

```
TNPA Tariff PDF (2024/25)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  INGESTION  (local only — not deployed)                         │
│                                                                  │
│  Docling (DocLayNet + TableFormer)                              │
│  Primary extraction — parses PDF structure directly             │
│         │                                                        │
│         ▼                                                        │
│  Quality Assessment per page                                     │
│  Checks markdown structure, numeric density, empty-cell ratio   │
│         │                        │                               │
│   Clean pages                Complex pages                      │
│         │                        │                               │
│         ▼                        ▼                               │
│  Gemini Text API         PyMuPDF → Gemini Vision                │
│  Single batched call     Double-pass verification               │
│                               │                                  │
│                          Flagged pages → MinerU (fallback)      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  KNOWLEDGE STORE                                                 │
│                                                                  │
│  JSON Rate Store (39 files)        ChromaDB                     │
│  Exact rate rows per charge type   Prose chunks + descriptions  │
│  Deterministic key-value reads     Cosine similarity search     │
│                                    text-embedding-004           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  MCP SERVERS  (FastMCP)                                          │
│                                                                  │
│  Rules Engine      Calculator         Tariff RAG   Vessel       │
│  Charge applicability  8 pure Python  JSON + Chroma  Profile    │
│  GT thresholds     functions          search_rules   validation │
│  Exemption logic   No LLM arithmetic  get_tariff_table          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT LAYER                                                     │
│                                                                  │
│  Chat Agent (chat_agent.py)      LangGraph Agent (tariff_agent) │
│  LangChain tool-calling          StateGraph ReAct               │
│  2 tools, session memory         17 tools, recursion limit 50   │
│  Gemini 2.5 Flash Lite, T=0      Powers /calculate endpoint     │
│  Powers /chat SSE endpoint                                       │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────┐        ┌────────────────────────────────────┐
│  FastAPI + SSE    │──────► │  Frontend                          │
│  /chat            │        │  Chat interface + debug drawer      │
│  /calculate       │        │  Real-time SSE event stream        │
│  /health          │        └────────────────────────────────────┘
└───────────────────┘
```

### Production Deployment

```
┌──────────────────────────────────────┐     ┌──────────────────────────┐
│  Railway (Docker)                    │     │  Vercel (Static CDN)     │
│  python:3.13-slim                    │     │                          │
│                                      │     │  index.html              │
│  JSON Rate Store (39 files)          │     │  app.js                  │
│  ChromaDB (pre-built)                │◄────│  style.css               │
│  MCP Servers                         │HTTPS│                          │
│  Chat Agent + LangGraph Agent        │     │  window.API_BASE →       │
│  FastAPI + SSE                       │     │  Railway URL             │
│                                      │     │                          │
│  GEMINI_API_KEY → env var            │     └──────────────────────────┘
└──────────────────────────────────────┘
```

---

## Charge Types

All eight charge categories from the TNPA 2024/25 tariff schedule are supported.

| Charge | Basis | Notes |
|---|---|---|
| Light Dues | Per 100 GT | All vessels, all SA ports |
| Vessel Traffic Services | Per GT | Flat rate per port call |
| Port Dues | Per 100 GT | Marine services levy |
| Cargo Dues | Per metric tonne | Rate varies by cargo category |
| Berth Dues | Per 100 GT per 24h | Pro-rated for partial periods |
| Running of Lines | Flat per service | Two services per port call |
| Pilotage | Banded by GT, per movement | Compulsory above 500 GT |
| Tug Assistance | Banded by GT, per tug per movement | Compulsory above 3,000 GT |

---

## Local Setup

### Prerequisites

- Python 3.11 or higher
- A Gemini API key from [aistudio.google.com](https://aistudio.google.com/apikey)

### Install

```bash
git clone https://github.com/hamxahbhatti/port-tariff-ai.git
cd port-tariff-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Run (tariff data already extracted)

The 39 JSON rate files are committed to the repository. You do not need to run ingestion.

```bash
python3 -m api.main
```

Open [http://localhost:8000](http://localhost:8000) to use the chat interface.

### Run ingestion (optional — only needed for a new port)

```bash
# Place the TNPA tariff PDF at ~/Downloads/Port Tariff.pdf
python3 -m ingestion.pipeline
```

Takes approximately 8 minutes. Progress is printed per page.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Chat UI (served from frontend/) |
| `/chat` | POST | Conversational agent — streams SSE events |
| `/calculate/quick/stream` | POST | Deterministic pipeline — streams SSE events |
| `/calculate/quick` | POST | Deterministic pipeline — returns JSON |
| `/calculate` | POST | Full LangGraph agent — returns JSON |
| `/health` | GET | Status check |
| `/ports` | GET | Ports with tariff data loaded |
| `/charges/{port}` | GET | Available charge types for a port |

### Chat endpoint

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "message": "51300 GT bulk carrier Durban 75000 MT iron ore 48 hours"}'
```

Pass the same `session_id` across requests to maintain conversation context. Each SSE event has a `type` field: `llm_call`, `tool_call`, `tool_result`, or `response`.

### Deterministic endpoint

```bash
curl -X POST http://localhost:8000/calculate/quick \
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
  }'
```

---

## Project Structure

```
port-tariff-ai/
├── config.py                        # Paths, model names, thresholds
├── requirements.txt                 # Full dev dependencies
├── requirements-prod.txt            # Runtime only (no ingestion stack)
├── Dockerfile                       # python:3.13-slim, Railway build
├── railway.toml                     # Railway deploy config
├── vercel.json                      # Vercel static site config
│
├── ingestion/                       # Runs once locally — not deployed
│   ├── pipeline.py                  # Orchestrator: Docling → routing → save
│   ├── docling_parser.py            # DocLayNet + TableFormer primary parse
│   ├── vision_extractor.py          # Quality assessment + Gemini Vision fallback
│   └── mineru_backup.py             # Final fallback for flagged pages
│
├── knowledge_store/
│   ├── tariff_store.py              # Load and save JSON rate files
│   └── vector_store.py              # ChromaDB prose and description store
│
├── mcp_servers/
│   ├── rules_engine/server.py       # Charge applicability + exemption logic
│   ├── calculator/server.py         # Eight deterministic arithmetic functions
│   ├── tariff_rag/server.py         # JSON store + ChromaDB query interface
│   └── vessel/server.py             # Profile validation and normalisation
│
├── agent/
│   ├── chat_agent.py                # Conversational agent with session memory
│   └── tariff_agent.py              # LangGraph StateGraph ReAct agent
│
├── api/
│   └── main.py                      # FastAPI — chat, calculate, health endpoints
│
├── frontend/
│   ├── index.html                   # Chat interface
│   ├── app.js                       # SSE client, debug drawer, calc card
│   └── style.css                    # Layout and component styles
│
├── tests/
│   └── test_mcp_servers.py          # Calculator arithmetic tests (no API key needed)
│
└── data/
    ├── tariffs/durban/              # 39 extracted JSON rate files (committed)
    └── chroma_db/                   # ChromaDB prose store (committed)
```

---

## Key Design Decisions

**Extraction separated from runtime.** The ingestion pipeline (Docling, Gemini Vision, MinerU) runs once on a developer machine. Only the structured output travels to production. The Docker image is lean and has no GPU or heavy CV dependencies.

**Deterministic calculators, no LLM arithmetic.** Every charge is computed by a pure Python function. The LLM handles conversation and tool sequencing only. This makes calculations auditable, unit-testable, and independent of model provider.

**Two-layer knowledge store.** Exact rate lookups go directly to the JSON store. Condition and exemption queries use ChromaDB semantic search. Keeping these separate prevents imprecision in rate lookups and avoids unnecessary latency on the common path.

**Streaming transparency.** Every agent step — LLM call, tool call, tool result — streams to the frontend as a typed SSE event. The debug drawer shows the full execution trace so any calculation can be traced back to its source data.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Conversational AI | Google Gemini 2.5 Flash Lite |
| Agent orchestration | LangChain, LangGraph StateGraph |
| Tool protocol | FastMCP (Model Context Protocol) |
| PDF parsing | Docling (DocLayNet + TableFormer) |
| Vision extraction | Gemini Vision Flash |
| Semantic store | ChromaDB with text-embedding-004 |
| API | FastAPI with Server-Sent Events |
| Backend hosting | Railway (Docker, python:3.13-slim) |
| Frontend hosting | Vercel (static, global CDN) |
| Runtime | Python 3.13 |
