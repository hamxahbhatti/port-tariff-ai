# Port Tariff AI

A system that reads a port tariff PDF and answers the question: *how much will this vessel owe when it docks?*

Instead of a shipping agent manually cross-referencing 23 pages of tariff tables, you describe a vessel in plain English — or submit a JSON profile — and the system returns an itemized breakdown of every applicable charge.

**Test case:** MV SUDESTADA — 51,300 GT Bulk Carrier, Port of Durban, exporting Iron Ore.

---

## How It Works

The system has two modes:

**Chat (conversational):** Type in natural language. The agent asks follow-up questions if it needs more information, then calculates. Subsequent messages like *"what if 72 hours instead?"* work because the agent remembers the conversation.

**API (deterministic):** Submit a vessel profile as JSON. No LLM involved — just the rules engine and calculator functions running directly. Fast and fully auditable.

Both modes use the same underlying tariff data, which is extracted from the PDF once during ingestion and stored as structured JSON.

---

## Architecture

```
Port Tariff PDF
      │
      ▼
┌─────────────────────────────────────┐
│ Ingestion (ingestion/)              │
│ Docling → quality check → Gemini   │
│ Output: data/tariffs/durban/*.json  │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ MCP Servers (mcp_servers/)          │
│ rules_engine  →  which charges?     │
│ calculator    →  exact amounts      │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ Agent (agent/)                      │
│ Gemini + LangChain tool-calling     │
│ Conversational, session memory      │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ API + Frontend (api/ · frontend/)   │
│ FastAPI · SSE streaming · Chat UI   │
└─────────────────────────────────────┘
```

### PDF Ingestion

The TNPA tariff document has complex tables — merged cells, multi-level headers, rate bands with footnotes. Standard PDF parsers can't handle this reliably.

The pipeline uses **Docling** (IBM's DocLayNet + TableFormer models) to do a first pass locally. Each extracted page is scored: if it has enough structure and numeric content, it goes to a single Gemini text call for cleanup. Pages that fail the quality check go through Gemini Vision. In practice, 14 of 23 pages were clean enough for the text path; 9 needed Vision. Total ingestion time: ~8 minutes.

Output is 39 JSON files — one per charge type — each containing the rate rows exactly as structured in the tariff document.

### Calculations

All arithmetic is deterministic. The LLM is never asked to compute a number. When the agent decides a calculation is needed, it calls a Python function that reads the relevant tariff JSON, finds the rate band matching the vessel's GT, and applies the exact formula.

This matters because port dues are legally binding. A 2% hallucination error on a $200K charge is not acceptable.

### The Agent

The chat agent uses LangChain's tool-calling interface with Gemini 2.5 Flash Lite. It has access to two tools:

- `determine_applicable_charges` — figures out which of the 8 charge types apply given the vessel profile
- `calculate_all_dues` — runs all applicable calculators and returns the full breakdown

The system prompt instructs it to ask one question at a time when something is missing, remember everything across turns, and never invent a tariff rate.

### Why the Charges Never Go Through the LLM

The rules engine determines *which* charges apply. The calculator functions compute *how much*. The LLM only handles natural language understanding and decides which tools to call and when. This separation means you can swap the LLM without touching any of the financial logic.

---

## Charges Calculated

| Charge | Basis |
|--------|-------|
| Light Dues | Per GT — lighthouse and buoy maintenance |
| VTS | Per GT — Vessel Traffic Services (port radio room) |
| Pilotage | Per GT band × number of movements |
| Tug Assistance | Per GT band × tug count × movements |
| Port Dues | Per GT — general port access |
| Cargo Dues | Per metric tonne × cargo type |
| Berth Dues | Per GT × hours alongside |
| Running of Lines | Flat rate × movements — mooring line handlers |

---

## Setup

### Requirements

- Python 3.11+
- A Gemini API key from [aistudio.google.com](https://aistudio.google.com/apikey)
- The TNPA Port Tariff PDF (attach at `~/Downloads/Port Tariff.pdf`)

### Install

```bash
git clone https://github.com/hamxahbhatti/port-tariff-ai.git
cd port-tariff-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

---

## Running the System

### Step 1 — Ingest the tariff PDF

This runs once. It extracts all rate tables from the PDF and saves them to `data/tariffs/`.

```bash
python -m ingestion.pipeline
```

Takes about 8 minutes. Progress is printed per page.

### Step 2 — Start the server

```bash
python -m api.main
```

Open `http://localhost:8000` to use the chat interface.

### Step 3 — Try the SUDESTADA test case

The reference vessel from the assignment. Via the API:

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
    "cargo_mt": 40000,
    "berthing": true,
    "hours_alongside": 81.4
  }'
```

Or just type into the chat: *"Calculate dues for a 51,300 GT bulk carrier at Durban exporting 40,000 MT of iron ore, 81 hours alongside"*

### Run tests

```bash
python -m tests.test_mcp_servers
```

The MCP server tests cover the calculator arithmetic against known values and don't require an API key.

---

## Project Structure

```
port-tariff-ai/
├── config.py                    # Paths, model names, thresholds
├── requirements.txt
│
├── ingestion/
│   ├── docling_parser.py        # Local PDF → table markdown
│   ├── vision_extractor.py      # Gemini Vision for complex pages
│   └── pipeline.py              # Orchestrates ingestion end-to-end
│
├── knowledge_store/
│   ├── tariff_store.py          # Load/save tariff JSON files
│   └── vector_store.py          # ChromaDB for prose sections
│
├── mcp_servers/
│   ├── rules_engine/server.py   # Which charges apply?
│   └── calculator/server.py     # Compute each charge
│
├── agent/
│   ├── chat_agent.py            # Conversational agent with memory
│   └── tariff_agent.py          # LangGraph ReAct variant
│
├── api/
│   └── main.py                  # FastAPI endpoints
│
├── frontend/
│   ├── index.html               # Chat interface
│   ├── app.js                   # SSE client, message rendering
│   └── style.css
│
├── tests/
│   └── test_mcp_servers.py      # Calculator arithmetic tests
│
└── data/
    ├── tariffs/durban/          # Extracted rate tables (JSON)
    └── chroma_db/               # Vector store
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat UI |
| `/chat` | POST | Conversational agent (SSE stream) |
| `/calculate/quick/stream` | POST | Deterministic calculation (SSE stream) |
| `/calculate/quick` | POST | Deterministic calculation (JSON) |
| `/health` | GET | Server status |
| `/ports` | GET | Ports with ingested data |
| `/charges/{port}` | GET | Available charge types for a port |

The `/chat` endpoint takes `{ session_id, message }` and streams events as SSE. Pass the same `session_id` across requests to maintain conversation context.

---

## Extending to Other Ports

Run ingestion on a different tariff PDF:

```bash
PORT_NAME=cape_town python -m ingestion.pipeline
```

Rate tables land in `data/tariffs/cape_town/`. The calculators and agent work with any port as long as the table JSON follows the same schema. No code changes required.

---

## Notes

- The free tier Gemini API allows 20 requests/day per project. If ingestion hits the limit mid-run, it saves progress and can resume.
- Tug count defaults to 3 for vessels above 20,000 GT. The actual count is set by the Harbour Master at arrival — this is an inherent uncertainty in any pre-arrival estimate.
- Cargo dues and berth dues depend on inputs (metric tonnes, hours alongside) that may not be final at the time of enquiry. The agent will ask for these if they're missing.
