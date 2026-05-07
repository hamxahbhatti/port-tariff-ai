"""
Port Tariff AI — FastAPI REST endpoint.

Endpoints:
  GET  /                        — Serve the web UI
  POST /calculate/quick/stream  — SSE: streaming calculation with debug trace
  POST /calculate/quick         — Deterministic calculation (JSON response)
  POST /calculate               — Full LangGraph ReAct agent
  GET  /ports                   — List ports with ingested tariff data
  GET  /charges/{port}          — List charge types available for a port
  GET  /health                  — Health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from knowledge_store import tariff_store

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(
    title="Port Tariff AI",
    description=(
        "AI-powered South African port tariff calculator. "
        "Accepts a vessel profile and computes all applicable port dues."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static assets (/static/style.css, /static/app.js, etc.)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Request / Response models ───────────────────────────────────────────────

class VesselProfile(BaseModel):
    name: str              = Field(..., example="SUDESTADA")
    vessel_type: Literal[
        "bulk_carrier", "tanker", "container", "general_cargo",
        "passenger", "ro_ro", "other"
    ]                      = Field(..., example="bulk_carrier")
    gt: float              = Field(..., gt=0, example=51300)
    loa_m: float           = Field(..., gt=0, example=229.2)
    port: str              = Field(..., example="durban")
    cargo_operation: bool  = Field(True)
    cargo_type: str        = Field("", example="iron_ore")
    cargo_mt: float        = Field(0.0, ge=0, example=75000)
    berthing: bool         = Field(True)
    hours_alongside: float = Field(24.0, gt=0, example=48)
    in_distress: bool      = Field(False)
    nrt: float | None      = Field(None)
    flag_state: str | None = Field(None, example="Panama")


class LineItem(BaseModel):
    charge_type: str
    port: str
    charge_zar: float
    formula: str


class CalculationResult(BaseModel):
    vessel: str
    port: str
    total_zar: float
    currency: str = "ZAR"
    line_items: list[LineItem]
    errors: list[dict] = []
    calculation_method: str
    duration_seconds: float


# ── In-memory session store ──────────────────────────────────────────────────

sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


# ── Helpers ─────────────────────────────────────────────────────────────────

def _build_calculators(vessel: VesselProfile, modifiers: dict):
    """Return a dict of {charge_type: callable(rows_json, port) -> str}."""
    from mcp_servers.calculator.server import (
        calculate_berth_dues, calculate_cargo_dues, calculate_light_dues,
        calculate_pilotage, calculate_port_dues, calculate_running_of_lines,
        calculate_tug_assistance, calculate_vts,
    )
    return {
        "light_dues": lambda r, p: calculate_light_dues(r, vessel.gt, p),
        "vts":        lambda r, p: calculate_vts(r, vessel.gt, p),
        "pilotage":   lambda r, p: calculate_pilotage(
            r, vessel.gt, p, modifiers.get("pilotage_movements", 2)
        ),
        "tug_assistance": lambda r, p: calculate_tug_assistance(
            r, vessel.gt, p,
            modifiers.get("tug_count", 3),
            modifiers.get("tug_movements", 2),
        ),
        "port_dues":  lambda r, p: calculate_port_dues(r, vessel.gt, p),
        "cargo_dues": lambda r, p: calculate_cargo_dues(
            r,
            modifiers.get("cargo_mt", vessel.cargo_mt),
            modifiers.get("cargo_type", vessel.cargo_type),
            p,
        ),
        "berth_dues": lambda r, p: calculate_berth_dues(
            r, vessel.gt, p,
            modifiers.get("hours_alongside", vessel.hours_alongside),
        ),
        "running_of_lines": lambda r, p: calculate_running_of_lines(r, p, 2),
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve the web frontend."""
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not built.")
    return FileResponse(str(index))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ports_with_data": tariff_store.list_ports(),
        "gemini_key_set": bool(config.GEMINI_API_KEY),
    }


@app.get("/ports")
def list_ports():
    ports = tariff_store.list_ports()
    return {"ports": ports, "count": len(ports)}


@app.get("/charges/{port_name}")
def list_charges(port_name: str):
    charges = tariff_store.list_charge_types(port_name)
    if not charges:
        raise HTTPException(
            status_code=404,
            detail=f"No tariff data for '{port_name}'. Run ingestion first.",
        )
    return {"port": port_name, "charge_types": charges, "count": len(charges)}


# ── SSE streaming endpoint ───────────────────────────────────────────────────

@app.post("/calculate/quick/stream")
async def calculate_quick_stream(vessel: VesselProfile):
    """
    Streaming version of /calculate/quick — returns Server-Sent Events.

    Each event is a JSON object on a line prefixed with 'data: '.
    The frontend reads these via fetch() + ReadableStream.

    Event types:
      profile  — vessel profile received
      rules    — applicable charges determined
      fetch    — tariff table loaded
      calc     — individual charge computed
      error    — charge could not be computed
      complete — all charges aggregated; includes final total & line_items
    """
    from mcp_servers.rules_engine.server import determine_applicable_charges

    async def generate():
        t0 = time.perf_counter()

        def evt(ev_type: str, **kwargs) -> str:
            elapsed = round((time.perf_counter() - t0) * 1000)
            payload = json.dumps({"type": ev_type, "elapsed_ms": elapsed, **kwargs})
            return f"data: {payload}\n\n"

        # ── Step 1: profile received ────────────────────────────────────
        yield evt(
            "profile",
            step="Vessel Profile Received",
            description=(
                f"MV {vessel.name} · "
                f"{vessel.vessel_type.replace('_', ' ').title()} · "
                f"{vessel.gt:,.0f} GT · Port of {vessel.port.title()}"
            ),
            data=vessel.model_dump(),
        )
        await asyncio.sleep(0.05)

        # ── Step 2: rules engine ────────────────────────────────────────
        profile_dict = vessel.model_dump()
        plan_str = determine_applicable_charges(json.dumps(profile_dict))
        plan = json.loads(plan_str)
        applicable = plan.get("applicable_charges", [])
        modifiers  = plan.get("modifiers", {})

        yield evt(
            "rules",
            step="Rules Engine · determine_applicable_charges",
            description=(
                f"Identified {len(applicable)} applicable charges: "
                + ", ".join(applicable)
            ),
            tool="rules_engine.determine_applicable_charges",
            input={
                "vessel_type":     vessel.vessel_type,
                "gt":              vessel.gt,
                "port":            vessel.port,
                "has_cargo":       vessel.cargo_operation,
                "requesting_berth": vessel.berthing,
            },
            output=plan,
        )
        await asyncio.sleep(0.04)

        # ── Steps 3-N: per charge ───────────────────────────────────────
        calculators = _build_calculators(vessel, modifiers)
        port        = vessel.port
        line_items  = []
        errors      = []

        for charge_type in applicable:
            # Load tariff table
            table = tariff_store.load_table(port, charge_type)

            if table is None:
                msg = f"No tariff table for '{charge_type}' at '{port}'"
                yield evt(
                    "error",
                    step=f"Tariff Not Found · {charge_type}",
                    description=msg,
                    charge_type=charge_type,
                )
                errors.append({"charge_type": charge_type, "error": msg})
                await asyncio.sleep(0.02)
                continue

            rows = table.get("rows", [])
            yield evt(
                "fetch",
                step=f"Tariff Store · {charge_type}",
                description=(
                    f"Loaded {len(rows)} row(s) from "
                    f"data/tariffs/{port}/{charge_type}.json"
                ),
                tool="tariff_store.load_table",
                input={"port": port, "charge_type": charge_type},
                rows_count=len(rows),
                sample_row=rows[0] if rows else None,
            )
            await asyncio.sleep(0.03)

            # Calculate
            calc_fn = calculators.get(charge_type)
            if calc_fn is None:
                errors.append({"charge_type": charge_type, "error": "No calculator"})
                continue

            try:
                result_str = calc_fn(json.dumps(rows), port)
                result = json.loads(result_str)

                if "error" in result:
                    yield evt(
                        "error",
                        step=f"Calculator Error · {charge_type}",
                        description=result["error"],
                        charge_type=charge_type,
                        detail=result,
                    )
                    errors.append({"charge_type": charge_type, "error": result["error"]})
                else:
                    charge_zar = result.get("charge_zar", 0)
                    formula    = result.get("formula", "")
                    yield evt(
                        "calc",
                        step=f"Calculator · calculate_{charge_type}",
                        description=formula,
                        tool=f"calculator.calculate_{charge_type}",
                        charge_type=charge_type,
                        charge_zar=charge_zar,
                        formula=formula,
                        result=result,
                    )
                    line_items.append({
                        "charge_type": charge_type,
                        "port":        port,
                        "charge_zar":  charge_zar,
                        "formula":     formula,
                    })

            except Exception as ex:
                errors.append({"charge_type": charge_type, "error": str(ex)})
                yield evt(
                    "error",
                    step=f"Exception · {charge_type}",
                    description=str(ex),
                    charge_type=charge_type,
                )

            await asyncio.sleep(0.02)

        # ── Final aggregation ───────────────────────────────────────────
        total_zar = sum(item["charge_zar"] for item in line_items)
        elapsed   = round((time.perf_counter() - t0) * 1000)

        yield evt(
            "complete",
            step="Calculation Complete",
            description=(
                f"Total port dues for MV {vessel.name}: "
                f"R {total_zar:,.2f}"
            ),
            vessel=vessel.name,
            port=port,
            total_zar=round(total_zar, 2),
            line_items=line_items,
            errors=errors,
            calculation_method="deterministic",
            elapsed_ms=elapsed,
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


# ── Non-streaming quick endpoint ─────────────────────────────────────────────

@app.post("/calculate/quick", response_model=CalculationResult)
def calculate_quick(vessel: VesselProfile):
    """
    Deterministic calculation — no LLM required. Returns JSON directly.
    Use /calculate/quick/stream for the real-time debug experience.
    """
    from mcp_servers.rules_engine.server import determine_applicable_charges

    start = time.time()
    profile_dict = vessel.model_dump()

    plan_str = determine_applicable_charges(json.dumps(profile_dict))
    plan = json.loads(plan_str)

    if "error" in plan:
        raise HTTPException(status_code=400, detail=plan["error"])

    applicable = plan["applicable_charges"]
    modifiers  = plan["modifiers"]
    port       = vessel.port
    calculators = _build_calculators(vessel, modifiers)

    line_items, errors = [], []

    for charge_type in applicable:
        table = tariff_store.load_table(port, charge_type)
        if table is None:
            errors.append({"charge_type": charge_type,
                           "error": f"No tariff table for '{charge_type}'"})
            continue

        calc_fn = calculators.get(charge_type)
        if calc_fn is None:
            errors.append({"charge_type": charge_type, "error": "No calculator"})
            continue

        try:
            result = json.loads(calc_fn(json.dumps(table.get("rows", [])), port))
            if "error" in result:
                errors.append({"charge_type": charge_type, "error": result["error"]})
            else:
                line_items.append(LineItem(
                    charge_type=result.get("charge_type", charge_type),
                    port=result.get("port", port),
                    charge_zar=result.get("charge_zar", 0.0),
                    formula=result.get("formula", ""),
                ))
        except Exception as ex:
            errors.append({"charge_type": charge_type, "error": str(ex)})

    return CalculationResult(
        vessel=vessel.name,
        port=vessel.port,
        total_zar=round(sum(i.charge_zar for i in line_items), 2),
        currency="ZAR",
        line_items=line_items,
        errors=errors,
        calculation_method="deterministic",
        duration_seconds=round(time.time() - start, 3),
    )


# ── Full agent endpoint ──────────────────────────────────────────────────────

@app.post("/calculate")
def calculate_with_agent(vessel: VesselProfile):
    """Full LangGraph ReAct agent — requires GEMINI_API_KEY."""
    from agent.tariff_agent import calculate_port_dues_for_vessel

    if not config.GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not set.")

    start = time.time()
    try:
        result   = calculate_port_dues_for_vessel(vessel.model_dump())
        duration = time.time() - start
        return {
            "vessel": vessel.name,
            "port": vessel.port,
            "agent_response": result,
            "duration_seconds": round(duration, 2),
        }
    except Exception as ex:
        logger.error(f"Agent error: {ex}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(ex))


# ── Conversational chat endpoint ─────────────────────────────────────────────

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Conversational SSE chat endpoint powered by the ChatAgent.

    Streams debug events (llm_call, tool_call, tool_result) followed by
    a final 'response' event containing the agent's reply and optional
    structured calculation data.

    Session history is persisted in-memory; pass the same session_id across
    turns to maintain conversation context.
    """
    from agent.chat_agent import get_agent

    if not config.GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not set.")

    history = list(sessions.get(req.session_id, []))
    agent = get_agent()

    async def generate():
        response_content = ""
        try:
            for event in agent.run(history, req.message):
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "response":
                    response_content = event.get("content", "")
                await asyncio.sleep(0)
        except Exception as ex:
            logger.error(f"Chat agent error: {ex}", exc_info=True)
            error_evt = {
                "type": "response",
                "content": "I'm sorry, I encountered an error. Please try again.",
                "calc_data": None,
            }
            yield f"data: {json.dumps(error_evt)}\n\n"
        finally:
            sessions[req.session_id] = history + [
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": response_content or "Error occurred."},
            ]

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a given session."""
    sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}



# ── Dev server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT") != "production"
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=reload)
