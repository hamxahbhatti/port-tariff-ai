"""
Port Tariff AI — FastAPI REST endpoint.

Endpoints:
  POST /calculate          — Run the full agent calculation for a vessel
  POST /calculate/quick    — Deterministic calculation (no LLM, uses known tariff data directly)
  GET  /ports              — List ports with ingested tariff data
  GET  /charges/{port}     — List charge types available for a port
  GET  /health             — Health check

The /calculate endpoint runs the full LangGraph ReAct agent.
The /calculate/quick endpoint bypasses the LLM and calls calculator tools directly
(useful when the tariff data is already in the store and you want fast, deterministic output).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from knowledge_store import tariff_store

logger = logging.getLogger(__name__)

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


# ── Request / Response models ───────────────────────────────────────────────

class VesselProfile(BaseModel):
    name: str = Field(..., description="Vessel name", example="SUDESTADA")
    vessel_type: Literal[
        "bulk_carrier", "tanker", "container", "general_cargo", "passenger", "ro_ro", "other"
    ] = Field(..., description="Vessel type category")
    gt: float = Field(..., gt=0, description="Gross Tonnage", example=51300)
    loa_m: float = Field(..., gt=0, description="Length Overall in metres", example=229.2)
    port: str = Field(..., description="Port of call (lowercase)", example="durban")
    cargo_operation: bool = Field(True, description="Is cargo being loaded/discharged?")
    cargo_type: str = Field("", description="Cargo type (e.g. 'iron_ore', 'containers')", example="iron_ore")
    cargo_mt: float = Field(0.0, ge=0, description="Cargo in metric tonnes", example=75000)
    berthing: bool = Field(True, description="Is vessel going alongside a berth?")
    hours_alongside: float = Field(24.0, gt=0, description="Hours at berth", example=48)
    in_distress: bool = Field(False, description="Is vessel in distress?")
    nrt: float | None = Field(None, description="Net Register Tonnage (optional)")
    flag_state: str | None = Field(None, description="Flag state (optional)", example="Panama")


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


class AgentCalculationResult(BaseModel):
    vessel: str
    port: str
    agent_response: dict
    duration_seconds: float
    message_count: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — confirms the API is running."""
    return {
        "status": "ok",
        "ports_with_data": tariff_store.list_ports(),
        "gemini_key_set": bool(config.GEMINI_API_KEY),
    }


@app.get("/ports")
def list_ports():
    """List all ports that have ingested tariff data."""
    ports = tariff_store.list_ports()
    return {"ports": ports, "count": len(ports)}


@app.get("/charges/{port_name}")
def list_charges(port_name: str):
    """List all charge types available in the tariff store for a port."""
    charges = tariff_store.list_charge_types(port_name)
    if not charges:
        raise HTTPException(
            status_code=404,
            detail=f"No tariff data found for port '{port_name}'. Run ingestion first.",
        )
    return {"port": port_name, "charge_types": charges, "count": len(charges)}


@app.post("/calculate", response_model=AgentCalculationResult)
def calculate_with_agent(vessel: VesselProfile):
    """
    Run the full LangGraph ReAct agent to calculate all applicable port dues.

    The agent:
    1. Determines which charges apply (rules engine)
    2. Retrieves tariff tables from the store
    3. Computes each charge deterministically
    4. Aggregates into a final invoice

    Requires a valid GEMINI_API_KEY and ingested tariff data for the port.
    """
    from agent.tariff_agent import calculate_port_dues_for_vessel

    if not config.GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not configured. Set via environment variable.",
        )

    start = time.time()
    try:
        result = calculate_port_dues_for_vessel(vessel.model_dump())
        duration = time.time() - start

        return AgentCalculationResult(
            vessel=vessel.name,
            port=vessel.port,
            agent_response=result,
            duration_seconds=round(duration, 2),
            message_count=result.get("_message_count", 0),
        )
    except Exception as e:
        logger.error(f"Agent calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate/quick", response_model=CalculationResult)
def calculate_quick(vessel: VesselProfile):
    """
    Deterministic calculation — no LLM required.

    Calls the rules engine and calculator tools directly using tariff data
    from the JSON store. Fast, auditable, and suitable for production batch use.

    Requires ingested tariff data for the port (run ingestion pipeline first).
    """
    from mcp_servers.calculator.server import (
        calculate_berth_dues,
        calculate_cargo_dues,
        calculate_light_dues,
        calculate_pilotage,
        calculate_port_dues,
        calculate_running_of_lines,
        calculate_tug_assistance,
        calculate_vts,
    )
    from mcp_servers.rules_engine.server import determine_applicable_charges

    start = time.time()
    profile = vessel.model_dump()

    # Step 1: get applicable charges
    plan_str = determine_applicable_charges(json.dumps(profile))
    plan = json.loads(plan_str)

    if "error" in plan:
        raise HTTPException(status_code=400, detail=plan["error"])

    applicable = plan["applicable_charges"]
    modifiers = plan["modifiers"]
    port = vessel.port

    # Step 2: for each charge, load table and calculate
    line_items = []
    errors = []

    CALCULATORS = {
        "light_dues": lambda rows, p: calculate_light_dues(
            rows_json=rows, gt=vessel.gt, port_name=p
        ),
        "vts": lambda rows, p: calculate_vts(
            rows_json=rows, gt=vessel.gt, port_name=p
        ),
        "pilotage": lambda rows, p: calculate_pilotage(
            rows_json=rows, gt=vessel.gt, port_name=p,
            movements=modifiers.get("pilotage_movements", 2)
        ),
        "tug_assistance": lambda rows, p: calculate_tug_assistance(
            rows_json=rows, gt=vessel.gt, port_name=p,
            num_tugs=modifiers.get("tug_count", 2),
            movements=modifiers.get("tug_movements", 2),
        ),
        "port_dues": lambda rows, p: calculate_port_dues(
            rows_json=rows, gt=vessel.gt, port_name=p
        ),
        "cargo_dues": lambda rows, p: calculate_cargo_dues(
            rows_json=rows,
            metric_tonnes=modifiers.get("cargo_mt", vessel.cargo_mt),
            cargo_type=modifiers.get("cargo_type", vessel.cargo_type),
            port_name=p,
        ),
        "berth_dues": lambda rows, p: calculate_berth_dues(
            rows_json=rows, gt=vessel.gt, port_name=p,
            hours_alongside=modifiers.get("hours_alongside", vessel.hours_alongside),
        ),
        "running_of_lines": lambda rows, p: calculate_running_of_lines(
            rows_json=rows, port_name=p, num_services=2
        ),
    }

    for charge_type in applicable:
        table = tariff_store.load_table(port, charge_type)
        if table is None:
            errors.append({
                "charge_type": charge_type,
                "error": f"No tariff table found for '{charge_type}' at '{port}'. Run ingestion first.",
            })
            continue

        rows_json = json.dumps(table.get("rows", []))
        calc_fn = CALCULATORS.get(charge_type)
        if calc_fn is None:
            errors.append({
                "charge_type": charge_type,
                "error": f"No calculator implemented for '{charge_type}'.",
            })
            continue

        try:
            result_str = calc_fn(rows_json, port)
            result = json.loads(result_str)

            if "error" in result:
                errors.append({"charge_type": charge_type, "error": result["error"]})
            else:
                line_items.append(LineItem(
                    charge_type=result.get("charge_type", charge_type),
                    port=result.get("port", port),
                    charge_zar=result.get("charge_zar", 0.0),
                    formula=result.get("formula", ""),
                ))
        except Exception as e:
            errors.append({"charge_type": charge_type, "error": str(e)})

    total_zar = sum(item.charge_zar for item in line_items)
    duration = time.time() - start

    return CalculationResult(
        vessel=vessel.name,
        port=vessel.port,
        total_zar=round(total_zar, 2),
        currency="ZAR",
        line_items=line_items,
        errors=errors,
        calculation_method="deterministic",
        duration_seconds=round(duration, 3),
    )


# ── Dev server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
