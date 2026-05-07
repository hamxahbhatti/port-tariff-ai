"""
Conversational port tariff agent with memory.

Three tools exposed to the LLM:
  - determine_applicable_charges  (rules engine)
  - calculate_all_dues            (full deterministic calculation — orchestrates all 8 calculators internally)
  - search_rules                  (ChromaDB semantic search for prose/exemption/policy questions)

The calculate_all_dues tool handles all arithmetic internally so the LLM never
needs to orchestrate individual calculator tools — keeps the model surface small
and prevents looping.
"""

from __future__ import annotations

import json
import logging
from typing import Generator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

import config

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Transnet National Ports Authority tariff assistant.
You help shipping agents and vessel operators with two types of requests:

1. CALCULATIONS — calculating exact port dues for a vessel call at a South African port.
2. RULE / POLICY QUESTIONS — answering questions about exemptions, conditions, surcharges,
   special vessel categories, and how charges work.

CONVERSATION RULES:
- Extract vessel parameters from what the user tells you (name, type, GT, port, cargo, berth hours).
- If required information is missing for a calculation, ask ONE specific question at a time.
- Required to calculate: vessel type, gross tonnage (GT), port of call.
- Also needed: cargo type + metric tonnes (if cargo is being worked), hours alongside berth.
- When you have enough information, call your tools and calculate immediately.
- Remember all vessel details from earlier in the conversation.
- For follow-ups like "what if 72 hours?" — use remembered vessel params, update only what changed.

TOOL USAGE:
For calculations:
  1. Call determine_applicable_charges first to see which charges apply.
  2. Then call calculate_all_dues with the full vessel details.
  3. Never call these tools more than once per user message.

For rule / policy / exemption questions:
  - Call search_rules with a clear natural-language query.
  - If results are returned, answer from those passages.
  - If no results are returned, say the specific rule is not documented in the indexed
    tariff content, and state the standard rule if known (e.g. pilotage is compulsory
    above 500 GT for all commercial vessels).
  - Never say the tools lack functionality.

RESPONSE FORMAT:
- Be concise and professional.
- For calculations: brief summary then the line-item breakdown.
- For rule questions: answer directly from the retrieved passages.
- Never estimate or invent a tariff rate or rule.
- If a port is not yet supported, say so clearly.

Supported ports: Durban (Richards Bay, Cape Town and others coming soon)."""


# ── Tools ─────────────────────────────────────────────────────────────────

@tool
def determine_applicable_charges(
    vessel_type: str,
    gt: float,
    port: str,
    has_cargo: bool = True,
    requesting_berth: bool = True,
) -> str:
    """
    Determine which charges apply to a vessel at a given South African port.

    Args:
        vessel_type: bulk_carrier | tanker | container | general_cargo | ro_ro | passenger
        gt: Gross Tonnage of the vessel
        port: Port name (e.g. 'durban')
        has_cargo: True if cargo is being loaded or discharged
        requesting_berth: True if vessel is going alongside a berth
    """
    from mcp_servers.rules_engine.server import determine_applicable_charges as _fn
    profile = {
        "name": "vessel", "vessel_type": vessel_type, "gt": gt,
        "loa_m": 200, "port": port, "cargo_operation": has_cargo,
        "cargo_type": "", "cargo_mt": 0, "berthing": requesting_berth,
        "hours_alongside": 24, "in_distress": False,
    }
    return _fn(json.dumps(profile))


@tool
def calculate_all_dues(
    vessel_type: str,
    gt: float,
    port: str,
    cargo_type: str = "",
    cargo_mt: float = 0,
    hours_alongside: float = 24,
    loa_m: float = 200,
) -> str:
    """
    Calculate all applicable port dues for a vessel. Returns a complete breakdown
    with each charge type, formula, and total in ZAR.

    Args:
        vessel_type: bulk_carrier | tanker | container | general_cargo | ro_ro | passenger
        gt: Gross Tonnage
        port: Port name (e.g. 'durban')
        cargo_type: Cargo description (e.g. 'iron_ore', 'coal', 'containers') — empty if no cargo
        cargo_mt: Cargo in metric tonnes — 0 if no cargo
        hours_alongside: Hours the vessel will spend at berth
        loa_m: Length Overall in metres
    """
    from knowledge_store.tariff_store import load_table
    from mcp_servers.calculator.server import (
        calculate_berth_dues, calculate_cargo_dues, calculate_light_dues,
        calculate_pilotage, calculate_port_dues, calculate_running_of_lines,
        calculate_tug_assistance, calculate_vts,
    )
    from mcp_servers.rules_engine.server import determine_applicable_charges as _rules

    has_cargo = bool(cargo_mt > 0 and cargo_type.strip())
    profile = {
        "name": "vessel", "vessel_type": vessel_type, "gt": gt, "loa_m": loa_m,
        "port": port, "cargo_operation": has_cargo, "cargo_type": cargo_type,
        "cargo_mt": cargo_mt, "berthing": True, "hours_alongside": hours_alongside,
        "in_distress": False,
    }
    plan = json.loads(_rules(json.dumps(profile)))
    applicable = plan.get("applicable_charges", [])
    modifiers = plan.get("modifiers", {})

    CALCS = {
        "light_dues":       lambda r, p: calculate_light_dues(r, gt, p),
        "vts":              lambda r, p: calculate_vts(r, gt, p),
        "pilotage":         lambda r, p: calculate_pilotage(r, gt, p, modifiers.get("pilotage_movements", 2)),
        "tug_assistance":   lambda r, p: calculate_tug_assistance(r, gt, p, modifiers.get("tug_count", 3), modifiers.get("tug_movements", 2)),
        "port_dues":        lambda r, p: calculate_port_dues(r, gt, p),
        "cargo_dues":       lambda r, p: calculate_cargo_dues(r, cargo_mt, cargo_type, p),
        "berth_dues":       lambda r, p: calculate_berth_dues(r, gt, p, hours_alongside),
        "running_of_lines": lambda r, p: calculate_running_of_lines(r, p, 2),
    }

    results, errors = [], []
    for charge in applicable:
        table = load_table(port, charge)
        if not table:
            errors.append({"charge_type": charge, "error": "No tariff table found"})
            continue
        calc_fn = CALCS.get(charge)
        if not calc_fn:
            continue
        try:
            r = json.loads(calc_fn(json.dumps(table.get("rows", [])), port))
            if "error" in r:
                errors.append({"charge_type": charge, "error": r["error"]})
            else:
                results.append({"charge_type": charge, "charge_zar": r.get("charge_zar", 0), "formula": r.get("formula", "")})
        except Exception as e:
            errors.append({"charge_type": charge, "error": str(e)})

    total = sum(r["charge_zar"] for r in results)
    return json.dumps({
        "success": True, "port": port, "vessel_type": vessel_type, "gt": gt,
        "cargo_type": cargo_type, "cargo_mt": cargo_mt, "hours_alongside": hours_alongside,
        "total_zar": round(total, 2), "line_items": results, "errors": errors,
    })


@tool
def search_rules(question: str, port_name: str = "durban") -> str:
    """
    Semantic search over the port tariff rules, exemptions, conditions, and
    table descriptions stored in ChromaDB. Use this for any question about
    policy, exemptions, surcharges, or how a charge works.

    Args:
        question: Natural-language question about tariff rules or exemptions
        port_name: Port to search (default 'durban')
    """
    from mcp_servers.tariff_rag.server import search_rules as _fn
    return _fn(question, port_name)


# ── Agent ─────────────────────────────────────────────────────────────────

TOOLS = [determine_applicable_charges, calculate_all_dues, search_rules]
TOOL_MAP = {t.name: t for t in TOOLS}


class ChatAgent:
    """Conversational agent with per-session memory."""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            base = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash-lite",
                temperature=0,
                google_api_key=config.GEMINI_API_KEY,
            )
            self._llm = base.bind_tools(TOOLS)
        return self._llm

    @staticmethod
    def _user_facing_error(exc: Exception) -> str:
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            import re
            delay = re.search(r"retry in (\d+)", msg)
            wait = f" Please try again in {delay.group(1)} seconds." if delay else ""
            return f"The AI model has hit its rate limit.{wait}"
        if "INVALID_ARGUMENT" in msg:
            return "The request was invalid. Please rephrase your question."
        if "UNAVAILABLE" in msg or "503" in msg:
            return "The AI model is temporarily unavailable. Please retry in a moment."
        return f"AI model error: {msg[:200]}"

    def run(self, history: list[dict], message: str) -> Generator[dict, None, None]:
        llm = self._get_llm()

        msgs: list = [SystemMessage(content=SYSTEM_PROMPT)]
        for h in history:
            if h["role"] == "user":
                msgs.append(HumanMessage(content=h["content"]))
            else:
                msgs.append(AIMessage(content=h["content"]))
        msgs.append(HumanMessage(content=message))

        yield {
            "type": "llm_call",
            "step": "Gemini · Processing message",
            "description": f"Analysing user message with {len(msgs)} context messages",
            "model": "gemini-2.5-flash-lite",
            "message_count": len(msgs),
            "user_message": message[:400],
            "system_prompt_preview": SYSTEM_PROMPT[:300],
        }

        try:
            response = llm.invoke(msgs)
        except Exception as exc:
            yield {"type": "response", "content": self._user_facing_error(exc), "calc_data": None}
            return

        iteration = 0
        while response.tool_calls and iteration < 8:
            iteration += 1
            msgs.append(response)
            tool_messages = []

            for tc in response.tool_calls:
                yield {
                    "type": "tool_call",
                    "step": f"Tool Call · {tc['name']}",
                    "description": (
                        f"Calling {tc['name']} with "
                        + ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
                    )[:200],
                    "tool": tc["name"],
                    "args": tc["args"],
                }

                tool_fn = TOOL_MAP.get(tc["name"])
                try:
                    raw = tool_fn.invoke(tc["args"]) if tool_fn else json.dumps({"error": f"Unknown tool: {tc['name']}"})
                except Exception as tool_exc:
                    raw = json.dumps({"error": str(tool_exc)})

                try:
                    result_data = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    result_data = raw

                yield {
                    "type": "tool_result",
                    "step": f"Tool Result · {tc['name']}",
                    "description": _summarise(tc["name"], result_data),
                    "tool": tc["name"],
                    "result": result_data,
                }

                tool_messages.append(ToolMessage(content=str(raw), tool_call_id=tc["id"]))

            msgs.extend(tool_messages)

            yield {
                "type": "llm_call",
                "step": "Gemini · Formulating response",
                "description": "Processing tool results and composing reply",
                "model": "gemini-2.5-flash-lite",
                "message_count": len(msgs),
            }

            try:
                response = llm.invoke(msgs)
            except Exception as exc:
                yield {"type": "response", "content": self._user_facing_error(exc), "calc_data": None}
                return

        calc_data = _extract_calc(msgs)
        yield {"type": "response", "content": response.content or "", "calc_data": calc_data}


def _summarise(tool_name: str, result) -> str:
    if isinstance(result, dict):
        if "total_zar" in result:
            n = len(result.get("line_items", []))
            return f"R {result['total_zar']:,.2f} total · {n} charge types calculated"
        if "applicable_charges" in result:
            return f"{len(result['applicable_charges'])} charges apply: {', '.join(result['applicable_charges'])}"
        if "results" in result:
            return f"{len(result.get('results', []))} rule passages retrieved from ChromaDB"
    return str(result)[:120]


def _extract_calc(msgs) -> dict | None:
    for msg in reversed(msgs):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and "total_zar" in data:
                    return data
            except Exception:
                pass
    return None


# ── Singleton ─────────────────────────────────────────────────────────────

_agent_instance: ChatAgent | None = None


def get_agent() -> ChatAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ChatAgent()
    return _agent_instance
