"""
Conversational port tariff agent with memory.

Uses Gemini (flash-lite) + LangChain tool-calling.
Supports iterative follow-up questions and remembers vessel parameters
across conversation turns.

Tools exposed to the LLM:
  - determine_applicable_charges  (rules engine)
  - calculate_all_dues            (full deterministic calculation)
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

SYSTEM_PROMPT = """You are an expert Transnet National Ports Authority tariff calculator.
You help shipping agents and vessel operators calculate exact port dues for vessels
calling at South African ports using the official 2024/25 tariff schedule.

CONVERSATION RULES:
1. Extract vessel parameters from what the user tells you (name, type, GT, port, cargo, berth hours).
2. If required information is missing, ask ONE specific question — not a list of questions.
3. Required to calculate: vessel type, gross tonnage (GT), port of call.
4. Also needed: cargo type + metric tonnes (if cargo is being worked), hours alongside berth.
5. When you have enough information, call your tools and calculate immediately.
6. Remember all vessel details from earlier in the conversation.
7. For follow-ups like "what if 72 hours?" — use remembered vessel params, update only what changed.

TOOL USAGE:
- Call determine_applicable_charges first to see which charges apply.
- Then call calculate_all_dues with the full vessel details.
- Never invent or estimate tariff rates — always use the tools.

RESPONSE FORMAT:
- Be concise and professional.
- When presenting a calculation, give a one-line summary then the tool will render a card.
- For follow-up questions, ask naturally — one question at a time.
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
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from mcp_servers.rules_engine.server import (
        determine_applicable_charges as _fn,
    )

    profile = {
        "name": "vessel",
        "vessel_type": vessel_type,
        "gt": gt,
        "loa_m": 200,
        "port": port,
        "cargo_operation": has_cargo,
        "cargo_type": "",
        "cargo_mt": 0,
        "berthing": requesting_berth,
        "hours_alongside": 24,
        "in_distress": False,
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
    with each charge type, formula, and total.

    Args:
        vessel_type: bulk_carrier | tanker | container | general_cargo | ro_ro | passenger
        gt: Gross Tonnage
        port: Port name (e.g. 'durban')
        cargo_type: Cargo description (e.g. 'iron_ore', 'coal', 'containers') — empty if no cargo
        cargo_mt: Cargo in metric tonnes — 0 if no cargo
        hours_alongside: Hours the vessel will spend at berth
        loa_m: Length Overall in metres (defaults to 200)
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from knowledge_store.tariff_store import load_table
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
    from mcp_servers.rules_engine.server import (
        determine_applicable_charges as _rules,
    )

    has_cargo = bool(cargo_mt > 0 and cargo_type.strip())
    profile = {
        "name": "vessel",
        "vessel_type": vessel_type,
        "gt": gt,
        "loa_m": loa_m,
        "port": port,
        "cargo_operation": has_cargo,
        "cargo_type": cargo_type,
        "cargo_mt": cargo_mt,
        "berthing": True,
        "hours_alongside": hours_alongside,
        "in_distress": False,
    }
    plan = json.loads(_rules(json.dumps(profile)))
    applicable = plan.get("applicable_charges", [])
    modifiers = plan.get("modifiers", {})

    CALCS = {
        "light_dues": lambda r, p: calculate_light_dues(r, gt, p),
        "vts": lambda r, p: calculate_vts(r, gt, p),
        "pilotage": lambda r, p: calculate_pilotage(
            r, gt, p, modifiers.get("pilotage_movements", 2)
        ),
        "tug_assistance": lambda r, p: calculate_tug_assistance(
            r, gt, p,
            modifiers.get("tug_count", 3),
            modifiers.get("tug_movements", 2),
        ),
        "port_dues": lambda r, p: calculate_port_dues(r, gt, p),
        "cargo_dues": lambda r, p: calculate_cargo_dues(r, cargo_mt, cargo_type, p),
        "berth_dues": lambda r, p: calculate_berth_dues(r, gt, p, hours_alongside),
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
                results.append({
                    "charge_type": charge,
                    "charge_zar": r.get("charge_zar", 0),
                    "formula": r.get("formula", ""),
                })
        except Exception as e:
            errors.append({"charge_type": charge, "error": str(e)})

    total = sum(r["charge_zar"] for r in results)

    return json.dumps({
        "success": True,
        "port": port,
        "vessel_type": vessel_type,
        "gt": gt,
        "cargo_type": cargo_type,
        "cargo_mt": cargo_mt,
        "hours_alongside": hours_alongside,
        "total_zar": round(total, 2),
        "line_items": results,
        "errors": errors,
    })


# ── Agent class ───────────────────────────────────────────────────────────

TOOLS = [determine_applicable_charges, calculate_all_dues]


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
        """Convert an LLM exception into a readable message for the user."""
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            import re
            delay = re.search(r"retry in (\d+)", msg)
            wait = f" Please try again in {delay.group(1)} seconds." if delay else ""
            return f"⚠ The AI model has hit its rate limit.{wait}"
        if "INVALID_ARGUMENT" in msg:
            return "⚠ The request was invalid. Please rephrase your question."
        if "UNAVAILABLE" in msg or "503" in msg:
            return "⚠ The AI model is temporarily unavailable. Please retry in a moment."
        return f"⚠ AI model error: {msg[:200]}"

    def run(
        self,
        history: list[dict],
        message: str,
    ) -> Generator[dict, None, None]:
        """
        Run one conversational turn.

        Args:
            history: Previous turns [{role: user|assistant, content: str}]
            message: The new user message

        Yields:
            Debug event dicts with 'type' field.
            Final event has type='response'.
        """
        llm = self._get_llm()
        tool_map = {t.name: t for t in TOOLS}

        # Build message list
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

        # Agentic loop
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

                tool_fn = tool_map.get(tc["name"])
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

                tool_messages.append(
                    ToolMessage(content=str(raw), tool_call_id=tc["id"])
                )

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

        # Extract calculation data from tool results (for card rendering)
        calc_data = _extract_calc(msgs)

        yield {
            "type": "response",
            "content": response.content or "",
            "calc_data": calc_data,
        }


def _summarise(tool_name: str, result) -> str:
    if isinstance(result, dict):
        if "total_zar" in result:
            n = len(result.get("line_items", []))
            return f"R {result['total_zar']:,.2f} total · {n} charge types calculated"
        if "applicable_charges" in result:
            charges = result["applicable_charges"]
            return f"{len(charges)} charges apply: {', '.join(charges)}"
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
