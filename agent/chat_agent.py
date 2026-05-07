"""
Conversational port tariff agent with memory.

Uses the full 17-tool set (same as tariff_agent) so the agent can answer
both calculation requests and rule/exemption/policy questions via ChromaDB.
Tools are imported lazily inside each wrapper to avoid module-level
initialisation issues in the Railway async environment.

Session memory is maintained per session_id in an in-process dict.
"""

from __future__ import annotations

import json
import logging
from typing import Generator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI

import config

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Transnet National Ports Authority tariff assistant.
You help shipping agents and vessel operators with two types of requests:

1. CALCULATIONS — calculating exact port dues for a vessel call at a South African port.
2. RULE / POLICY QUESTIONS — answering questions about exemptions, conditions, surcharges,
   special vessel categories, and how charges work, using the official tariff rules.

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
  1. Call get_vessel_charge_plan to get the applicable charges and modifiers.
  2. For each charge: call get_tariff_table then the matching calculate_* tool.
  3. Call aggregate_charges at the end to sum the result.

For rule / policy / exemption questions:
  - Call search_rules with a clear natural-language query to retrieve relevant tariff prose.
  - Use the returned passages to answer the question accurately.
  - Do not invent rules — only use what search_rules returns.

RESPONSE FORMAT:
- Be concise and professional.
- For calculations: give a brief summary then present the line-item breakdown.
- For rule questions: answer directly from the retrieved passages.
- Never estimate or invent a tariff rate or rule.

Supported ports: Durban (Richards Bay, Cape Town and others coming soon)."""


# ── Tool wrappers (lazy imports keep module load lightweight) ──────────────

def _st(fn) -> StructuredTool:
    return StructuredTool.from_function(func=fn)


def get_vessel_charge_plan(vessel_profile_json: str) -> str:
    """Get the full charge plan and modifiers for a vessel at a port."""
    from mcp_servers.rules_engine.server import get_vessel_charge_plan as _fn
    return _fn(vessel_profile_json)


def determine_applicable_charges(vessel_profile_json: str) -> str:
    """Determine which charges apply to a vessel at a given South African port."""
    from mcp_servers.rules_engine.server import determine_applicable_charges as _fn
    return _fn(vessel_profile_json)


def check_exemptions(vessel_profile_json: str) -> str:
    """Check whether any exemptions apply to a vessel."""
    from mcp_servers.rules_engine.server import check_exemptions as _fn
    return _fn(vessel_profile_json)


def get_tariff_table(port_name: str, charge_type: str) -> str:
    """Retrieve the exact numeric rate rows for a charge type at a port."""
    from mcp_servers.tariff_rag.server import get_tariff_table as _fn
    return _fn(port_name, charge_type)


def list_available_charges(port_name: str) -> str:
    """List all charge types available in the tariff store for a port."""
    from mcp_servers.tariff_rag.server import list_available_charges as _fn
    return _fn(port_name)


def search_rules(question: str, port_name: str | None = None) -> str:
    """
    Semantic search over port tariff rules, exemptions, conditions,
    and table descriptions. Use this for any policy or exemption question.
    """
    from mcp_servers.tariff_rag.server import search_rules as _fn
    return _fn(question, port_name)


def register_vessel(vessel_profile_json: str) -> str:
    """Register and validate a vessel profile."""
    from mcp_servers.vessel.server import register_vessel as _fn
    return _fn(vessel_profile_json)


def classify_vessel_for_tariff(vessel_profile_json: str) -> str:
    """Classify a vessel type and GT band for tariff purposes."""
    from mcp_servers.vessel.server import classify_vessel_for_tariff as _fn
    return _fn(vessel_profile_json)


def calculate_light_dues(rows_json: str, gt: float, port: str) -> str:
    """Calculate light dues for a vessel."""
    from mcp_servers.calculator.server import calculate_light_dues as _fn
    return _fn(rows_json, gt, port)


def calculate_vts(rows_json: str, gt: float, port: str) -> str:
    """Calculate Vessel Traffic Services dues."""
    from mcp_servers.calculator.server import calculate_vts as _fn
    return _fn(rows_json, gt, port)


def calculate_pilotage(rows_json: str, gt: float, port: str, movements: int = 2) -> str:
    """Calculate pilotage dues for a vessel."""
    from mcp_servers.calculator.server import calculate_pilotage as _fn
    return _fn(rows_json, gt, port, movements)


def calculate_tug_assistance(rows_json: str, gt: float, port: str,
                              tug_count: int = 3, movements: int = 2) -> str:
    """Calculate tug assistance dues."""
    from mcp_servers.calculator.server import calculate_tug_assistance as _fn
    return _fn(rows_json, gt, port, tug_count, movements)


def calculate_port_dues(rows_json: str, gt: float, port: str) -> str:
    """Calculate port dues for a vessel."""
    from mcp_servers.calculator.server import calculate_port_dues as _fn
    return _fn(rows_json, gt, port)


def calculate_cargo_dues(rows_json: str, cargo_mt: float,
                          cargo_type: str, port: str) -> str:
    """Calculate cargo dues based on metric tonnes and cargo type."""
    from mcp_servers.calculator.server import calculate_cargo_dues as _fn
    return _fn(rows_json, cargo_mt, cargo_type, port)


def calculate_berth_dues(rows_json: str, gt: float, port: str,
                          hours_alongside: float = 24) -> str:
    """Calculate berth dues for a vessel."""
    from mcp_servers.calculator.server import calculate_berth_dues as _fn
    return _fn(rows_json, gt, port, hours_alongside)


def calculate_running_of_lines(rows_json: str, port: str,
                                 services: int = 2) -> str:
    """Calculate running of lines charge."""
    from mcp_servers.calculator.server import calculate_running_of_lines as _fn
    return _fn(rows_json, port, services)


def aggregate_charges(charges_json: str) -> str:
    """Aggregate all calculated charges into a final total."""
    from mcp_servers.calculator.server import aggregate_charges as _fn
    return _fn(charges_json)


AGENT_TOOLS = [
    _st(get_vessel_charge_plan),
    _st(determine_applicable_charges),
    _st(check_exemptions),
    _st(get_tariff_table),
    _st(list_available_charges),
    _st(search_rules),
    _st(register_vessel),
    _st(classify_vessel_for_tariff),
    _st(calculate_light_dues),
    _st(calculate_vts),
    _st(calculate_pilotage),
    _st(calculate_tug_assistance),
    _st(calculate_port_dues),
    _st(calculate_cargo_dues),
    _st(calculate_berth_dues),
    _st(calculate_running_of_lines),
    _st(aggregate_charges),
]

TOOL_BY_NAME = {t.name: t for t in AGENT_TOOLS}


# ── Agent class ───────────────────────────────────────────────────────────

class ChatAgent:
    """Conversational agent with per-session memory and full tool access."""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            base = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash-lite",
                temperature=0,
                google_api_key=config.GEMINI_API_KEY,
            )
            self._llm = base.bind_tools(AGENT_TOOLS)
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

        while response.tool_calls and iteration < 20:
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

                tool_fn = TOOL_BY_NAME.get(tc["name"])
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
        if "results" in result:
            hits = result.get("results", [])
            return f"{len(hits)} rule passages retrieved from ChromaDB"
    if isinstance(result, str) and len(result) > 20:
        return result[:120]
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
