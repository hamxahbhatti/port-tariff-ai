"""
Conversational port tariff agent with memory.

Uses the full LangGraph tool set (17 tools) so the agent can answer both
calculation requests and rule/exemption/policy questions via ChromaDB.
Session memory is maintained per session_id in an in-process dict.

Tools available (same set as tariff_agent):
  Rules Engine  — determine_applicable_charges, check_exemptions, get_vessel_charge_plan
  Tariff RAG    — get_tariff_table, list_available_charges, search_rules (ChromaDB)
  Calculator    — 8 deterministic charge functions + aggregate_charges
  Vessel        — register_vessel, classify_vessel_for_tariff
"""

from __future__ import annotations

import json
import logging
from typing import Generator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from agent.tariff_agent import AGENT_TOOLS, TOOL_BY_NAME

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

        # Build message list from history + new message
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
