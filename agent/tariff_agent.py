"""
Port Tariff Calculation Agent (LangGraph ReAct)

Architecture:
  LangGraph StateGraph with a ReAct loop:
    identify_charges → (for each charge) retrieve_table → compute → validate → aggregate

The agent is given a vessel profile and a port name. It:
  1. Calls rules_engine to determine which charges apply and get modifiers
  2. For each applicable charge:
     a. Retrieves the tariff table from tariff_rag
     b. Calls the calculator tool with the table rows + vessel params
  3. Aggregates all charges into a final invoice
  4. Returns a structured JSON result with line-by-line breakdown

The LLM (Gemini) acts as the orchestrator — deciding tool call sequences,
handling edge cases, and generating the final narrative summary.
Tools are bound as LangChain tools so LangGraph can route them automatically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

import config

logger = logging.getLogger(__name__)

# ── Tool imports from MCP servers ──────────────────────────────────────────
# We import the underlying functions directly (not via MCP transport) for
# local use. For a production deployment these would be called via MCP client.

from mcp_servers.calculator.server import (
    aggregate_charges,
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
    check_exemptions,
    determine_applicable_charges,
    get_vessel_charge_plan,
)
from mcp_servers.tariff_rag.server import (
    get_tariff_table,
    list_available_charges,
    search_rules,
)
from mcp_servers.vessel.server import (
    classify_vessel_for_tariff,
    register_vessel,
)


# ── Wrap as LangChain tools ────────────────────────────────────────────────

def _st(fn) -> StructuredTool:
    """Wrap a function as a StructuredTool, using its docstring as description."""
    return StructuredTool.from_function(func=fn)


# Build the tool list the agent can call
AGENT_TOOLS = [
    # Rules
    _st(get_vessel_charge_plan),
    _st(determine_applicable_charges),
    _st(check_exemptions),

    # Data retrieval
    _st(get_tariff_table),
    _st(list_available_charges),
    _st(search_rules),

    # Vessel
    _st(register_vessel),
    _st(classify_vessel_for_tariff),

    # Calculator
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


# ── Agent state ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    vessel_profile: dict
    port_name: str
    final_result: dict | None


# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a South African port tariff calculation specialist.

Your task: given a vessel profile, calculate ALL applicable port dues for a port call.

WORKFLOW (follow in order):
1. Call get_vessel_charge_plan with the vessel profile JSON to get the charge list and modifiers
2. Call list_available_charges to confirm what's in the tariff store
3. For EACH applicable charge that is NOT skipped/exempt:
   a. Call get_tariff_table(port, charge_type) to get the rate rows
   b. Call the matching calculate_* tool with rows_json and vessel parameters
4. Collect ALL charge results into a JSON array
5. Call aggregate_charges with the array to get the total
6. Return a final structured response

IMPORTANT RULES:
- Use EXACT numeric values from tariff tables — never estimate
- For pilotage and tug_assistance: use is_incremental rows with parent_band matching
- For VTS: the table is ports_as_rows — rows have a 'port' field, not column headers
- Always pass the full 'rows' array from the tariff table as rows_json (JSON stringified)
- If a tariff table is missing (store is empty), note it clearly and skip that charge
- At the end, produce a JSON invoice with all line items and total_zar

VESSEL PARAMETERS you will receive:
- name, vessel_type, gt (Gross Tonnage), loa_m (Length Overall), port
- cargo_operation, cargo_type, cargo_mt, berthing, hours_alongside
"""


# ── Graph nodes ─────────────────────────────────────────────────────────────

def _build_llm():
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_TEXT_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0,
    ).bind_tools(AGENT_TOOLS)


def agent_node(state: AgentState) -> dict:
    """Call the LLM with the current message history."""
    llm = _build_llm()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route: if last message has tool calls → tools node, else → END."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


def extract_final_result(state: AgentState) -> dict:
    """Parse the final agent message to extract the structured result."""
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # Try to extract JSON from the response
    try:
        # Look for JSON block in the content
        import re
        json_match = re.search(r'\{[\s\S]*"total_zar"[\s\S]*\}', content)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    return {"raw_response": content}


# ── Graph construction ───────────────────────────────────────────────────────

def build_graph():
    tool_node = ToolNode(AGENT_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Public API ───────────────────────────────────────────────────────────────

def calculate_port_dues_for_vessel(
    vessel_profile: dict,
    port_name: str | None = None,
) -> dict:
    """
    Main entry point: calculate all port dues for a vessel.

    Args:
        vessel_profile: Dict with vessel details (name, gt, loa_m, vessel_type, port, etc.)
        port_name:      Override port from vessel_profile if needed.

    Returns:
        Dict with total_zar, line_items, and full message history.
    """
    if port_name:
        vessel_profile = {**vessel_profile, "port": port_name}

    graph = build_graph()

    # Initial message to the agent
    user_msg = HumanMessage(content=(
        f"Calculate all applicable port dues for this vessel:\n\n"
        f"```json\n{json.dumps(vessel_profile, indent=2)}\n```\n\n"
        f"Follow the workflow: get charge plan → retrieve each tariff table → "
        f"calculate each charge → aggregate → return structured JSON invoice."
    ))

    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    initial_state: AgentState = {
        "messages": [system_msg, user_msg],
        "vessel_profile": vessel_profile,
        "port_name": vessel_profile.get("port", "durban"),
        "final_result": None,
    }

    logger.info(f"Agent: starting calculation for {vessel_profile.get('name')} at {vessel_profile.get('port')}")

    final_state = graph.invoke(initial_state, config={"recursion_limit": 50})

    result = extract_final_result(final_state)
    result["_message_count"] = len(final_state["messages"])

    return result


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    SUDESTADA = {
        "name": "SUDESTADA",
        "vessel_type": "bulk_carrier",
        "gt": 51300,
        "loa_m": 229.2,
        "port": "durban",
        "cargo_operation": True,
        "cargo_type": "iron_ore",
        "cargo_mt": 75000,
        "berthing": True,
        "hours_alongside": 48,
    }

    print("=" * 60)
    print("  PORT TARIFF AGENT — SUDESTADA @ PORT OF DURBAN")
    print("=" * 60)

    result = calculate_port_dues_for_vessel(SUDESTADA)
    print(json.dumps(result, indent=2))
