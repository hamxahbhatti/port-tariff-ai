"""
MCP Server: tariff_rag

Two tools:
  - search_rules(question, port?)  → semantic search over prose + table descriptions
  - get_tariff_table(port, charge_type)  → exact numeric JSON for a charge type

The agent uses search_rules to find conditions/exemptions and get_tariff_table
to retrieve the actual rate rows for calculation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastmcp import FastMCP

import config
from knowledge_store import tariff_store, vector_store

mcp = FastMCP(
    name="tariff_rag",
    instructions=(
        "Use search_rules to find prose conditions, exemptions, and table descriptions. "
        "Use get_tariff_table to retrieve exact numeric rate rows for a specific charge type. "
        "Always use get_tariff_table for actual calculation — never rely on search_rules for numbers."
    ),
)


@mcp.tool()
def search_rules(question: str, port_name: str | None = None) -> str:
    """
    Semantic search over port tariff rules and table descriptions.

    Use this to answer questions like:
    - "What conditions apply to bulk carriers?"
    - "Are there exemptions for vessels in distress?"
    - "What is VTS and how is it calculated?"

    Args:
        question: Natural language question about tariff rules or conditions.
        port_name: Optional port name to narrow search (e.g. 'durban', 'cape_town').

    Returns:
        JSON array of matching passages with text and metadata.
    """
    try:
        results = vector_store.query(question, port_name=port_name, n_results=5)
        if not results:
            return json.dumps({"results": [], "message": "No matching rules found."})
        return json.dumps({"results": results}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "results": []})


@mcp.tool()
def get_tariff_table(port_name: str, charge_type: str) -> str:
    """
    Retrieve the exact numeric tariff table for a specific charge type at a port.

    Use this to get the rate rows needed for calculation. The returned JSON contains:
    - rows: list of rate rows with tonnage_band, values per port, unit, is_incremental flag
    - general_conditions: conditions applying to the whole table

    Args:
        port_name: Port name in lowercase_underscore format (e.g. 'durban', 'cape_town').
        charge_type: Charge type (e.g. 'pilotage', 'tug_assistance', 'vts', 'light_dues',
                     'port_dues', 'cargo_dues', 'berthing', 'running_of_lines', 'berth_dues').

    Returns:
        JSON with full tariff table including rows, units, and conditions. Returns an error
        message if the table does not exist in the store.
    """
    try:
        table = tariff_store.load_table(port_name, charge_type)
        if table is None:
            available = tariff_store.list_charge_types(port_name)
            return json.dumps({
                "error": f"No tariff table found for '{charge_type}' at '{port_name}'.",
                "available_charge_types": available,
            })
        return json.dumps(table, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def list_available_charges(port_name: str) -> str:
    """
    List all charge types available in the tariff store for a port.

    Args:
        port_name: Port name (e.g. 'durban').

    Returns:
        JSON list of available charge type names.
    """
    try:
        charges = tariff_store.list_charge_types(port_name)
        ports = tariff_store.list_ports()
        return json.dumps({
            "port": port_name,
            "available_charges": charges,
            "all_ports": ports,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
