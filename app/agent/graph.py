"""LangGraph graph assembly for the Country Information Agent."""
from __future__ import annotations

import logging
from functools import partial
from typing import Any

import httpx
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import intent_node, synthesis_node, tool_node
from app.agent.state import AgentState, IntentStatus

logger = logging.getLogger(__name__)


def _route_after_intent(state: AgentState) -> str:
    """
    After intent_node:
    - VALID   → call the REST Countries API
    - INVALID → skip the API, go straight to synthesis
    """
    if state.get("intent_status") == IntentStatus.VALID:
        return "tool"
    return "synthesis"


def build_graph(http_client: httpx.AsyncClient):
    """
    Build and compile the LangGraph for the Country Information Agent.

    The http_client is injected at graph-build time so tool_node can
    reuse a single shared AsyncClient across all requests.

    Graph flow:
        START → intent → [tool | synthesis] → synthesis → END
    """
    # Bind the shared HTTP client into tool_node at startup
    bound_tool_node = partial(tool_node, http_client=http_client)

    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("intent", intent_node)
    builder.add_node("tool", bound_tool_node)
    builder.add_node("synthesis", synthesis_node)

    builder.add_edge(START, "intent")

    builder.add_conditional_edges(
        "intent",
        _route_after_intent,
        {"tool": "tool", "synthesis": "synthesis"},
    )

    builder.add_edge("tool", "synthesis")
    builder.add_edge("synthesis", END)

    compiled = builder.compile()
    logger.info("LangGraph compiled successfully")
    return compiled


async def run_agent(graph: Any, user_query: str) -> dict[str, Any]:
    """
    Invoke the compiled graph with the user query and return the final state.
    """
    initial_state: AgentState = {
        "user_query": user_query,
        "intent_status": None,
        "country_name": None,
        "requested_fields": [],
        "intent_error_msg": None,
        "was_corrected": False,
        "tool_status": None,
        "raw_api_data": None,
        "tool_error_msg": None,
        "final_answer": None,
    }

    logger.info("Running agent for query: %r", user_query)
    result: AgentState = await graph.ainvoke(initial_state)
    logger.info("Agent completed — answer: %r", result.get("final_answer", "")[:80])
    return result
