"""POST /ask router."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.agent.graph import run_agent
from app.agent.state import ToolStatus
from app.models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
async def ask(request: Request, body: QueryRequest) -> QueryResponse:
    """
    Answer a question about a country.

    Example questions:
    - "What is the population of Germany?"
    - "What currency does Japan use?"
    - "What is the capital and population of Brazil?"
    """
    graph = request.app.state.graph

    try:
        result = await run_agent(graph, body.query)
    except Exception as exc:
        logger.exception("Unhandled error in agent for query %r: %s", body.query, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal agent error. Please try again."},
        )

    tool_status_value = (
        result.get("tool_status").value
        if isinstance(result.get("tool_status"), ToolStatus)
        else result.get("tool_status")
    )

    return QueryResponse(
        answer=result.get("final_answer") or "No answer was generated.",
        country=result.get("country_name"),
        fields_requested=result.get("requested_fields") or [],
        tool_status=tool_status_value,
    )
