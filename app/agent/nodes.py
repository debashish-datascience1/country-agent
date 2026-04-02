"""LangGraph node functions for the Country Information Agent."""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.agent.state import AgentState, IntentStatus, ToolStatus
from app.agent.tools import CountryAPIError, CountryNotFoundError, fetch_country
from app.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output (Node 1)
# ---------------------------------------------------------------------------

class IntentResult(BaseModel):
    is_valid: bool = Field(
        description="True if the query is a recognizable question about a country's attributes."
    )
    country_name: str | None = Field(
        None,
        description="The canonical English name of the country (e.g. 'France', 'South Korea'). "
                    "Null if the query is invalid or no country is mentioned.",
    )
    requested_fields: list[str] = Field(
        default_factory=list,
        description=(
            "List of data fields the user is asking about. "
            "Use these standard names where applicable: "
            "capital, population, area, currency, languages, flag, region, "
            "subregion, timezones, borders, coordinates, independence, un_member. "
            "Can include multiple fields."
        ),
    )
    was_corrected: bool = Field(
        False,
        description=(
            "True if the country name was inferred from a misspelling or typo "
            "(e.g. 'Cndia' interpreted as 'India'). False if the name was stated clearly."
        ),
    )
    error_reason: str | None = Field(
        None,
        description="Brief explanation if is_valid is False (e.g. 'No country name found in query').",
    )


# ---------------------------------------------------------------------------
# Node 1 — Intent / field identification
# ---------------------------------------------------------------------------

async def intent_node(state: AgentState) -> dict[str, Any]:
    """
    Identify the country and requested fields from the user query.
    Uses a structured LLM output call — no free-form generation.
    """
    settings = get_settings()
    llm = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=0,
    ).with_structured_output(IntentResult)

    system_prompt = (
        "You are an intent parser for a Country Information service. "
        "Given a user question, extract: (1) whether it is a valid country-info query, "
        "(2) the canonical English country name, and (3) which fields the user wants.\n\n"
        "Important rules:\n"
        "- If the country name is clearly misspelled but still recognizable "
        "(e.g. 'Cndia' → 'India', 'Germny' → 'Germany'), set is_valid=true, "
        "country_name to the correct spelling, and was_corrected=true.\n"
        "- If the input is too garbled to identify any country with confidence "
        "(e.g. 'dndia', 'xzqrt'), set is_valid=false.\n"
        "- Do not answer the question itself."
    )

    try:
        result: IntentResult = await llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["user_query"]},
            ]
        )
    except Exception as exc:
        logger.error("Intent node LLM call failed: %s", exc)
        return {
            "intent_status": IntentStatus.INVALID,
            "country_name": None,
            "requested_fields": [],
            "was_corrected": False,
            "intent_error_msg": "Failed to parse your query. Please try again.",
        }

    if result.is_valid and result.country_name:
        logger.info(
            "Intent identified — country: '%s', fields: %s, corrected: %s",
            result.country_name,
            result.requested_fields,
            result.was_corrected,
        )
        return {
            "intent_status": IntentStatus.VALID,
            "country_name": result.country_name,
            "requested_fields": result.requested_fields or ["general"],
            "was_corrected": result.was_corrected,
            "intent_error_msg": None,
        }

    logger.info("Invalid query: %s", result.error_reason)
    return {
        "intent_status": IntentStatus.INVALID,
        "country_name": result.country_name,
        "requested_fields": [],
        "was_corrected": False,
        "intent_error_msg": result.error_reason or "Could not understand the query.",
    }


# ---------------------------------------------------------------------------
# Node 2 — Tool invocation (REST Countries API)
# ---------------------------------------------------------------------------

async def tool_node(state: AgentState, http_client: httpx.AsyncClient) -> dict[str, Any]:
    """
    Fetch country data from the REST Countries API.
    Classifies errors into ToolStatus variants so synthesis can respond appropriately.
    """
    country_name: str = state["country_name"]  # guaranteed non-null by routing

    try:
        data = await fetch_country(country_name, http_client)
    except CountryNotFoundError as exc:
        logger.info("Country not found: %s", exc)
        return {
            "tool_status": ToolStatus.NOT_FOUND,
            "raw_api_data": None,
            "tool_error_msg": str(exc),
        }
    except CountryAPIError as exc:
        logger.warning("API error: %s", exc)
        return {
            "tool_status": ToolStatus.API_ERROR,
            "raw_api_data": None,
            "tool_error_msg": str(exc),
        }
    except Exception as exc:
        logger.error("Unexpected error in tool_node: %s", exc)
        return {
            "tool_status": ToolStatus.API_ERROR,
            "raw_api_data": None,
            "tool_error_msg": "Unexpected error while fetching country data.",
        }

    # Detect partial data: check if any requested field is missing
    requested = state.get("requested_fields", [])
    _field_map = {
        "capital": "capital",
        "population": "population",
        "area": "area",
        "currency": "currencies",
        "languages": "languages",
        "flag": "flags",
        "region": "region",
        "subregion": "subregion",
        "timezones": "timezones",
        "borders": "borders",
        "coordinates": "latlng",
    }
    missing = [
        f for f in requested
        if _field_map.get(f) and _field_map[f] not in data
    ]
    status = ToolStatus.PARTIAL_DATA if missing else ToolStatus.SUCCESS

    logger.info(
        "Tool node success for '%s' — status: %s, missing fields: %s",
        country_name,
        status,
        missing,
    )
    return {
        "tool_status": status,
        "raw_api_data": data,
        "tool_error_msg": None,
    }


# ---------------------------------------------------------------------------
# Node 3 — Answer synthesis
# ---------------------------------------------------------------------------

def _extract_relevant_data(raw: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Extract only the fields the user asked about from the raw API payload."""
    field_map: dict[str, str] = {
        "capital": "capital",
        "population": "population",
        "area": "area",
        "currency": "currencies",
        "languages": "languages",
        "flag": "flags",
        "region": "region",
        "subregion": "subregion",
        "timezones": "timezones",
        "borders": "borders",
        "coordinates": "latlng",
        "independence": "independent",
        "un_member": "unMember",
        "general": None,  # include name + region as fallback
    }

    if not fields or fields == ["general"]:
        # Return a curated subset for general queries
        return {
            "name": raw.get("name", {}).get("common"),
            "capital": raw.get("capital"),
            "population": raw.get("population"),
            "region": raw.get("region"),
            "subregion": raw.get("subregion"),
            "currencies": raw.get("currencies"),
            "languages": raw.get("languages"),
        }

    result: dict[str, Any] = {
        "name": raw.get("name", {}).get("common")
    }
    for field in fields:
        api_key = field_map.get(field)
        if api_key and api_key in raw:
            result[api_key] = raw[api_key]
        elif api_key and api_key not in raw:
            result[api_key] = None  # explicitly show missing

    return result


async def synthesis_node(state: AgentState) -> dict[str, Any]:
    """
    Generate the final human-readable answer using the LLM.
    Handles all tool_status variants gracefully.
    """
    settings = get_settings()
    llm = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=0.2,
    )

    query = state["user_query"]
    intent_status = state.get("intent_status")
    tool_status = state.get("tool_status")

    # --- Case 1: Invalid intent ---
    if intent_status == IntentStatus.INVALID:
        reason = state.get("intent_error_msg", "Could not understand the query.")
        prompt = (
            f"The user asked: \"{query}\"\n\n"
            f"This could not be answered because: {reason}\n\n"
            "Politely explain that you can only answer questions about countries "
            "(e.g. population, capital, currency) and give an example of a valid question."
        )
        answer = await _call_llm(llm, prompt)
        return {"final_answer": answer}

    # --- Case 2: Country not found ---
    if tool_status == ToolStatus.NOT_FOUND:
        country = state.get("country_name", "the specified country")
        prompt = (
            f"The user asked: \"{query}\"\n\n"
            f"A search for '{country}' in the REST Countries database returned no results.\n\n"
            "Politely inform the user that this country could not be found and suggest "
            "they check the spelling or try an alternative name."
        )
        answer = await _call_llm(llm, prompt)
        return {"final_answer": answer}

    # --- Case 3: API error ---
    if tool_status == ToolStatus.API_ERROR:
        prompt = (
            f"The user asked: \"{query}\"\n\n"
            "The external country data service is temporarily unavailable.\n\n"
            "Apologize briefly and ask them to try again in a moment."
        )
        answer = await _call_llm(llm, prompt)
        return {"final_answer": answer}

    # --- Case 4: Success or partial data ---
    raw = state.get("raw_api_data", {})
    fields = state.get("requested_fields", [])
    data_subset = _extract_relevant_data(raw, fields)
    data_json = json.dumps(data_subset, ensure_ascii=False, indent=2)

    correction_note = ""
    if state.get("was_corrected"):
        original = query  # the raw user input already contains the typo
        country = state.get("country_name", "")
        correction_note = (
            f"\nNote: The user's input appeared to contain a misspelling. "
            f"You interpreted it as '{country}'. "
            f"Begin your answer by briefly acknowledging this, e.g. "
            f"\"I interpreted your query as being about {country}.\""
        )

    partial_note = ""
    if tool_status == ToolStatus.PARTIAL_DATA:
        partial_note = (
            "\nNote: Some requested fields are missing from the data (shown as null). "
            "Acknowledge what is unavailable instead of guessing."
        )

    prompt = (
        f"The user asked: \"{query}\"\n\n"
        f"Here is the relevant country data (JSON):\n{data_json}\n"
        f"{correction_note}{partial_note}\n\n"
        "Answer the user's question directly and naturally using only the data above. "
        "Do not add information that is not in the data. Be concise."
    )

    answer = await _call_llm(llm, prompt)
    return {"final_answer": answer}


async def _call_llm(llm: ChatGroq, prompt: str) -> str:
    """Helper to invoke the LLM and return the text content."""
    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content
    except Exception as exc:
        logger.error("Synthesis LLM call failed: %s", exc)
        return "I'm sorry, I encountered an error while generating the answer. Please try again."
