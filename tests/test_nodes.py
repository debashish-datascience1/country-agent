"""Unit tests for individual LangGraph nodes (synthesis logic)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.nodes import _extract_relevant_data, synthesis_node
from app.agent.state import AgentState, IntentStatus, ToolStatus


GERMANY_DATA = {
    "name": {"common": "Germany"},
    "capital": ["Berlin"],
    "population": 83240525,
    "currencies": {"EUR": {"name": "Euro", "symbol": "€"}},
    "languages": {"deu": "German"},
    "region": "Europe",
    "subregion": "Western Europe",
}


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "user_query": "What is the capital of Germany?",
        "intent_status": IntentStatus.VALID,
        "country_name": "Germany",
        "requested_fields": ["capital"],
        "intent_error_msg": None,
        "tool_status": ToolStatus.SUCCESS,
        "raw_api_data": GERMANY_DATA,
        "tool_error_msg": None,
        "final_answer": None,
    }
    base.update(overrides)
    return base


class TestExtractRelevantData:
    def test_extracts_capital(self):
        result = _extract_relevant_data(GERMANY_DATA, ["capital"])
        assert "capital" in result
        assert result["capital"] == ["Berlin"]

    def test_extracts_currency(self):
        result = _extract_relevant_data(GERMANY_DATA, ["currency"])
        assert "currencies" in result

    def test_general_query_returns_curated_subset(self):
        result = _extract_relevant_data(GERMANY_DATA, ["general"])
        assert "name" in result
        assert "population" in result
        assert "capital" in result

    def test_missing_field_returns_none(self):
        result = _extract_relevant_data(GERMANY_DATA, ["borders"])
        assert result.get("borders") is None

    def test_empty_fields_returns_general(self):
        result = _extract_relevant_data(GERMANY_DATA, [])
        assert "name" in result


@pytest.mark.asyncio
async def test_synthesis_invalid_intent():
    state = _make_state(
        intent_status=IntentStatus.INVALID,
        intent_error_msg="No country name found in query.",
        user_query="What is the weather like?",
    )
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Please ask about a country."))

    with patch("app.agent.nodes.ChatAnthropic", return_value=mock_llm):
        result = await synthesis_node(state)

    assert result["final_answer"] == "Please ask about a country."


@pytest.mark.asyncio
async def test_synthesis_not_found():
    state = _make_state(
        tool_status=ToolStatus.NOT_FOUND,
        country_name="Xlandia",
        raw_api_data=None,
    )
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(content="Xlandia could not be found.")
    )

    with patch("app.agent.nodes.ChatAnthropic", return_value=mock_llm):
        result = await synthesis_node(state)

    assert "Xlandia" in result["final_answer"]


@pytest.mark.asyncio
async def test_synthesis_success():
    state = _make_state()
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(content="The capital of Germany is Berlin.")
    )

    with patch("app.agent.nodes.ChatAnthropic", return_value=mock_llm):
        result = await synthesis_node(state)

    assert result["final_answer"] == "The capital of Germany is Berlin."


@pytest.mark.asyncio
async def test_synthesis_llm_failure_returns_fallback():
    state = _make_state()
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))

    with patch("app.agent.nodes.ChatAnthropic", return_value=mock_llm):
        result = await synthesis_node(state)

    assert result["final_answer"] is not None
    assert "error" in result["final_answer"].lower()
