"""Integration tests for the FastAPI endpoints."""
import pytest
import httpx
import respx
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


GERMANY_PAYLOAD = [
    {
        "name": {"common": "Germany", "official": "Federal Republic of Germany"},
        "capital": ["Berlin"],
        "population": 83240525,
        "currencies": {"EUR": {"name": "Euro", "symbol": "€"}},
        "languages": {"deu": "German"},
        "region": "Europe",
        "subregion": "Western Europe",
        "flags": {"svg": "https://example.com/de.svg"},
        "area": 357114.0,
        "timezones": ["UTC+01:00"],
        "borders": ["AUT", "BEL", "CZE", "DNK", "FRA", "LUX", "NLD", "POL", "CHE"],
        "latlng": [51.0, 9.0],
        "independent": True,
        "unMember": True,
    }
]


def _make_mock_graph(answer: str, country: str = "Germany", fields: list = None, tool_status=None):
    from app.agent.state import IntentStatus, ToolStatus

    mock_result = {
        "user_query": "test",
        "intent_status": IntentStatus.VALID,
        "country_name": country,
        "requested_fields": fields or ["population"],
        "intent_error_msg": None,
        "tool_status": tool_status or ToolStatus.SUCCESS,
        "raw_api_data": GERMANY_PAYLOAD[0],
        "tool_error_msg": None,
        "final_answer": answer,
    }
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=mock_result)
    return graph


@pytest.fixture
def client_with_mock_graph():
    """FastAPI test client with a mocked LangGraph."""
    from app.main import create_app

    with patch("app.agent.graph.build_graph") as mock_build:
        mock_graph = _make_mock_graph("The population of Germany is approximately 83 million.")
        mock_build.return_value = mock_graph

        app = create_app()
        with TestClient(app) as client:
            # Override the graph on app state directly
            client.app.state.graph = mock_graph
            yield client


def test_health(client_with_mock_graph):
    response = client_with_mock_graph.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_ask_valid_query(client_with_mock_graph):
    response = client_with_mock_graph.post(
        "/api/ask", json={"query": "What is the population of Germany?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["country"] == "Germany"


def test_ask_empty_query(client_with_mock_graph):
    response = client_with_mock_graph.post("/api/ask", json={"query": ""})
    assert response.status_code == 422  # Pydantic min_length validation


def test_root_redirects(client_with_mock_graph):
    response = client_with_mock_graph.get("/", follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)
