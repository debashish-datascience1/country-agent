# Country Information AI Agent

An AI agent built with **LangGraph** and **Claude** that answers natural language questions about countries using the public [REST Countries API](https://restcountries.com).

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      LangGraph Agent                        │
│                                                             │
│  ┌──────────────┐     ┌──────────────┐   ┌──────────────┐  │
│  │ intent_node  │────▶│  tool_node   │──▶│synthesis_node│  │
│  │              │     │              │   │              │  │
│  │ LLM extracts │     │ Calls REST   │   │ LLM formats  │  │
│  │ country name │     │ Countries API│   │ the answer   │  │
│  │ + fields     │     │ (httpx)      │   │              │  │
│  └──────────────┘     └──────────────┘   └──────────────┘  │
│        │ (invalid intent)                        ▲          │
│        └────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Final Answer
```

### Agent Nodes

| Node | Role | LLM? |
|------|------|------|
| `intent_node` | Extracts country name and requested fields via structured output | Yes |
| `tool_node` | Fetches data from REST Countries API | No |
| `synthesis_node` | Generates a natural language answer from the data | Yes |

### Conditional Routing

- If intent is **invalid** (no country found in query) → skip API call → synthesis explains the issue
- If intent is **valid** → fetch API → synthesis answers using real data
- All error cases (API not found, timeout, partial data) are handled gracefully in synthesis

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd cloudeagle
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

### 3. Run

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to use the interactive API docs.

### 4. Example requests

```bash
# Capital and currency
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital and currency of Japan?"}'

# Population
curl -X POST http://localhost:8000/ask \
  -d '{"query": "What is the population of Germany?"}' \
  -H "Content-Type: application/json"

# Multiple fields
curl -X POST http://localhost:8000/ask \
  -d '{"query": "What languages are spoken in Brazil and what is its population?"}' \
  -H "Content-Type: application/json"

# Health check
curl http://localhost:8000/health
```

### Example response

```json
{
  "answer": "The capital of Japan is Tokyo. Japan uses the Japanese Yen (JPY, ¥) as its currency.",
  "country": "Japan",
  "fields_requested": ["capital", "currency"],
  "tool_status": "success"
}
```

## Running Tests

```bash
pip install pytest pytest-asyncio respx
pytest tests/ -v
```

## Docker

```bash
docker build -t country-agent .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_key country-agent
```

## Production Considerations

- **Stateless**: The agent has no persistent state — safe to scale horizontally behind a load balancer.
- **Shared HTTP client**: `httpx.AsyncClient` is created once at startup (via FastAPI lifespan) and reused across requests — avoids connection overhead.
- **Structured output**: The intent node uses `with_structured_output()` to ensure deterministic field extraction (no JSON parsing errors).
- **Error containment**: Every node catches its own exceptions and writes them into the state. `synthesis_node` always runs and always produces an answer, even for error cases.
- **Token efficiency**: Only the fields the user asked about are passed to the synthesis LLM, not the full API payload.

## Known Limitations & Trade-offs

- **Country name disambiguation**: The REST Countries API is queried by name. Ambiguous names (e.g., "Congo") may return the first match. A production system could add a disambiguation step.
- **No caching**: Each query hits the REST Countries API live. A Redis TTL cache on `country_name` would reduce latency and API load.
- **LLM latency**: Two LLM calls per request (intent + synthesis). Intent could be replaced with a cheaper regex/NER approach for speed-sensitive deployments.
- **English only**: The intent node is English-optimized. Multi-language support would require prompt changes.
- **API dependency**: The service degrades gracefully when REST Countries is down, but cannot answer without it.
