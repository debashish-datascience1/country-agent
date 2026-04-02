# Country Information AI Agent

A production-grade AI agent that answers natural language questions about any country — built with **LangGraph**, **Groq (LLaMA 3.3 70B)**, and the public [REST Countries API](https://restcountries.com).

**Live Demo:** [https://country-agent.onrender.com](https://country-agent.onrender.com)

---

## What it does

Ask questions like:
- *"What is the capital and currency of Japan?"*
- *"What languages are spoken in Switzerland?"*
- *"What is the population of Brazil?"*

The agent understands natural language, fetches real data, and returns a grounded plain-English answer. Typos are handled gracefully — asking about *"Cndia"* returns the correct answer for India with a note that the name was corrected.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                       LangGraph Agent                        │
│                                                              │
│  ┌─────────────┐      ┌─────────────┐    ┌───────────────┐  │
│  │ intent_node │─────▶│  tool_node  │───▶│synthesis_node │  │
│  │             │      │             │    │               │  │
│  │ LLM detects │      │ REST        │    │ LLM formats   │  │
│  │ country +   │      │ Countries   │    │ plain-English │  │
│  │ fields      │      │ API (httpx) │    │ answer        │  │
│  └─────────────┘      └─────────────┘    └───────────────┘  │
│        │ (invalid / unrecognised)                 ▲          │
│        └─────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Final Answer
```

### The three nodes

| Node | Responsibility | Uses LLM |
|------|---------------|----------|
| `intent_node` | Extracts country name and requested fields using structured output. Detects and flags typos (`was_corrected`). | Yes |
| `tool_node` | Calls REST Countries API. Tries exact match first (`fullText=true`), falls back to scored partial match to avoid wrong results for "China", "India", etc. | No |
| `synthesis_node` | Produces a grounded natural language answer. Acknowledges typo corrections, partial data, and API errors transparently. | Yes |

### Conditional routing

```
intent_node
    ├── VALID   → tool_node → synthesis_node
    └── INVALID → synthesis_node  (API call skipped)
```

Error cases (`not_found`, `api_error`, `partial_data`) are classified in `tool_node` and handled gracefully in `synthesis_node` — the agent always returns a meaningful response.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph |
| LLM | Groq — LLaMA 3.3 70B Versatile |
| API framework | FastAPI (async) |
| HTTP client | httpx (async, shared client) |
| Data source | REST Countries API (free, no auth) |
| Config | pydantic-settings |
| Deployment | Render |

---

## Local Setup

### 1. Clone and install

```bash
git clone https://github.com/debashish-datascience1/country-agent.git
cd country-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq key at [console.groq.com](https://console.groq.com) — no credit card required.

### 3. Run

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) for the web UI, or [http://localhost:8000/docs](http://localhost:8000/docs) for the API docs.

---

## API Reference

### `POST /api/ask`

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital and currency of Japan?"}'
```

**Response**
```json
{
  "answer": "The capital of Japan is Tokyo. Japan uses the Japanese Yen (JPY, ¥).",
  "country": "Japan",
  "fields_requested": ["capital", "currency"],
  "tool_status": "success"
}
```

**`tool_status` values**

| Value | Meaning |
|-------|---------|
| `success` | Full data returned |
| `partial_data` | Country found, some requested fields missing |
| `not_found` | Country not found in REST Countries API |
| `api_error` | Network error or API unavailable |

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "model": "llama-3.3-70b-versatile"}
```

---

## Example Scenarios

**Normal query**
```
Q: What languages are spoken in Switzerland?
A: Switzerland has four official languages: German, French, Italian, and Romansh.
```

**Typo correction**
```
Q: What is the capital of Cndia?
A: I interpreted your query as being about India. The capital of India is New Delhi.
```

**Unrecognisable input**
```
Q: What is the capital of dndia?
A: I couldn't identify a country name in your query. Could you check the spelling?
    Example: "What is the capital of India?"
```

**Country not found**
```
Q: What is the capital of Xlandia?
A: I couldn't find 'Xlandia' in the countries database. Please check the spelling.
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: API client (mocked HTTP), node logic (mocked LLM), and FastAPI endpoints.

---

## Docker

```bash
docker build -t country-agent .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key country-agent
```

---

## Production Design Decisions

- **Exact-match-first lookup** — queries the REST Countries API with `fullText=true` before falling back to partial search, preventing wrong results for countries like China and India.
- **Typo transparency** — when the LLM infers a country from a misspelling, the answer explicitly states what was assumed, keeping responses grounded.
- **Shared async HTTP client** — `httpx.AsyncClient` is created once at startup via FastAPI lifespan and reused across all requests.
- **Structured LLM output** — `intent_node` uses `with_structured_output()` so field extraction is always a typed Pydantic object, never free-form text to parse.
- **Error containment** — every node catches its own exceptions and writes them into the LangGraph state. `synthesis_node` always runs and always produces an answer.
- **Token efficiency** — only the fields the user asked about are passed to the synthesis LLM, not the full 50-field API payload.
- **Stateless** — no database, no session state. Safe to scale horizontally.

## Known Limitations & Trade-offs

- **No caching** — each request hits the REST Countries API live. A Redis TTL cache keyed on country name would reduce latency significantly at scale.
- **Two LLM calls per request** — intent + synthesis. For high-throughput deployments, intent detection could be replaced with a fine-tuned NER model.
- **English-optimised** — the intent node prompt is in English. Multi-language queries work partially but are not guaranteed.
- **Groq rate limits** — the free tier allows ~30 requests/min. A production deployment would need a paid plan or request queuing.
