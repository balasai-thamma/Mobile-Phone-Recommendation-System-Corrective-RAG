# 📱 Mobile Phone Recommendation System — Corrective RAG

A Retrieval-Augmented Generation (RAG) pipeline that recommends mobile phones based on natural-language queries, with a **Corrective RAG (CRAG)** layer that grades retrieved results against the constraints in your query (budget, RAM, battery, brand, year) and self-corrects when vector search alone isn't enough.

Built with **Weaviate** (vector DB), **sentence-transformers** (embeddings), **FastAPI** (serving), and **Ollama** (local LLMs) — no external API keys, fully self-hosted.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector database | [Weaviate](https://weaviate.io/) (self-hosted via Docker) |
| Embedding model | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| LLM inference | [Ollama](https://ollama.com/) — `qwen2.5:0.5b`, `llama3.2:1b`, `gemma3:270m` |
| API layer | FastAPI |
| Constraint extraction | Regex fast-path + LLM fallback (hybrid) |
| Chunking strategy | Row-based (one phone row = one chunk = one vector) |

---

## Project Structure

```
mobile-recommender/
├── app.py              # FastAPI service — retrieval, voting, and the CRAG layer
├── crag_utils.py        # Constraint extraction + phone grading (pure functions)
├── ingest.py             # Loads the CSV, chunks it, embeds it, stores it in Weaviate
├── test.py                # One-shot Weaviate connectivity check
├── docker-compose.yml      # Spins up the Weaviate container
└── Data/
    └── Mobiles Dataset.csv  # Source catalogue
```

## API

### `GET /`
Returns project metadata and current record count.

### `GET /recommend?query=<your query>`
Runs the full pipeline: extract → retrieve → grade → correct → vote.

**Example:**
```
GET /recommend?query=gaming phone under 15000 with 8gb ram
```

**Response (abridged):**
```json
{
  "user_query": "gaming phone under 15000 with 8gb ram",
  "corrective_rag_details": {
    "constraints_extracted": {
      "max_price": 15000.0,
      "min_ram_gb": 8.0,
      "min_price": null,
      "min_battery_mah": null,
      "brand": null,
      "min_year": null
    },
    "correction_action": "ambiguous_combine",
    "retrieval_rounds": 2,
    "relaxed_constraints": [],
    "warning": null
  },
  "top_recommendation": {
    "recommended_by": "Qwen2.5",
    "model": "Poco X6",
    "price": "INR 14,999",
    "ram": "8GB",
    "match_type": "exact"
  },
  "other_model_recommendations": [ ... ]
}
```


## License

MIT (or update to whatever you're using).
