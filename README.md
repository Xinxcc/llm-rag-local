# Local RAG Service

A fully **on-premise** retrieval-augmented generation (RAG) service for question answering over a private document collection — no data leaves the machine. Everything runs locally via Docker: the API, a local LLM (Ollama) and a vector database (Qdrant).

## Features

- **FastAPI** service with three endpoints: `/health`, `/ingest`, `/query`, protected by an API key.
- **Multi-format ingestion** — PDF, DOCX, Markdown and plain text, with text cleaning and overlapping chunking (deterministic chunk IDs for idempotent upserts).
- **Local embeddings & generation** through **Ollama** (`nomic-embed-text` for embeddings, a small instruct model for generation).
- **Vector search** with **Qdrant** (cosine similarity).
- **Access-group filtering** — each chunk is tagged with an `access_group`, and retrieval is filtered so answers only use documents the caller is allowed to see.
- **Cited answers** — every response returns the source chunks it used (file + page).
- **Container orchestration** — `docker-compose` brings up the API, Ollama and Qdrant with persistent volumes.

## Architecture

```
            ┌─────────────────────────────┐
   client → │  FastAPI  (/ingest /query)  │
            └───────┬──────────────┬──────┘
                    │              │
          embeddings│              │ vector search (cosine,
          + generate│              │ access-group filtered)
                    ▼              ▼
              ┌──────────┐   ┌──────────┐
              │  Ollama  │   │  Qdrant  │
              │ (local)  │   │ (vectors)│
              └──────────┘   └──────────┘
```

**Query flow:** embed the question → retrieve top-k chunks from Qdrant (filtered by access group) → build a grounded prompt that forces citation-based answers → generate with the local LLM → return the answer plus citations.

## Quick start

```bash
cp .env.example .env          # then set a strong API_KEY
docker compose up -d --build

# pull the models once (inside the ollama container)
docker exec -it ollama ollama pull nomic-embed-text
docker exec -it ollama ollama pull gemma3:1b

# put documents under ./knowledge_sources, then ingest:
curl -X POST localhost:8000/ingest -H "x-api-key: $API_KEY" \
     -H "content-type: application/json" -d '{"recreate_collection": true}'

# ask a question:
curl -X POST localhost:8000/query -H "x-api-key: $API_KEY" \
     -H "content-type: application/json" \
     -d '{"query": "What is the refund policy?"}'
```

## Configuration

All settings come from environment variables (see `.env.example`): `API_KEY`, `OLLAMA_BASE_URL`, `LLM_MODEL`, `EMBED_MODEL`, `QDRANT_URL`, `QDRANT_COLLECTION`, `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`.

## Tech stack

FastAPI · Pydantic · Ollama · Qdrant · Docker Compose · pypdf · python-docx

## Status & roadmap

Working prototype. Planned improvements:
- **Batch embeddings** (currently embedded one chunk at a time).
- **Evaluation harness** (retrieval/answer quality, RAGAS-style).
- **Streaming responses** over Server-Sent Events / WebSocket.
- **Richer access control** and per-user audit logging.
- Swappable larger generation models.

## Notes

This repository contains no private documents or secrets: the knowledge base and `.env` are git-ignored. The service is intended for local/internal use.
