---
title: Indian Budget RAG
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Indian Budget RAG API

Retrieval-augmented Q&A over Indian Union Budget 2026-27 documents. FastAPI backend, Qdrant vector store, Groq LLM, local BGE embeddings.

API docs: `/api/v1/docs`

## Required Space secrets

| Name | Value |
|---|---|
| `GROQ_API_KEY` | Groq API key |
| `QDRANT_URL` | Qdrant Cloud endpoint (no port — must resolve on 443) |
| `QDRANT_API_KEY` | Qdrant Cloud API key |

## Required Space variables

| Name | Value | Why |
|---|---|---|
| `QDRANT_COLLECTION` | `indian_budget_2026_v2` | re-chunked with the embedding model's own tokenizer and `heading_path`; the unsuffixed `indian_budget_2026` collection predates this and should not be used |
| `RATE_LIMIT_TRUST_FORWARDED_FOR` | `true` | Space traffic arrives via HF's reverse proxy; without this every visitor shares one rate-limit bucket |
