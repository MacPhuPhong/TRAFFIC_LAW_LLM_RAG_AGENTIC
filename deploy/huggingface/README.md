---
title: Traffic RAG Backend
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Traffic-law RAG backend (FastAPI + LangGraph)
---

# Traffic RAG Backend — Hugging Face Space

FastAPI + LangGraph backend cho hệ thống tra cứu luật giao thông Việt Nam.

## Stack

- **FastAPI** — REST API
- **LangGraph** — agentic flow (router → retrieve → grade → generate → web-HITL)
- **multilingual-e5-base** — embedding (đã pre-download trong image)
- **Qdrant Cloud** — vector store (hybrid dense + BM25)
- **Gemini 2.5 Flash** — generator
- **Tavily** — web search fallback
- **Supabase Postgres** — LangGraph checkpoint persistence

## Build context

Space này được upload từ folder cha của repo. Dockerfile expects:

```
context/
  ├── requirements.txt
  └── traffic_rag/
      ├── api/
      ├── source/
      ├── Data/
      └── ...
```

## Required Secrets (Space Settings → Secrets)

| Key                    | Mô tả                                      |
| ---------------------- | ------------------------------------------ |
| `API_KEY`              | Google Gemini API key                      |
| `TAVILY_API_KEY`       | Tavily search key                          |
| `QDRANT_URL`           | https://xxx.qdrant.io (Qdrant Cloud)       |
| `QDRANT_API_KEY`       | Qdrant Cloud API key                       |
| `CHECKPOINT_DB_URL`    | postgresql://...supabase.co:5432/postgres  |
| `CORS_ALLOWED_ORIGINS` | https://your-app.vercel.app                |
| `LANGCHAIN_API_KEY`    | (optional) LangSmith tracing               |
| `LANGCHAIN_TRACING_V2` | (optional) `true`                          |
| `LANGCHAIN_PROJECT`    | (optional) `Traffic-RAG-Production`        |
| `GENERATOR_MODEL`      | (optional) `gemini-3.1-flash-lite-preview` |

## Endpoints

- `GET /health` — liveness
- `POST /chat` — agentic chat (streaming SSE)
- `POST /resume` — resume from HITL pause
- `GET /pending` — list pending HITL items
- `GET /docs` — Swagger UI

Xem chi tiết deploy: [traffic_rag/deploy/DEPLOY-HYBRID.md](https://github.com/...)
