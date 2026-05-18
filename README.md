# Traffic RAG — Trợ lý Pháp luật Giao thông Việt Nam

Hệ thống **Agentic RAG** trả lời câu hỏi về pháp luật giao thông đường bộ Việt Nam, có **Human-in-the-Loop** (HITL) phê duyệt câu trả lời từ web khi corpus nội bộ không đủ.

- **🌐 Production**: https://traffic-rag.vercel.app (free, deployed 2026-05-17)
- **Corpus:** 26 văn bản (Luật + Nghị định + Thông tư hiệu lực 2025–2026) → **2 818 chunk** trong Qdrant.
- **Stack:** FastAPI · LangGraph · Qdrant (hybrid dense + BM25) · Gemini 3.1 Flash Lite · Tavily · Next.js 14.
- **Deploy production**: Vercel (FE) + Hugging Face Spaces (BE) + Qdrant Cloud + Supabase Postgres — xem [deploy/DEPLOY-PRODUCTION.md](deploy/DEPLOY-PRODUCTION.md).

---

## 1. Kiến trúc hệ thống

```
                 +-----------------------+
   User UI       |   Next.js 14 (3000)   |
   (browser) --> |  Chat + Citations     |
                 |  Admin /admin (HITL)  |
                 +-----------+-----------+
                             | POST /api/chat (proxy)
                             v
                 +-----------------------+
                 |   FastAPI (8000)      |
                 |   /chat /resume       |
                 |   /pending /health    |
                 +-----------+-----------+
                             |
                 +-----------v-----------+
                 |   LangGraph agent     |
                 |   router -> retrieve  |
                 |       -> grade        |
                 |       -> generate     |
                 |       -> [web HITL]   |
                 +-----+-----------+-----+
                       |           |
                       v           v
              +---------------+   +-------------+
              | Qdrant 6333   |   |  Tavily API |
              | hybrid search |   |  web search |
              | 2 818 vectors |   +-------------+
              +---------------+
```

**Pipeline ingestion** (offline, idempotent — xem [doc/bao_cao_he_thong.md](doc/bao_cao_he_thong.md) §2):

```
Data/raw/*.pdf
   --[clean_{luat,nghidinh,thongtu}_pdfs.py]-->  Data/cleaned/*.md
   --[semantic_chunker.py]-->                    Data/all_chunks.jsonl
   --[indexer.py --recreate]-->                  Qdrant collection "Traffic_Law_Hybrid"
```

---

## 2. Yêu cầu môi trường

| Thành phần | Phiên bản đã test |
|---|---|
| Python | 3.10 |
| Node.js | ≥ 18 |
| Docker + Docker Compose | bất kỳ bản hỗ trợ Compose v2 |
| Hệ điều hành | Linux (Ubuntu 24.04), macOS |
| RAM | ≥ 8 GB (encoder e5-small + Gemini stream) |

API key cần có:
- `API_KEY` — Google Gemini (free tier ổn cho demo)
- `TAVILY_API_KEY` — web search fallback
- `LANGCHAIN_API_KEY` *(tùy chọn)* — bật LangSmith tracing để quay video

---

## 3. Quickstart — chạy local trong 5 phút

> Tất cả lệnh chạy từ thư mục `traffic_rag/` trừ khi ghi rõ.

### 3.1 Clone + chuẩn bị `.env`

```bash
git clone <repo-url> GitHub1
cd GitHub1
cp .env.example .env   # nếu repo có sẵn .env.example
# Mở .env, điền API_KEY và TAVILY_API_KEY
```

File `.env` đặt ở **gốc repo** (`GitHub1/.env`), không phải trong `traffic_rag/`.

### 3.2 Cài Python deps

```bash
python3 -m venv ~/venv/LLM_Agentic
source ~/venv/LLM_Agentic/bin/activate
pip install -r requirements.txt
```

### 3.3 Chạy Qdrant (Docker)

```bash
cd traffic_rag
docker compose up -d qdrant
```

- Dashboard: http://localhost:6334/dashboard
- Collection sau khi index: `Traffic_Law_Hybrid`

### 3.4 Index dữ liệu (lần đầu)

```bash
# Đảm bảo Data/all_chunks.jsonl đã tồn tại (đã commit kèm repo)
python -m source.indexing.indexer --recreate
```

Mất ~6 phút trên CPU x86. Nếu chỉ chạy demo, bỏ qua bước này nếu Qdrant đã được preload.

### 3.5 Chạy Backend (FastAPI)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Health: http://localhost:8000/health
- Swagger: http://localhost:8000/docs

### 3.6 Chạy Frontend (Next.js)

```bash
cd app/nextjs-app
npm install         # lần đầu
cp .env.example .env.local   # nếu chưa có
npm run dev
```

- UI: http://localhost:3000
- Admin (HITL approval): http://localhost:3000/admin

---

## 4. Test nhanh

### Hỏi qua UI
Mở http://localhost:3000, nhập:

> *Vượt đèn đỏ ô tô bị phạt bao nhiêu tiền?*

→ Câu trả lời stream về với citations `[1][2]` (click để xem nguồn).

### Hỏi qua API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Vượt đèn đỏ ô tô bị phạt bao nhiêu?"}'
```

### Test HITL (web search fallback)
Hỏi câu **ngoài corpus** (vd. *"Phí đăng kiểm xe điện 2026?"*) → backend gọi Tavily, **dừng** ở node `web_finalize`, tạo entry trong `/pending`. Vào `/admin` để duyệt/từ chối.

---

## 5. Cấu trúc thư mục

```
traffic_rag/
├── api/                       # FastAPI app
│   ├── main.py                # endpoints: /chat /resume /pending /health
│   └── schemas.py             # Pydantic models
├── source/
│   ├── ingestion/             # PDF -> markdown -> chunk
│   │   ├── clean_luat_pdfs.py
│   │   ├── clean_nghidinh_pdfs.py
│   │   ├── clean_thongtu_pdfs.py
│   │   └── semantic_chunker.py
│   ├── indexing/              # chunk -> Qdrant
│   │   └── indexer.py
│   ├── rag_core/              # retriever + generator
│   │   ├── retriever.py       # hybrid dense (e5-small) + BM25
│   │   └── generator.py       # Gemini answer formatter
│   └── agent/                 # LangGraph agent
│       ├── graph.py           # state graph build
│       ├── nodes.py           # router/retrieve/grade/generate/web_*
│       ├── router.py          # category classifier
│       ├── tools.py           # Tavily wrapper
│       └── state.py           # AgentState TypedDict
├── app/nextjs-app/            # Next.js 14 frontend
├── Data/
│   ├── raw/{luat,nghidinh,thongtu,phuluc,quychuan}/   # PDF gốc
│   ├── cleaned/                                        # markdown sau clean
│   └── all_chunks.jsonl                                # 2 818 chunk
├── checkpoints/graph.db       # SQLite cho LangGraph (HITL state)
├── doc/
│   ├── bao_cao_he_thong.md    # Báo cáo đồ án (full)
│   ├── implementation_plan.md # Lệnh chạy chi tiết
│   └── video_script.md        # Kịch bản video demo
├── tests/                     # pytest suite
├── docker-compose.yml         # qdrant + (optional) backend container
└── requirements.txt           # ở gốc GitHub1/, không phải trong traffic_rag/
```

---

## 6. Các tính năng nổi bật

- **Hybrid retrieval** — dense (e5-small-v2 multilingual) + BM25 (rank-bm25, có tokenize tiếng Việt qua pyvi) với reciprocal rank fusion.
- **Semantic chunker** nhận diện hierarchy `Chương → Điều → Khoản → Điểm` của văn bản pháp luật VN; mỗi chunk giữ metadata `doc_id`, `dieu`, `khoan`, `effective_date`.
- **Status-aware retrieval** — chunk từ Luật 2008 (đã hết hiệu lực) hoặc Nghị định bị bãi bỏ được đánh `status="repealed"` và lọc khỏi mặc định search.
- **Agentic flow với HITL** — khi corpus không đủ, agent fallback Tavily → tạo `draft_answer` → **dừng** chờ admin duyệt qua `/admin`. Trạng thái persist trong SQLite (`checkpoints/graph.db`) để resume sau reload.
- **Citation grounding** — mỗi câu trả lời gắn `[n]` mapping về `{doc_id, dieu, khoan, page}`. Frontend hiển thị popover khi click.

---

## 7. Tham khảo

- Báo cáo đầy đủ: [doc/bao_cao_he_thong.md](doc/bao_cao_he_thong.md)
- Hướng dẫn chạy chi tiết từng bước: [doc/implementation_plan.md](doc/implementation_plan.md)
- Kịch bản video demo: [doc/video_script.md](doc/video_script.md)
- **Deploy production** (free, hybrid cloud): [deploy/DEPLOY-PRODUCTION.md](deploy/DEPLOY-PRODUCTION.md)
- Deploy alternative (1 VPS Oracle ARM): [deploy/DEPLOY.md](deploy/DEPLOY.md)
- Sơ đồ kiến trúc: [doc/so_do_he_thong.md §9](doc/so_do_he_thong.md#9-deployment)

---

## 8. License & tác giả

Đồ án tốt nghiệp 2026 — Mạc Phú Phong.
