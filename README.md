# Traffic-Law RAG — Trợ lý Pháp lý Giao thông Việt Nam

[![CI](https://github.com/MacPhuPhong/TRAFFIC_LAW_LLM_RAG_AGENTIC/actions/workflows/ci.yml/badge.svg)](https://github.com/MacPhuPhong/TRAFFIC_LAW_LLM_RAG_AGENTIC/actions/workflows/ci.yml)

Hệ thống **Agentic RAG** trả lời câu hỏi pháp luật giao thông đường bộ Việt Nam theo quy định mới 2024–2026, kèm **Human-in-the-Loop (HITL)**: câu trả lời lấy từ web search phải được admin phê duyệt trước khi trả về người dùng.

🌐 **Demo:** https://traffic-rag.vercel.app

**Mục tiêu thiết kế:**

- **Trả lời đúng theo văn bản gốc** — mỗi câu trả lời trích dẫn chính xác tới cấp **Điểm/Khoản/Điều** của Luật, Nghị định, Thông tư.
- **Chống ảo giác** — từ chối khi ngữ cảnh không đủ thay vì bịa số tiền phạt/số điểm trừ.
- **Xử lý đa ý (multi-intent)** — một câu hỏi có thể chứa nhiều câu con (mức phạt + trừ điểm GPLX + tước bằng + xử lý phương tiện).
- **Giải quyết cross-reference** — khi Điều A dẫn tới Điều B, hệ thống tự kéo cả B vào ngữ cảnh.
- **Định tuyến đa intent** — tách biệt Q&A pháp luật / chit-chat / tra cứu web / out-of-scope.

**Corpus (v7.0):** 24 văn bản pháp luật còn hiệu lực (2 Luật 2024, 5 Nghị định, 17 Thông tư) → **4 423 hierarchical chunks**, index trong Qdrant.

---

## 1. Mọi lựa chọn thiết kế đều có ablation chứng minh

Hệ thống được xây theo phương pháp **Research → Quyết định → Triển khai**: 10 research question (RQ1–RQ10) chạy ablation trên eval set 40 câu (35 legal + 5 out-of-scope, 8 category):

| RQ   | Câu hỏi                             | Kết quả                                                 | Quyết định                  |
| ---- | ----------------------------------- | ------------------------------------------------------- | --------------------------- |
| RQ1  | Gemini-only / Vanilla / Agentic?    | Agentic: Category Acc **0.975**                          | **Agentic RAG (LangGraph)** |
| RQ2  | Hierarchical vs Fixed-512 chunking? | Cit-R(khoản) **0.41 vs 0.17** (×2.4)                     | **Hierarchical chunker v6** |
| RQ3  | sbert / mpnet / e5-small / e5-base? | **e5-base R@10 0.42**, MRR 0.28 (cao nhất)               | **multilingual-e5-base** (768d) |
| RQ4  | Qdrant / Chroma / FAISS?            | R@10 tương đương; FAISS nhanh hơn nhưng không filter     | **Qdrant** (payload filter) |
| RQ5  | Prompt P0 zero / P1 role / P2 rules?| P2 Cit-F1 **0.28 vs 0.07** (P1)                          | **P2** (role + 14 rules)    |
| RQ6  | Retrieval-only mạnh tới đâu?        | R@10 0.32, MRR 0.33                                      | Cân nhắc reranker GPU       |
| RQ7  | RAGAS judge (gpt-4o-mini)?          | context_precision **0.89**, faithfulness 0.55            | Retrieval còn dư địa        |
| RQ8  | Cost / latency?                     | **$0.00344/câu**, mean 33s (LangSmith, 1 964 runs)       | ~$45/tháng cho 10k query    |
| RQ9  | Query rewriting có giúp không?      | R@10 **0.58 vs 0.32** (+26 điểm)                         | **BẬT rewrite**             |
| RQ10 | Reranker có đáng bật?               | +48% R@10 nhưng latency ×156 trên CPU                    | **Tắt** đến khi có GPU      |

> Chi tiết đầy đủ (thuật toán, công thức metric, diễn giải) nằm trong báo cáo kỹ thuật `doc/bao_cao_he_thong.md` và thư mục `research/` — hai thư mục này **không công bố kèm repo** (đồ án tốt nghiệp, công bố sau).

---

## 2. Kiến trúc

**Offline pipeline** — chạy một lần, tái chạy khi bổ sung văn bản mới:

```
Data/raw/*.pdf
  --[clean_{luat,nghidinh,thongtu}_pdfs.py]-->  Data/cleaned/*.md   (pdfplumber, giữ bảng biểu,
                                                                     filter văn bản hết hiệu lực)
  --[semantic_chunker.py v6]-->                 Data/all_chunks.jsonl  (4 423 chunks,
                                                                     Chương→Điều→Khoản→Điểm)
  --[indexer.py --recreate]-->                  Qdrant "Traffic_Law_Hybrid"  (e5-base 768d,
                                                                     prefix "passage: ")
```

**Online pipeline** — mỗi request đi qua state machine LangGraph:

```
User query + chat_history
        │
        ▼
   analyzer (Gemini structured output:
   category, standalone_query, expanded_queries, intent, vehicle_type)
        │
        ├─ legal_rag ──── TrafficHybridRetriever          ── LegalAnswerGenerator
        │                 Dense(e5-base) ⊕ BM25 ⊕ RRF        prompt P2 (role + 14 rules)
        │                 → sibling ±2 khoản                 + citation sanitation
        │                 → cross-ref regex → top_k=10       │
        │                      │ (corpus không đủ)           ▼
        │                      └──────────┐        {answer, sources:[{doc_id, dieu, khoan, diem}]}
        ├─ web_legal_search ── web_search (Tavily) → web_finalize  ⏸ HITL interrupt
        ├─ chit_chat ─── template, không gọi LLM
        └─ out_of_scope ─ refusal template
```

- Node `web_finalize` compile với `interrupt_before` — graph **tạm dừng**, admin xem snippet ở `/admin` rồi Approve/Reject; khi resume mới tổng hợp câu trả lời (gắn nhãn *⚠ Tra cứu từ internet*).
- Trạng thái chờ duyệt persist qua LangGraph checkpointer (SQLite `checkpoints/graph.db`) — restart không mất.
- Node `legal_rag`: loop retriever theo từng `expanded_queries[i]` → RRF fuse các ranked list → diversity rerank (tối đa 3 chunk mỗi cặp doc/Điều/Khoản) → generator với hint `intent` + `vehicle_type`.

---

## 3. Tech stack

| Layer         | Công nghệ                                          | Lý do chọn                              |
| ------------- | -------------------------------------------------- | ---------------------------------------- |
| Orchestration | LangGraph                                          | state machine, HITL `interrupt_before`   |
| LLM           | Google Gemini (`gemini-3.1-flash-lite-preview`)    | 1M context, free-tier đủ demo            |
| Embedding     | `intfloat/multilingual-e5-base` (768d)             | RQ3 winner (R@10 0.42)                   |
| Vector DB     | Qdrant (Docker)                                    | payload filter, hybrid native            |
| Sparse        | `rank_bm25` (BM25Okapi) + `pyvi` tokenize          | thuần Python, đồng bộ Qdrant id          |
| Reranker      | `BAAI/bge-reranker-v2-m3` (opt-in)                 | RQ10: chỉ bật khi có GPU                 |
| Web search    | Tavily                                             | fallback khi corpus không đủ             |
| Backend       | FastAPI + uvicorn                                  | async, SSE streaming                     |
| Frontend      | Next.js 14 (App Router, TypeScript, Tailwind)      | chat UI + admin console, NextAuth        |
| Observability | LangSmith tracing + `GET /metrics`                 | phân tích cost/latency (RQ8)             |

---

## 4. Cài đặt và chạy local

### Yêu cầu

- Python 3.10+ · Node.js ≥ 18 · Docker (Compose v2)
- API key: **Google Gemini** và **Tavily** (free tier đủ dùng)

### 4.1 Clone + cấu hình

```bash
git clone https://github.com/MacPhuPhong/TRAFFIC_LAW_LLM_RAG_AGENTIC.git
cd TRAFFIC_LAW_LLM_RAG_AGENTIC

cp .env.example .env      # điền API_KEY (Gemini) và TAVILY_API_KEY

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2 Qdrant + index dữ liệu

```bash
docker compose up -d qdrant                    # dashboard: http://localhost:6333/dashboard
python -m source.indexing.indexer --recreate   # cần Data/all_chunks.jsonl (xem lưu ý dưới)
```

> ⚠️ Thư mục `Data/` (PDF gốc + chunks đã xử lý) **không kèm theo repo**. Muốn tự build index, chạy pipeline ingestion ở §2 với văn bản PDF của bạn, hoặc liên hệ tác giả.

### 4.3 Backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Health: http://localhost:8000/health · Swagger: http://localhost:8000/docs

### 4.4 Frontend

```bash
cd app/nextjs-app
npm install
cp .env.example .env.local   # sửa NEXTAUTH_SECRET
npm run dev
```

Chat UI: http://localhost:3000 · Admin (HITL): http://localhost:3000/admin

### 4.5 Dùng thử

Hỏi qua UI: *"Vượt đèn đỏ ô tô bị phạt bao nhiêu tiền?"* → câu trả lời kèm citation `[n]` map về `{doc_id, dieu, khoan, diem}`, click để xem trích dẫn.

Hỏi qua API:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Vượt đèn đỏ ô tô bị phạt bao nhiêu?"}'
```

Test HITL: hỏi câu ngoài corpus (vd. *"Phí đăng kiểm xe điện 2026?"*) → agent gọi Tavily rồi dừng chờ duyệt, vào `/admin` để Approve/Reject.

### API endpoints

| Method | Path                   | Mô tả                                        |
| ------ | ---------------------- | -------------------------------------------- |
| `POST` | `/chat`                | Gửi câu hỏi, nhận answer + sources           |
| `POST` | `/resume/{thread_id}`  | Admin approve/reject câu trả lời HITL        |
| `GET`  | `/pending`             | Danh sách thread chờ duyệt                   |
| `GET`  | `/pending/{thread_id}` | Chi tiết một thread chờ duyệt                |
| `GET`  | `/health`              | Health check                                 |
| `GET`  | `/metrics`             | Uptime, request/error count, avg latency     |

---

## 5. Cấu trúc repo

```
├── source/                        # production code
│   ├── ingestion/
│   │   ├── clean_luat_pdfs.py     #
│   │   ├── clean_nghidinh_pdfs.py # PDF -> markdown (pdfplumber, giữ bảng,
│   │   ├── clean_thongtu_pdfs.py  # filter văn bản hết hiệu lực)
│   │   └── semantic_chunker.py    # HierarchicalLegalSplitter v6 (Điều/Khoản/Điểm)
│   ├── indexing/
│   │   └── indexer.py             # all_chunks.jsonl -> Qdrant (e5-base, batch)
│   ├── rag_core/
│   │   ├── retriever.py           # TrafficHybridRetriever (Dense + BM25 + RRF)
│   │   ├── retriever_pgvector.py  # biến thể pgvector cho production cloud
│   │   ├── reranker.py            # bge-reranker-v2-m3 (opt-in)
│   │   └── generator.py           # LegalAnswerGenerator (prompt P2, citation sanitation)
│   └── agent/                     # LangGraph
│       ├── graph.py               # state machine + interrupt_before=["web_finalize"]
│       ├── nodes.py               # analyzer / legal_rag / web_search / ...
│       ├── tools.py               # Tavily wrapper
│       └── state.py               # AgentState
├── api/                           # FastAPI backend (main.py + schemas.py)
├── app/nextjs-app/                # Next.js 14: chat UI + /admin HITL console
├── deploy/                        # Dockerfile, HF Spaces, hướng dẫn deploy production
├── tests/                         # smoke tests + retrieval regression gate
├── .github/workflows/ci.yml       # CI 2 job (xem §6)
├── docker-compose.yml             # Qdrant local
├── requirements.txt
└── .env.example
```

Không kèm repo (công bố sau): `Data/` (corpus PDF + chunks), `doc/` (báo cáo kỹ thuật đồ án), `research/` (scripts + kết quả ablation RQ1–RQ10).

---

## 6. Testing & CI

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) gồm 2 job:

- **smoke** — unit tests cơ bản, không cần service ngoài.
- **regression** — khởi động Qdrant ephemeral trong GitHub Actions, chạy retrieval regression gate: assert **mean Recall@10 ≥ 0.30** trên bộ câu legal.

```bash
pytest tests/test_smoke.py -m "not requires_qdrant"   # local, không cần Qdrant
pytest tests/test_retrieval_regression.py             # cần Qdrant đã index
```

---

## 7. Deploy production

Kiến trúc free-tier: **Vercel** (frontend) + **Hugging Face Spaces** (backend Docker) + **Qdrant Cloud** + **Supabase Postgres/pgvector**.

- Hướng dẫn đầy đủ: [deploy/DEPLOY-PRODUCTION.md](deploy/DEPLOY-PRODUCTION.md)
- Phương án 1 VPS (Docker Compose + nginx): [deploy/DEPLOY-HYBRID.md](deploy/DEPLOY-HYBRID.md)

---

## 8. Hạn chế hiện tại & hướng phát triển

Theo đánh giá v7.0 (02/06/2026):

- **context_recall 0.45** là bottleneck chính — retrieval mới bắt được ~45% facts cần; reranker GPU (RQ10 cho thấy +48% R@10) là hướng cải thiện rõ nhất.
- **Eval set 35 câu legal còn nhỏ** — cần mở rộng lên 80–100 câu.
- **Gemini preview model** error rate cao khi bị rate-limit — production nên chuyển sang model stable.
- Hướng xa hơn: multi-hop agent cho câu hỏi liên quan nhiều điều luật, streaming + prompt caching để giảm latency.

---

## Tác giả

Đồ án tốt nghiệp 2026 — **Mạc Phú Phong** (Phong Mac).
