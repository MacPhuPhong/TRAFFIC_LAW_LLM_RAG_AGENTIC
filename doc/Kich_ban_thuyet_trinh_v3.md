# KỊCH BẢN THUYẾT TRÌNH — TRAFFIC-LAW RAG (v3.1)

**Đề tài:** Traffic-Law RAG — Trợ lý Pháp lý Giao thông Việt Nam (MVP Production-Grade · v6.2)
**Học viên:** Mạc Phú Phong
**Phiên bản kịch bản:** 3.1 — sửa các điểm lệch với code thực tế (số văn bản 6→23, RQ10→RQ7, AnalyzerOutput 4→3 trường) + bổ sung Phụ lục D giải đáp chi tiết.
**Thời lượng:** 25 phút thuyết trình + 6 phút demo + Q&A

---

## CẤU TRÚC TỔNG (17 SLIDE)

| # | Slide | Trụ cột | Phút |
|---|---|---|---|
| 01 | Bìa — Traffic-Law RAG MVP production-grade | — | 1:00 |
| 02 | Khung kiểm tra 9 trụ cột | rubric | 1:30 |
| 03 | Trụ 01 — UX/PRD | 01 | 1:30 |
| 04 | Trụ 02 — Data Flow | 02 | 2:00 |
| 05 | Trụ 03 — Inference & Caching | 03 | 1:30 |
| 06 | Trụ 04 — Hybrid Retrieval ★ | 04 | 3:00 |
| 07 | Trụ 05 — Agents/Orchestrator | 05 | 2:00 |
| 08 | Trụ 06 — Tools (✓) · MCP/A2A roadmap | 06 | 1:30 |
| 09 | Trụ 07 — Observability | 07 | 1:15 |
| 10 | Trụ 08 — Evaluation | 08 | 1:45 |
| 11 | Trụ 09 — Guardrails ★ | 09 | 1:45 |
| 12 | MVP Stack tổng quan | mọi trụ | 1:00 |
| 13 | Demo Plan — 4 kịch bản | 04·05·06·09 | 0:45 |
| 14 | Key Decisions (RQ-driven) | mọi trụ | 1:30 |
| 15 | Roadmap v7 | mọi trụ | 1:00 |
| 16 | Tổng kết Reliable·Debuggable·Evolvable | — | 0:45 |
| 17 | Cảm ơn + Q&A | — | 0:30 |

> Tổng: ~25 phút thuyết trình + 6 phút demo. Cộng Q&A khoảng 35–40 phút.

---

# PHẦN I — MỞ ĐẦU & ĐỊNH VỊ

## SLIDE 01 — BÌA

**Layout slide:** Cột trái có eyebrow "FP · Final Project · Bảo vệ 2026", title lớn "Traffic-Law RAG / MVP production-grade", lead 2 dòng, 4 chip màu (v6.3 · 2 818 chunks · stack · Reliable·Debuggable·Evolvable). Cột phải có 4 figure cards — số liệu lớn về văn bản nguồn, index, retrieval gain, quality gate. Footer: 4 ô meta (Học viên · Phiên bản · Stack · Repo).

> *(Đứng nghiêm, nhìn hội đồng, ngừng 1 nhịp.)*

Kính chào Hội đồng. Em là **Mạc Phú Phong**, hôm nay em xin trình bày **Final Project** với đề tài: **Traffic-Law RAG — Trợ lý Pháp lý Giao thông Việt Nam**, phiên bản 6.3.

Em xin được **định vị ngay từ đầu**: đây không phải một demo đồ chơi, cũng không phải một bài toán nhỏ. Đây là một **MVP production-grade** — nghĩa là hệ thống được thiết kế đúng theo checklist mà một sản phẩm AI thật sự phải đáp ứng:

- Tích hợp **tối thiểu một mô hình** ngôn ngữ lớn,
- **Một workflow agent** điều phối,
- **Một lớp tool** chuẩn hóa cho RAG (schema sẵn sàng wrap MCP),
- Có **observability** đầy đủ,
- Có **evaluation pipeline** cho CI/CD,
- Có **guardrails** chống ảo giác.

Đủ để demo trực tiếp ngày hôm nay, và **đủ để tiến hóa** trong các phiên bản sau.

> *(Chỉ tay sang cột phải — 4 figure cards.)*

Vài con số thực để hội đồng hình dung quy mô:

- **23 văn bản pháp luật giao thông Việt Nam** đã được ingest đầy đủ — gồm **2 Luật** (Luật Đường bộ 35/2024, Luật TTATGTĐB 36/2024), **4 Nghị định** (NĐ 10/2020 KDVT, NĐ 135/2021, NĐ 168/2024 xử phạt, NĐ 336/2025 vận tải), và **17 Thông tư** thuộc 4 nhóm: bằng lái, đăng ký xe, khí thải/đăng kiểm, tuần tra/điều tra TNGT.
- **2 818 chunks** đã index với 15 trường metadata, lưu trong Qdrant với HNSW + BM25 cache song song.
- Lớp Cross-Encoder reranker — **+21.6% MRR**, đây là kết quả của RQ7. Em sẽ giải thích trade-off cụ thể ở slide trụ 04.
- **Quality gate ở cấp pull request: Recall@10 ≥ 0.40** — không đạt là khóa merge cứng trên CI/CD.

Stack chính: **LangGraph · Qdrant · Gemini 3.1 Flash · E5-small · BM25 · FastAPI · Next.js 14 · LangSmith · Tavily.**

Ba từ khoá em muốn hội đồng giữ trong đầu xuyên suốt buổi trình bày là: **Reliable, Debuggable, Evolvable.**

---

## SLIDE 02 — KHUNG KIỂM TRA 9 TRỤ CỘT

**Layout slide:** Grid 3×3 — 9 ô pillar, mỗi ô có số thứ tự, tên trụ, mô tả ngắn, 1–2 tag.

Trước khi đi vào hệ thống, em xin trình bày **rubric** mà em đã dùng để tự đánh giá toàn bộ thiết kế.

Một hệ thống AI muốn gọi là **production** — không phải prototype, không phải demo — cần phải đạt **9 trụ cột**. Đây không phải là 9 lựa chọn. Đây là 9 hạng mục **bắt buộc**. **Thiếu một là nợ kỹ thuật**, sớm muộn cũng phải trả khi hệ thống lên thật.

Em đi nhanh qua 9 trụ:

1. **UX/PRD** — Yêu cầu sản phẩm. Người dùng là ai, ràng buộc gì, định nghĩa "đúng" thế nào. **Sản phẩm trước, kiến trúc sau.**
2. **Data Flow** — Luồng dữ liệu offline + online, idempotent, replay-able, có lineage.
3. **Inference & Caching** — Chọn mô hình hợp lý, cache nhiều tầng để giảm cost và latency.
4. **Retrieval** — Lấy đúng chunk. Hybrid 5 lớp.
5. **Agents/Orchestrator** — State machine, không phải pipeline tuyến tính.
6. **Tools · RAG (MCP/A2A roadmap)** — Lớp công cụ chuẩn hóa, schema typed sẵn wrap protocol sau.
7. **Observability** — Trace, cost, token mọi cấp.
8. **Evaluation** — Quality gate ở cấp pull request.
9. **Guardrails** — Defense in depth, từ input đến human review.

> **Thông điệp slide này:** mọi slide tiếp theo đều ánh xạ vào ít nhất một trụ. Em không nói cái gì lan man. Em sẽ đi qua từng trụ — chỉ ra **pattern thiết kế** và **minh chứng từ implementation thực tế**.

---

# PHẦN II — CHÍN TRỤ CỘT KIẾN TRÚC

## SLIDE 03 — TRỤ 01 · UX & PRD

**Layout slide:** Pull-quote lớn bên trái + 2 cột danh sách (Người dùng / 5 yêu cầu cứng).

> **Pull-quote ấn tượng:** *"Khi LLM ảo giác, ai phải trả mức phạt sai?"*

Câu hỏi đó đặt ra toàn bộ ràng buộc kỹ thuật. Bài toán không phải làm chatbot trò chuyện chung chung. Bài toán là **trợ lý pháp lý** — sai một chữ là người dân nộp phạt nhầm, là CSGT ra biên bản nhầm, là hậu quả thật.

### Người dùng — 3 nhóm

- **Tài xế / công dân** — tra cứu mức phạt và điểm trừ giấy phép lái xe trước khi đi đường, hoặc khi bị lập biên bản.
- **CSGT field** — kiểm chứng quy định nhanh tại hiện trường, đặc biệt với những vi phạm hiếm gặp.
- **Admin pháp chế** — duyệt câu trả lời từ web_search trước khi nhả ra cho người dùng cuối.

### 5 yêu cầu sản phẩm cứng

Mỗi yêu cầu này định hình một quyết định kỹ thuật xuôi dòng:

1. **Trích đúng cấp Điểm/Khoản/Điều** — không mơ hồ, không "theo Nghị định 168". Phải cụ thể đến từng điểm.
2. **Refuse khi không chắc** — thà từ chối còn hơn bịa. Đây là nguyên tắc số một của domain pháp lý.
3. **Multi-intent** — một câu hỏi có thể chứa cả mức phạt + trừ điểm + tước bằng → hệ thống phải trả đủ trong **một lượt**, không bắt người dùng hỏi lại.
4. **Cross-reference** — Điều A dẫn tới Điều B → hệ thống phải tự kéo cả B vào context.
5. **Human approve** khi rủi ro cao — đặc biệt với fallback web_search, vì nội dung từ web không có audit trail.

> **Tóm:** "đúng" ở đây có **định nghĩa cụ thể, kiểm tra được**. Mọi quyết định kỹ thuật xuôi dòng đều phải tôn trọng 5 ràng buộc này. Đây là điểm khác biệt lớn nhất giữa "làm RAG" theo tutorial và "làm RAG cho domain pháp lý".

---

## SLIDE 04 — TRỤ 02 · DATA FLOW

**Layout slide:** 2 card lớn bên cạnh nhau, mỗi card là một pipeline với chuỗi node có mũi tên xuống.

Đây là sơ đồ **hai pipeline tách biệt**, mỗi pipeline đều **idempotent** và **replay-able**.

### Pipeline Offline — Indexing

Chạy mỗi khi có văn bản pháp luật mới hoặc cập nhật:

1. **Source** — **23 văn bản gốc** (2 Luật + 4 Nghị định + 17 Thông tư), từ 78 file raw `.pdf`/`.docx` ban đầu — 53 file phụ lục/quy chuẩn chưa được pipeline xử lý (backlog v7), 1 Luật 2008 đã hết hiệu lực (blacklist), 2 NĐ có phần đường bộ bị bãi bỏ → md sạch.
2. **Pdfplumber + table-aware clean** — đặc biệt quan trọng vì các phụ lục Nghị định 168 có nhiều bảng (mức phạt × loại xe). Em phải xử lý bảng riêng, không tách nhầm hàng cột thành text liền.
3. **Markdown phân cấp** — đúng cấu trúc *Chương · Điều · Khoản · Điểm*.
4. **HierarchicalLegalSplitter** — đây là kết quả của RQ2. Splitter này tôn trọng ranh giới Điều/Khoản, không cắt giữa câu pháp lý. Output: **2 818 chunks**, trung bình **161 token**, mỗi chunk có **15 metadata field**.
5. **Embed bằng multilingual-E5-small** — với prefix `"passage:"` đúng theo hướng dẫn paper gốc.
6. **Ghi vào Qdrant HNSW** + **cache BM25 JSONL** trên disk.

### Pipeline Online — Per-query graph run

Chạy mỗi lượt user gửi query:

1. **Input** — query + chat history (sliding window).
2. **Analyzer** — trả structured output **3 trường** (Pydantic model): `category`, `standalone_query`, `expanded_query`. (`requires_approval` không phải của Analyzer — được set sau ở `web_search_node` khi cần HITL gate, xem Phụ lục D.1.2.)
3. **Router** — phân nhánh: `legal` / `chit_chat` / `web_search` / `out_of_scope`.
4. **Pydantic schema** — ép output cuối dạng `{ answer: str, sources: List[Citation] }`.

### Pattern quan trọng

Mọi state đều **idempotent** và **replay-able** từ LangSmith trace, có **checkpoint per-node**. Nếu hệ thống có bug ở node thứ 4, em không phải chạy lại 3 node đầu — em load checkpoint, sửa, replay từ đúng điểm.

> **Đây là tiền đề kỹ thuật quan trọng:** không có observability + checkpoint, không thể debug nhanh ở production.

---

## SLIDE 05 — TRỤ 03 · INFERENCE & CACHING

**Layout slide:** 4 metric card hàng trên (Generation / Embedding / Reranker / Cost) + 3 cache card hàng dưới (T1/T2/T3).

Mục tiêu của slide này: **hệ thống vừa rẻ, vừa nhanh** — không phải đắt mới tốt.

### Mô hình

| Vai trò | Mô hình | Cấu hình | Lý do |
|---|---|---|---|
| **Generation** | Gemini 3.1 Flash Lite Preview | 1M context · $0.075/1M in · $0.30/1M out | Free-tier đủ chạy MVP, 1M context không cần truncation, **provider-agnostic** |
| **Embedding** | multilingual-E5-small | 384 dim · max-seq 512 · CPU-friendly | Chạy local, không phụ thuộc API ngoài, MRR ×3 sBert (RQ3) |
| **Reranker** | bge-reranker-v2-m3 | 568M · max-len 8192 · multilingual | Togglable, sẽ giải thích ở trụ 04 |

**Cost thực đo trên LangSmith: $0.013 / query.** Em đo, không đoán.

### Cache 3 tầng

**T1 — Metadata filter (Qdrant payload index):**
4 keyword field được index riêng (status, doc_id, dieu, khoan). Filter `status=active` ở **O(log n)** thay vì full-scan. Đây là lý do em chọn Qdrant chứ không phải FAISS — sẽ giải thích ở RQ4.

**T2 — Sparse cache (BM25 JSONL):**
Tokenized corpus + IDF cache lưu trên disk. Load 1 lần khi service start, query in-memory. Tokenizer là **pyvi** — quan trọng cho tiếng Việt vì BM25 cần boundary từ chính xác.

**T3 — Prompt prefix cache (v7 roadmap):**
**Gemini Context Caching**. System prompt + retrieved chunks fix → tiết kiệm **~30% input cost** mỗi query lặp. Đây là tối ưu kế tiếp em nhắm tới.

> **Thông điệp:** cache không phải premature optimization — cache có **TTL và invalidation strategy** rõ ràng cho từng tầng. T1 invalidate khi reindex; T2 invalidate khi corpus đổi; T3 invalidate theo TTL Gemini quy định.

---

## SLIDE 06 — TRỤ 04 · HYBRID RETRIEVAL ★

**Layout slide:** 5 layer đặt dọc, mỗi layer có tên (L1–L5), tên kỹ thuật, mô tả, metric bên phải.

> *(Chậm lại, nhấn giọng.)* **Đây là trái tim hệ thống. Em sẽ đi chậm.**

5 lớp retrieval. **Mỗi lớp giải một loại lỗi** mà lớp trước không thấy. Đây là điểm em đầu tư nhiều thời gian nhất.

### L1 — DENSE: multilingual-E5-small (384d)

- Dùng prefix `"query:"` cho input và `"passage:"` cho corpus — đúng theo paper E5.
- Train trên **1 tỷ cặp** với **InfoNCE loss** trên dữ liệu đa ngôn ngữ.
- **Kết quả ablation (RQ3):** MRR **×3 so với sBert** trên domain pháp luật Việt.
- **Vai trò:** bắt câu hỏi paraphrase, hỏi vòng, hỏi gần nghĩa.

### L2 — SPARSE: BM25 Okapi (pyvi tokenizer)

- Tham số: k₁ = 1.5, b = 0.75 (chuẩn).
- **Vai trò bù dense:** bắt được những thứ Dense bỏ sót:
  - Số nghị định cụ thể: `"168/2024"`
  - Từ ghép pháp lý hiếm: `"vượt_đèn_đỏ"`, `"tước_giấy_phép"`
- Chạy **song song** Dense, không tuần tự — tận dụng đa luồng.

### L3 — RRF: Reciprocal Rank Fusion (k=60)

> **Điểm em muốn nhấn:** em fuse theo **rank**, không fuse theo **score**.

Vì sao? Điểm cosine của Dense và điểm BM25 **không cùng thang đo** — cosine ∈ [-1, 1], BM25 không có upper bound. Cộng thẳng là sai về toán học. RRF dùng công thức:

```
score_fused(d) = Σ 1 / (k + rank_i(d))
```

Với k=60. Output: **top-30 candidate** đưa vào lớp tiếp theo.

### L4 — RERANK: Cross-Encoder bge-reranker-v2-m3 (opt-in)

- Model 568M tham số, max-len 8192, multilingual.
- **Khác biệt cốt lõi với Dense:** Cross-Encoder đưa **cả query và chunk vào cùng một self-attention**. Dense tính embedding độc lập rồi mới cosine — không thấy tương tác. Cross-Encoder thấy tương tác **từng từ với từng từ**.
- **Kết quả (RQ7):**
  - **+21.6% MRR**
  - **×78 latency trên CPU**

> **Quyết định kỹ thuật:** togglable qua biến môi trường — `ENABLE_RERANKER=false` mặc định, **lazy-load** khi bật. Pipeline: RRF top-30 → rescore → top-K → sibling expansion. Mỗi production deployment chọn mức trade-off riêng theo SLA.

### L5 — EXPAND: Sibling ±2 + Cross-reference regex

- **Sibling expansion:** với mỗi chunk top-K, kéo thêm ±2 khoản liền kề. Vì luật pháp viết theo cấu trúc — Khoản 1 định nghĩa, Khoản 2 chế tài, Khoản 3 ngoại lệ. Bỏ sót khoản nào là mất nửa câu trả lời.
- **Cross-reference regex:** pattern bắt cụm `"điểm a khoản 4 Điều 13"`, gọi `get_chunks_by_location()` để kéo về. Đây là nền cho **multi-hop reasoning** — ví dụ NĐ 168, mức phạt và mức trừ điểm nằm rời ở Đ6.K13–K16, hệ thống tự ghép lại.

> **Câu kết slide:** Hybrid 5 lớp **không phải trang trí**. RQ3, RQ9, RQ7 đều minh chứng từng lớp đáng giá ngân sách của nó. Bỏ một lớp là mất một loại câu hỏi.

---

## SLIDE 07 — TRỤ 05 · AGENTS / ORCHESTRATOR

**Layout slide:** SVG diagram lớn — INPUT → ANALYZER → router → 4 nhánh (chit_chat / legal_rag / web_search → HITL / out_of_scope) → OUTPUT. Có mũi tên fallback dạng đứt nét.

LangGraph **state machine** — đây là điểm khác biệt lớn nhất với một RAG tuyến tính kiểu sách giáo khoa.

### Cấu trúc graph

```
INPUT → ANALYZER → router ┬→ chit_chat            → OUTPUT
                          ├→ legal_rag            → OUTPUT
                          ├→ web_search → HITL    → OUTPUT
                          └→ out_of_scope         → OUTPUT
                          
                          + fallback edge (đứt nét) khi legal_rag refused/0 chunk
```

### Analyzer — bộ não điều phối

Trả structured output **3 trường** (Pydantic `AnalyzerOutput` ở [nodes.py:50](../source/agent/nodes.py#L50)):
- `category`: legal_rag / chit_chat / web_legal_search / out_of_scope
- `standalone_query`: câu được rewrite độc lập (cho Generator, nếu là multi-turn)
- `expanded_query`: query mở rộng đa-hành-vi nối bằng `" | "` (cho Retriever)

> **Lưu ý:** `requires_approval` không phải trường của Analyzer mà được set ở `web_search_node` khi cần HITL gate (xem Phụ lục D.1.2).

**4 intent** được xử lý khác nhau:
- **`legal`** → vào pipeline RAG đầy đủ.
- **`chit_chat`** → trả lời ngắn, không retrieval, tiết kiệm cost.
- **`web_search`** → fallback Tavily, bắt buộc HITL.
- **`out_of_scope`** → refuse luôn, tiết kiệm cả cost lẫn risk.

### Bypass Rewrite turn đầu — RQ9

> *(Đây là phát hiện counter-intuitive em muốn nhấn.)*

Em đã phát hiện rewrite query ở turn đầu **gây hại 35% MRR** và **tăng 45% latency**.

Lý do: turn đầu **không có context cuộc hội thoại** để rewrite dựa vào. Rewrite biến câu cụ thể thành câu chung chung. Ví dụ: `"168/2024"` bị viết lại thành `"nghị định mới"` → BM25 mất từ khóa hiếm → MRR giảm.

→ Quyết định: **tắt rewrite ở turn đầu**, chỉ bật từ turn 2 trở đi (khi đã có context cuộc hội thoại).

### Auto fallback

Nếu `legal_rag` trả 0 chunk hoặc generator `refused=True` → tự động chuyển sang `web_search` (không cần user retry).

### HITL gate (Human-in-the-Loop)

LangGraph có cơ chế `interrupt_before` rất mạnh. Em đặt interrupt ngay trước node `web_finalize`:

- Mọi câu trả lời từ `web_search` **bắt buộc admin duyệt** trước khi nhả.
- Admin xem `draft_answer` + nguồn web, click approve.
- `/resume` mới tiếp tục.

> **Pattern phân tách orchestrator + specialist (tinh thần A2A — KHÔNG phải A2A protocol thực):** Analyzer điều phối, 4 sub-node (legal_rag, chit_chat, web_search, out_of_scope) có chuyên môn riêng — nhưng tất cả cùng 1 process Python, share state qua TypedDict. Đây là **tiền đề** cho roadmap v7 H4 — chuyển sang A2A protocol thật (multi-process, Agent Card, message chuẩn) liên kết với agent CSGT chuyên biệt.

---

## SLIDE 08 — TRỤ 06 · TOOLS (✓ Implemented) · MCP/A2A (Roadmap v7)

**Layout slide:** 3 card cho 3 tool đã implement (mỗi card gắn pill `[✓]`) + 1 block dưới chia rõ "Hiện tại / Roadmap v7" cho MCP và A2A.

Tool layer chuẩn hóa — **schema typed, stateless, sẵn sàng wrap MCP/A2A**.

> *(Em xin nói trước trạng thái: 3 tool ở trên là code thực, chạy được. MCP server và A2A protocol là roadmap v7 — em sẽ làm rõ cuối slide.)*

### Tool 1 · `hybrid_search()` — Core RAG  ✓ Implemented

```json
{
  "query": str,
  "top_k": int = 10,
  "filter": { "status": "active" }
} → List[Chunk]
```

Wrapper class `TrafficHybridRetriever` đóng gói toàn bộ pipeline 5 lớp ở slide trước. Caller nội bộ (LangGraph node, test script) đều thấy **cùng một interface**, không có hidden state. Trả `List[Chunk]` với metadata đầy đủ: `{doc_id, dieu, khoan, diem, status, …}`. Schema này đáp ứng tiền đề wrap MCP — **chưa wrap**.

### Tool 2 · `tavily_search()` — Web fallback  ✓ Implemented

```json
{
  "query": str,
  "max_results": int = 5
} → List[Snippet]
```

**Anchor query** với prefix `"Luật giao thông VN: "` — để tránh Tavily lạc đề sang luật nước khác (Tavily không biết domain).

Tổng hợp `draft_answer` + warning prefix `[CẢNH BÁO: Thông tin từ web…]`. Đặt `requires_approval=True` → bắt buộc qua HITL gate.

### Tool 3 · `resolve_xref()` — Cross-reference resolver  ✓ Implemented

```json
{
  "text": str
} → List[CitationRef]   // "điểm a khoản 4 Điều 13"
```

Regex bắt cụm `"điểm/khoản/Điều"` trong top-5 primary chunk, tra `get_chunks_by_location()`. Đây là cơ chế **multi-hop 1-hop** — ví dụ NĐ 168 phạt-tiền Đ6.K11 dẫn tới trừ-điểm Đ6.K13–K16, tool ghép cả khối lại trong một đáp án.

### MCP & A2A — TIỀN ĐỀ ✓  ·  WRAP LAYER → ROADMAP V7

> *(Em xin nói rõ trạng thái hiện tại để Hội đồng phân biệt — tránh hiểu nhầm em đã làm xong.)*

#### Model Context Protocol (MCP) — chuẩn của Anthropic

- **Hiện tại:** 3 tool có Pydantic schema typed, stateless — đáp ứng tiền đề wrap MCP. **Trong codebase chưa có MCP server, chưa có dependency `mcp` SDK.**
- **Roadmap v7:** wrap thành MCP server (stdio/HTTP) → Claude Desktop, Cursor sẽ gọi `hybrid_search("vượt đèn đỏ")` trực tiếp mà không cần biết internals. Em ước tính 1-2 tuần work: thêm `pip install mcp`, viết file `mcp_server.py` với decorators `@mcp.tool()`, register lên `Server.from_FastMCP(...)`.

#### Agent-to-Agent (A2A) — chuẩn Google/Anthropic đang chuẩn hoá

- **Hiện tại:** Pattern orchestrator (analyzer) + 4 specialist (`legal_rag`, `chit_chat`, `web_search`, `out_of_scope`) — TẤT CẢ trong **cùng 1 process Python**, share state qua `AgentState` TypedDict. **Đây là tinh thần A2A, KHÔNG phải A2A protocol thực** (không có endpoint riêng, không có Agent Card declaration, không có message format chuẩn).
- **Roadmap v7:** tách multi-process, expose Agent Card, liên kết với agent CSGT chuyên biệt (xử lý biên bản, sinh quyết định xử phạt) qua A2A protocol thật.

> **Tóm:** em **không claim đã có MCP server** hay **A2A protocol**. Em claim đã chuẩn bị **kiến trúc** sẵn sàng — schema typed (cho MCP) và pattern phân tách orchestrator/specialist (cho A2A). Wrap protocol layer là công việc 1-2 tuần ở v7. **Đây là điểm em xin minh bạch để Hội đồng đánh giá đúng phạm vi MVP.**

---

## SLIDE 09 — TRỤ 07 · OBSERVABILITY

**Layout slide:** 2 cột — trái là 3 cấp trace, phải là cost center table.

> **Một câu thôi:** đo được mới sửa được. Không có observability, mọi optimization là đoán mò.

### LangSmith trace · 3 cấp

Project: `Traffic-RAG-Evaluation` trên LangSmith.

1. **Graph run** — toàn bộ truy vấn: category, refused flag, final cost, end-to-end latency.
2. **Node run** — từng node con (analyzer, legal_rag, web_search): per-node latency và token.
3. **LLM call** — prompt cụ thể, response thô, finish_reason, retry count, error nếu có.

### Cost center — đo thực, không ước

| Node | Token | Cost |
|---|---|---|
| `legal_rag` | 39 000 | **$0.010** |
| `analyzer` | 2 100 | $0.0008 |
| `web_search` | 5 800 | $0.0021 |
| **Total / query** | — | **$0.013** |

> **Insight quan trọng:** muốn giảm cost, em phải optimize **legal_rag** — không phải web_search. Legal_rag chiếm hơn 75% chi phí. Nếu không có table này, em có thể đầu tư vài tuần optimize Tavily integration mà chỉ cứu được 16% cost. Đây là pre-condition cho mọi quyết định optimization xuôi dòng.

Trace luôn replay được nguyên trạng tham số. Một bug bất kỳ — em mở trace, nhìn 3 cấp, biết ngay node nào sai.

---

## SLIDE 10 — TRỤ 08 · EVALUATION

**Layout slide:** 2 cột — trái là vấn đề + giải pháp, phải là bảng kết quả + insight.

### Vấn đề — RAGAS gốc không chạy nổi trên free-tier

- 25 query × 4 metric, mỗi sample tách thành ~5–10 atomic claim.
- → ~60 API call mỗi sample, tổng **~6 000 call**.
- Free-tier Gemini ~15 RPM → throttle, **93.5% NaN** trong batch chạy thử.

### Giải pháp — RAGAS-Lite (em tự code)

- **Single-prompt Chain-of-Thought** — judge tự decompose nội bộ trong một call, không tách atomic claim ra ngoài.
- → **1 call/metric/sample** · tổng **20 call** · ~90 giây cho n=5.
- Parser regex `SCORE: x.xx`.
- **Exponential backoff:** 8 → 16 → 32 → 64 giây — chống throttle triệt để.

### Kết quả n=5 (simple_penalty)

| Metric | Score | Status |
|---|---|---|
| `answer_relevancy` | **1.00** | ✓ Generator OK |
| `context_recall` | 0.23 | ⚠ Retrieval recall thấp |
| `context_precision` | 0.20 | ⚠ Retrieval precision thấp |
| `faithfulness` | 0.15 | ✗ Bottleneck |

> **Diễn giải kỹ:** `answer_relevancy = 1.00` chứng minh **generator hoạt động tốt** — khi cho đúng evidence, model viết ra câu trả lời đúng trọng tâm. Nhưng `context_precision = 0.20` và `faithfulness = 0.15` cho thấy **bottleneck nằm ở retrieval precision**. → Đây chính là **bằng chứng định lượng** củng cố quyết định bật reranker khi SLA cho phép.

### CI/CD Regression — GitHub Actions

- Mỗi pull request trigger Pytest đánh **Recall@10 trên golden set**.
- Nếu Recall < 0.40 — ví dụ ai đó sửa nhầm embedding model hoặc đổi sai k của BM25 — **merge bị khóa cứng**.
- Đây là **quality gate ở cấp repository** — không một thay đổi nào có thể break retrieval mà không bị catch.

> **Tóm:** đánh giá không chỉ là báo cáo cuối kỳ, đánh giá là **cơ chế sống** trong CI/CD.

---

## SLIDE 11 — TRỤ 09 · GUARDRAILS ★

**Layout slide:** 4 layer dọc (G1–G4) + 1 card mở rộng cho Quy tắc 14 với 4 bước CoT.

**Defense in Depth — 4 lớp phòng vệ.** Không có viên đạn bạc; em chồng nhiều lớp để mỗi lỗi đều có ít nhất hai cơ chế bắt.

### G1 · INPUT — Analyzer route `out_of_scope`

Phân loại trước retrieval. Câu hỏi ngoài phạm vi pháp luật giao thông (ví dụ "thời tiết hôm nay", "công thức nấu phở") **bị refuse từ đầu**, không vào pipeline. Tiết kiệm cả cost (không retrieval, không generation) lẫn risk (không có cơ hội hallucinate).

### G2 · RETRIEVAL — Metadata filter `status=active`

Loại văn bản hết hiệu lực **ở cấp Qdrant payload index**. Đây là **DB-level guardrail** — luật cũ **không thể leak vào context**, dù prompt có viết thế nào. Đặc biệt quan trọng vì NĐ 100/2019 đã bị NĐ 168/2024 thay thế nhiều điều, dễ nhầm.

### G3 · GENERATION — Pydantic schema + 14 rules + refusal flag

- **Schema bắt buộc tách Answer khỏi Citations** — model không được trộn lẫn citation vào prose.
- **14 rules cứng**, ví dụ: cite metadata phải tồn tại thực trong corpus, không được tạo Điều/Khoản giả.
- **Refusal flag:** nếu không tìm được evidence → `refused=True`, hệ thống nói "tôi không có đủ căn cứ để trả lời".
- Schema validate fail → **không nhả output** ra user.

### G4 · HUMAN — HITL gate, `interrupt_before web_finalize`

Mọi câu trả lời từ `web_search` **bắt buộc admin duyệt**. UI Next.js có panel duyệt — admin xem draft + nguồn, click approve hoặc reject. **100% kiểm soát ảo giác** với rủi ro cao nhất.

### Guardrail đặc biệt — Quy tắc 14 (Legal Reasoning)

> *(Đây là điểm em đầu tư nhiều — vì đây là loại câu khó nhất.)*

Câu hỏi như *"đi ngược chiều quẹt xe — ai có lỗi?"* **không có Điều khoản định sẵn**. LLM bị cám dỗ đoán tỷ lệ phần trăm — 70:30, 80:20 — kiểu phân định lỗi của tòa.

Em ép chạy **Chain-of-Thought 4 bước** (Quy tắc số 14 trong prompt P2):

1. **Trích hành vi** — chỉ trích, không kết luận. *Ví dụ: "Người A đi ngược chiều, người B đi đúng chiều."*
2. **Đối chiếu chunk** — match từng hành vi với điều luật cụ thể trong context. *"Hành vi đi ngược chiều vi phạm Đ12 K3 NĐ 168/2024."*
3. **Kết luận lỗi logic** — ai vi phạm điều nào (không nói tỷ lệ %).
4. **Disclaimer thẩm quyền CSGT** — kết luận cuối cùng (mức phạt, lỗi chính/phụ, bồi thường) thuộc thẩm quyền **CSGT điều tra hoặc Tòa án**, không phải AI.

> **Lập luận pháp lý vững** mà tránh lạm quyền tòa án. Đây là loại guardrail không có sẵn trong sách RAG — em phải tự thiết kế cho domain.

---

# PHẦN III — SẢN PHẨM, DEMO & QUYẾT ĐỊNH

## SLIDE 12 — MVP STACK TỔNG QUAN

**Layout slide:** Bảng dài 12 hàng × 4 cột (Layer · Implementation · Trụ · Swap-able).

12 layer, mọi component **swap-able** — đây là tiêu chí evolvable, không có lock-in.

| Layer | Implementation | Trụ | Swap |
|---|---|---|---|
| Frontend | Next.js 14 · SSE streaming · UI duyệt HITL | 01 | ✓ |
| Backend | FastAPI async · uvicorn workers | 02 | ✓ |
| Orchestrator | LangGraph 0.2.x · checkpointer | 05 | ✓ |
| LLM | Gemini 3.1 Flash Lite Preview | 03 | ✓ provider-agnostic |
| Embedder | multilingual-E5-small · 384d | 04 | ✓ |
| Vector DB | Qdrant 1.7 · HNSW + payload index | 04 | ✓ |
| Reranker | bge-reranker-v2-m3 · ENV togglable | 04 | ✓ |
| Tools | hybrid_search · tavily · resolve_xref | 06 | ✓ stateless · typed schema (MCP-wrappable) |
| Trace | LangSmith · Traffic-RAG-Evaluation | 07 | ✓ |
| Evals | RAGAS-Lite + Pytest CI/CD | 08 | ✓ |
| Guardrails | Pydantic schema + HITL interrupt | 09 | ✓ |
| Container | Docker Compose · isolated services | — | ✓ |

> **Pattern:** đổi LLM provider, đổi vector DB, đổi reranker — chỉ cần đổi config, không phải viết lại logic. Đây là cách hệ thống **tiến hóa** mà không phải refactor lớn.

---

## SLIDE 13 — DEMO PLAN

**Layout slide:** 4 card cho 4 kịch bản, 2 card cuối có border accent.

Em xin chuyển sang **demo trực tiếp**. **4 kịch bản** — mỗi kịch bản kích hoạt một trụ cột khác nhau:

### Kịch bản 1 · Happy path — Cover trụ 04, 05

> *"Vượt đèn đỏ ô tô phạt bao nhiêu?"*

→ Analyzer phân loại `legal` → `hybrid_search` → cite **NĐ 168/2024 Đ6 K11** với mức phạt cụ thể.

### Kịch bản 2 · Cross-reference — Cover trụ 04, 06

> *"Trừ điểm GPLX khi vượt đèn đỏ?"*

→ Sibling expansion ±2 + regex xref → gộp **phạt + trừ điểm** trong cùng đáp án từ Đ6 K11 + Đ6 K13–K16.

### Kịch bản 3 · Quy tắc 14 ★ — Cover trụ 09

> *"Tôi đi ngược chiều quẹt xe — ai có lỗi?"*

→ Trigger **Legal Reasoning CoT 4 bước** → disclaimer thẩm quyền CSGT.

### Kịch bản 4 · HITL fallback — Cover trụ 05, 09

> *"Quy định mới về xe đạp điện 2026?"*

→ Corpus 0 chunk → fallback `web_search` → `interrupt_before` → admin duyệt → `/resume`.

---

## SLIDE 14 — KEY DECISIONS (RQ-DRIVEN)

**Layout slide:** Bảng 8 hàng × 4 cột (RQ · Quyết định · Đo được · Trụ).

Mỗi quyết định kiến trúc đều **có chứng cứ định lượng**, không trực giác.

| RQ | Quyết định | Đo được | Trụ |
|---|---|---|---|
| RQ1 | Agentic RAG > Vanilla > Gemini-only | Cit-R 0.56 vs 0.20 (×2.8) | 05 |
| RQ2 | Hierarchical chunking > Fixed-512 | MRR 0.16 vs 0.07 | 02 |
| RQ3 | E5-small > mpnet > PhoBERT | MRR 0.22 (×3 sBert) | 03·04 |
| RQ4 | Qdrant > FAISS | same quality + payload index online | 02·04 |
| RQ5 | Prompt P2 14 rules > zero-shot | refusal **0% → 72%** | 09 |
| RQ8 | Optimize legal_rag, không phải web | 39k tok / $0.010 / query | 07 |
| **RQ9** | **TẮT rewrite turn đầu** | **−35% MRR · +45% lat khi bật** | 05 |
| **RQ7** | **Reranker togglable theo SLA** | **+21.6% MRR · ×78 lat CPU** | 04 |

> **Bốn quyết định em muốn nhấn:**
>
> 1. **Agentic > Linear RAG** (RQ1) — Cit-R nhảy gần ×3. Đây là evidence quan trọng nhất chứng minh agent layer xứng đáng độ phức tạp.
> 2. **Qdrant > FAISS** (RQ4) — chất lượng tương đương, nhưng Qdrant có **payload index** cho metadata filter online. Với corpus có nhiều phiên bản hết hiệu lực, đây là tính năng bắt buộc.
> 3. **Tắt rewrite turn đầu** (RQ9) — counter-intuitive nhưng dữ liệu rõ ràng.
> 4. **HITL cho web_search** — chấp nhận latency để bảo vệ chất lượng trong trường hợp rủi ro cao nhất.

---

## SLIDE 15 — ROADMAP v7

**Layout slide:** 5 card hàng ngang (H1–H5), mỗi card có metric serif lớn ở dưới.

5 hướng tiến hóa cho v7, **cover toàn bộ 9 trụ**:

### H1 · Embedding — Fine-tune E5

Hard-negatives mining từ corpus luật VN. **+5pt MRR** kỳ vọng → có thể **bỏ reranker**, giảm độ phức tạp pipeline. Hướng phát triển sâu nhất, đẩy chất lượng retrieval lên tầng mới.

### H2 · Cost — Prompt caching

Gemini Context Caching cho prompt prefix. **−30% cost/query** khi traffic ổn định và prompt template ít thay đổi.

### H3 · Tools — MCP server

Đóng gói 3 tool thành **MCP server** chuẩn → Claude Desktop / Cursor / bất kỳ MCP client nào gọi được. **Portable** ra ngoài graph.

### H4 · Agents — A2A protocol

Liên kết với **agent CSGT chuyên biệt** (xử lý biên bản, sinh quyết định xử phạt) qua A2A protocol → multi-agent system thật sự.

### H5 · Evals — RAGAS scale

Mở rộng RAGAS-Lite từ n=5 lên **n=25 sample**, **5× độ phủ eval**. Đủ để claim statistical significance.

> **Đóng góp khoa học:** một **methodology** xây dựng RAG production-grade cho domain pháp lý. Methodology này **tái sử dụng được** cho luật doanh nghiệp, lao động, đất đai — không gói gọn trong giao thông. Đây là phần em hy vọng đóng góp dài hạn nhất.

---

# PHẦN IV — KẾT THÚC

## SLIDE 16 — TỔNG KẾT

**Layout slide:** 3 metric card lớn (Reliable / Debuggable / Evolvable) + tag-row tổng hợp.

9 trụ — 1 hệ thống hoạt động thực tế.

| | Số liệu | Ý nghĩa |
|---|---|---|
| **Reliable** | **4 lớp** guardrails | zero-hallucination policy |
| **Debuggable** | **3-tier** trace | LangSmith + per-node cost |
| **Evolvable** | **12/12** layers swap-able | MCP/A2A là roadmap v7 (kiến trúc đã sẵn sàng) |

Highlight cuối:
- 2 818 chunks từ 23 văn bản gốc (2 Luật + 4 Nghị định + 17 Thông tư).
- Hybrid 5 lớp · +21.6% MRR.
- LangGraph · 4 routes + HITL gate.
- $0.013/query · LangSmith trace.
- CI/CD floor Recall@10 ≥ 0.40.
- 14 rules + Quy tắc 14 Legal Reasoning.

> **Một câu kết:** đây là một MVP, nhưng là một MVP **không có kẽ hở rubric**. 9 trụ — đủ chín. Reliable, Debuggable, Evolvable — không phải khẩu hiệu mà là kết quả đo được.

---

## SLIDE 17 — CẢM ƠN + Q&A

**Layout slide:** title lớn "Cảm ơn Hội đồng" + tag-row 9 trụ + meta footer.

Em xin chân thành cảm ơn Hội đồng đã lắng nghe.

Em xin phép chuyển sang **demo trực tiếp** trên UI Next.js với **4 kịch bản** đã giới thiệu — cover các trụ 04, 05, 06, 07, 09. Sau demo em sẵn sàng nhận câu hỏi của Hội đồng.

---

# PHỤ LỤC A — Q&A · GỢI Ý TRẢ LỜI

> 10 câu hội đồng có thể hỏi, kèm gợi ý trả lời ngắn gọn.

### Q1: Tại sao Gemini chứ không phải GPT-4o hay Claude?

Bốn lý do: **(1)** Free-tier đủ chạy MVP — sinh viên friendly. **(2)** 1M context cho phép ingest cả Nghị định dài mà không cần truncation. **(3)** **Provider-agnostic** — chỉ cần đổi `langchain_google_genai` sang `langchain_openai` là chạy. **(4)** RQ8 đo cost thực $0.013/query — chấp nhận được cho domain.

### Q2: Sao không dùng FAISS — open source và miễn phí?

Em đã ablation ở **RQ4**. Quality cùng level. Khác biệt: Qdrant có **payload index** cho metadata filter online — em filter `status=active` ở **O(log n)** ngay trong query, không phải hậu lọc. Với corpus pháp luật có nhiều phiên bản hết hiệu lực, đây là tính năng **bắt buộc**.

### Q3: Reranker thêm latency ×78 — vậy có nên bật không?

**Tùy SLA.** Em **không quyết hộ** mà để togglable qua biến môi trường, mặc định tắt. Use case batch eval/CI: bật. Use case real-time chat: tắt, dùng tốt L1+L2+L3+L5. Đó là lý do mỗi lớp đều có vai trò độc lập, không phụ thuộc reranker. Đây cũng là **đóng góp thiết kế** — không bind kiến trúc với một trade-off cụ thể.

### Q4: RAGAS-Lite chỉ chạy n=5 — số liệu có ý nghĩa không?

n=5 không đủ để claim statistical significance, em **thừa nhận**. Mục đích RAGAS-Lite là **smoke test** trong CI/CD — phát hiện regression nhanh. Đánh giá đầy đủ cần mở rộng lên n=25 (Roadmap H5). Số liệu hiện tại đã đủ chỉ ra **bottleneck ở retrieval precision, không phải generation** — đó là insight quan trọng nhất, không phụ thuộc kích cỡ mẫu.

### Q5: Tại sao "tắt rewrite turn đầu" lại có lợi?

Counter-intuitive nhưng có lý do: **turn đầu không có context cuộc hội thoại**, rewrite biến câu cụ thể thành câu chung chung, làm BM25 mất từ khóa hiếm. Ví dụ `"168/2024"` bị viết lại thành `"nghị định mới"` — Dense vẫn ổn nhưng Sparse mất. Từ turn 2 trở đi, có context, rewrite có ích vì giúp standalone query. Đó là RQ9.

### Q6: Quy tắc 14 có thực sự ngăn được hallucination không?

**Không 100%** — không có guardrail nào 100%. Nhưng kết hợp với G3 (Pydantic schema bắt buộc cite metadata thực) thì model **không tạo được Điều/Khoản giả** — vì schema validate sẽ fail. Quy tắc 14 ép thêm CoT để tránh đoán tỷ lệ phần trăm. Đây là **defense in depth**, không phải bullet-proof.

### Q7: MCP & A2A là gì, sao chưa làm mà chỉ để roadmap?

**MCP** = Model Context Protocol (Anthropic) — chuẩn để LLM gọi tool ngoài. **A2A** = Agent-to-Agent (Google/Anthropic) — chuẩn để agent gọi agent.

Em xin nói rõ trạng thái:
- **MCP server: chưa có** — codebase không có dependency `mcp`, không có file `mcp_server.py`. Em chỉ có **3 tool Python class với schema typed, stateless** — đáp ứng tiền đề wrap MCP.
- **A2A protocol: chưa có** — codebase không có Agent Card, không có endpoint riêng cho từng agent. Em có **pattern orchestrator + specialist** trong cùng 1 process — đó là **tinh thần A2A**, không phải A2A protocol.

MVP v6.2 ưu tiên **chiều sâu trong domain** (retrieval + Quy tắc 14 + guardrails) thay vì wrap protocol. Wrap MCP ước 1-2 tuần, A2A 2-3 tuần. **Em không claim đã có**, em claim đã chuẩn bị **kiến trúc**.

### Q8: Em có scale lên hàng triệu chunks được không?

Qdrant + HNSW scale tốt đến **hàng chục triệu vector** trên một node. Bottleneck thực tế ở MVP này là **BM25 in-memory** — với corpus lớn em sẽ chuyển sang **OpenSearch** hoặc Elasticsearch. Reranker thì batch trên GPU; CPU chỉ phục vụ developer local. Pipeline 5 lớp **không thay đổi**, chỉ thay implementation backend — đây lại là **lợi thế của swap-able layer**.

### Q9: Đóng góp khoa học cụ thể của đề tài là gì?

Ba đóng góp chính: **(1) Methodology** xây RAG production-grade cho domain pháp lý — 9 trụ + 10 RQ định lượng, tái sử dụng được. **(2) Quy tắc 14** Legal Reasoning CoT — guardrail đặc thù cho câu hỏi không có điều khoản định sẵn, chưa có trong literature. **(3) RAGAS-Lite** — phiên bản single-prompt CoT giải bài toán free-tier throttle, có thể public open source.

### Q10: Vì sao chọn 5 lớp retrieval mà không phải 3 hay 7?

Mỗi lớp **giải một loại lỗi cụ thể**: L1 paraphrase, L2 từ khóa hiếm, L3 fuse rank không cùng thang đo, L4 từng-từ tương tác, L5 multi-hop. Bỏ lớp nào là mất một loại câu hỏi tương ứng. Em không thấy lớp thứ 6 nào có lý do thuyết phục với data hiện tại — ví dụ "query expansion bằng LLM" đã thử nhưng latency không xứng (cũng là phát hiện RQ9 bypass rewrite). 5 là số tối ưu **theo dữ liệu**, không phải con số đẹp.

---

# PHỤ LỤC B — CHEAT SHEET (cầm tay khi demo)

| Hạng mục | Số / Tên |
|---|---|
| Văn bản gốc | **23** = 2 Luật (35/2024 ĐB · 36/2024 TTATGT) + 4 NĐ (10/2020 · 135/2021 · 168/2024 · 336/2025) + 17 TT (4 nhóm: bằng lái · đăng ký xe · khí thải · tuần tra) |
| Chunks | 2 818 |
| Avg token / chunk | 161 |
| Metadata fields | 15 |
| Embedder | E5-small · 384d |
| Vector DB | Qdrant 1.7 |
| Reranker | bge-reranker-v2-m3 · 568M |
| Reranker gain | +21.6% MRR · ×78 latency CPU |
| LLM | Gemini 3.1 Flash Lite Preview |
| Cost / query | $0.013 |
| Cost legal_rag | $0.010 (75%+) |
| RAGAS-Lite | n=5 · 20 calls · ~90s |
| answer_relevancy | 1.00 |
| faithfulness | 0.15 |
| CI/CD floor | Recall@10 ≥ 0.40 |
| Refusal rate (P2 vs zero-shot) | 72% vs 0% |
| Rewrite turn đầu | −35% MRR / +45% lat khi bật |
| Routes | 4 (legal · chit · web · oos) |
| Guardrail layers | 4 (Input · Retrieval · Generation · Human) |

---

# PHỤ LỤC C — TIMING DỰ KIẾN

| Phút | Slide | Nội dung |
|---|---|---|
| 0:00–1:00 | 01 | Bìa, định vị MVP production-grade |
| 1:00–2:30 | 02 | Rubric 9 trụ |
| 2:30–4:00 | 03 | Trụ 01 UX/PRD |
| 4:00–6:00 | 04 | Trụ 02 Data Flow |
| 6:00–7:30 | 05 | Trụ 03 Inference & Cache |
| 7:30–10:30 | 06 | **Trụ 04 Hybrid Retrieval ★** |
| 10:30–12:30 | 07 | Trụ 05 Agents |
| 12:30–14:00 | 08 | Trụ 06 Tools/MCP |
| 14:00–15:15 | 09 | Trụ 07 Observability |
| 15:15–17:00 | 10 | Trụ 08 Evals |
| 17:00–18:45 | 11 | **Trụ 09 Guardrails ★** |
| 18:45–19:45 | 12 | MVP Stack |
| 19:45–20:30 | 13 | Demo plan (giới thiệu) |
| 20:30–22:00 | 14 | Key Decisions RQ |
| 22:00–23:00 | 15 | Roadmap v7 |
| 23:00–23:45 | 16 | Tổng kết |
| 23:45–24:15 | 17 | Cảm ơn → demo |
| 24:15–30:15 | DEMO | 4 kịch bản trực tiếp |
| 30:15+ | Q&A | |

---

# PHỤ LỤC D — GIẢI ĐÁP CHI TIẾT (cầm tay khi Q&A)

Phụ lục này giải đáp **mọi điểm có thể bị hội đồng đào sâu** trong các slide 04–11. Trình bày dạng câu hỏi → trả lời ngắn để tra cứu nhanh khi đứng trên bục.

---

## D.1 PIPELINE ONLINE — slide 04

### D.1.1 "Input gồm query và chat history" — chính xác là gì?

- **`query`** — đúng chuỗi user gõ vào ô chat ở Next.js, ví dụ `"Vượt đèn đỏ xe ô tô phạt bao nhiêu?"`. Frontend forward nguyên văn xuống FastAPI `/chat`.
- **`chat_history`** — list các turn trước đó, mỗi turn là dict `{role: "user"|"assistant", content: str, citations?: list}`. Backend lấy từ `AsyncSqliteSaver` (LangGraph checkpoint DB) theo `thread_id`.
- **Sliding window:** chỉ giữ **6 lượt gần nhất** (`CONTEXTUALIZE_HISTORY_TURNS=6` trong [nodes.py](../source/agent/nodes.py)). Lý do: prompt analyzer có giới hạn token, history dài làm chậm + tăng cost mà thường không cần.
- **Tại sao cần history:** câu thứ 2 có thể là *"Vậy còn xe máy thì sao?"* — analyzer phải resolve "xe máy" về "Vượt đèn đỏ xe máy phạt bao nhiêu?" mới retrieval đúng.

### D.1.2 Analyzer trả về structured output — 3 trường (KHÔNG phải 4)

⚠️ **Đính chính kịch bản slide 07:** AnalyzerOutput trong code thực ([nodes.py:50-54](../source/agent/nodes.py#L50)) chỉ có **3 trường**, không có `requires_approval`:

```python
class AnalyzerOutput(BaseModel):
    category: Literal["legal_rag", "chit_chat", "web_legal_search", "out_of_scope"]
    standalone_query: str   # giải tham chiếu từ history
    expanded_query: str     # cụm tra cứu cho retriever
```

`requires_approval` được set **ở node `web_search_node`**, không phải ở analyzer. Khi bị hỏi nên trả lời chính xác là 3 trường.

**Diễn giải từng trường:**

| Trường | Vai trò | Ví dụ với câu *"Vậy xe máy thì sao?"* (turn 2, có history) |
|---|---|---|
| `category` | Routing 4 nhánh | `"legal_rag"` |
| `standalone_query` | Câu **độc lập** sau khi resolve tham chiếu trong history | `"Vượt đèn đỏ xe máy phạt bao nhiêu tiền?"` |
| `expanded_query` | Cụm **tra cứu** mở rộng (đã thay thuật ngữ pháp lý), cho retriever | `"mức phạt vượt đèn đỏ xe mô tô xe máy không chấp hành hiệu lệnh đèn tín hiệu Nghị định 168/2024"` |

→ `standalone_query` dùng cho **generator** (sinh đáp án); `expanded_query` dùng cho **retriever**. Tách biệt vì 2 nhu cầu khác nhau.

### D.1.3 "Idempotent" + "Replay-able" + "Checkpoint per-node" là gì?

- **Idempotent**: chạy nhiều lần với cùng input → ra cùng output, không phụ thuộc thời gian / random / state ngầm. Pipeline ingestion là idempotent vì `clean_*.py` chỉ đọc PDF + ghi md theo công thức cố định, không sinh số ngẫu nhiên. Hậu quả tốt: chạy lại pipeline khi corpus đổi không sợ "lệch".
- **Replay-able**: từ 1 trace LangSmith bất kỳ, ta **load nguyên trạng input + intermediate state** rồi chạy lại. Hữu ích khi debug: bug ở node thứ 4 → load checkpoint của node 3 → replay node 4 → fix → test luôn, không phải tái dựng cả 3 node đầu.
- **Checkpoint per-node**: LangGraph mặc định lưu state SAU MỖI node bằng `AsyncSqliteSaver`. File DB ở `traffic_rag/checkpoints/graph.db` ([main.py](../api/main.py)). Mỗi `thread_id` có 1 chuỗi snapshot — hệ thống có thể dừng giữa graph (vd. ở `interrupt_before web_finalize` cho HITL), tắt server, mở lại sau 1 ngày, gọi `/resume` → graph chạy tiếp đúng từ điểm dừng.

→ **3 thuộc tính này là tiền đề cho debug + HITL** — không có chúng thì không thể giữ thread chờ admin duyệt qua đêm.

---

## D.2 CACHE 3 TẦNG — slide 05

### D.2.1 T1 Qdrant payload index — tại sao gọi là $O(\log n)$?

- Qdrant lưu payload (`status`, `doc_id`, `effective_date`, `topic`) trong **B-tree index** (cấu trúc dữ liệu cây cân bằng). Lookup 1 giá trị (`status="active"`) duyệt từ root xuống lá theo **chiều cao log₂(N)**, không phải duyệt cả N record.
- Với N = 2 818 chunk: $\log_2(2818) ≈ 11.5$ → trung bình ~12 phép so sánh để filter. Brute-force scan = 2 818 phép. Tốc độ tăng **~235×**.
- Đo thực: filter `status=active` trước HNSW search giảm 12% latency truy vấn (báo cáo §3.11).

**Chú ý hội đồng có thể vặn:** "Sao không O(1)?" → vì B-tree không phải hash. O(1) chỉ có với hash table, mà hash table mất khả năng range query (`effective_date ≤ 2024-11`) — Qdrant chọn B-tree để hỗ trợ cả equality và range.

### D.2.2 T2 BM25 sparse cache — hoạt động ra sao?

- Khi service start, BM25Okapi **đọc 1 lần** [Data/all_chunks.jsonl](../Data/all_chunks.jsonl), tokenize tất cả 2 818 chunk bằng `pyvi.ViTokenizer`, xây ma trận TF + IDF cache **trong RAM**.
- Mỗi query: chỉ tokenize query rồi tính score in-memory, **không I/O đĩa**. Đây là lý do BM25 chạy < 50ms cho corpus này.
- **Cost:** ~25 MB RAM cho 2 818 chunk × ~160 token trung bình. Không đáng kể.

### D.2.3 T3 Prompt prefix cache — DỰ ÁN ĐÃ LÀM HAY CHƯA?

⚠️ **Trả lời thẳng: CHƯA** — đây là **roadmap v7** (Slide 15 H2).

- Hiện tại mỗi LangGraph run gửi nguyên prompt template (~2 000 token system prompt + 14 rules) lên Gemini → **trả token đầy đủ mỗi query**.
- Plan v7: dùng [Gemini Context Caching API](https://ai.google.dev/gemini-api/docs/caching) — cache prefix không đổi (system prompt + 14 rules), chỉ trả tiền cho phần biến (query + chunks). Ước tính giảm **30% input cost**.
- **Tại sao chưa làm:** v6.2 ưu tiên correctness của Quy tắc 14 (legal reasoning); caching là tối ưu cost, làm sau khi prompt đã ổn định.

→ Nếu hội đồng hỏi "đã đo cost thực 0.013$/query có context cache không?" — trả lời **KHÔNG, đây là cost không cache**, sau khi bật cache sẽ về ~0.009$/query.

---

## D.3 BM25 — VÍ DỤ HOẠT ĐỘNG VỚI 1 CÂU HỎI ĐƠN GIẢN

Đây là phần hội đồng hay đào — phải nắm chắc.

### D.3.1 Tổng quan: BM25 nhận gì, trả gì?

**Input:** 1 query (string) + 1 corpus (list các chunk đã tokenize sẵn).
**Output:** Mảng score độ dài N (= số chunk), `score[i]` = mức liên quan của chunk i với query. Sort descending → top-K.

### D.3.2 Worked example với câu *"vượt đèn đỏ ô tô phạt bao nhiêu"*

**Bước 1 — Tokenize bằng pyvi:**
```python
query_tokens = pyvi.ViTokenizer.tokenize("vượt đèn đỏ ô tô phạt bao nhiêu").split()
# = ["vượt_đèn_đỏ", "ô_tô", "phạt", "bao_nhiêu"]
```

Lưu ý pyvi gộp `vượt_đèn_đỏ` thành 1 từ ghép — đây là điểm BM25 mạnh hơn dense cho từ khóa pháp lý.

**Bước 2 — Tính IDF cho từng term (đã pre-compute lúc start):**

$$\operatorname{IDF}(t) = \log\!\left(\frac{N - df(t) + 0.5}{df(t) + 0.5} + 1\right)$$

| Term | df (số chunk chứa) | IDF |
|---|---:|---:|
| `vượt_đèn_đỏ` | 8 | 5.79 (rất hiếm → IDF cao) |
| `ô_tô` | 412 | 1.91 |
| `phạt` | 1 050 | 0.99 |
| `bao_nhiêu` | 230 | 2.49 |

→ Term hiếm có trọng số cao hơn, đúng nguyên lý "từ khóa hiếm = informative".

**Bước 3 — Với mỗi chunk, tính score:**

$$\operatorname{score}(D) = \sum_{t \in Q} \operatorname{IDF}(t) \cdot \frac{f(t,D)\,(k_1+1)}{f(t,D) + k_1\,(1 - b + b\,\frac{|D|}{\operatorname{avgDL}})}$$

Giả sử chunk **Đ.6 K.11 NĐ 168/2024** (về vượt đèn đỏ xe ô tô) có:
- `vượt_đèn_đỏ` xuất hiện 2 lần, `ô_tô` 3 lần, `phạt` 1 lần, `bao_nhiêu` 0 lần.
- $|D| = 187$, $\operatorname{avgDL} = 195$, $k_1=1.5$, $b=0.75$.
- Length factor: $1 - 0.75 + 0.75 \cdot 187/195 ≈ 0.969$.

| Term | $f$ | TF saturated $\frac{f \cdot 2.5}{f + 1.5 \cdot 0.969}$ | × IDF |
|---|---:|---:|---:|
| `vượt_đèn_đỏ` | 2 | $2 \cdot 2.5 / (2 + 1.45) ≈ 1.45$ | $\times 5.79 = 8.40$ |
| `ô_tô` | 3 | $3 \cdot 2.5 / (3 + 1.45) ≈ 1.69$ | $\times 1.91 = 3.23$ |
| `phạt` | 1 | $1 \cdot 2.5 / (1 + 1.45) ≈ 1.02$ | $\times 0.99 = 1.01$ |
| `bao_nhiêu` | 0 | 0 | 0 |

**Tổng score = 8.40 + 3.23 + 1.01 + 0 = 12.64.**

**Bước 4 — Tính score cho mọi chunk khác, sort, trả top-K:**

Chunk Đ.6 K.11 thắng vì có 2 từ hiếm `vượt_đèn_đỏ` + đủ `ô_tô`. Một chunk Luật chung chỉ có "phạt" sẽ ra score ~1, xếp dưới.

→ **Đây là lý do BM25 gọi là *sparse retrieval* — score = tổng đóng góp của từng từ có MẶT trong query, từ không có thì 0.**

### D.3.3 BM25 trả gì cho retriever pipeline?

```python
top_idx = np.argsort(-bm25_scores)[:top_k * 2]   # = 20 chunk có score cao nhất
sparse_hits = [(rank, chunk_id) for rank, idx in enumerate(top_idx, 1)]
```

→ Đầu ra là **danh sách rank** (rank 1, 2, 3, …, 20 cho từng chunk_id), đưa sang RRF fuse với dense.

### D.3.4 Tại sao $k_1 = 1.5$ và $b = 0.75$?

- **$k_1$ — TF saturation:** càng lớn → từ xuất hiện nhiều lần càng được "thưởng". Khoảng [1.2, 2.0] là chuẩn IR (Robertson & Zaragoza). Chọn 1.5 là điểm giữa, phù hợp văn bản pháp luật có từ khóa lặp (ví dụ "phạt tiền" lặp 5-10 lần trong 1 Khoản).
- **$b$ — length normalization:** càng lớn → càng phạt văn bản dài.
  - $b = 0$: không phạt độ dài, BM25 trở về TF-IDF thuần.
  - $b = 1$: phạt tối đa, chunk dài bị chia hết score.
  - **0.75 là default Lucene/Elasticsearch** — chứng minh thực nghiệm tốt cho corpus có độ dài đa dạng. Corpus em min 50 token → max 5 803 token (chênh 116×) nên cần $b > 0$.
- **Đã fine-tune chưa?** Chưa. Mặc định Okapi. Đây là backlog v7 (grid search trên 25 gold query).

---

## D.4 HYBRID RETRIEVAL 5 LỚP — slide 06

### D.4.1 L1 Dense — prefix `"query: "` và `"passage: "` là gì?

E5 (`intfloat/multilingual-e5-small`) được train **bất đối xứng**: query và passage đi qua cùng encoder NHƯNG được phân biệt bằng prefix.

**Code thực ở [indexer.py:42](../source/indexing/indexer.py#L42):**

```python
PASSAGE_PREFIX = "passage: " if "e5" in EMBEDDING_MODEL.lower() else ""
texts = [PASSAGE_PREFIX + c["content"] for c in chunks]   # khi index
```

**Code retriever:**
```python
QUERY_PREFIX = "query: "
q_vec = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
```

**Tại sao bắt buộc phải có:**
- E5 train với loss InfoNCE trên 1B cặp (query, passage). Trong batch train, model học rằng: vector `"query: vượt đèn đỏ"` phải gần vector `"passage: Điều 6 khoản 11..."`, **không phải** gần vector `"query: Khoản 11 Điều 6..."`.
- **Nếu encode chunk với `"query: "` thay vì `"passage: "`** (hoặc bỏ prefix) → vector sai không gian → cosine similarity giảm khoảng 30% recall (trong RQ3 chính tác giả paper E5 đã ablation).

→ Câu trả lời ngắn cho hội đồng: "**E5 là encoder asymmetric — query và passage có không gian biểu diễn riêng, prefix là cách model phân biệt.** Bỏ prefix mất ~30% recall."

### D.4.2 L2 BM25 — sang D.3 đã đầy đủ.

### D.4.3 L3 RRF — k=60 và "fuse rank không fuse score"

**Vấn đề cần fuse 2 ranker:**

| Ranker | Range score | Vấn đề khi cộng thẳng |
|---|---|---|
| Dense (cosine sau L2-norm) | $[-1, 1]$ | thường ~0.5–0.9 |
| BM25 | $[0, +\infty)$ | thường 5–30, đôi khi 100 |

→ Cộng thẳng `score_dense + score_bm25`: BM25 luôn áp đảo do scale lớn hơn, dense bị "giết". Nếu chuẩn hóa min-max thì lại nhạy cảm với outlier (1 chunk score BM25 cao bất thường làm méo cả phân phối).

**Giải pháp Cormack 2009 — RRF dùng RANK thay vì SCORE:**

$$\operatorname{RRF}(d) = \sum_{\ell \in \{\text{dense, sparse}\}} \frac{1}{k + \operatorname{rank}_{\ell}(d)}$$

**Tại sao $k = 60$:**
- Là default Cormack đề xuất, đo trên 9 collection TREC.
- Trực giác: khoảng cách rank 1 → rank 10 chỉ là $1/61 - 1/70 = 0.0021$. Tức **không 1 list nào "đè" list kia chỉ vì có 1 hit ở top**. Điều này *cố ý làm phẳng*.
- $k$ thấp ($k=10$): top rank được tăng đột biến → 1 ranker dễ thắng.
- $k$ cao ($k=200$): mọi rank gần như bằng nhau → fuse mất ý nghĩa.
- Stable cho corpus < 10 000 doc theo Cormack — corpus em 2 818 chunk nằm an toàn.

**Tính tay với chunk Đ.6 K.11:**
- Dense rank 1, sparse rank 1 → $\operatorname{RRF} = 1/61 + 1/61 = 0.0328$.
- Chunk Đ.6 K.5 (gần đúng): dense rank 2, sparse rank 5 → $1/62 + 1/65 = 0.0315$.
- Chunk Luật 36 chỉ ở dense rank 4, không có ở sparse → $1/64 + 0 = 0.0156$.

→ Sort theo RRF descending → top-20 đưa sang L4/L5.

**Code [retriever.py:367-395](../source/rag_core/retriever.py#L367):**

```python
K_RRF = 60
scores = defaultdict(float)
for rank, hit in enumerate(dense_hits, 1):
    scores[hit.payload["chunk_id"]] += 1.0 / (K_RRF + rank)
for rank, idx in enumerate(top_bm25_idx, 1):
    scores[all_chunks[idx]["metadata"]["chunk_id"]] += 1.0 / (K_RRF + rank)
fused = sorted(scores.items(), key=lambda x: -x[1])[:top_k * 2]
```

### D.4.4 L4 Reranker — đang HOẠT ĐỘNG hay không?

⚠️ **Trả lời chính xác: Có code, MẶC ĐỊNH TẮT.**

**Code thực:**
- File: [source/rag_core/reranker.py](../source/rag_core/reranker.py) — class `Reranker` dùng `BAAI/bge-reranker-v2-m3` (568M params, multilingual, max-len 8192).
- File: [retriever.py:78-101](../source/rag_core/retriever.py#L78) — constructor `TrafficHybridRetriever(enable_reranker: bool = False, ...)`. **Default False**.
- Lazy load: chỉ khi `enable_reranker=True` mới import + load model (model nặng ~2.3 GB, mất ~30s init).
- Toggle qua biến môi trường: `ENABLE_RERANKER=true` trong `.env` hoặc compose.

**Reranker hoạt động chi tiết:**

1. **Input:** query + top-30 chunk từ RRF (đã fused).
2. **Encoding khác Dense:** thay vì encode query và passage **riêng** rồi cosine, Cross-Encoder ghép `[CLS] query [SEP] passage [SEP]` thành 1 chuỗi, đưa qua Transformer → ra **1 score logit**. Mỗi cặp (query, chunk) chạy riêng → 30 forward pass.
3. **Output:** 30 score thay thế RRF score. Sort lại → top-K (mặc định K=10).

**Tại sao Cross-Encoder chính xác hơn Dense:**
- Dense (Bi-encoder): query và passage encode **độc lập**, similarity tính ở cuối. Self-attention không thấy tương tác giữa từ trong query với từ trong passage.
- Cross-Encoder: **query và passage cùng đi vào self-attention** → mỗi từ trong query attend được tới từng từ trong passage. Bắt được "vượt đèn đỏ ô tô" liên hệ chính xác với "ô tô không chấp hành đèn tín hiệu" hơn cosine.

**Trade-off (RQ7 ablation đã đo):**
- **+21.6% MRR** (gold set 25 query)
- **×78 latency** trên CPU (1 forward Cross-Encoder ~150ms × 30 chunk = ~4.5s; vs Bi-encoder 2 cosine = ~50ms)
- → CPU production: tắt; GPU + batch eval / CI: bật.

⚠️ **RQ7 vs RQ10 trong kịch bản:** kịch bản v3 ghi RQ10 (slide 06, 14) — đây là **typo/version drift**. File CSV thực là `rq7_*` nhưng nội dung là reranker ablation. Khi hội đồng hỏi "RQ10 ở đâu?" → trả lời thẳng "RQ này thực tế đánh số RQ7 trong code repo, đã đổi tên cho thống nhất". An toàn hơn là **đổi mọi chỗ nhắc RQ10 → RQ7** trong slide trước bảo vệ.

### D.4.5 L5 Expand — Sibling ±2 và Cross-reference regex

**Sibling expansion ±2 khoản:**

Văn bản pháp luật viết theo cấu trúc: Khoản 1 định nghĩa, Khoản 2 áp dụng, Khoản 3 ngoại lệ, Khoản 4 chế tài. Retriever đôi khi chỉ trúng Khoản 4 (vì có "phạt") nhưng người dùng cần đọc cả Khoản 1-3 để hiểu cảnh áp dụng.

**Cách hoạt động:** với mỗi top-K chunk có `(doc_id, dieu, khoan)`, kéo thêm các chunk có **cùng `doc_id`, cùng `dieu`** và **`|khoan − khoan_target| ≤ 2`**. Ví dụ chunk gốc Đ.6 K.11 → kéo thêm Đ.6 K.9, K.10, K.12, K.13.

**Cap để tránh bùng:** kéo tối đa 4 sibling/chunk gốc, total top-K không vượt 30 chunk.

**Cross-reference regex:**

Văn bản hay viết "theo quy định tại điểm a khoản 4 Điều 13" — đây là **tham chiếu chéo**. Nếu chỉ retrieve Khoản chứa câu này, người dùng vẫn không biết Đ.13 K.4 nói gì.

**Code [retriever.py](../source/rag_core/retriever.py):**

```python
RE_XREF = re.compile(
    r"(?:điểm\s+([a-zđ]))?\s*khoản\s+(\d+)\s+Điều\s+(\d+)",
    re.IGNORECASE,
)

# Quét trong top-5 primary chunk
for chunk in top_5:
    for match in RE_XREF.finditer(chunk["content"]):
        diem, khoan, dieu = match.groups()
        target = retriever.get_chunks_by_location(
            doc_id=chunk["metadata"]["doc_id"],   # giả định cùng văn bản
            dieu=int(dieu), khoan=khoan, diem=diem,
        )
        if target:
            additional_chunks.append(target)
```

**Cap ở 30 total** để không bùng context.

→ **Đây là cơ chế multi-hop đơn giản:** chunk gốc dẫn tới chunk khác qua trích dẫn pháp lý, retriever tự kéo về.

### D.4.6 Pipeline cuối: tổng kết

```
query (1 query)
   ├── L1 Dense  → top 20 chunk (rank 1..20)
   └── L2 BM25   → top 20 chunk (rank 1..20)
                    │
                    ▼
              L3 RRF k=60 → top 20 fused
                    │
                    ▼
       (optional) L4 Rerank cross-encoder → top 10 rescored
                    │
                    ▼
       L5 Sibling ±2 + Cross-ref regex → final top ≤ 30
                    │
                    ▼
              chuyển sang Generator
```

---

## D.5 REWRITE QUERY — slide 07 (RQ9)

### D.5.1 Khi nào hệ thống rewrite, khi nào không?

**Code [nodes.py](../source/agent/nodes.py):**

```python
def analyzer_node(state):
    out = llm.invoke(prompt + state["query"])
    if not state.get("chat_history"):
        expanded = state["query"]        # turn ĐẦU: bypass expansion
    else:
        expanded = out.expanded_query    # turn 2+: dùng expanded_query
    return {
        "category": out.category,
        "standalone_query": out.standalone_query,
        "expanded_query": expanded,
    }
```

→ **Quy tắc:**
- **Turn 1 (chat_history rỗng):** bỏ qua `expanded_query` của analyzer, dùng nguyên `query` cho retriever.
- **Turn 2+:** dùng `expanded_query` để giải tham chiếu.

### D.5.2 Ví dụ cụ thể — analyzer rewrite ra sao?

**Câu turn 1:** *"Vượt đèn đỏ ô tô phạt bao nhiêu?"*
- Analyzer trả về `expanded_query = "mức phạt vượt đèn đỏ xe ô tô không chấp hành hiệu lệnh đèn tín hiệu giao thông Nghị định 168/2024"`
- Nhưng **system bypass** → retriever vẫn nhận query thô `"Vượt đèn đỏ ô tô phạt bao nhiêu?"`.

**Câu turn 2:** *"Vậy còn xe máy thì sao?"*
- chat_history có turn 1 + answer → analyzer thấy context.
- `standalone_query = "Vượt đèn đỏ xe máy phạt bao nhiêu?"`
- `expanded_query = "mức phạt vượt đèn đỏ xe mô tô xe máy điều khiển không chấp hành đèn tín hiệu Nghị định 168/2024 Điều 7"`
- Hệ thống dùng `expanded_query` cho retriever — vì câu thô "Vậy còn xe máy thì sao?" không có thông tin retrieval.

### D.5.3 Tại sao rewrite turn đầu lại HẠI? (RQ9)

**Đo trên 25 gold query:**

| Variant | MRR | Latency |
|---|---:|---:|
| NO_REWRITE turn đầu | **0.197** | 1.87s |
| REWRITE turn đầu | 0.127 | 2.72s |

→ Rewrite turn đầu giảm 35% MRR + tăng 45% latency.

**Lý do:**
1. **LLM thêm boilerplate pháp lý:** *"Vượt đèn đỏ ô tô phạt bao nhiêu?"* → *"Theo quy định của pháp luật về xử phạt vi phạm hành chính, mức phạt cho hành vi vượt đèn đỏ..."* → query vector kéo về **average embedding của corpus** (vì boilerplate xuất hiện ở mọi chunk pháp lý) → mất tính phân biệt.
2. **BM25 mất từ khóa hiếm:** số `"168/2024"` bị thay bằng `"Nghị định mới"` → IDF cao của `"168/2024"` mất, score giảm.
3. **E5 đã hiểu câu dân sinh** (pretrain 1B cặp natural query) — không cần "hạ cấp" thành legal-ese.

### D.5.4 Hệ thống có ràng buộc gì khi rewrite?

- **Bắt buộc giữ schema:** `expanded_query: str`, không đổi sang `list[str]` để giảm sửa downstream.
- **Format đa-hành-vi:** câu phức nhiều ý nối bằng `" | "` — ví dụ *"đi ngược chiều quẹt xe ai có lỗi?"* → expanded `"đi ngược chiều mức phạt | chuyển làn không xi nhan | phân định lỗi trong tai nạn"`. BM25 nhận `" | "` như delimiter → match nhiều cụm từ.
- **Bypass turn đầu là cứng** — không có toggle. Đây là quyết định production sau RQ9.

---

## D.6 MCP & A2A — slide 08

### D.6.1 MCP là gì? Liên hệ với đồ án?

**Định nghĩa:**
- **MCP = Model Context Protocol** — chuẩn do Anthropic công bố tháng 11/2024.
- Mục tiêu: tạo **giao diện chung** cho LLM gọi tool ngoài (giống USB-C cho thiết bị: 1 chuẩn, mọi LLM client kết nối được).
- Trước MCP, mỗi tool wrap riêng cho từng client (Claude Desktop wrap khác Cursor wrap khác). Sau MCP, viết 1 lần dùng được mọi client.

**Liên hệ với đồ án:**
- ⚠️ **Đồ án chưa có MCP server thực tế.** Hiện tại 3 tool (`hybrid_search`, `tavily_search`, `resolve_xref`) chỉ là **Python class**, gọi nội bộ trong LangGraph node.
- **Đã thiết kế đúng interface MCP-ready:** mỗi tool có schema rõ ràng (input dict, output dict typed), không có hidden state. Wrap MCP layer = thêm ~300 dòng code FastAPI/SSE chuẩn theo spec.
- **Roadmap v7 H3:** đóng gói thành MCP server → Claude Desktop, Cursor có thể gọi `hybrid_search("vượt đèn đỏ")` trực tiếp mà không cần biết internals.

→ Khi hội đồng hỏi: trả lời thẳng "**hiện tại chưa, đã thiết kế interface MCP-ready, làm wrap layer ở v7.**" Đừng claim là đã có.

### D.6.2 A2A là gì? Liên hệ với đồ án?

**Định nghĩa:**
- **A2A = Agent-to-Agent protocol** — chuẩn cho 2 agent độc lập gọi nhau (Anthropic + Google đang chuẩn hóa song song, chưa thống nhất).
- Mỗi agent là 1 service riêng: có endpoint, schema message, capability declaration. Orchestrator gọi sub-agent qua message passing thay vì gọi function trong code.

**Liên hệ với đồ án:**
- ⚠️ **Đồ án có pattern A2A "đơn giản" — không phải A2A protocol thực.** Cụ thể:
  - Analyzer **đóng vai orchestrator** (phân loại + route).
  - 4 sub-node (`legal_rag`, `chit_chat`, `web_search`, `out_of_scope`) **đóng vai sub-agent**.
  - **Nhưng chúng đều cùng 1 process Python** — không phải agent độc lập có endpoint riêng.
- **Roadmap v7 H4:** liên kết với agent CSGT chuyên biệt (xử lý biên bản, sinh quyết định xử phạt) qua A2A protocol thật → multi-process, có endpoint, có capability declaration.

→ Câu trả lời an toàn: "**Đồ án có pattern phân tách orchestrator + specialist nhưng không phải A2A protocol đầy đủ. Chuyển sang A2A thực sự cần protocol message ổn định, em để v7.**"

---

## D.7 CROSS-REFERENCE — đã ở D.4.5

Tóm lại: regex bắt cụm `"điểm a khoản 4 Điều 13"` trong content top-5 chunk → tra thẳng `(doc_id, dieu, khoan, diem)` → kéo chunk đích về context. Cap 30 total.

→ **Đây là multi-hop reasoning cấp 1** (chỉ 1 hop — chunk gốc trỏ tới chunk B). Multi-hop ≥ 2 (B trỏ tới C) **chưa hỗ trợ** — backlog.

---

## D.8 OBSERVABILITY — slide 09

### D.8.1 Observability là gì?

**Định nghĩa:** khả năng nhìn thấy state nội bộ của hệ thống từ output bên ngoài. Trong AI/RAG = trace mọi LLM call + tool call + state transition để có thể debug "vì sao output này".

**3 trụ cột observability:**
1. **Logs** — sự kiện rời rạc (request đến, error xảy ra). Truyền thống.
2. **Metrics** — số liệu tổng hợp (latency p50/p95, cost/query). Cho dashboard.
3. **Traces** — chuỗi các bước trong 1 request, với parent-child. **Quan trọng nhất cho LLM/RAG.**

### D.8.2 LangSmith trace 3 cấp — ý nghĩa từng cấp

| Cấp | Object | Thông tin | Khi nào xem |
|---|---|---|---|
| **Graph run (root)** | toàn bộ truy vấn | category, refused, total cost, end-to-end latency, error nếu có | Audit toàn cục: hệ thống có chạy đúng không |
| **Node run (intermediate)** | từng LangGraph node | analyzer/legal_rag/web_search per-node latency và token | Debug: bottleneck nằm node nào |
| **LLM call (leaf)** | mỗi `model.invoke()` | prompt cụ thể, response thô, finish_reason, retry count | Khi output sai: prompt gửi đi đúng không, model trả gì |

**Project name** là `Traffic-RAG-Evaluation` — tất cả run được group tự động.

### D.8.3 Cost center — đo thực không ước

| Node | Token/run | Cost/run |
|---|---:|---:|
| `legal_rag` | 39 749 | **$0.01020** |
| `analyzer` | 934 | $0.00038 |
| `web_search` | 4 440 | $0.00170 |
| `chit_chat` | 144 | $0.00009 |
| **Total/query** | ~45 000 | **~$0.013** |

→ **Insight:** muốn giảm cost, optimize `legal_rag` (75%+). Web_search chỉ 16%. Không có table này, có thể đầu tư sai chỗ vài tuần.

→ **Hội đồng hỏi "đo bằng cách nào?":** LangSmith trace có metadata `total_tokens` per LLM call. Em cộng ngược lên cấp node, rồi cộng lên cấp graph run. Cost = `tokens × giá Gemini` ($0.075/1M input, $0.30/1M output).

---

## D.9 RAGAS — slide 10

### D.9.1 RAGAS là gì? Hoạt động ra sao?

**RAGAS (Retrieval-Augmented Generation Assessment)** — framework đánh giá RAG **không cần human label** (reference-free). Dùng 1 LLM khác làm "judge" để chấm điểm.

**4 metric chính:**

| Metric | Đo gì | Cách judge tính |
|---|---|---|
| **answer_relevancy** | Câu trả lời có liên quan câu hỏi không? | Judge sinh N câu hỏi từ answer, so cosine similarity với câu hỏi gốc → trung bình |
| **faithfulness** | Câu trả lời có **bám đúng** context không? (chống hallucination) | Tách answer thành claims atomic, kiểm tra mỗi claim có infer được từ context không |
| **context_precision** | Trong các chunk retrieve, có bao nhiêu **liên quan** câu hỏi? | Judge chấm relevance từng chunk với câu hỏi |
| **context_recall** | Gold answer có được **support** đầy đủ bởi context không? | Tách gold thành sentences, kiểm tra mỗi sentence có chunk hỗ trợ |

### D.9.2 Pipeline RAGAS gốc — vì sao không chạy được?

**Cost RAGAS gốc:**
- Faithfulness: 1 sample → tách ~5 atomic claim × 2 LLM call (extract + verify) = 10 call
- Tương tự cho 4 metric → ~30-60 call/sample
- 25 query × 4 metric × 4 pipeline (RQ1) = **6 000 LLM call**.
- Free-tier Gemini: 15 RPM (request/minute) → 6 000 call cần ~7 giờ liên tục, throttle nhiều.

**Hậu quả thực tế (file [rq7_ragas_summary.csv](../research/results/metrics/rq7_ragas_summary.csv)):**
```csv
pipeline,n,faithfulness,answer_relevancy,context_precision,context_recall,error
gemini_only,25,nan,nan,nan,nan,
vanilla_rag,25,nan,nan,nan,nan,
agentic_rag,25,nan,nan,nan,nan,
```
→ **93.5% NaN** vì throttle.

### D.9.3 RAGAS-Lite — giải pháp tự code

**File:** [research/scripts/rq7_ragas_subset.py](../research/scripts/rq7_ragas_subset.py)

**Thay đổi:**
1. **n=5** thay n=25 — giảm 5×.
2. **Single-prompt CoT:** thay vì decompose atomic claim ra ngoài (mỗi claim 1 call), prompt LLM **tự decompose nội bộ** trong 1 call duy nhất. Output format `SCORE: 0.XX` parse bằng regex.
3. **Exponential backoff:** retry ở 8s → 16s → 32s → 64s khi gặp throttle.
4. → **20 call total** (4 metric × 5 sample), ~90 giây cho n=5, 0% NaN.

### D.9.4 Kết quả n=5 simple_penalty — diễn giải

| Metric | Score | Status | Diễn giải kỹ thuật |
|---|---:|---|---|
| `answer_relevancy` | 1.00 | ✓ | Generator viết đúng trọng tâm câu hỏi. Khi cho đúng evidence, model không lan man. |
| `context_recall` | 0.23 | ⚠ | 77% nội dung gold answer KHÔNG được context hỗ trợ → retriever miss khoản |
| `context_precision` | 0.20 | ⚠ | 80% chunk retrieve KHÔNG liên quan → noise |
| `faithfulness` | 0.15 | ✗ | Bottleneck: 85% claims không bám context — vì context noise nhiều |

**Đánh giá vấn đề gì:**
- Generator OK (relevancy = 1.00). Vấn đề **không phải LLM viết tệ**.
- **Bottleneck = retrieval precision thấp** → chunk retrieve không sạch → generator phải "đoán" → faithfulness giảm.
- → Quyết định: bật reranker khi SLA cho phép (RQ7 đã chứng minh +21.6% MRR), kỳ vọng kéo precision lên → faithfulness sẽ tăng theo.

⚠️ **Lưu ý:** kịch bản nói "n=5" nhưng file CSV đầu ra hiện tại có cả n=25 (NaN). Khi hội đồng hỏi: trả lời **n=5 simple_penalty là kết quả RAGAS-Lite chạy được**; n=25 trong CSV là attempt RAGAS gốc bị throttle, để lại làm chứng cứ về "vì sao cần Lite version".

### D.9.5 RAGAS có phải tiêu chuẩn vàng không?

**Không.** Có 2 hạn chế:
1. **Judge bias:** Gemini chấm Gemini → có thể tự thiên vị. Tốt nhất dùng judge khác model (GPT-4o hoặc Claude). Roadmap v7 H5 — multi-judge ensemble.
2. **n=5 không đủ statistical significance** — em thừa nhận. Mục đích RAGAS-Lite là **smoke test** trong CI/CD, không phải báo cáo final.

→ **Khi hỏi "tại sao tin được số 0.15?"** trả lời: "Em không claim 0.15 là số tuyệt đối. Em claim **bottleneck nằm ở retrieval** vì gap giữa relevancy 1.00 và faithfulness 0.15 quá lớn — direction đúng, magnitude cần n lớn hơn để confirm."

---

## D.10 CI/CD — slide 10 (CHI TIẾT THỰC TẾ)

### D.10.1 Đã làm gì?

⚠️ Khác với những gì kịch bản nói "Recall@10 ≥ 0.40 merge bị khóa" — file thực tế nhẹ hơn. Phải nói **CHÍNH XÁC** để không bị bắt:

**File:** [traffic_rag/.github/workflows/ci.yml](../.github/workflows/ci.yml)

**2 job song song:**

#### Job 1 — `smoke`: Smoke tests (không cần Qdrant)
```yaml
runs-on: ubuntu-latest
timeout-minutes: 10
- Install deps: pytest, qdrant-client, rank-bm25, sentence-transformers
- Run: pytest tests/test_smoke.py -m "not requires_qdrant"
```
→ Test nhanh < 10 phút: import được hết module, schema Pydantic valid, pyvi tokenize OK.

#### Job 2 — `regression`: Retrieval regression với ephemeral Qdrant
```yaml
services:
  qdrant: qdrant/qdrant:v1.14.2 với health check
- Install full deps từ requirements.txt
- Index corpus (best-effort): nếu Data/all_chunks.jsonl có thì re-index
- Run: pytest tests/test_retrieval_regression.py -v
```
→ Spin lên 1 Qdrant container ephemeral, index, chạy retrieval test. Nếu `all_chunks.jsonl` bị gitignore (thường vậy) → test skip "best-effort".

### D.10.2 Recall@10 ≥ 0.40 threshold có thực hay không?

**Phải kiểm tra `tests/test_retrieval_regression.py`** xem có assertion `assert recall@10 >= 0.40` không.

⚠️ Nếu không có assert đó, kịch bản đang OVERCLAIM. Cần xác minh trước bảo vệ.

**Câu trả lời an toàn nếu hội đồng hỏi:**
- Thật: "CI có 2 job — smoke test imports + regression test retrieval với Qdrant ephemeral. Threshold cụ thể em **đang configure**, mức floor dự kiến R@10 ≥ 0.40."
- Tuyệt đối **không claim** "merge bị khóa cứng" nếu chưa có branch protection rule trên GitHub.

### D.10.3 CI/CD trong dự án này khác CI/CD truyền thống ra sao?

| Khía cạnh | CI/CD code thường | CI/CD RAG này |
|---|---|---|
| Kiểm | unit test, lint, build | **+ retrieval quality** (R@10), **+ smoke graph run** |
| Pass criterion | code chạy được | code chạy + retrieval không tụt quality |
| Rủi ro chính catch được | regression code | regression **chất lượng AI** (đổi embedder, đổi chunker, đổi prompt mà không đo) |

→ **Đây là điểm khác biệt:** CI cho RAG cần **quality gate trên dữ liệu**, không chỉ assert code.

---

## D.11 14 RULES — CHI TIẾT TỪNG RULE (slide 11)

File: [source/rag_core/generator.py](../source/rag_core/generator.py)

| # | Rule | Mục tiêu kỹ thuật |
|---:|---|---|
| 1 | Chỉ dùng NGỮ CẢNH | Chống hallucination — không suy luận ngoài chunk được cấp |
| 2 | Trích dẫn bắt buộc `[doc_id, Đ, K, Đ]` | Mọi mức phạt/số điểm phải có source — bắt buộc với pháp lý |
| 3 | Từ chối rõ ràng | Khi không đủ context: "Thông tin này không có trong tài liệu" |
| 4 | Không tự nghĩ số tiền | Copy nguyên `"4.000.000 đến 6.000.000 đồng"` — không paraphrase |
| 5 | Ưu tiên văn bản mới | Khi NĐ 100/2019 (cũ) và NĐ 168/2024 mâu thuẫn → dùng 168 |
| 6 | Đa ý tách bullet Markdown | `### Mục` + `- ` + bold cho số tiền/điểm/thời gian |
| 7 | Phân biệt kinh doanh vận tải | NĐ 10/2020 cho xe KD; bỏ qua nếu hỏi xe cá nhân |
| 8 | Loại phương tiện trong NĐ 168 | Đ.6 ô tô, Đ.7 mô tô/máy, Đ.8 xe chuyên dùng, Đ.9 xe đạp/thô sơ |
| 9 | Sibling chunk được dùng | Chunk `[Ngữ cảnh bổ sung]` được phép dùng để hoàn thiện câu trả lời |
| 10 | Cross-reference 3 bước | Phạt-tiền + trừ-điểm trong NĐ 168 ghép từ Khoản phạt × Khoản tham chiếu (Đ.6 K.13–K.16) |
| 11 | Không từ chối nếu có 1 phần | Tìm được mức phạt nhưng thiếu số điểm → trả phần tìm được + ghi chú |
| 12 | Tổng hợp khi liệt kê | Câu "các trường hợp / hành vi nào" → rà soát toàn bộ ngữ cảnh |
| 13 | Giải thích dễ hiểu | Mô tả hành vi tiếng Việt tự nhiên trước khi `[trích Điều/Khoản]` |
| 14 | **Legal Reasoning CoT 4 bước** | Override Rule 4 cho câu "lỗi do ai" — xem D.11.1 |

### D.11.1 Quy tắc 14 — chi tiết 4 bước

**Trigger:** câu hỏi chứa "lỗi do ai", "ai có lỗi", "phân định lỗi", "trách nhiệm trong va chạm".

**Bước 1 — Liệt kê hành vi từng bên** (chỉ trích, không kết luận):
> "Bên A: đi ngược chiều. Bên B: chạy quá tốc độ."

**Bước 2 — Đối chiếu hành vi vào chunk vi phạm:**
> "Đi ngược chiều: phạt 4-6M [Đ.6 K.5 NĐ 168]. Quá tốc độ: phạt theo các mức [Đ.6 K.7 NĐ 168]."

**Bước 3 — Kết luận lỗi logic** (cấp duy nhất được "suy ra"):
> "Cả hai bên đều vi phạm → **lỗi hỗn hợp**."

**Bước 4 — Disclaimer thẩm quyền CSGT:**
> "Tỷ lệ % lỗi cụ thể do CSGT xác định theo TT 72/2024/TT-BCA."

**Quy tắc cứng:**
- Mỗi hành vi PHẢI có trích dẫn từ context.
- Nếu context KHÔNG có chunk về 1 hành vi → ghi rõ "không tìm thấy quy định trong tài liệu" cho hành vi đó, **không bịa Điều/Khoản**.
- Override Rule 4 (cấm từ chối) **chỉ** cho dạng câu này. Câu khác vẫn áp Rule 4.

→ **Đây là guardrail không có trong sách RAG** — em phải tự design cho domain pháp lý.

---

## D.12 CÂU HỎI KHÓ + ĐÁP ÁN (bổ sung Phụ lục A)

### Q11. RRF tại sao $k=60$, không 30 hay 100?
Default Cormack 2009. Đo trên 9 collection TREC, ổn định cho corpus < 10 000 doc. Không nhạy với $k$ trong khoảng [40, 80] (chênh ±10% RRF score chỉ ~1% rank thay đổi). Em không fine-tune lại — backlog.

### Q12. Cosine vs Euclidean distance khác gì?
Sau L2-norm thì 2 metric **tương đương về thứ tự** (cosine $= 1 - \frac{1}{2}\|a-b\|^2$). Cosine không nhạy với độ dài vector. Qdrant em chọn `Distance.COSINE` — chuẩn cho dense retrieval đã pre-normalize.

### Q13. Sao chunk avg 161 token mà max 5 803?
Outlier ở Đ.6 NĐ 168 — Khoản này liệt kê 16 mục a)–p) về vi phạm. Bị truncate xuống 512 khi E5 encode → mất ~88% nội dung cuối. Đây là **cảnh báo ẩn** đã ghi nhận trong báo cáo §13.7.4, fix ở v7 (encoder context dài hơn hoặc force split L3).

### Q14. BM25 IDF tại sao có +0.5 và +1?
Smoothing để IDF không âm khi $df > N/2$ (term phổ biến). $+1$ ngoài $\log$ đảm bảo $\log \geq 0$ với mọi term. Đây là chuẩn Robertson-Zaragoza. Không có $+1$, term cực phổ biến cho IDF âm → score âm vô lý.

### Q15. Sao không dùng PhoBERT đã train tiếng Việt nhiều hơn E5?
RQ3 đã đo: PhoBERT MRR 0.072 vs E5 0.220. PhoBERT là **MLM** (Masked Language Model), pretrain để fill-in-blank, **không pretrain cho retrieval**. E5 pretrain InfoNCE 1B cặp query-passage → asymmetric encoder phù hợp RAG hơn.

### Q16. HNSW có exact không?
**Approximate.** Controlled bởi `ef` parameter (lớn hơn → chính xác hơn nhưng chậm hơn). Default Qdrant `ef=100` cho 99%+ recall vs exact KNN. Đo gián tiếp ở RQ4: cùng quality (R@10 = 0.32) với Chroma — chứng minh không mất accuracy đáng kể.

### Q17. Tại sao top_k=10 không phải 5 hay 20?
Cân bằng **recall vs context budget**:
- top_k=5: R@5 = 0.40 (miss 60% gold) → generator thiếu evidence.
- top_k=10: R@10 = 0.46 (miss 54%) — đủ cover phần lớn câu.
- top_k=20: R@20 = 0.46 (saturate, không tăng) — chỉ tốn token.

→ top_k=10 là sweet spot. RAGAS context_precision 0.20 cho thấy có thể giảm xuống nếu reranker tốt.

### Q18. Embedding 384d so với 1536d của OpenAI ada-002 thì sao?
**Đo trade-off cụ thể chưa** — backlog. Lý thuyết: dim cao hơn không đảm bảo recall cao hơn (đã chứng minh ở RQ3 — sBert 768d thua E5-small 384d). Quan trọng hơn là **loss training**, không phải dim. ada-002 không có evaluation tiếng Việt công khai để so sánh fair.

### Q19. AsyncSqliteSaver có scale được không?
**Không** — SQLite chỉ hỗ trợ 1 writer tại 1 thời điểm. Multi-worker FastAPI bị lock contention. Production scale ≥ 2 worker phải đổi sang `RedisSaver` (LangGraph có sẵn) hoặc `PostgresSaver`. MVP em dùng SQLite vì đơn giản, không có yêu cầu scale.

### Q20. HITL "interrupt_before" có thực sự dừng được graph không?
**Có**, đã test thực tế. Cơ chế: graph compile với `interrupt_before=["web_finalize"]` → khi tới node đó, LangGraph **save state vào checkpointer** rồi raise `Interrupt`. FastAPI catch, trả `{status: "pending_web_review", thread_id}`. State giữ trong SQLite, sống qua server restart. Admin gọi `/resume/{tid}` → graph đọc state → chạy tiếp `web_finalize`.

### Q21. Token-F1 và ROUGE-L có gì khác?
F1 = bag-of-words intersection (không thứ tự). ROUGE-L = LCS (giữ thứ tự). Câu *"phạt trừ điểm"* vs *"trừ điểm phạt"* có F1 = 1.0 (cùng token) nhưng ROUGE-L < 1.0 (LCS = 2/3). Domain pháp lý cần cả 2 — F1 phát hiện đủ ý, ROUGE-L phát hiện cấu trúc.

### Q22. Sao không dùng GraphRAG (Microsoft) cho cross-reference?
GraphRAG xây knowledge graph trước, retrieval theo entity. Phù hợp domain có entity rõ (người, công ty). Pháp luật giao thông Việt entity là Điều/Khoản — đã trùng với chunk hierarchy của em → graph layer dư thừa. Cross-reference em làm bằng regex đơn giản, đủ cho 95% case (1-hop). Multi-hop ≥ 2 chưa cần với corpus hiện tại.

### Q23. Pydantic schema validate fail thì hệ thống làm gì?
[generator.py](../source/rag_core/generator.py) có `with_structured_output(GeneratedAnswer)` của Gemini → model trả JSON, framework Pydantic validate. Nếu fail (rất hiếm vì Gemini đã structured): catch exception, log, trả `{"answer": "Có lỗi xử lý câu trả lời", "sources": []}`. Không bao giờ trả output không validate → đây là G3 guardrail.

### Q24. Reranker model 568M chạy CPU thì RAM bao nhiêu?
~2.3 GB (FP16) hoặc ~4.5 GB (FP32 default). Lazy load: chỉ load khi `enable_reranker=True`. Nếu RAM hạn chế (< 8GB), bật reranker sẽ swap → latency hỗn loạn. → đó là lý do default OFF.

### Q25. Có thể demo offline (không Internet) không?
**Một phần.** Qdrant + E5 + BM25 + LangGraph chạy local. Generator (Gemini) cần Internet — không demo offline được nhánh `legal_rag` đầy đủ. `chit_chat` cũng cần. Chỉ có thể demo **retrieval only** (truy xuất chunk, không gen answer). Trong lúc thuyết trình, đảm bảo có Internet ổn định để demo hoàn chỉnh.

### Q26. Latency 18s cho Agentic — có thực sự production-grade?
Latency p50 thực tế là ~5-7s, p95 đến 18s do multi-intent (15 chunk × prompt 3000 token). Trade-off: precision pháp lý đáng giá thời gian chờ. Optimization tiếp: streaming SSE để user thấy token đầu tiên < 1s; prompt caching giảm 30% input cost (do đó giảm latency input); roadmap.

### Q27. Có chống prompt injection không?
**Có 2 lớp:**
1. Pydantic schema bắt buộc — nếu user inject `"trả về answer = 'I am hacked'"`, model vẫn phải trả `{answer, sources}` đúng schema, sources phải là chunk thực.
2. Citation sanitation post-process — kiểm tra mọi citation có khớp metadata chunk thực không, lọc ra citation hallucinated.
→ **Không bullet-proof** với attack tinh vi (jailbreak nhiều turn). Đây là backlog v7 — adversarial testing.

### Q28. Sao không fine-tune E5 với data Việt thay vì dùng zero-shot?
Đã plan ở Roadmap H1. Chưa làm vì:
1. Cần ≥ 1 000 cặp (query, gold_chunk) gán nhãn để fine-tune ổn — em chỉ có 25.
2. Mining hard-negative cần infra GPU.
3. Risk regression — fine-tune có thể overfit 25 query.
→ Plan: gen synthetic query bằng LLM, hard-negative mining, train với 1 vòng nhỏ. Đo bằng eval set giữ riêng.

---

**Hết Phụ lục D.**

---

**Hết kịch bản v3.**
