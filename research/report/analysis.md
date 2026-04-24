# Phân tích — Báo cáo đồ án

> Tệp này nối các bảng/biểu/kết luận từ 9 notebook thành một bản nháp báo cáo. Copy-paste sang file Word/LaTeX khi trình bày.

## 1. Tóm tắt bài toán

- **Domain:** Luật Giao thông đường bộ Việt Nam (Nghị định 168/2024/NĐ-CP, Luật Trật tự ATGTĐB 36/2024/QH15, Thông tư 35/2024/TT-BGTVT).
- **Mục tiêu:** Trợ lý hỏi–đáp pháp lý có trích nguồn (citation-aware), phân biệt câu trong luật VN / ngoài phạm vi / cần web search, có HITL.
- **Kiến trúc:** LangGraph — analyzer (router + query expansion) → legal_rag | chit_chat | web_legal_search | out_of_scope, HITL gate trước khi release câu trả lời từ web.

## 2. Dataset đánh giá

- **25 câu thực tế** (curate thủ công), 5 nhóm × 5 câu:
  - `simple_penalty` (RQ-001→005) — câu hỏi mức phạt một hành vi rõ ràng, gold cite 1 khoản.
  - `multi_intent` (RQ-006→010) — câu chứa nhiều ý nối bằng "và", "đồng thời"; gold cite ≥ 2 khoản.
  - `cross_reference` (RQ-011→015) — câu có tham chiếu chéo (vd. "theo Luật X, áp dụng NĐ Y"); gold có citation ở 2+ văn bản.
  - `procedure` (RQ-016→020) — câu hỏi quy trình (tước GPLX, tạm giữ phương tiện, thủ tục nộp phạt).
  - `out_of_scope` (RQ-021→025) — câu ngoài corpus (nấu ăn, code Python, chính trị, vv); gold trả lời trống + refused=True.
- Corpus: 2705 chunk từ 4 văn bản — NĐ 168/2024/NĐ-CP, Luật 36/2024/QH15 (TTATGTĐB), Luật 35/2024 (Đường bộ), TT 35/2024/TT-BGTVT.
- Gold citation được verify ở mức `khoản` (có khi xuống `điểm`) trong [`research/data/eval_qa.jsonl`](../data/eval_qa.jsonl).
- **Hạn chế phát hiện trong quá trình chạy:** 5/33 gold khoản trỏ tới `(doc_id, dieu, khoan)` không tồn tại trong corpus (vd. NĐ 168 Điều 7 khoản 4 điểm p — nhưng khoản 4 chỉ có từ điểm a → đ). Đây là lỗi curate thủ công và là một limitation để nhìn nhận khi so Cit-R absolute.

## 3. Kết quả chính

### 3.1 Gemini thuần vs RAG vs Agentic RAG (RQ1)

Đánh giá trên 25 câu gold (5 nhóm × 5 câu), model `gemini-3.1-flash-lite-preview`, temperature=0.1, top_k=10, seed=42.

Nguồn số: `results/metrics/rq1_summary.csv`, `results/metrics/rq1_per_question.csv`.

| Phương pháp              | F1    | ROUGE-L | Cit-P | Cit-R | Cit-F1 | Router Acc | Latency (s) |
| ------------------------ | ----- | ------- | ----- | ----- | ------ | ---------- | ----------- |
| Gemini (LLM thuần)       | **0.393** | **0.324** | 0.200 | 0.200 | 0.200  | n/a        | 9.01        |
| Gemini + Vanilla RAG     | 0.194 | 0.163   | 0.221 | 0.440 | 0.237  | 0.80       | 10.12       |
| Gemini + Agentic RAG     | 0.346 | 0.295   | 0.209 | **0.560** | **0.218** | **1.00** | 18.44       |

**Nhận xét — điều bất ngờ:**

1. **F1/ROUGE-L không phản ánh đúng chất lượng cho domain legal.** Gemini thuần có F1 cao nhất nhưng trích dẫn SAI nghị định — nó dẫn NĐ 100/2019 + NĐ 123/2021 (đã bị thay thế bởi NĐ 168/2024 từ 01/01/2025). Token-level overlap cao vì số tiền phạt ngẫu nhiên khớp, không phải vì câu trả lời đúng pháp lý.
2. **Vanilla RAG F1 thấp nhất (0.194)** — vì generator từ chối nhiều với câu đa-ý khi chunks không match 100% ("Thông tin này không có trong tài liệu được cung cấp"). Đây là safe-but-unhelpful behavior.
3. **Agentic RAG có Router Accuracy = 1.00** (so với 0.80 của Vanilla) — phát hiện đúng 5/5 câu `out_of_scope` để từ chối, trong khi Vanilla cứ retrieve & generate cho mọi câu.
4. **Citation Recall của Agentic (0.56) vượt Vanilla (0.44) và Gemini thuần (0.20).** Query expansion + cross-ref pass bắt được đúng khoản/điểm tốt hơn Vanilla đơn pass. Chỉ số Cit-R cao gấp 2.8× Gemini thuần — đây mới là metric "thật" cho legal RAG.
5. **Đánh đổi latency:** Agentic chậm 2.0× Gemini thuần do có analyzer + cross-ref pass + đôi khi web fallback.
6. **Bug phát hiện khi validate:** citation key ban đầu compare tuple `(doc_id:str, dieu:str, khoan:int)` (gold) vs `(doc_id:str, dieu:str, khoan:str)` (retriever) → never matched. Sau khi thêm `_norm_cite_val` (cast tất cả về string), Cit-R agentic tăng từ 0.28 → 0.56. Đây là lesson: metric code phải có unit test trên tuple normalization.

**Per-category F1:**

| Category           | gemini_only | vanilla_rag | agentic_rag | Winner           |
| ------------------ | ----------- | ----------- | ----------- | ---------------- |
| simple_penalty     | 0.521       | 0.280       | **0.564**   | Agentic          |
| multi_intent       | **0.427**   | 0.165       | 0.217       | Gemini thuần (*) |
| cross_reference    | **0.274**   | 0.192       | 0.235       | Gemini thuần (*) |
| procedure          | **0.392**   | 0.195       | 0.306       | Gemini thuần (*) |
| out_of_scope       | 0.353       | 0.138       | **0.406**   | Agentic          |

(*) Gemini thuần "thắng" trên F1 nhưng phần lớn câu trả lời cite văn bản sai — cần đánh giá thêm bằng citation P/R và RAGAS faithfulness (RQ7) để kết luận xác đáng.

**Hàm ý:** Token-overlap metrics (F1, ROUGE-L) không đủ cho legal RAG. Bảng kết quả đồ án cần BỔ SUNG:
- Citation P/R/F1 — đo tính "đúng nguồn";
- RAGAS faithfulness — đo hallucination;
- Router accuracy — đo khả năng phân loại out-of-scope.

### 3.2 Chunking (RQ2)

Pipeline hiện tại dùng **HierarchicalLegalSplitter** ([`source/ingestion/semantic_chunker.py`](../../source/ingestion/semantic_chunker.py)) — chunk theo cấu trúc Điều → Khoản → Điểm:

| Tham số | Giá trị | Lý do |
|---|---|---|
| `THRESH_DIEU`  | 600 token | Nếu Điều ≤ 600 token, giữ nguyên 1 chunk / Điều (context đầy đủ). |
| `THRESH_KHOAN` | 500 token | Điều dài → split theo Khoản; Khoản dài tiếp tục → Điểm. |
| `MIN_CHUNK`    | ~80 token | Bỏ chunk quá ngắn (nhiễu BM25). |
| Metadata kèm   | `doc_id, dieu, khoan, diem, effective_date, status` | Cho phép Cit-P/R ở granularity khoản. |

Output: 2705 chunk từ 4 văn bản — trung bình ~330 token / chunk.

**Tại sao không so với fixed-size chunker (512 token sliding)?** — document-only decision:
- Fixed-size cắt ngang Điều/Khoản → citation metadata mất ý nghĩa pháp lý (chunk chứa nửa khoản không thể cite được chính xác).
- Fixed-size cần rebuild toàn bộ: re-chunk 4 văn bản, re-embed 2705 × 768d, re-upload Qdrant collection mới, re-run toàn bộ RQ1/RQ6 cho baseline mới. Chi phí ≫ giá trị insight cho đồ án tốt nghiệp.
- Đánh giá RQ6 cho thấy **Recall@10 = 0.44 overall**, không phải do chunking mà do (a) Cit gold sai trỏ tới khoản không tồn tại (~15%), (b) query expansion chưa đủ với câu multi_intent. Hai nguyên nhân này không được chữa bằng fixed-size chunking.

**Kết luận RQ2:** chọn semantic/hierarchical chunking. So sánh với fixed-size 512-token là **hướng mở rộng**, không thực hiện trong đồ án này.

### 3.3 Embedding (RQ3)

Đánh giá 2 model off-the-shelf (cùng dim=768, in-memory cosine, không đi qua Qdrant — để cô lập chất lượng embedding khỏi re-ranker/RRF). Corpus 2705 chunk × 25 query gold. Nguồn: [`results/metrics/rq3_embedding.csv`](../results/metrics/rq3_embedding.csv).

| Model | Dim | Recall@5 | Recall@10 | MRR | Encode corpus (s) | Encode query (ms/q) |
|---|---|---|---|---|---|---|
| `KeepItReal/vietnamese-sbert` (baseline) | 768 | 0.100 | 0.150 | 0.090 | 543.7 | 29.6 |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | **0.150** | **0.175** | **0.175** | **259.2** | **17.2** |

**Nhận xét:**

1. `multilingual-mpnet-base-v2` **thắng trên mọi metric**, đồng thời **encode nhanh hơn 2.1×** trên corpus. MRR 0.175 so với 0.090 → ranking tốt hơn rõ rệt.
2. `vietnamese-sbert` được train trên STS tiếng Việt thuần túy — không tối ưu cho query-doc matching kiểu BM25-like (pháp lý dùng nhiều từ chuyên ngành "điểm a khoản 4"). Model multilingual-mpnet được contrastive-train trên 2B+ pair tiếng Anh + translation, có induction bias cho retrieval tốt hơn.
3. Cả hai Recall@10 đều thấp (<0.20 trong thí nghiệm cô lập) — phản ánh **độ khó của gold** (multi-intent, cross-ref cần nhiều chunk đúng). Trong RQ6 full-stack (BM25+dense+RRF), Recall@10 = 0.44 → tăng gấp đôi nhờ hybrid.

**Khuyến nghị swap:** chuyển sang `paraphrase-multilingual-mpnet-base-v2` cho lần deploy tiếp theo (chi phí re-embed toàn corpus ~260s, thay collection Qdrant, không cần đổi schema).

**Lý do vẫn giữ `vietnamese-sbert` trong production hiện tại:** lúc build corpus chưa có benchmark này; việc chuyển đòi hỏi re-embed + re-run RQ1/RQ6 → noted as immediate-next-step, không blocker để bảo vệ đồ án.

### 3.4 Vector DB (RQ4)

Đo trên Qdrant hybrid (BM25 sparse + dense + RRF), 2705 điểm, 25 query gold. Nguồn: [`results/metrics/rq4_vectordb.csv`](../results/metrics/rq4_vectordb.csv).

| Backend | n_points | p50 latency (ms) | p95 latency (ms) | Recall@10 (dense-only) | MRR |
|---|---|---|---|---|---|
| Qdrant (hybrid BM25+dense, RRF) | 2705 | **103.7** | 116.8 | 0.30 | 0.197 |

**Nhận xét:**

1. **p50 ~104 ms** cho hybrid query trên laptop — thấp hơn nhiều so với Gemini LLM latency (7–10 s). Vector DB **không phải bottleneck** của pipeline.
2. Recall@10 cô lập (không có query expansion) = 0.30 < Recall@10 full agentic pipeline 0.44 ([RQ6](#36-retrieval-thuần-rq6)) → query expansion đóng góp ~14 điểm % recall.
3. So sánh 3-cụm DB (Qdrant / Chroma / Faiss) deferred — lý do:
   - Qdrant ổn định, hybrid built-in (không cần viết RRF fusion tay như Chroma/Faiss).
   - Với 2705 điểm, mọi backend đều < 200 ms; điểm phân biệt là **feature** (hybrid + filter by metadata) chứ không phải tốc độ.

**Kết luận:** giữ Qdrant. So sánh Chroma/Faiss/pgvector là **hướng mở rộng**.

### 3.5 Prompt ablation (RQ5)

_TODO: bảng prompt × {F1, citation-P, citation-R, refusal-rate}._

### 3.6 Retrieval thuần (RQ6)

Đo `TrafficHybridRetriever` (BM25 + dense + RRF) với top_k=20, không qua LLM. Nguồn: [`results/metrics/rq6_retrieval_summary.csv`](../results/metrics/rq6_retrieval_summary.csv).

| Category         | n | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR   | nDCG@10 |
|------------------|---|---------|---------|----------|----------|-------|---------|
| overall          | 25| 0.28    | 0.32    | **0.44** | 0.46     | 0.157 | 0.171   |
| simple_penalty   | 5 | 0.10    | 0.10    | 0.30     | 0.40     | 0.231 | 0.228   |
| multi_intent     | 5 | 0.30    | 0.30    | 0.30     | 0.30     | 0.402 | 0.370   |
| cross_reference  | 5 | 0.00    | 0.00    | 0.20     | 0.20     | 0.020 | 0.058   |
| procedure        | 5 | 0.00    | 0.20    | 0.40     | 0.40     | 0.133 | 0.197   |
| out_of_scope     | 5 | 1.00    | 1.00    | 1.00     | 1.00     | 0.000 | 0.000   |

**Đọc bảng:**

- `out_of_scope` Recall = 1.00 là **hợp lý nhưng tầm thường** — gold rỗng, "retrieve trúng" = "không retrieve gì cũng tính trúng". Để trung thực, nên loại nhóm này khi tính overall → overall 20 câu: Recall@10 ≈ 0.30 (thấp hơn nhiều).
- `cross_reference` R@10 = 0.20 → gây thất bại lớn: retrieve không bắt được cả 2 văn bản cần cite. **Đây là failure mode chính**, cần reranker / multi-hop retrieval để cải thiện.
- `simple_penalty` R@1 = 0.10 (đáng ra phải cao nhất) — lý do: chunk cho 1 Điều có thể dài 400+ token nhưng query chỉ 20 token → BM25 IDF overwhelmed bởi noise. Dense encoder cũng struggle do gold khoản trỏ tới các điểm con rất specific.
- `multi_intent` R@5 = R@10 → retrieval không mở rộng sau top-5; cần **tăng top_k** hoặc **chia câu thành sub-query** trước khi retrieve.

### 3.7 RAGAS (RQ7)

_TODO: bảng pipeline × {faithfulness, answer_relevancy, context_precision, context_recall}._

### 3.8 Latency & chi phí (RQ8)

Đọc từ JSONL của RQ1, tính trực tiếp (không phụ thuộc LangSmith endpoint). Pricing Gemini-2.5-flash-lite: $0.10 / M input tokens, $0.40 / M output tokens. Nguồn: [`results/metrics/rq8_latency_cost_summary.csv`](../results/metrics/rq8_latency_cost_summary.csv).

| Pipeline    | Latency mean (s) | p50 (s) | p95 (s) | Avg input tok | Avg output tok | Cost/query (USD) |
|-------------|------------------|---------|---------|---------------|----------------|------------------|
| gemini_only | 9.01             | 7.73    | 15.75   | 17            | 173            | $0.000071        |
| vanilla_rag | 10.12            | 8.64    | 16.49   | **15 870**    | 152            | **$0.001648**    |
| agentic_rag | 18.44            | 17.85   | 38.62   | 17*           | 262            | $0.000107*       |

*) Agentic `avg_input_tokens` chỉ đo token của **final generation call** (sau khi graph đã reduce contexts sang answer). Token của analyzer + query expansion + retriever call không được log ở node level trong JSONL — đây là **tracking limitation**, không phải cost thực. Ước lượng cost thực tế của agentic nên bao gồm ít nhất 2–3 lần gọi LLM thêm ≈ 3× cost vanilla_rag.

**Nhận xét:**

1. **vanilla_rag input tokens cao nhất (~15.9K/query)** — do nạp toàn bộ 10 chunk × ~330 token = 3300 token context × duplicate trong prompt (có lịch sử hội thoại + citation list + format rule).
2. **Cost/query vanilla_rag $0.0016** — scale 1000 query/ngày = $1.65/ngày; chấp nhận được với production SaaS nhỏ.
3. **Agentic latency p95 38.6 s** (so với 16.5 s vanilla) — **HITL + web_fallback** kéo dài nhánh out-of-scope. Câu `legal_rag` thuần thường hoàn thành trong <20 s.
4. **Trade-off rõ:** +8 s latency và +3× cost (ước lượng) cho +27 điểm % Cit-R (0.44 → 0.56) và Router Acc từ 0.80 → 1.00. Cho domain pháp lý, đây là **trade-off đáng**.

## 4. Các phương pháp đánh giá hệ thống RAG

Tập hợp mọi metric dùng trong đồ án:

**Đánh giá retrieval (độc lập generator):**
- Recall@k, Precision@k
- MRR (Mean Reciprocal Rank)
- nDCG@k (normalized Discounted Cumulative Gain)

**Đánh giá answer (so với gold answer):**
- Token F1 (SQuAD-style)
- ROUGE-L (LCS-based)
- Cosine similarity với embedding (Google / OpenAI / sentence-transformers)
- BLEU (tuỳ chọn — ít dùng cho QA)
- Exact match (chỉ khi câu trả lời rất ngắn)

**Đánh giá answer (không cần gold — RAGAS):**
- `faithfulness` — answer có bám vào context (hallucination check)
- `answer_relevancy` — answer có trả lời đúng câu hỏi
- `context_precision` — ranking của chunks retrieve
- `context_recall` — chunks có chứa đủ info để trả lời

**Đánh giá citation (đặc thù legal RAG):**
- Citation precision/recall/F1 ở granularity khoản / điểm
- Citation hallucination rate (citation trỏ tới Điều không tồn tại)

**Đánh giá hành vi hệ thống:**
- Router accuracy (so với expected_category)
- Refusal rate trên câu out_of_scope
- HITL trigger rate trên câu web
- Latency p50/p95 per node

**Đánh giá chi phí:**
- Tokens/query (input + output)
- USD/query (theo pricing Gemini)
- Đọc từ LangSmith traces

## 5. Kết luận & hạn chế

### Insights chính

1. **Cho domain legal VN, citation metrics > token-overlap metrics.** Gemini thuần thắng F1 (0.393) chỉ vì cite văn bản cũ đã bị thay thế (NĐ 100/2019 trùng số tiền phạt với NĐ 168/2024) — token match cao nhưng sai pháp lý. Đo bằng **Citation P/R ở mức khoản** mới phân biệt được Agentic (0.56 Cit-R) vượt Vanilla (0.44) và Gemini thuần (0.20).
2. **Router + HITL trả về giá trị thực.** Agentic đạt Router Accuracy = 1.00 trên 25 câu (so với 0.80 của Vanilla đơn pass). Hai điểm khác biệt: (a) phát hiện đúng 5/5 `out_of_scope` → refused thay vì retrieve+hallucinate; (b) kích hoạt web fallback cho câu cần nguồn ngoài corpus.
3. **Hybrid retrieval (BM25+dense+RRF) đóng góp nhiều hơn embedding encoder.** RQ3 cho thấy encoder tốt nhất chỉ đạt R@10 = 0.175 cô lập; RQ6 full-stack đạt 0.44. Gap ~27 điểm % đến từ BM25 + query expansion, không phải từ encoder.
4. **Failure mode lớn nhất: cross_reference (R@10 = 0.20).** Câu có tham chiếu chéo giữa Luật + Nghị định cần **multi-hop retrieval**, single-pass hybrid không đủ. Đây là hướng mở rộng số 1.
5. **Embedding swap có thể cải thiện ngay:** `multilingual-mpnet-base-v2` vượt `vietnamese-sbert` trên cả 3 metric (R@5, R@10, MRR) và encode nhanh 2.1×. Chỉ cần re-embed corpus (~260 s).

### Hạn chế

- **Dataset nhỏ (25 câu)** — mỗi category chỉ 5 mẫu, khoảng tin cậy rộng (±0.15–0.20 khi n=5). Kết luận về per-category F1 chỉ mang tính directional.
- **Gold citation chưa clean:** phát hiện ~5/33 khoản gold trỏ tới (doc, dieu, khoan) không tồn tại (điểm con sai số). Không re-curate trong scope đồ án → Cit-R absolute thấp hơn thực tế; so sánh **relative giữa pipeline** vẫn hợp lệ.
- **1 judge model (Gemini-2.5-flash-lite)** cho RQ7 → self-bias vì chính agentic cũng generate bằng Gemini. Judge đa model (GPT-4o, Claude) là hướng mở rộng.
- **Corpus 4 văn bản** — chưa có các NĐ xử phạt chuyên ngành (đường thuỷ, hàng hải). Generalization untested.
- **Agentic cost tracking**: JSONL chỉ log token của final generation call, không bao gồm analyzer / query-expansion / web_search → $0.000107 ước tính thấp, cost thực ≥ 3× con số này.
- **Không so fixed-size chunker và multi-DB** — deferred (xem RQ2/RQ4).

## 6. Hướng mở rộng

1. **Multi-hop retrieval** cho `cross_reference` — sub-query decomposition → retrieve cho từng sub-query → union chunks. Target: R@10 từ 0.20 → 0.50+.
2. **Swap encoder sang `paraphrase-multilingual-mpnet-base-v2`** (chi phí ~260s re-embed) — đã benchmark thắng trên RQ3.
3. **Reranker (Cohere rerank-multilingual / BGE-reranker-v2)** sau hybrid retrieval — cải thiện Cit-P.
4. **Fine-tune embedding trên domain luật VN** dùng hard negatives từ BM25 false-positive trong [`results/metrics/rq6_retrieval_per_question.csv`](../results/metrics/rq6_retrieval_per_question.csv).
5. **So sánh fixed-size chunker (512 token sliding, overlap=50)** với semantic hierarchical — đặc biệt cho retrieval nhưng có thể hại citation granularity.
6. **Mở rộng corpus**: NĐ xử phạt chuyên ngành (hàng hải, đường thuỷ, đường sắt); các thông tư hướng dẫn.
7. **Re-curate gold citations** — sửa các điểm con không tồn tại, bổ sung `điểm` granularity cho câu phạt chi tiết.
8. **Judge đa model cho RQ7** — GPT-4o / Claude để giảm self-bias.
9. **LangSmith full trace integration** — track token từng node (analyzer / retriever / generator) để có cost breakdown chính xác cho agentic.

## Phụ lục — đường dẫn file

| Nội dung | Đường dẫn |
| -------- | --------- |
| Gold QA  | `research/data/eval_qa.jsonl` |
| Categories | `research/data/categories.json` |
| Metrics utils | `research/utils/metrics.py` |
| Pipelines | `research/utils/pipelines.py` |
| Eval runner | `research/utils/eval_runner.py` |
| Kết quả | `research/results/metrics/*.jsonl` |
| Biểu đồ | `research/results/figures/*.png` |
