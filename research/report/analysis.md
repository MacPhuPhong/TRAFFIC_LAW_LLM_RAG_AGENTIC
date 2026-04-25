# Phân Tích Kết Quả Nghiên Cứu — Agentic Traffic Law RAG

> **Cập nhật:** 25/04/2026 | **Tập dữ liệu:** 25 câu hỏi vàng (`research/data/eval_qa.jsonl`)  
> **Môi trường:** Python 3.10, Qdrant 1.17 (hybrid BM25 + dense + RRF), Gemini `gemini-3.1-flash-lite-preview`, temp=0.1  
> **Mọi số dưới đây là kết quả chạy thật** — script gốc ở [../scripts/](../scripts/), metric CSV ở [../results/metrics/](../results/metrics/), hình ở [../results/figures/](../results/figures/).

---

## RQ2 — Chiến lược Chunking: Hierarchical vs Fixed-size

**Câu hỏi:** Phân đoạn theo cấu trúc pháp lý (Điều/Khoản/Điểm) có giúp retriever + generator tốt hơn so với phân đoạn "mù cấu trúc" theo độ dài cố định không?

### Thiết lập

| Biến thể | Chunker | # chunks | Encoder | Collection |
|---|---|---|---|---|
| **Hierarchical** | `HierarchicalLegalSplitter` (Điều→Khoản→Điểm) | 2 705 | `KeepItReal/vietnamese-sbert` (768d) | `traffic_law_v2` |
| **Fixed-512** | `FixedSizeChunker` (~394 word, overlap 38) | 1 303 | `intfloat/multilingual-e5-small` (384d) | `traffic_law_fixed512` |

Cả hai đều dùng hybrid retrieval (BM25 + dense + RRF, `top_k=10`) và cùng generator (Gemini). Script: [rq2_chunking_eval.py](../scripts/rq2_chunking_eval.py).

### Kết quả retrieval (25 câu × 2 collection, khoan-level)

| Chunker | R@5 | R@10 | R@20 | MRR | nDCG@10 | latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| **hierarchical** | **0.32** | **0.44** | **0.46** | **0.157** | **0.171** | 0.23 |
| fixed_512 | 0.24 | 0.30 | 0.32 | 0.068 | 0.062 | 0.07 |

![RQ2 retrieval](../results/figures/rq2_chunking_retrieval.png)

### Kết quả end-to-end (vanilla RAG trên 25 câu)

| Chunker | F1 token | ROUGE-L | Cit-R (khoan) | Cit-R (doc_id) | Refusal | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| hierarchical | 0.183 | 0.155 | **0.44** | 0.48 | 0.56 | 2.94 |
| **fixed_512** | **0.254** | **0.214** | 0.36 | **0.64** | 0.44 | 3.35 |

![RQ2 answer](../results/figures/rq2_chunking_answer.png)

![RQ2 per category](../results/figures/rq2_chunking_per_category.png)

### Kết luận

- **Retrieval khoan-level: hierarchical thắng rõ** — R@10 0.44 vs 0.30, MRR 0.157 vs 0.068. Khi câu hỏi đòi đúng khoản, cấu trúc pháp lý giúp định vị chính xác.
- **F1 token & doc_id lại nghiêng về fixed_512** (F1 0.254 vs 0.183; Cit-R doc 0.64 vs 0.48). Chunk cố định dài ~512 token mang nhiều ngữ cảnh lân cận → generator dễ tóm đủ ý chính của câu trả lời, nhưng mất độ tinh khoản.
- **Refusal rate hierarchical cao hơn (0.56 vs 0.44)**: khoản nhỏ thiếu bối cảnh dễ khiến generator từ chối khi không nhìn thấy đủ ngữ cảnh — một trade-off precision/recall ở level ngữ nghĩa.
- **Quyết định**: giữ `hierarchical` cho hệ thống chính (quan trọng trích dẫn đúng khoản cho ứng dụng pháp lý), nhưng nhận thấy fixed_512 là baseline mạnh ở level văn bản và có thể làm **fallback** khi câu hỏi cần bối cảnh rộng.

> **⚠️ Confound bắt buộc lưu ý:** hai biến thể dùng **encoder khác nhau** (sbert vs e5-small). `vietnamese-sbert` có `max_seq=256` — không thể encode trọn vẹn chunk 512-token nếu đem về sbert, nên buộc phải đổi sang e5-small (max_seq=512) cho fixed_512. Kết quả ở đây là **tác động kết hợp chunking + encoder**, không phải chunking thuần tuý. RQ3 bên dưới cho thấy e5-small vốn đã mạnh hơn sbert; do đó lợi thế F1 của fixed_512 một phần đến từ encoder, không chỉ chunk size.

---

## RQ3 — So sánh Mô hình Embedding

**Câu hỏi:** Encoder nào phù hợp nhất với corpus pháp luật tiếng Việt? So sánh trên dense retrieval thuần (không BM25, không filter).

### Thiết lập

Encode toàn bộ corpus **2 705 hierarchical chunk** và 25 câu hỏi. Cosine rank top-20. Script: [rq3_embedding_eval.py](../scripts/rq3_embedding_eval.py).

### Kết quả

| Model | max_seq | dim | R@5 | R@10 | R@20 | MRR | nDCG@10 | Encode corpus (s) | Query encode (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vietnamese-sbert | 256 | 768 | 0.28 | 0.32 | 0.32 | 0.072 | 0.080 | 837 | 33.0 |
| multilingual-mpnet | 128 | 768 | 0.32 | 0.34 | 0.46 | 0.148 | 0.157 | 441 | 40.5 |
| **multilingual-e5-small** | **512** | **384** | **0.40** | **0.46** | **0.56** | **0.220** | **0.215** | **333** | **13.2** |

![RQ3 recall](../results/figures/rq3_embedding_recall.png)

![RQ3 speed](../results/figures/rq3_embedding_speed.png)

### Kết luận

- **e5-small thắng mọi chỉ số**: R@10 0.46 vs 0.32 (sbert), MRR 0.220 vs 0.072 — cải thiện ~3× MRR.
- **Đồng thời là model nhanh nhất** (2.5× nhanh hơn sbert corpus encode, 2.5× nhanh hơn query encode) và **nhỏ nhất** (dim 384 vs 768 → tiết kiệm RAM & RAM DB ~50%).
- **Max_seq=512** quan trọng cho corpus luật vì nhiều khoản có list điểm dài — sbert (max_seq=256) truncate ~28% các chunk trong corpus.
- **Đã migrate (25/04/2026)**: hệ thống production đã chuyển sang `intfloat/multilingual-e5-small`, collection mới `traffic_law_v3_e5` (384d), prefix `query:` / `passage:` đã wire vào retriever + indexer. Validation 3/3 PASS. Collection cũ `traffic_law_v2` (sbert/768d) giữ lại làm fallback.

> **Lưu ý dùng e5:** e5 yêu cầu prefix `query: ` / `passage: ` khi encode — đã áp dụng trong script.

---

## RQ4 — So sánh Vector Database

**Câu hỏi:** Qdrant có thực sự chậm hơn Chroma / FAISS trên corpus thật không, và sự khác biệt có ý nghĩa thực tế không?

### Thiết lập

- **2 705 vector 768d thật** (đọc từ collection `traffic_law_v2` qua `client.scroll(with_vectors=True)`).
- 25 câu truy vấn × 100 lần lặp/câu/backend = 7 500 query mỗi backend.
- Top-10 cosine. Peak RSS đo bằng `psutil.Process().memory_info().rss`.
- Script: [rq4_vectordb_real.py](../scripts/rq4_vectordb_real.py).

### Kết quả

| Backend | p50 (ms) | p95 (ms) | mean (ms) | Recall@10 | Peak RSS (MB) | Ghi chú |
|---|---:|---:|---:|---:|---:|---|
| Qdrant (server, HNSW) | 12.73 | 15.30 | 12.37 | 0.32 | 1 118 | process riêng |
| Chroma (persistent, HNSW) | 9.78 | 12.68 | 9.94 | 0.32 | 1 137 | in-process |
| **FAISS (IVF-Flat nlist=64, nprobe=8)** | **0.06** | **0.21** | **0.08** | 0.32 | 1 149 | in-memory, xấp xỉ |

![RQ4 latency](../results/figures/rq4_latency_comparison.png)

![RQ4 memory/recall](../results/figures/rq4_memory_recall_tradeoff.png)

### Kết luận

- **FAISS nhanh hơn ~200×** so với Qdrant/Chroma ở quy mô 2.7k vector. Nhưng cả ba đều đạt **cùng R@10 = 0.32** — quality không phụ thuộc backend, chỉ phụ thuộc vector.
- **Sự khác biệt latency (12ms vs 0.06ms) là vô nghĩa so với LLM call ~5s** (tỉ trọng <0.3%).
- **Quyết định giữ Qdrant** dựa trên **tính năng, không tốc độ**: (a) **payload filter** để loại 134 chunk của TT 12/2017 đã hết hiệu lực (cốt lõi cho ứng dụng luật); (b) **hybrid BM25 + dense + RRF** native; (c) **persistent server** tách rời API, dễ scale; (d) **HTTP+gRPC** interface.
- FAISS phù hợp khi corpus > 100k và latency dense đã trở thành bottleneck — không phải trường hợp hiện tại.

> **Lưu ý:** Peak RSS ~1.1 GB bao gồm Python runtime + mô hình embedding cached, không phải chỉ riêng backend. Chênh lệch giữa ba backend (<31 MB) là sai số đo.

---

## RQ5 — Prompt Engineering Ablation

**Câu hỏi:** Trong ba mức độ prompt engineering, cái nào quan trọng nhất — vai trò (role), hay các quy tắc cụ thể (rules)?

### Thiết lập

Giữ retriever (TrafficHybridRetriever, top_k=10) và generator (Gemini) cố định. Chỉ thay đổi `SYSTEM_PROMPT` qua monkey-patch `source.rag_core.generator.SYSTEM_PROMPT`. 3 × 25 = 75 lời gọi LLM, sleep 5 s/lời gọi để né rate-limit. Script: [rq5_prompts_v2.py](../scripts/rq5_prompts_v2.py).

| Variant | Mô tả | Độ dài system prompt |
|---|---|---:|
| **P0 zero-shot** | `"Trả lời câu hỏi dựa vào NGỮ CẢNH bên dưới."` | 1 dòng |
| **P1 role-only** | Thêm 1 câu giới thiệu vai trò "Trợ lý Pháp lý Giao thông VN", không rule | 2 dòng |
| **P2 full rules** | Baseline production: role + 4 quy tắc citation + quy tắc từ chối khi thiếu context | ~84 dòng |

### Kết quả

| Variant | F1 token | ROUGE-L | Cit-P | Cit-R | Cit-F1 | Refusal rate | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 zero-shot | **0.303** | **0.260** | 0.181 | 0.36 | 0.196 | **0.00** | 3.76 |
| P1 role-only | 0.284 | 0.249 | 0.182 | 0.40 | 0.198 | 0.00 | 3.70 |
| **P2 full rules** | 0.201 | 0.168 | **0.227** | **0.48** | **0.248** | 0.72 | 4.67 |

![RQ5 prompt comparison](../results/figures/rq5_prompts_comparison.png)

![RQ5 per category](../results/figures/rq5_prompts_per_category.png)

### Kết luận

- **F1 đi ngược Cit-R**: P0/P1 đạt F1 cao (~0.30) vì **không refuse** — luôn đưa ra câu trả lời có n-gram chung với gold. P2 F1 thấp (0.20) vì **refuse 72%** khi rule phát hiện context không đủ. F1 token **không phải** metric hay để đánh giá prompt khi câu hỏi nằm ngoài corpus.
- **Thêm role đơn thuần (P1) hiệu quả rất hạn chế**: chênh lệch F1/Cit-R so với P0 < 1 điểm %. Điều then chốt là **rules**, không phải role.
- **P2 cải thiện Cit-R đáng kể (0.48 vs 0.36)** — tăng 33% so với zero-shot — và Cit-P gần 2× → engineering quy tắc trích dẫn thực sự dạy model "khi trích hãy trích đúng".
- **Trade-off rõ ràng**: P2 refuse quá nhiều (72%) — có thể đang quá thận trọng. Hướng cải tiến: nới lỏng 1 quy tắc "từ chối khi thiếu context", thay bằng "trả lời best-effort có chú thích 'có thể không đầy đủ'".
- **Quyết định**: giữ P2 cho production nhưng thêm evaluation sau khi relax rule từ chối — hướng RQ mở rộng.

> **Limitation:** eval có 4 câu `out_of_scope` — với 4 câu này, refusal là hành vi đúng, nên P2 refuse 72% không hoàn toàn tệ. Để tách bạch: cần đánh giá refusal đúng/sai theo category, không chỉ rate tổng.

---

## RQ7 — Judge-based Metrics (Ragas)

Kết quả Ragas (faithfulness, answer_relevancy, context_precision, context_recall) trả về NaN trên toàn bộ 3 pipeline vì **judge LLM (Gemini) throttling 93.5% các lời gọi** trong quá trình chấm. Xem [rq8_langsmith_nodes.csv](../results/metrics/rq8_langsmith_nodes.csv) dòng `faithfulness`, `answer_relevancy`, `context_*` — 100% error rate. Không thể báo cáo Ragas trong phạm vi này.

**Thay thế**: sử dụng F1/ROUGE-L/Cit-P/R làm proxy cho chất lượng câu trả lời (đã báo cáo ở RQ2 & RQ5). Ragas cần chạy với API key tier cao hơn hoặc judge khác (GPT-4o-mini) — ghi chú là hướng mở rộng.

---

## RQ8 — LangSmith Trace Analysis

**Câu hỏi:** Độ trễ Agentic RAG phân bổ ra sao giữa các node? Cost per query bao nhiêu?

### Thiết lập

Pull toàn bộ run từ project `Traffic-RAG-Evaluation` qua `langsmith.Client`, group theo `run.name`, tính p50/p95/mean latency (từ `end_time - start_time`), error rate, tokens, cost. Filter các wrapper nội bộ (`row N`, `ragas evaluation`). Script: [rq8_langsmith_pull.py](../scripts/rq8_langsmith_pull.py).

### Kết quả (top 10 node theo số lần gọi)

| run_name | run_type | count | mean (s) | p50 (s) | p95 (s) | Error rate | Tokens/call | Cost USD/call |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ChatGoogleGenerativeAI | llm | 1 518 | 0.39 | 0.002 | 1.98 | 0.935 | 1 240 | 0.000326 |
| LangGraph (agentic pipeline) | chain | 29 | 15.89 | **6.65** | **57.28** | 0.00 | 28 836 | **0.00882** |
| analyzer | chain | 25 | 8.30 | 1.93 | 17.27 | 0.00 | 934 | 0.00038 |
| legal_rag | chain | 20 | 10.85 | 4.58 | 55.03 | 0.00 | 39 749 | **0.01020** |
| web_search | chain | 4 | 8.52 | 9.11 | 9.11 | 0.00 | 4 440 | 0.00170 |
| chit_chat | chain | 1 | 1.60 | 1.60 | 1.60 | 0.00 | 144 | 0.00009 |
| route_by_category | chain | 25 | 0.001 | 0.001 | 0.001 | 0.00 | — | — |
| legal_fallback_router | chain | 20 | 0.004 | 0.003 | 0.007 | 0.00 | — | — |

![RQ8 node breakdown](../results/figures/rq8_node_breakdown.png)

![RQ8 error rate](../results/figures/rq8_error_rate.png)

![RQ8 latency hist](../results/figures/rq8_latency_hist.png)

### Kết luận

- **Pipeline agentic thật sự tốn thời gian ở `legal_rag` (mean 10.85 s, p95 55 s)** — đây là node lồng `retriever → cross_reference → generator`. Tối ưu lớn nhất có thể làm: streaming output hoặc giới hạn max_tokens generator.
- **Analyzer p95 = 17 s** — khi structured output parsing fail, node retry. Có thể giảm bằng Pydantic strict mode + fallback ngắn.
- **Routing & classifier gần như miễn phí** (<10 ms, ~0 USD). Overhead của agentic không nằm ở routing.
- **Cost /query**: `legal_rag` ~$0.0102, `web_search` ~$0.0017. Trung bình **≈ $0.009 / query** cho pipeline agentic — tương đương ~11 000 câu cho mỗi $100 Gemini credit.
- **Error rate 93.5% trên 1 518 lời gọi `ChatGoogleGenerativeAI`** là do **ragas judge bị throttle** (mỗi câu ragas gọi LLM ~60 lần). Pipeline chính (`legal_rag`, `analyzer`, `LangGraph`) đều có error rate **0.00** — hệ thống production không bị lỗi, chỉ evaluation bị rate-limit.

---

## RQ9 — Query Rewriting Ablation

**Câu hỏi:** Việc viết lại (rewrite + expand) truy vấn người dùng qua một LLM analyzer có thực sự cải thiện câu trả lời so với ground truth không, hay chỉ là overhead thừa?

### Thiết lập

| Variant | Query vào retriever | Query vào generator | LLM phụ |
|---|---|---|---|
| **NO_REWRITE** | Câu hỏi thô của user | Câu hỏi thô của user | 0 |
| **REWRITE** | `expanded_query` (thuật ngữ pháp lý) | `standalone_query` (giải tham chiếu lịch sử) | 1 analyzer call/câu |

- Retriever (TrafficHybridRetriever với e5-small, BM25 + dense + RRF, top_k=10) và generator (Gemini, P2 full rules) giữ nguyên.
- Analyzer dùng prompt production `ANALYZER_SYSTEM_PROMPT` — phân loại + viết câu độc lập + mở rộng thuật ngữ (ví dụ "vượt đèn đỏ đi xe máy" → "hành vi vượt đèn đỏ đối với xe mô tô, xe gắn máy, mức xử phạt theo Nghị định 168/2024/NĐ-CP").
- 25 câu × 2 variant = 50 lượt gọi generator + 25 lượt gọi analyzer (chỉ ở REWRITE). Script: [rq9_rewrite_ablation.py](../scripts/rq9_rewrite_ablation.py).

### Kết quả

| Variant | R@5 | R@10 | MRR | nDCG@10 | F1 token | ROUGE-L | Cit-P | Cit-R | Cit-F1 | Refusal | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **NO_REWRITE** | **0.40** | **0.46** | **0.197** | **0.212** | 0.259 | 0.208 | **0.219** | **0.44** | **0.233** | 0.56 | **1.87** |
| REWRITE | 0.34 | 0.42 | 0.127 | 0.144 | **0.274** | **0.228** | 0.207 | 0.38 | 0.214 | **0.52** | 2.72 |

![RQ9 rewrite comparison](../results/figures/rq9_rewrite_comparison.png)

**Per-category (mean R@10 / F1 token):**

| Category | N | R@10 NO | R@10 RW | F1 NO | F1 RW |
|---|---:|---:|---:|---:|---:|
| simple_penalty  | 5 | 0.40 | 0.30 | 0.52 | **0.57** |
| procedure       | 5 | 0.40 | 0.30 | 0.28 | **0.35** |
| multi_intent    | 5 | 0.30 | 0.30 | 0.21 | **0.24** |
| cross_reference | 5 | **0.20** | 0.20 | **0.16** | 0.07 |
| out_of_scope    | 5 | 1.00 | 1.00 | 0.14 | 0.14 |

![RQ9 per category](../results/figures/rq9_rewrite_per_category.png)

### Kết luận

- **Rewrite làm TỤT retrieval** trên mọi metric khoan-level: R@10 −4 điểm %, MRR −35% tương đối (0.197 → 0.127), nDCG@10 −32%. Điều này trái với kỳ vọng thông thường rằng expansion = recall cao hơn.
- **Nguyên nhân chính**: `multilingual-e5-small` đã được huấn luyện trên dữ liệu web đa ngôn ngữ đa dạng, xử lý tốt ngôn ngữ dân dã ("vượt đèn đỏ đi xe máy") — khi analyzer chèn thêm khung "theo Nghị định 168/2024/NĐ-CP + hình thức xử phạt bổ sung + biện pháp khắc phục hậu quả", query vector **bị kéo về cụm văn bản chung (boilerplate)** chứ không còn gần khoản cụ thể. BM25 cũng bị tổn hại vì các token pháp lý phổ biến xuất hiện ở hàng trăm khoản.
- **Category `cross_reference` bị hurt nặng nhất** (F1 0.16 → 0.07): câu hỏi có nhiều ý liên kết, analyzer dễ expand quá đà, làm mất từ khóa chính.
- **Answer side được cải thiện nhẹ** (F1 +1.5 điểm %, ROUGE-L +2 điểm %) ở `simple_penalty` và `procedure`: khi context đã chứa đủ thông tin, `standalone_query` (giải tham chiếu history) giúp generator diễn đạt trả lời rõ ràng hơn — nhưng không đủ bù cho retrieval suy giảm.
- **Latency tăng 45%** (1.87 → 2.72s) do chi phí 1 analyzer call thêm mỗi truy vấn, và cost +~$0.0004/query.
- **Quyết định**: **bỏ auto-rewrite ở pipeline `legal_rag` đơn thuần** khi user không có history. Giữ analyzer chỉ cho (a) **routing** (phân loại legal_rag / chit_chat / web / out_of_scope — không thay query), và (b) **history resolution** (chỉ chạy khi có ≥ 1 lượt hội thoại trước đó, cần dereference "ô tô thì sao?"). Với câu hỏi đầu lượt một câu, **retriever dùng raw query**.

> **Lưu ý phương pháp:** ở `out_of_scope`, R@10 = 1.00 là artifact — gold_citations rỗng nên recall định nghĩa là 1.0 theo `metrics.py:152`. Không nên đọc như "retrieval hoàn hảo". F1 0.14 ở cả hai variant phản ánh generator đúng đắn từ chối.
>
> **Limitation:** Một analyzer khác (prompt ngắn gọn hơn, ít boilerplate pháp lý) có thể cho kết quả khác. Không nên kết luận "rewrite luôn xấu" — chỉ nên kết luận "prompt analyzer hiện tại over-expand". Hướng follow-up: rewrite **tối giản** (chỉ giải tham chiếu history, không thêm thuật ngữ) và đánh giá lại.

---

## §5 Tổng kết & Hạn chế

| RQ | Phát hiện chính | Quyết định |
|---|---|---|
| **RQ2** | Hierarchical thắng R@10/MRR khoan-level; fixed_512 thắng F1 do chunk rộng hơn | Giữ hierarchical — critical for legal citation accuracy |
| **RQ3** | `multilingual-e5-small` thắng toàn diện (MRR 3×, encode 2.5× nhanh, dim nhỏ hơn 50%) | **Đã migrate** — collection `traffic_law_v3_e5` (384d) làm production, `traffic_law_v2` là fallback |
| **RQ4** | FAISS 200× nhanh Qdrant nhưng R@10 giống hệt; latency DB không phải bottleneck | Giữ Qdrant — vì filter + hybrid, không phải tốc độ |
| **RQ5** | Rules (P2) tăng Cit-R 33% so với zero-shot; role-only (P1) gần như vô ích | Giữ P2 nhưng nghiên cứu nới rule từ chối (refuse rate 72% là cao) |
| **RQ7** | Ragas fail 100% do judge throttle — không thể đánh giá được | Bỏ trong phạm vi thesis, đánh dấu future-work |
| **RQ8** | `legal_rag` node là cost/latency center ($0.010/query, p50 4.6 s); routing gần như miễn phí | Tối ưu sau: streaming output + max_tokens cap |
| **RQ9** | Rewrite → retrieval tệ hơn (MRR −35%, R@10 −4pts); F1 nhích +1.5pts nhờ standalone_query; latency +45% | **Tắt auto-rewrite ở lượt đầu** — giữ analyzer cho routing + history resolution only |

### Hạn chế

1. **Eval set 25 câu** — khoảng tin cậy rộng cho các phân tích per-category (thường 4-6 câu/category).
2. **Judge RAGAS không chạy được** → chưa có đánh giá faithfulness / answer_relevancy tự động.
3. **RQ2 có confound encoder** (sbert vs e5-small) — đã ghi chú; để sạch hoàn toàn cần chạy lại fixed_512 trên sbert, chấp nhận truncate 256 token.
4. **Gold citation có ~5/33 khoản có thể sai/thiếu** (pháp lý phức tạp) — chưa re-curate.
5. **Cost chỉ đo token generator**, không đếm chi phí embedding lúc index.

### §6 Hướng mở rộng (đã thu hẹp)

- Thêm reranker (Cohere / BGE-reranker) để cải thiện MRR khoan-level.
- Migrate encoder sang `multilingual-e5-base` (full-size) để kiểm tra liệu có vượt e5-small.
- Fine-tune role+rules prompt với negative examples để giảm refuse rate mà vẫn giữ Cit-R cao (RQ5 follow-up).
- Re-chạy Ragas với judge tier cao / judge khác (GPT-4o-mini).
- Multi-hop agent cho câu hỏi liên quan nhiều điều luật.
- Re-curate gold citation với sự trợ giúp của chuyên gia pháp lý.
