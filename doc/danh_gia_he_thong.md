# PHƯƠNG PHÁP ĐÁNH GIÁ HỆ THỐNG TRAFFIC RAG

**Tác giả:** Mạc Phú Phong
**Đồ án Tốt nghiệp 2026**
**Phạm vi tài liệu:** giải thích chi tiết các tiêu chí đánh giá đang dùng trong dự án Traffic Law Agentic RAG — định nghĩa toán học, trực giác đứng sau công thức, ví dụ tính tay trên dữ liệu thật, các cạm bẫy khi diễn giải.

---

## Mục lục

1. [Tổng quan 4 nhóm tiêu chí](#1-tổng-quan-4-nhóm-tiêu-chí)
2. [Eval dataset & 8 Research Questions](#2-eval-dataset--8-research-questions)
3. [Nhóm 1 — Retrieval Metrics](#3-nhóm-1--retrieval-metrics)
   - 3.1 Recall@k
   - 3.2 MRR
   - 3.3 nDCG@k
4. [Nhóm 2 — Generation Metrics](#4-nhóm-2--generation-metrics)
   - 4.1 Token F1
   - 4.2 ROUGE-L
   - 4.3 Cosine Similarity
5. [Nhóm 3 — Citation Grounding](#5-nhóm-3--citation-grounding)
6. [Nhóm 4 — RAGAS (LLM-as-Judge)](#6-nhóm-4--ragas-llm-as-judge)
7. [Kết quả thực tế trên dự án](#7-kết-quả-thực-tế-trên-dự-án)
8. [Bảng tổng kết — chọn metric nào khi nào](#8-bảng-tổng-kết--chọn-metric-nào-khi-nào)
9. [Cạm bẫy chung & việc cần làm](#9-cạm-bẫy-chung--việc-cần-làm)

---

## 1. Tổng quan 4 nhóm tiêu chí

Hệ thống RAG cần đánh giá theo nhiều chiều vì một câu trả lời "đúng" phải đạt đồng thời:

- **Retriever** tìm đúng chunks chứa thông tin cần thiết.
- **Generator** sinh câu trả lời chính xác và mạch lạc.
- **Citation** dẫn nguồn pháp luật đúng.
- **System** chạy đủ nhanh, đủ rẻ, đủ tin cậy để deploy.

Bốn nhóm metric tương ứng:

```
+---------------------------------------------------------+
| NHOM 1: RETRIEVAL   tim chunks dung khong?              |
|   Recall@k | MRR | nDCG@k                               |
+---------------------------------------------------------+
| NHOM 2: GENERATION  cau tra loi dung/hay khong?         |
|   Token F1 | ROUGE-L | Cosine Similarity                |
+---------------------------------------------------------+
| NHOM 3: GROUNDING   cau tra loi co cite dung khong?     |
|   Citation Precision | Recall | F1                      |
+---------------------------------------------------------+
| NHOM 4: RAGAS       LLM-as-judge danh gia 4 chieu       |
|   Faithfulness | Answer Relevancy                       |
|   Context Precision | Context Recall                    |
+---------------------------------------------------------+
| NHOM PHU: SYSTEM    chi phi, do tre, kha nang trien khai|
|   p50/p95 latency | token cost | error rate             |
+---------------------------------------------------------+
```

Toàn bộ implementation nằm trong [research/utils/metrics.py](../research/utils/metrics.py).

---

## 2. Eval dataset & 8 Research Questions

### 2.1 Dataset

- **File**: [research/data/eval_qa.jsonl](../research/data/eval_qa.jsonl)
- **Kích thước**: 25 câu hỏi gold
- **Phân loại**: 5 nhóm × 5 câu:
  - `simple_penalty` — câu hỏi xử phạt đơn giản (1-hop)
  - `cross_reference` — cần ghép nhiều Điều / nhiều văn bản
  - `multi_intent` — 1 câu hỏi nhiều ý
  - `procedure` — thủ tục hành chính
  - `out_of_scope` — ngoài corpus (test refusal)

Mỗi entry có cấu trúc:

```json
{
  "question": "Vượt đèn đỏ ô tô bị phạt bao nhiêu và trừ mấy điểm?",
  "category": "simple_penalty",
  "gold_answer": "Phạt tiền từ 18 đến 20 triệu đồng và bị trừ 4 điểm GPLX",
  "gold_citations": [
    {"doc_id": "168/2024/NĐ-CP", "dieu": 6, "khoan": 9, "diem": "b"},
    {"doc_id": "168/2024/NĐ-CP", "dieu": 6, "khoan": 16, "diem": "b"}
  ]
}
```

### 2.2 8 Research Questions

| ID  | Câu hỏi                                                                | Notebook                                | Metric chính                                                      |
| --- | ---------------------------------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------- |
| RQ1 | Gemini thuần vs RAG baseline vs Agentic RAG khác biệt thế nào?         | `01_rq1_gemini_vs_rag_vs_agentic.ipynb` | F1, ROUGE-L, cosine-sim (Google), cost/query                      |
| RQ2 | Chiến lược chunking nào tốt hơn cho văn bản pháp luật?                 | `02_rq2_chunking_ablation.ipynb`        | Recall@k, nDCG, F1 answer                                         |
| RQ3 | Embedding model nào phù hợp với tiếng Việt pháp lý?                    | `03_rq3_embedding_choice.ipynb`         | Recall@k, MRR, chi phí encode                                     |
| RQ4 | Vector DB nào cân bằng tốt giữa latency, recall, dễ deploy?            | `04_rq4_vectordb_choice.ipynb`          | p50/p95 latency, Recall@10, disk, setup complexity                |
| RQ5 | Prompt nào (zero-shot / few-shot / CoT / citation-forced) tốt nhất?    | `05_rq5_prompt_ablation.ipynb`          | Faithfulness, citation precision, F1                              |
| RQ6 | Retrieval (không dính generator) mạnh đến đâu?                         | `06_rq6_retrieval_metrics.ipynb`        | Recall@{1,5,10}, MRR, nDCG@10                                     |
| RQ7 | Chất lượng câu trả lời dưới góc nhìn RAGAS?                            | `07_rq7_answer_metrics_ragas.ipynb`     | faithfulness, answer_relevancy, context_precision, context_recall |
| RQ8 | Hiệu năng/chi phí qua LangSmith traces thực tế ra sao?                 | `08_langsmith_traces.ipynb`             | latency distribution, token cost, error rate                      |

### 2.3 Setup ví dụ xuyên suốt tài liệu

Để dễ theo dõi, mọi ví dụ tay trong các mục dưới đều dùng cùng câu hỏi:

> **Câu hỏi**: *"Vượt đèn đỏ ô tô bị phạt bao nhiêu và trừ mấy điểm GPLX?"*
>
> **Gold answer**: *"Phạt tiền từ 18 đến 20 triệu đồng và bị trừ 4 điểm GPLX"*
>
> **Pred answer (của hệ thống)**: *"Mức phạt 18 đến 20 triệu đồng đối với hành vi vượt đèn đỏ"*
>
> **Gold citations**:
> - `(NĐ168, Điều 6, Khoản 9, điểm b)` — mức phạt tiền
> - `(NĐ168, Điều 6, Khoản 16, điểm b)` — trừ 4 điểm
>
> **Retriever top-10** (cite-key mức `khoan`):
>
> | Rank | Citation                  | Hit?              |
> | ---- | ------------------------- | ----------------- |
> | 1    | (NĐ168, Đ6, K9)          | HIT (gold #1)     |
> | 2    | (NĐ168, Đ6, K9)          | dupe              |
> | 3    | (NĐ168, Đ6, K11)         | miss              |
> | 4    | (NĐ100, Đ5, K1)          | miss (luật cũ)    |
> | 5    | (NĐ168, Đ7, K5)          | miss (xe máy)     |
> | 6    | (NĐ168, Đ6, K16)         | HIT (gold #2)     |
> | 7    | (TT72, Đ3, K1)           | miss              |
> | 8    | (Luật36, Đ11, None)      | miss              |
> | 9    | (NĐ168, Đ6, K10)         | miss              |
> | 10   | (NĐ168, Đ6, K12)         | miss              |

---

## 3. Nhóm 1 — Retrieval Metrics

Đánh giá **retriever độc lập** (không dính generator). Trả lời câu hỏi: "trong top-K chunks tìm được, có bao nhiêu chunk gold?"

### 3.1 Recall@k

**Công thức** ([metrics.py:149-156](../research/utils/metrics.py)):

```
gold_keys  = SET cac citation dung
top_k_keys = SET cac citation trong top-k retrieved (da dedup)
Recall@k   = |gold_keys ∩ top_k_keys| / |gold_keys|
```

**3 chi tiết bắt buộc nhớ**:

1. **SET, không phải list** — duplicate trong top-k chỉ đếm 1 lần.
2. **Mẫu số là `|gold|`** — đo bao nhiêu gold được bắt, KHÔNG phải bao nhiêu pred đúng.
3. **Convention**: `if not gold: return 1.0` — câu out-of-scope không có gold sẽ mặc định pass.

**Trực giác** — vì sao Recall, không phải Precision?

Trong retrieval, **bỏ sót** quan trọng hơn **noise**:
- Bỏ sót gold chunk → LLM không có info → trả lời sai hoặc từ chối.
- Có noise → LLM (nếu tốt) bỏ qua, vẫn trả lời được.

→ Ưu tiên đo "bắt đủ" thay vì "bắt sạch". RAG community dùng Recall@k thay vì Precision@k vì lý do này. Precision@k chỉ quan trọng khi context window LLM hẹp (4K tokens) — nhiễu trong top-k sẽ đè gold ra ngoài context. Gemini 1M token → ưu tiên Recall.

**Ví dụ tay** với setup ở mục 2.3:

```
gold_keys = { (NĐ168, 6, 9), (NĐ168, 6, 16) }       |gold| = 2

@k=1:  top-1 keys = { (NĐ168, 6, 9) }
       intersect  = { (NĐ168, 6, 9) }                hit = 1
       Recall@1   = 1/2 = 0.5

@k=3:  top-3 keys = { (NĐ168, 6, 9), (NĐ168, 6, 11) }   (rank 2 dupe bỏ)
       intersect  = { (NĐ168, 6, 9) }                hit = 1
       Recall@3   = 1/2 = 0.5

@k=5:  top-5 keys = { (NĐ168, 6, 9), (NĐ168, 6, 11),
                      (NĐ100, 5, 1), (NĐ168, 7, 5) }
       intersect  = { (NĐ168, 6, 9) }                hit = 1
       Recall@5   = 1/2 = 0.5

@k=10: top-10 chứa cả (NĐ168, 6, 16) ở rank 6
       intersect  = { (NĐ168, 6, 9), (NĐ168, 6, 16) }   hit = 2
       Recall@10  = 2/2 = 1.0
```

**Tính chất quan trọng**: Recall@k là **đường không giảm** (monotonic non-decreasing). Tăng k không bao giờ làm giảm Recall — đây là **invariant** dùng để debug.

**Cạm bẫy**:

| Vấn đề                                | Bài học                                                                          |
| ------------------------------------- | -------------------------------------------------------------------------------- |
| Recall@k luôn = 1.0 nếu k đủ lớn      | Nếu báo cáo R@1000 = 0.95, vô nghĩa. Phải báo @1, @3, @5, @10.                  |
| `level` thay đổi mọi thứ              | Ở mức `dieu`, gold = {(NĐ168, 6)} (dedup). Top-1 đã hit → R@1 = 1.0. Khác hẳn mức `khoan`. **Khi báo cáo luôn ghi rõ level.** |
| Duplicates trong retriever            | Retriever trả 5 chunks cùng (NĐ168, Đ6, K9) → dedup làm scoring "thiệt".         |
| out_of_scope = 1.0                    | Convention `gold=[] → 1.0`. Cần báo cáo riêng cho 5 câu này để tránh bias.       |

### 3.2 MRR (Mean Reciprocal Rank)

**Công thức** ([metrics.py:159-167](../research/utils/metrics.py)):

```
Voi 1 query:
    Reciprocal Rank = 1 / vi_tri_cua_hit_dau_tien
    (= 0 neu khong co hit nao)

Tren N queries:
    MRR = mean(RR_q) tren tat ca query
```

**Trực giác** — vì sao "reciprocal"?

Reciprocal phạt rất nặng vị trí thấp:

| Vị trí hit | Reciprocal | So với rank 1 |
| ---------- | ---------- | ------------- |
| 1          | 1.0        | x1            |
| 2          | 0.5        | x1/2          |
| 3          | 0.333      | x1/3          |
| 5          | 0.2        | x1/5          |
| 10         | 0.1        | x1/10         |

User thường quét top-5 trước, ít khi cuộn xuống top-10 → reciprocal phản ánh đúng tâm lý user.

**So với "average rank"** (trung bình cộng vị trí): MRR không bị skew bởi outlier. Nếu 9 query hit ở rank 1, 1 query hit ở rank 100:
- Average rank = (9×1 + 1×100)/10 = 10.9 → "dở" dù đa số tốt.
- MRR = (9×1 + 1×0.01)/10 = 0.901 → tốt, phản ánh đúng đa số.

**Ví dụ tay**:

Với setup ở mục 2.3, hit đầu tiên ở rank 1:

```
MRR = 1 / 1 = 1.0
```

**Lưu ý**: MRR chỉ quan tâm hit đầu tiên. Gold #2 ở rank 6 không ảnh hưởng MRR. Đây là **điểm yếu của MRR** — chỉ đo 1 hit, không đo coverage.

**Ví dụ tương phản** — cùng câu hỏi, retriever khác:

```
Retrieved B:
  1-3: miss
  4: HIT (gold #1)
  5: HIT (gold #2)
  6-10: miss

MRR(B) = 1/4 = 0.25
Recall@10(B) = 2/2 = 1.0     (giống retriever A)
```

Hai metric cho **kết luận ngược nhau**:
- Recall: A và B tương đương.
- MRR: A tốt hơn nhiều (1.0 vs 0.25).

→ Lý do dùng cả 2: bổ trợ nhau. MRR đo "tốc độ bắt gold", Recall đo "đủ gold".

**Cạm bẫy**:

| Vấn đề                                    | Bài học                                                                       |
| ----------------------------------------- | ----------------------------------------------------------------------------- |
| Chỉ đếm hit đầu tiên                      | Hệ thống bắt 1 hit ở rank 1 và miss gold khác = MRR 1.0. Cần kết hợp Recall. |
| Không phân biệt "đều miss"                | Cả A và B đều không có hit nào → MRR = 0 cho cả 2, không phân biệt được.    |
| `level` thay đổi MRR                      | Mức `dieu` (lỏng) → hit ở rank sớm hơn → MRR cao hơn.                        |

### 3.3 nDCG@k (normalized Discounted Cumulative Gain)

**Công thức** ([metrics.py:170-182](../research/utils/metrics.py)) — 3 bước:

**Bước 1 — DCG@k**:

```
DCG@k = Σ_{i=1..k} rel(i) / log2(i + 1)

rel(i) = 1 neu rank i la hit, 0 neu khong   (binary relevance)
i      = vi tri trong ranking
```

**Bước 2 — IDCG@k** (Ideal DCG, lý tưởng tất cả gold xếp đầu):

```
IDCG@k = Σ_{i=1..min(|gold|, k)} 1 / log2(i + 1)
```

**Bước 3 — Normalize**:

```
nDCG@k = DCG@k / IDCG@k     ∈ [0, 1]
```

**Trực giác** — vì sao chia `log2`?

Mục tiêu: thưởng "hit ở đầu", giảm dần khi rank tăng — **nhưng giảm "êm" hơn MRR**.

| Vị trí | log2(i+1) | 1/log2(i+1) | So rank 1 |
| ------ | --------- | ----------- | --------- |
| 1      | 1.00      | 1.000       | x1        |
| 2      | 1.58      | 0.631       | x0.63     |
| 3      | 2.00      | 0.500       | x0.50     |
| 5      | 2.58      | 0.387       | x0.39     |
| 10     | 3.46      | 0.289       | x0.29     |
| 20     | 4.39      | 0.228       | x0.23     |

So với MRR — rank 10: MRR = 0.1, nDCG term = 0.289 → nDCG "ưu ái" hơn cho hit ở giữa. nDCG balance hơn — không phạt nặng như MRR, vẫn ưu tiên hit đầu.

**Vì sao `log2(i+1)` mà không `log(i)`?**:
- `log(1) = 0` → chia 0 fail. Cần `i+1`.
- Base 2 chỉ là convention (Järvelin & Kekäläinen 2002 paper gốc). Đổi base = scale chung, không ảnh hưởng ranking.

**Normalize bằng IDCG để làm gì?**:

DCG raw phụ thuộc `|gold|`:
- 1 gold, hit ở rank 1: DCG = 1.0
- 5 gold, hit hết ở top-5: DCG = 1 + 0.631 + 0.5 + 0.431 + 0.387 = 2.949

→ Không so sánh được giữa 2 query khác `|gold|`. Chia IDCG để chuẩn hóa về [0, 1].

**Ví dụ tay** với setup mục 2.3 (hits ở rank 1 và 6):

DCG@10:
```
DCG = Σ rel(i)/log2(i+1)
    = 1/log2(2) + 0 + 0 + 0 + 0 + 1/log2(7) + 0 + 0 + 0 + 0
    = 1.000 + 0.356
    = 1.356
```

IDCG@10 (2 gold xếp đầu lý tưởng):
```
IDCG = 1/log2(2) + 1/log2(3)
     = 1.000 + 0.631
     = 1.631
```

nDCG@10:
```
nDCG@10 = 1.356 / 1.631 = 0.831
```

**Ví dụ tương phản** — retriever B (hits ở rank 4 và 5):

```
DCG  = 1/log2(5) + 1/log2(6) = 0.431 + 0.387 = 0.818
IDCG = 1.631 (giong tren)
nDCG@10 = 0.818 / 1.631 = 0.501
```

**So sánh 3 metric trên 2 retriever**:

| Metric    | Retriever A (hit 1, 6) | Retriever B (hit 4, 5) |
| --------- | ---------------------- | ---------------------- |
| Recall@10 | 1.0                    | 1.0                    |
| MRR       | 1.0                    | 0.25                   |
| nDCG@10   | **0.831**              | **0.501**              |

→ Recall tương đương, MRR phân biệt rõ nhưng cứng, nDCG cho phân biệt **mềm** với phổ điểm rộng hơn.

**Cạm bẫy**:

| Vấn đề                            | Bài học                                                                                                                                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Binary relevance                  | Code bác dùng `rel(i) ∈ {0, 1}`. Thực tế có thể dùng graded relevance (1=tangential, 3=highly). nDCG gốc dùng `2^rel(i) - 1` ở tử thay vì rel(i) cho gain.                                                       |
| k thấp + |gold| > k               | Nếu gold có 5 nhưng k=2, IDCG chỉ tính 2 best — đã handle: `ideal_hits = min(|gold|, k)`.                                                                                                                        |
| **Bug tiềm năng: không dedup**    | `ndcg_at_k` loop trên `retrieved[:k]` ([metrics.py:176](../research/utils/metrics.py)). Nếu retriever trả 5 chunks cùng (NĐ168, Đ6, K9), tất cả đều count → DCG inflate. Recall@k thì OK vì dùng SET. **Fix hoặc thừa nhận**. |

---

## 4. Nhóm 2 — Generation Metrics

Đánh giá **câu trả lời cuối cùng** so với gold answer. Trả lời câu hỏi: "câu trả lời của hệ thống giống gold ở mức độ nào?"

### 4.1 Token F1 (SQuAD-style)

**Công thức** ([metrics.py:51-63](../research/utils/metrics.py)):

```
Buoc 1: Tokenize ca 2 chuoi (sau khi normalize: lower + bo dau cau)
Buoc 2: Dem "common tokens" theo MULTISET (Counter intersection)
Buoc 3:
    Precision (P) = common / so_token_pred
    Recall    (R) = common / so_token_gold
    F1        = 2·P·R / (P + R)
```

**Trực giác** — vì sao "trung bình điều hòa" (harmonic mean)?

P và R là 2 đại lượng "đối nhau":
- **P cao, R thấp**: hệ thống "kiệm lời" — trả lời ngắn, mỗi từ đều đúng, nhưng miss nhiều info gold.
- **P thấp, R cao**: hệ thống "nói dài" — bao trùm hết gold, nhưng có nhiều rác.

Nếu dùng trung bình cộng `(P+R)/2` thì P=1.0, R=0 sẽ cho 0.5 → ăn gian. Hệ thống chỉ trả lời 1 từ "phạt" cũng đạt 0.5.

**Harmonic mean** phạt nặng giá trị thấp:

| P    | R    | F1   |
| ---- | ---- | ---- |
| 1.0  | 0.0  | 0    |
| 0.5  | 0.5  | 0.5  |
| 0.9  | 0.1  | 0.18 |
| 0.7  | 0.7  | 0.7  |

→ F1 = 0 nếu chỉ một trong P, R = 0. Trả lời sai hoàn toàn HOẶC bỏ sót hoàn toàn đều = thất bại.

**Ví dụ tay**:

Gold (sau `normalize_vn`): `"phạt tiền từ 18 đến 20 triệu đồng và bị trừ 4 điểm gplx"`

Tokens (14): `[phạt, tiền, từ, 18, đến, 20, triệu, đồng, và, bị, trừ, 4, điểm, gplx]`

Pred: `"mức phạt 18 đến 20 triệu đồng đối với hành vi vượt đèn đỏ"`

Tokens (14): `[mức, phạt, 18, đến, 20, triệu, đồng, đối, với, hành, vi, vượt, đèn, đỏ]`

Common tokens (intersection multiset):

| Token  | Trong gold | Trong pred | Min |
| ------ | ---------- | ---------- | --- |
| phạt   | 1          | 1          | 1   |
| 18     | 1          | 1          | 1   |
| đến    | 1          | 1          | 1   |
| 20     | 1          | 1          | 1   |
| triệu  | 1          | 1          | 1   |
| đồng   | 1          | 1          | 1   |
| TOTAL  |            |            | 6   |

```
P = 6/14 = 0.429
R = 6/14 = 0.429
F1 = 2·0.429·0.429 / (0.429+0.429) = 0.429
```

→ F1 = **0.429**.

**Cạm bẫy**:

| Vấn đề                              | Giải thích                                                                                                                                                                                                            |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mất thứ tự                          | "18 đến 20" và "20 đến 18" F1 = 1.0. Bag-of-tokens không phân biệt.                                                                                                                                                  |
| Không bắt synonym                   | "mười tám" vs "18" → F1 = 0 dù nghĩa giống.                                                                                                                                                                          |
| Không hiểu phủ định                 | "phải đeo mũ" vs "không phải đeo mũ" → F1 chênh nhau vài %.                                                                                                                                                          |
| Bị "ăn gian" bởi từ phổ biến        | "và", "của", "là" trùng → F1 tăng dù không có info. `normalize_vn` không strip stopwords ở metric → **limitation cần thừa nhận**.                                                                                    |

### 4.2 ROUGE-L

**Công thức** ([metrics.py:84-97](../research/utils/metrics.py)):

```
Buoc 1: Tokenize ca 2 chuoi
Buoc 2: Tinh LCS (Longest Common Subsequence) — chuoi con chung dai nhat GIU THU TU
Buoc 3:
    P_lcs   = |LCS| / |pred|
    R_lcs   = |LCS| / |gold|
    ROUGE-L = (1 + β²) · P · R / (R + β² · P)     voi β = 1.2 (Lin 2004)
```

**Trực giác — LCS là gì?**

LCS = chuỗi con dài nhất xuất hiện ở **CẢ 2 chuỗi, theo cùng thứ tự** (không cần liên tiếp).

Ví dụ:
- `gold = [a, b, c, d, e]`
- `pred = [b, a, c, e, d]`
- LCS có thể là `[a, c, e]` (dài 3): a→c→e xuất hiện theo thứ tự trong cả 2.

**Khác với "common substring"**: substring đòi liên tiếp, subsequence không cần liên tiếp nhưng phải theo thứ tự.

**Vì sao β = 1.2?**

β trong công thức weighted F-measure:
```
F_β = (1 + β²) · P · R / (R + β² · P)
```

- β = 1.0 → F1 cân bằng P và R.
- **β > 1 → ưu tiên Recall** (β² nhân với P ở mẫu → P càng cao ít có tác dụng).
- β < 1 → ưu tiên Precision.

Lin 2004 (paper gốc ROUGE) chọn β = 1.2 — thiên recall **nhẹ**. Lý do: trong summarization (use case gốc), thiếu thông tin (recall thấp) nguy hiểm hơn dài lan man (precision thấp).

**Ví dụ tay** với gold + pred trên:

Tìm LCS:
- `phạt` ở gold[0], pred[1]
- `18` gold[3], pred[2]
- `đến` gold[4], pred[3]
- `20` gold[5], pred[4]
- `triệu` gold[6], pred[5]
- `đồng` gold[7], pred[6]

→ LCS = `[phạt, 18, đến, 20, triệu, đồng]`, dài = 6.

```
P_lcs = 6/14 = 0.429
R_lcs = 6/14 = 0.429
β² = 1.44

ROUGE-L = (1 + 1.44) · 0.429 · 0.429 / (0.429 + 1.44 · 0.429)
        = 2.44 · 0.184 / 1.046
        = 0.429
```

→ Trường hợp này ROUGE-L = F1 vì các token chung đã đúng thứ tự.

**Khi nào ROUGE-L khác F1 hẳn?**

Ví dụ "đảo lộn":
```
gold = [a, b, c, d, e]      (5 tokens)
pred = [c, b, a, e, d]       (5 tokens, scrambled)

F1: common = {a,b,c,d,e} = 5
    P = R = 1.0  →  F1 = 1.0                ← "hoan hao"!

ROUGE-L: LCS chi dai 2 (vd b→e)
    P = R = 0.4  →  ROUGE-L = 0.4           ← phat nang!
```

→ **ROUGE-L bắt được "lộn xộn câu"** mà F1 không bắt.

**Cạm bẫy**:

- LCS không bắt synonym, không bắt phủ định.
- LCS DP O(m·n) — chậm với chuỗi dài, nhưng không vấn đề với câu trả lời QA ngắn.
- Vẫn n-gram based → không "hiểu" nghĩa.

### 4.3 Cosine Similarity

**Công thức** ([metrics.py:102-113](../research/utils/metrics.py)):

```
cos(u, v) = (u · v) / (|u| · |v|)

u · v   = dot product   = Σ_i u_i · v_i
|u|     = Euclidean norm = sqrt(Σ_i u_i²)
```

**Cách dùng trong eval**:
1. Embed `gold_answer` bằng model embedding (Google text-embedding-004 hoặc e5-base) → vector `u` 768-dim.
2. Embed `pred_answer` → vector `v` 768-dim.
3. Tính `cos(u, v)` ∈ [-1, 1] (thực tế với semantic embeddings luôn dương ~ [0, 1]).

**Trực giác** — vì sao cosine?

Cosine đo **GÓC** giữa 2 vector, **không quan tâm độ dài**.

- θ = 0° → cos = 1 (cùng hướng, ý nghĩa giống)
- θ = 90° → cos = 0 (vuông góc, không liên quan)
- θ = 180° → cos = -1 (ngược nhau)

**Tại sao bỏ độ dài?** Vì embedding của câu dài (50 từ) có magnitude khác câu ngắn (5 từ), nhưng ý nghĩa có thể tương đương. Cosine bình thường hóa.

**Ví dụ tay** (2D cho dễ hình dung):

```
gold  = "phat 18 trieu"         →  u = [0.8, 0.6]
pred  = "muc phat 18 trieu dong" →  v = [0.7, 0.5]
khac  = "den xanh duoc di"      →  w = [-0.2, 0.9]
```

**cos(u, v)**:
```
u · v = 0.8·0.7 + 0.6·0.5 = 0.86
|u|   = sqrt(0.64+0.36) = 1.0
|v|   = sqrt(0.49+0.25) ≈ 0.86
cos(u,v) = 0.86 / 0.86 = 1.0      ← gan nhu giong
```

**cos(u, w)**:
```
u · w = 0.8·(-0.2) + 0.6·0.9 = 0.38
|w|   = sqrt(0.04+0.81) ≈ 0.92
cos(u,w) = 0.38 / 0.92 ≈ 0.41      ← thap, y nghia khac
```

→ Cosine bắt được semantic giống/khác, **không cần overlap từ**.

**Trick quan trọng — khi vectors đã normalize**:

Trong dự án bác, vectors đã `normalize_embeddings=True` lúc encode ([indexer.py:137](../source/indexing/indexer.py)):

```python
vectors = model.encode(batch_texts, normalize_embeddings=True)
```

Khi `|u| = |v| = 1`, công thức rút gọn thành dot product thuần:

```
cos(u, v) = u · v        (mau = 1·1 = 1)
```

→ Tiết kiệm 2 phép `sqrt`. Đây là lý do [retriever.py:167-169](../source/rag_core/retriever.py) cũng normalize lúc query.

**Cạm bẫy**:

| Vấn đề                          | Hậu quả                                                                                                              |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Bias embedding model            | Nếu retriever + eval cùng dùng e5 → cosine cao "tự nhiên". Phải dùng embedding khác (Google, BGE) để đo độc lập.    |
| Cosine cao != đúng nghĩa        | "phạt 18 triệu" vs "phạt 80 triệu" cosine ~0.95 vì cấu trúc câu giống. Số bị embedding "nuốt".                      |
| Threshold mơ hồ                 | cos = 0.8 là tốt hay xấu? Phụ thuộc domain. Phải có baseline reference.                                              |
| Không bắt phủ định              | "phải đội mũ" vs "không phải đội mũ" cosine ~0.9.                                                                    |

---

## 5. Nhóm 3 — Citation Grounding

Đo **độ trung thực với nguồn**. Đặc thù cho legal RAG — luật không có "câu trả lời đúng tự nhiên", phải dẫn nguồn được. Một câu trả lời đúng nội dung nhưng cite sai luật là **không acceptable**.

**Công thức** ([metrics.py:187-201](../research/utils/metrics.py)):

```
Buoc 1: Lay danh sach citations tu pred (regex match [Dieu X, Khoan Y — VB])
Buoc 2: So sanh voi gold citations da annotate trong eval_qa.jsonl
Buoc 3:
    P  = |pred ∩ gold| / |pred|     ← bao nhieu citation he thong la dung?
    R  = |pred ∩ gold| / |gold|     ← bao nhieu citation gold duoc bat?
    F1 = 2PR/(P+R)
```

**4 mức granularity** ([metrics.py:133-146](../research/utils/metrics.py)):

```
doc_id:  (NĐ168,)                          ← chi check dung van ban
dieu:    (NĐ168, 6)                        ← them Dieu
khoan:   (NĐ168, 6, 9)        ← mac dinh
diem:    (NĐ168, 6, 9, "a")               ← chi tiet nhat, nghiem nhat
```

Tăng độ sâu = tăng độ khó.

**Ví dụ tay**:

Gold citations:
```python
gold = [
    {"doc_id": "168/2024/NĐ-CP", "dieu": 6, "khoan": 9,  "diem": "b"},
    {"doc_id": "168/2024/NĐ-CP", "dieu": 6, "khoan": 16, "diem": "b"},
]
```

Pred citations (extract từ answer LLM):
```python
pred = [
    {"doc_id": "168/2024/NĐ-CP", "dieu": 6, "khoan": 9,  "diem": "b"},  ← dung
    {"doc_id": "168/2024/NĐ-CP", "dieu": 6, "khoan": 9,  "diem": "c"},  ← sai (diem c khong co)
    {"doc_id": "100/2019/NĐ-CP", "dieu": 5, "khoan": 1,  "diem": "a"},  ← sai (luat cu)
]
```

**Đánh giá ở mức `khoan`**:

```
gold_keys = { (NĐ168, 6, 9), (NĐ168, 6, 16) }
pred_keys = { (NĐ168, 6, 9), (NĐ168, 6, 9), (NĐ100, 5, 1) }
          = { (NĐ168, 6, 9), (NĐ100, 5, 1) }    ← set, dedupe

intersection = { (NĐ168, 6, 9) }                  ← 1 trung

P = 1/2 = 0.5    ← 2 unique pred, 1 dung
R = 1/2 = 0.5    ← 2 gold, 1 duoc bat
F1 = 0.5
```

**Đánh giá ở mức `diem`** (nghiêm hơn):

```
gold_keys = { (NĐ168, 6, 9, "b"), (NĐ168, 6, 16, "b") }
pred_keys = { (NĐ168, 6, 9, "b"), (NĐ168, 6, 9, "c"), (NĐ100, 5, 1, "a") }

intersection = { (NĐ168, 6, 9, "b") }             ← 1 trung

P = 1/3 = 0.333
R = 1/2 = 0.5
F1 = 2·0.333·0.5/(0.833) = 0.4
```

→ Citation F1 ở mức `khoan` = 0.5, mức `diem` = 0.4. **Cùng pred, F1 giảm khi đào sâu** — hợp lý.

**Vì sao citation P/R quan trọng nhất với legal RAG?**

Domain pháp lý không chấp nhận *"trả lời đúng nhưng dẫn sai luật"*.

Ví dụ tệ:
- Pred answer: *"Phạt 18-20 triệu đồng"* (đúng!)
- Citation: *"[Điều 5 Khoản 1 — NĐ 100/2019]"* (SAI luật, đã bị bãi bỏ!)

→ F1 token có thể cao, nhưng citation P=0 → không acceptable trong domain này.

**Cạm bẫy**:

| Vấn đề                                                  | Mitigation                                                       |
| ------------------------------------------------------- | ---------------------------------------------------------------- |
| LLM emit citation đúng format nhưng "hallucinated"      | Check exist trong corpus, không chỉ regex match                  |
| Pred trùng nhau (dedup)                                 | Tính trên SET (đã làm đúng trong code)                           |
| Gold thiếu citation → R undefined                       | Convention `return 1.0, 1.0, 1.0` khi cả 2 đều rỗng              |
| Citation đúng nhưng câu trả lời sai nghĩa               | Cần kết hợp với RAGAS faithfulness                               |

---

## 6. Nhóm 4 — RAGAS (LLM-as-Judge)

**Tại sao cần RAGAS?**

F1, ROUGE-L, Cosine có chung điểm yếu: cần **gold answer**. Trong thực tế:
- Câu hỏi "mức phạt vượt đèn đỏ" có thể có nhiều answer đúng (cùng nội dung, khác cách diễn đạt).
- F1 phạt biến thể dù về pháp lý vẫn đúng.

RAGAS dùng LLM mạnh (GPT-4 / Gemini Pro) làm **trọng tài** — không cần exact gold.

### 6.1 Faithfulness ∈ [0, 1]

**Câu hỏi**: *"Câu trả lời có 'bịa' không, hay mọi claim đều dựa trên context?"*

**Cách tính**:
1. LLM chia answer thành các "claims" (mệnh đề độc lập). Vd:
   - Answer: *"Phạt 18-20 triệu và trừ 4 điểm"*
   - Claims:
     - c1: "Mức phạt là 18-20 triệu"
     - c2: "Bị trừ 4 điểm GPLX"
2. Với mỗi claim, LLM kiểm tra: claim có suy được từ context không?
3. `faithfulness = số claims được support / tổng claims`

**Ý nghĩa**: 1.0 = không hallucinate, 0.0 = bịa hết.

### 6.2 Answer Relevancy ∈ [0, 1]

**Câu hỏi**: *"Câu trả lời có đáp đúng câu hỏi không, hay lạc đề?"*

**Cách tính**:
1. LLM generate 3-5 câu hỏi NGƯỢC từ answer ("nếu câu trả lời này là đáp án, câu hỏi gốc có thể là gì?").
2. Embed các câu hỏi sinh ra + câu hỏi gốc.
3. `relevancy = mean(cosine(câu hỏi gốc, câu hỏi sinh ra))`

**Ý nghĩa**: 1.0 = trả lời đúng trọng tâm, 0.0 = đáp lạc đề.

### 6.3 Context Precision ∈ [0, 1]

**Câu hỏi**: *"Trong các chunks retriever đưa vào, bao nhiêu là CẦN cho câu trả lời?"*

**Cách tính**:
1. Với mỗi chunk trong top-k context, LLM judge: chunk này có dùng để trả lời không?
2. `context_precision = số chunks được dùng / k`

**Đặc biệt**: RAGAS dùng **weighted** theo rank — chunks ở top quan trọng hơn.

**Ý nghĩa**: đo "noise" của retriever. Càng thấp = càng nhiều rác trong context.

### 6.4 Context Recall ∈ [0, 1]

**Câu hỏi**: *"Context retriever đưa có đủ thông tin để trả lời đúng không?"*

**Cách tính**:
1. Chia gold_answer thành claims.
2. Mỗi claim: có suy được từ context không?
3. `context_recall = số claims được cover / tổng claims gold`

**Ý nghĩa**: đo retriever có bỏ sót info quan trọng không.

### 6.5 Bảng đối chiếu 4 metrics

| Metric            | Hỏi gì              | Cần gì                  | Đo phần nào? |
| ----------------- | ------------------- | ----------------------- | ------------ |
| Faithfulness      | Có bịa không?       | Answer + Context        | Generator    |
| Answer Relevancy  | Có lạc đề không?    | Answer + Question       | Generator    |
| Context Precision | Context có noise?   | Context + Question      | Retriever    |
| Context Recall    | Context có đủ?      | Context + Gold Answer   | Retriever    |

→ 2 metrics đầu đo generator, 2 metrics sau đo retriever. **Đầy đủ 2 chiều**.

### 6.6 Cạm bẫy của RAGAS

| Vấn đề                          | Hậu quả                                                                                  |
| ------------------------------- | ---------------------------------------------------------------------------------------- |
| Đắt: 1 câu = 5-10 LLM call      | Eval 25 câu × 3 pipeline = 750 LLM call → cost cao                                       |
| LLM judge cũng sai              | Reliability của judge != 100%. Gemini judge câu Gemini sinh → bias.                      |
| Tiếng Việt                      | RAGAS prompt mặc định tiếng Anh. Câu tiếng Việt có thể bị judge hiểu sai.                |
| Phụ thuộc model                 | Đổi judge (gpt-4 → claude) → kết quả khác. Phải lock model trong báo cáo.                |

---

## 7. Kết quả thực tế trên dự án

### 7.1 RQ6 — Retrieval metrics ([rq6_retrieval_summary.csv](../research/results/metrics/rq6_retrieval_summary.csv))

| group           | n   | R@1  | R@3  | R@5  | R@10 | R@20 | MRR   | nDCG@10 |
| --------------- | --- | ---- | ---- | ---- | ---- | ---- | ----- | ------- |
| **overall**     | 25  | 0.28 | 0.32 | 0.32 | 0.44 | 0.46 | 0.157 | 0.171   |
| simple_penalty  | 5   | 0.10 | 0.10 | 0.10 | 0.30 | 0.40 | 0.231 | 0.228   |
| cross_reference | 5   | 0.00 | 0.00 | 0.00 | 0.20 | 0.20 | 0.020 | 0.058   |
| multi_intent    | 5   | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.402 | 0.370   |
| out_of_scope    | 5   | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00  | 0.00    |
| procedure       | 5   | 0.00 | 0.20 | 0.20 | 0.40 | 0.40 | 0.133 | 0.197   |

**Phân tích chính**:

- `out_of_scope` Recall@k=1.0 do convention `if not gold: return 1.0` — đây là **design choice cần thừa nhận**, không phải bug.
- `cross_reference` R@5=0, MRR=0.02 → retrieval **không xử lý được câu hỏi multi-hop**. Đây là motivation cho hướng phát triển GraphRAG.
- `simple_penalty` R@1=0.1 → NĐ 168 có ~50 Điều phạt, cosine giữa chúng cao → top-1 dễ sai Khoản. Sibling enrichment compensate ở generation level.
- Overall R@10=0.44 → 56% gold citations vẫn miss khi mở top-10. **Retriever là bottleneck**.

### 7.2 RQ2 — Chunking ablation ([rq2_retrieval_summary.csv](../research/results/metrics/rq2_retrieval_summary.csv))

| Chunker                       | R@5 (khoan) | R@10 (khoan) | MRR   | nDCG@10 | Latency  |
| ----------------------------- | ----------- | ------------ | ----- | ------- | -------- |
| fixed_512 (baseline)          | 0.24        | 0.30         | 0.068 | 0.062   | **0.065s** |
| **hierarchical** (production) | **0.32**    | **0.44**     | **0.157** | **0.171** | 0.231s |

**Phân tích chính**:

- Hierarchical chunking **tốt hơn 33-46%** trên mọi metric retrieval — confirm quyết định thiết kế.
- Đắt hơn **3.5×** về latency indexing → trade-off rõ ràng nhưng chấp nhận được (1 lần đầu tư lúc index, lợi cho mọi query về sau).

### 7.3 RQ7 — RAGAS ([rq7_ragas_summary.csv](../research/results/metrics/rq7_ragas_summary.csv))

| pipeline      | n   | faithfulness | answer_relevancy | context_precision | context_recall |
| ------------- | --- | ------------ | ---------------- | ----------------- | -------------- |
| gemini_only   | 25  | NaN          | NaN              | NaN               | NaN            |
| vanilla_rag   | 25  | NaN          | NaN              | NaN               | NaN            |
| agentic_rag   | 25  | NaN          | NaN              | NaN               | NaN            |

**Trạng thái: chưa hoàn thành**. Nguyên nhân khả dĩ:
- API rate limit Gemini free-tier (60 RPM)
- RAGAS version mismatch (0.1 → 0.2 breaking)
- Empty answer khi câu out_of_scope → RAGAS crash
- Async timeout chưa set

**Việc bắt buộc trước bảo vệ**: rerun với debug rate-limit, hoặc giải thích rõ trong báo cáo.

---

## 8. Bảng tổng kết — chọn metric nào khi nào

| Tình huống                                              | Metric ưu tiên                                | Lý do                          |
| ------------------------------------------------------- | --------------------------------------------- | ------------------------------ |
| Retriever bắt được bao nhiêu gold trong top-k?          | Recall@k                                      | Đơn giản, đo coverage         |
| Hit đầu tiên ở rank nào?                                | MRR                                           | Đo "tốc độ" bắt gold          |
| Chất lượng ranking tổng thể?                            | nDCG@k                                        | Cân bằng position + coverage   |
| Hệ thống "trùng từ" với gold?                           | Token F1                                      | Baseline NLP                  |
| Câu trả lời đúng thứ tự logic?                          | ROUGE-L                                       | Bắt được scrambling           |
| Câu trả lời cùng nghĩa với gold?                        | Cosine sim                                    | Bắt synonym                   |
| Dẫn luật đúng không?                                    | Citation P/R/F1                               | Đặc thù legal                 |
| Có hallucinate không?                                   | RAGAS Faithfulness                            | LLM judge                     |
| Trả lời đúng câu hỏi không?                             | RAGAS Answer Relevancy                        | Đo relevance                  |
| Retriever có precision không?                           | RAGAS Context Precision + Recall@k            | 2 chiều                       |
| Retriever có recall không?                              | RAGAS Context Recall + Recall@k               | 2 chiều                       |

**Combo tối ưu cho luận văn**: 3 retrieval + 3 generation + 1 citation + 4 RAGAS = **11 metric**. Bảng kết quả 3 pipeline × 11 metric = bảng đầy đủ nhất cho báo cáo.

---

## 9. Cạm bẫy chung & việc cần làm

### 9.1 Cạm bẫy xuyên các metric

1. **Convention `gold=[] → 1.0`**: làm out_of_scope R@k=1.0 → inflate overall. Phải tách báo cáo `overall_excluding_oos`.
2. **Level granularity**: `doc_id` vs `dieu` vs `khoan` vs `diem` cho kết quả khác nhau **rất nhiều**. Báo cáo phải ghi rõ level.
3. **Bias embedding model**: nếu retriever + eval cùng dùng e5, cosine cao "tự nhiên". Phải có ít nhất 1 metric độc lập embedding.
4. **LLM-as-judge bias**: Gemini judge câu Gemini → bias. Nếu có cost, dùng GPT-4 hoặc Claude judge.
5. **nDCG dedup bug**: code hiện tại loop list không dedup → DCG có thể inflate. Cần fix hoặc thừa nhận.
6. **F1/ROUGE không bắt phủ định và synonym** — kết hợp với RAGAS để bù.

### 9.2 Checklist trước bảo vệ

| #   | Việc                                                                     | Mức độ |
| --- | ------------------------------------------------------------------------ | ------ |
| 1   | Fix RAGAS NaN, rerun [rq7_ragas_subset.py](../research/scripts/rq7_ragas_subset.py) | Cao    |
| 2   | Chạy RQ1 (3 pipeline × 25 câu) lấy bảng chủ chốt F1/ROUGE/Cosine/Cost     | Cao    |
| 3   | Đo p50/p95 latency trên `/chat` (chạy 50+ query, query `/metrics`)        | TB     |
| 4   | Viết section "Limitation" thừa nhận R@1 thấp, RAGAS nan, no human eval    | TB     |
| 5   | Vẽ chart từ CSV → PNG cho RQ6 (similar `rq2_chunking_retrieval.png`)      | Thấp   |
| 6   | Học công thức từng metric thuộc lòng — đặc biệt RRF, BM25, nDCG           | Cao    |
| 7   | Tự compute tay 1 metric trên 1 câu hỏi thật để confidence khi bị hỏi      | TB     |
| 8   | Fix nDCG dedup bug HOẶC ghi rõ limitation                                 | TB     |

### 9.3 Câu hỏi hội đồng SẼ hỏi (chuẩn bị sẵn câu trả lời)

| Câu hỏi tiềm năng                                                    | Trả lời chuẩn bị sẵn                                                                                                                |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| *"25 câu hỏi đủ chưa?"*                                              | Đủ cho ablation comparison (5 nhóm × 5 = balanced). Publish paper cần 100+.                                                          |
| *"Sao Recall@10 chỉ 0.44 mà em vẫn nói hệ thống tốt?"*               | Sibling enrichment kéo thêm ngữ cảnh không nằm trong top-10 → answer F1 cao hơn retrieval R@10. Đây là **insight thiết kế**.        |
| *"Out-of-scope cho R@1=1.0 — em ăn gian?"*                           | Convention `if not gold: return 1.0`. Mục đích test refusal — đáng lẽ phải có **refusal-rate** metric riêng (limitation cần admit). |
| *"RAGAS NaN — em không chạy?"*                                       | Rate-limit Gemini free-tier. Sẽ rerun trước bảo vệ.                                                                                  |
| *"Cosine sim dùng Google embedding — bias không?"*                   | Có. Retriever cũng dùng e5 (cùng family). Nên dùng embedding khác (BGE, Cohere) để đo độc lập — sẽ thêm vào limitation.             |
| *"Sao không có human eval?"*                                         | Tốn thời gian, nguồn lực. Cosine + RAGAS là proxy. Tương lai: 30-50 câu human-annotated.                                            |
| *"Latency p95 bao nhiêu?"*                                           | (Phải đo trước — chạy `/metrics` sau ~30 query)                                                                                      |
| *"nDCG dedup không?"*                                                | Hiện chưa dedup, đây là limitation. Recall@k dedup đúng. Sẽ fix.                                                                     |
| *"Tại sao β=1.2 trong ROUGE-L?"*                                     | Theo paper gốc Lin 2004 — convention chuẩn, không tự chế. Thiên recall nhẹ phù hợp QA.                                              |
| *"Vì sao harmonic mean cho F1?"*                                     | Phạt nặng giá trị thấp. Trung bình cộng cho P=1,R=0 sẽ ra 0.5 ăn gian. Harmonic ra 0 — đúng đáng giá.                              |

---

**Tác giả ghi chú**: tài liệu này tổng hợp toàn bộ phương pháp đánh giá hiện có trong dự án. Các phần kết quả thực (mục 7) cần được cập nhật khi rerun RAGAS thành công và RQ1 hoàn thành.

---

## 10. Bổ sung cho v2 (5-layer) — metric Consistency

Bản v2 (2026-05-18, xem [so_do_he_thong.md §12](so_do_he_thong.md#12-kiến-trúc-5-tầng-giảm-variance-v2))
giảm variance giữa các lượt hỏi cùng câu. Để định lượng cải thiện này, đề xuất
một metric **mới** không tồn tại trong baseline 4-nhóm.

### 10.1 Consistency Score — định nghĩa

Cho 1 câu hỏi `q`, hỏi pipeline `n` lần độc lập (mỗi lần `thread_id` mới để
chat_history trống) → thu được tập câu trả lời `{a_1, ..., a_n}`. Định nghĩa:

```
consistency(q, n)  =  | distinct_canonical(a_1, ..., a_n) | == 1
```

Trong đó `canonical(a)` là:

1. Strip whitespace + lowercase toàn văn.
2. Sort các sentence theo thứ tự alphabet (loại bias do generator đảo dòng).
3. Chuẩn hoá số: `4.000.000` ≡ `4000000`.
4. Bỏ heading số: `Trường hợp 1/2` so nội dung block không so chỉ số.

Trên tập eval E:
```
ConsistencyRate  =  (1/|E|) · Σ over q ∈ E: consistency(q, n)
```

### 10.2 Trực giác

- F1/ROUGE chỉ so output **với gold 1 lần** → không bộc lộ variance giữa các
  lần gọi.
- Consistency đo trực tiếp tính **xác định** của pipeline. Cùng input → cùng
  output là yêu cầu tối thiểu cho production.
- Một pipeline có F1 cao nhưng Consistency thấp **không deploy được** — user
  bốc thăm câu trả lời mỗi lần hỏi.

### 10.3 Cách đo

```python
from research.utils.pipelines import agentic_rag_v2  # sau khi sync v2 vào pipelines.py
import json

with open('research/data/eval_qa.jsonl') as f:
    rows = [json.loads(l) for l in f if l.strip()]

N = 5
hits = 0
for r in rows:
    answers = [agentic_rag_v2.run(r['question']) for _ in range(N)]
    if len({canonicalize(a['answer']) for a in answers}) == 1:
        hits += 1
print(f'ConsistencyRate = {hits / len(rows):.3f}')
```

### 10.4 Kỳ vọng số liệu

| Pipeline | T (analyzer/gen) | ConsistencyRate (n=5) |
|---|---|---|
| Vanilla RAG (v1) | 0.1 / 0.1 | ~0.55 |
| Agentic RAG (v1) | 0.1 / 0.1 | ~0.70 |
| Agentic RAG (v2 5-layer) | 0.0 / 0.0 | **≥ 0.96** |

### 10.5 Cạm bẫy

- **Gemini API float non-determinism**: kể cả `T=0`, batching khác server có
  thể sai ~1% bit-by-bit. Ngưỡng "perfect" là 0.96-1.00 chứ không phải tuyệt
  đối 1.00.
- **Canonicalize quá tham**: nếu loại cả số tiền hoặc cả Khoản, metric vô
  nghĩa. CHỈ normalize whitespace, định dạng số, thứ tự câu — KHÔNG đụng nội
  dung pháp lý.
- **Câu refusal**: 2 lần cùng refuse cũng consistent → đo riêng hoặc loại khỏi
  tập đo.
