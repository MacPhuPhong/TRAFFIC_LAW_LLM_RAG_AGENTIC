# Research — Legal Traffic RAG

Thư mục nghiên cứu kiểu đồ án tốt nghiệp, tách riêng khỏi code production (`traffic_rag/source/`, `traffic_rag/api/`) để:

- so sánh có kiểm soát giữa nhiều biến thể pipeline;
- dựng số liệu cho báo cáo đồ án (bảng, biểu đồ, phân tích);
- tái sử dụng eval set vàng cho các lần refactor về sau.

Mọi notebook đọc code production qua import (`from traffic_rag.source...`), không fork logic.

## Câu hỏi nghiên cứu (RQ)

| ID  | Câu hỏi                                                                    | Notebook                                 | Metric chính                                        |
| --- | -------------------------------------------------------------------------- | ---------------------------------------- | --------------------------------------------------- |
| RQ1 | Gemini thuần vs RAG baseline vs Agentic RAG khác biệt thế nào?             | `01_rq1_gemini_vs_rag_vs_agentic.ipynb`  | F1, ROUGE-L, cosine-sim (Google), cost/query        |
| RQ2 | Chiến lược chunking nào tốt hơn cho văn bản pháp luật?                     | `02_rq2_chunking_ablation.ipynb`         | Recall@k, nDCG, F1 answer                           |
| RQ3 | Embedding model nào phù hợp với tiếng Việt pháp lý?                        | `03_rq3_embedding_choice.ipynb`          | Recall@k, MRR, chi phí encode                       |
| RQ4 | Vector DB nào cân bằng tốt giữa latency, recall, dễ deploy?                | `04_rq4_vectordb_choice.ipynb`           | p50/p95 latency, Recall@10, disk, setup complexity  |
| RQ5 | Prompt nào (zero-shot / few-shot / CoT / citation-forced) cho câu trả lời tốt nhất? | `05_rq5_prompt_ablation.ipynb`  | Faithfulness, citation precision, F1                |
| RQ6 | Retrieval (không dính generator) mạnh đến đâu?                             | `06_rq6_retrieval_metrics.ipynb`         | Recall@{1,5,10}, MRR, nDCG@10                       |
| RQ7 | Chất lượng câu trả lời dưới góc nhìn RAGAS?                                | `07_rq7_answer_metrics_ragas.ipynb`      | faithfulness, answer_relevancy, context_precision, context_recall |
| RQ8 | Hiệu năng/chi phí qua LangSmith traces thực tế ra sao?                     | `08_langsmith_traces.ipynb`              | latency distribution, token cost, error rate        |

RQ1 là bảng "hàng đầu" cho đồ án (tái tạo bảng 4 phương pháp với F1 / ROUGE-L / cosine-sim / cost).

## Cấu trúc

```
research/
├── data/
│   ├── eval_qa.jsonl        # TODO: build 50-100 câu gold (xem notebook 00)
│   └── categories.json      # mô tả 5 nhóm câu hỏi
├── notebooks/
│   ├── 00_build_eval_dataset.ipynb
│   ├── 01_rq1_gemini_vs_rag_vs_agentic.ipynb
│   ├── 02_rq2_chunking_ablation.ipynb
│   ├── 03_rq3_embedding_choice.ipynb
│   ├── 04_rq4_vectordb_choice.ipynb
│   ├── 05_rq5_prompt_ablation.ipynb
│   ├── 06_rq6_retrieval_metrics.ipynb
│   ├── 07_rq7_answer_metrics_ragas.ipynb
│   └── 08_langsmith_traces.ipynb
├── utils/
│   ├── metrics.py           # F1 token-level, ROUGE-L, cosine-sim, citation-P/R
│   ├── pipelines.py         # 3 pipeline biến thể (gemini_only | vanilla_rag | agentic_rag)
│   ├── eval_runner.py       # chạy 1 pipeline qua eval set → log JSONL
│   └── langsmith_setup.py   # bật LANGCHAIN_TRACING + tag run
├── results/
│   ├── metrics/             # *.csv/json output mỗi lần chạy
│   ├── figures/             # *.png biểu đồ
│   └── traces/              # dump LangSmith traces (nếu cần)
└── report/
    └── analysis.md          # phần viết (nối bảng/biểu vào đồ án)
```

## Quy ước

- **Không sửa production code** từ trong `research/`. Nếu cần option mới (vd. bật/tắt cross-ref pass), expose qua tham số của hàm factory ở `source/`, rồi dùng từ notebook.
- **Mọi số liệu** lưu dưới `results/metrics/<rq_id>_<timestamp>.json` + CSV copy để viết bảng.
- **Không commit** file trong `results/traces/` nếu chứa nội dung nhạy cảm.
- **Random seed cố định** (`SEED = 42`) ở mọi notebook có mẫu ngẫu nhiên.

## Chạy

```bash
# 1. activate venv
source /home/pphong/venv/LLM_Agentic/bin/activate

# 2. bật LangSmith tracing (tuỳ chọn, cần LANGCHAIN_API_KEY trong .env)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=traffic-rag-research

# 3. jupyter
cd /media/pphong/D:/Do_An_Tot_Nghiep/GitHub1
jupyter lab traffic_rag/research/
```

## Dependency cần thêm

```
ragas>=0.2
rouge-score>=0.1.2
pandas>=2.0
matplotlib>=3.7
seaborn>=0.13
```

Đã có sẵn trong `requirements.txt`: `langsmith`, `sentence-transformers`, `qdrant-client`, `rank-bm25`, `langchain-google-genai`.
