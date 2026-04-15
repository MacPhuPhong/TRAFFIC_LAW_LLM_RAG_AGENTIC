import json
import os

notebook_path = '/media/pphong/D:/Do_An_Tot_Nghiep/GitHub1/traffic_rag/source_research/Eval_System/benchmark_end_to_end.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define new cells
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.  Nâng cao: Đánh giá bằng RAGAS (Faithfulness, Answer Relevance)\n",
            "Mục này sử dụng thư viện **Ragas** để đánh giá tính trung thực và độ phù hợp của câu trả lời bằng cách dùng chính LLM làm giám khảo (LLM-as-a-judge)."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys\n",
            "PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))\n",
            "if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)\n",
            "\n",
            "from source.generation.rag_pipeline import TrafficRAGPipeline\n",
            "from source.evaluation.evaluator_ragas import RagasEvaluator\n",
            "from source.core.config import Settings\n",
            "\n",
            "settings = Settings()\n",
            "pipeline = TrafficRAGPipeline(settings)\n",
            "# Index data for retriever\n",
            "CHUNKS_PATH = os.path.join(PROJECT_ROOT, 'Data', 'chunks', 'traffic_chunks.json')\n",
            "pipeline.retriever.index_data(CHUNKS_PATH)\n",
            "\n",
            "evaluator = RagasEvaluator(settings, model_name='gemini-1.5-flash')\n",
            "GT_PATH = os.path.join(PROJECT_ROOT, 'source_research', 'Eval_System', 'dataset', 'ground_truth_traffic.csv')\n",
            "\n",
            "# Chạy đánh giá (sẽ mất thời gian và gọi API liên tục)\n",
            "print(\"🚀 Đang bắt đầu đánh giá chuyên sâu bằng Ragas...\")\n",
            "results = evaluator.run_evaluation(pipeline, GT_PATH, output_path='ragas_report.json')\n",
            "\n",
            "if results:\n",
            "    # Hiển thị bảng kết quả rút gọn\n",
            "    import pandas as pd\n",
            "    df = pd.read_json('ragas_report.json')\n",
            "    display(df[['question', 'faithfulness', 'answer_relevance', 'context_precision']].round(3))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Trực quan hóa kết quả\n",
            "Biểu đồ Radar giúp so sánh tổng thể các chỉ số của RAG."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "if os.path.exists('ragas_report.json'):\n",
            "    df = pd.read_json('ragas_report.json')\n",
            "    metrics = ['faithfulness', 'answer_relevance', 'context_precision', 'context_recall']\n",
            "    values = [df[m].mean() for m in metrics]\n",
            "\n",
            "    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()\n",
            "    values += values[:1]\n",
            "    angles += angles[:1]\n",
            "\n",
            "    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))\n",
            "    ax.fill(angles, values, color='blue', alpha=0.25)\n",
            "    ax.plot(angles, values, color='blue', linewidth=2)\n",
            "    ax.set_yticklabels([])\n",
            "    ax.set_xticks(angles[:-1])\n",
            "    ax.set_xticklabels(metrics)\n",
            "    plt.title('RAG Triad Analysis')\n",
            "    plt.show()"
        ]
    }
]

# Insert before final metadata close
nb['cells'].extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f" Đã thêm phần đánh giá nâng cao vào notebook: {notebook_path}")
