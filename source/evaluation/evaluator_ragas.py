import os
import time
import json
import pandas as pd
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from source.generation.rag_pipeline import TrafficRAGPipeline
from source.core.config import Settings
from dotenv import load_dotenv

class RagasEvaluator:
    def __init__(self, settings: Settings, model_name: str = "gemini-flash-latest"):
        # Load environment variables for LangSmith and API Keys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', '.env'))
        load_dotenv(dotenv_path=env_path)

        self.settings = settings
        api_key = settings.api_key or os.getenv("API_KEY")

        self.evaluator_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0
        )
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        for metric in self.metrics:
            metric.llm = self.evaluator_llm

    def run_evaluation(self, pipeline: TrafficRAGPipeline, ground_truth_path: str, output_path: str = "evaluation_results.json"):
        """Chạy đánh giá end-to-end với checkpointing để chạy tiếp khi bị lỗi."""

        # 1. Load Ground Truth
        if ground_truth_path.endswith('.csv'):
            df_gt = pd.read_csv(ground_truth_path)
        else:
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                df_gt = pd.DataFrame(json.load(f))

        # CHECKPOINT: Đọc dữ liệu đã xử lý (nếu có)
        existing_results = []
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                print(f"📂 Tìm thấy {len(existing_results)} kết quả cũ. Đang chạy tiếp...")
            except Exception:
                pass

        processed_questions = {res['question'] for res in existing_results if 'answer' in res}
        print(f"🚀 Bắt đầu đánh giá {len(df_gt)} câu hỏi...")

        results = existing_results
        for i, row in df_gt.iterrows():
            question = row['question']
            if question in processed_questions:
                continue

            print(f"[{i+1}/{len(df_gt)}] Processing: {question[:50]}...")
            try:
                res = pipeline.run(question)
                results.append({
                    "question": question,
                    "answer": res['answer'],
                    "contexts": [doc['chunk']['content'] for doc in res['references']],
                    "ground_truth": row.get('expected_answer', row.get('expected_article', ''))
                })
                # Lưu checkpoint ngay sau mỗi câu
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"   ❌ Lỗi: {e}")
                continue
            time.sleep(2)

        # 2. Chấm điểm Ragas
        return self.evaluate_existing_results(results, output_path)

    def evaluate_existing_results(self, results: List[Dict], output_path: str = "ragas_report.json"):
        """Chấm điểm Ragas từ danh sách kết quả đã có sẵn."""
        if not results:
            print("❌ Không có kết quả nào để đánh giá.")
            return None

        print(f"📊 Đang chấm điểm Ragas trên {len(results)} kết quả...")
        dataset = Dataset.from_list(results)
        try:
            score = evaluate(dataset, metrics=self.metrics)
            df_score = score.to_pandas()
            df_score.to_json(output_path, orient="records", indent=2, force_ascii=False)
            print(f"✅ Đã chấm điểm xong! Lưu tại: {output_path}")
            return score
        except Exception as e:
            print(f"❌ Lỗi khi chấm điểm: {e}")
            return None


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    settings = Settings()
    pipeline = TrafficRAGPipeline(settings)
    evaluator = RagasEvaluator(settings)
    GT_PATH = os.path.abspath(os.path.join(os.getcwd(), 'source_research', 'Eval_System', 'dataset', 'ground_truth_traffic.csv'))
    evaluator.run_evaluation(pipeline, GT_PATH)
