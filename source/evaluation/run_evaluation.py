from traffic_rag.source.generation.rag_pipeline import TrafficRAGPipeline
from traffic_rag.source.evaluation.eval_manager import EvaluationManager
from source.core.config import Settings
import json

def main():
    settings = Settings()
    pipeline = TrafficRAGPipeline(settings)
    eval_mgr = EvaluationManager()
    
    # Load ground truth
    with open("/media/pphong/D:/Do_An_Tot_Nghiep/GitHub1/Retrieval_Information_System_With_Agentic_RAG-main/traffic_rag/data/eval/ground_truth.json", "r") as f:
        ground_truth = json.load(f)
        
    pipeline_results = []
    for item in ground_truth:
        print(f"Evaluating: {item['question']}")
        res = pipeline.run(item['question'])
        pipeline_results.append(res)
        
    summary = eval_mgr.run_eval_suite(pipeline_results, ground_truth)
    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
