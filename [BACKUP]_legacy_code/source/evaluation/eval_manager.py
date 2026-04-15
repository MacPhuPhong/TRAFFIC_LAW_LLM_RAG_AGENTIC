import json
from typing import List, Dict

class EvaluationManager:
    def __init__(self):
        pass

    def calculate_precision_at_k(self, retrieved_docs: List[Dict], expected_article: str, k: int = 5) -> float:
        """
        Check if the expected article is within the top K retrieved documents.
        """
        top_k_docs = retrieved_docs[:k]
        hits = 0
        for doc in top_k_docs:
            # check if expected_article (e.g. "Điều 9") is in the chunk's metadata 'dieu'
            if expected_article in doc['chunk']['metadata']['dieu']:
                hits = 1
                break
        return float(hits)

    def evaluate_hallucination(self, answer: str, context: str) -> bool:
        """
        Simple heuristic check: Does the answer mention things NOT in the context?
        In a production system, this would use an NLI model or LLM-as-a-judge.
        For now, we use a basic keyword overlap check or placeholder.
        """
        # Placeholder for complex hallucination check
        return False # False means NO hallucination detected

    def run_eval_suite(self, pipeline_results: List[Dict], ground_truth: List[Dict]):
        results = []
        for i, res in enumerate(pipeline_results):
            gt = ground_truth[i]
            p_at_k = self.calculate_precision_at_k(res['references'], gt['expected_article'])
            
            results.append({
                "question": res['query'],
                "precision_at_5": p_at_k,
                "hallucination": self.evaluate_hallucination(res['answer'], " ".join([d['chunk']['content'] for d in res['references']]))
            })
            
        avg_p = sum([r['precision_at_5'] for r in results]) / len(results)
        return {
            "average_precision_at_5": avg_p,
            "detailed_results": results
        }
