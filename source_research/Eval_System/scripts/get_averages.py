import pandas as pd

df = pd.read_csv('/media/pphong/D:/Do_An_Tot_Nghiep/GitHub1/traffic_rag/source_research/Eval_System/dataset/eval_responses_final_eval.csv')
print("Cosine Pure:", df['cosine_pure'].mean())
print("Cosine RAG:", df['cosine_rag'].mean())
print("Precision@5:", df['precision_at_5'].mean())
print("Rouge-L Pure:", df['rouge_l_pure'].mean())
print("Rouge-L RAG:", df['rouge_l_rag'].mean())
print("F1 Pure:", df['f1_pure'].mean())
print("F1 RAG:", df['f1_rag'].mean())
print("Latency:", df['latency'].mean())

