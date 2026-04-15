import os
from traffic_rag.source.retrieval.hybrid_retriever import HybridRetriever
from source.core.config import Settings

def main():
    settings = Settings()
    retriever = HybridRetriever(settings)
    chunks_file = "/media/pphong/D:/Do_An_Tot_Nghiep/GitHub1/Retrieval_Information_System_With_Agentic_RAG-main/traffic_rag/data/traffic_chunks.json"
    
    if os.path.exists(chunks_file):
        retriever.index_data(chunks_file)
    else:
        print(f"Error: {chunks_file} not found. Run ingestion first.")

if __name__ == "__main__":
    main()
