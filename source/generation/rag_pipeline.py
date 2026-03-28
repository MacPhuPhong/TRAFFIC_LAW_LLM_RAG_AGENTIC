import yaml
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from traffic_rag.source.retrieval.hybrid_retriever import HybridRetriever
from source.generate.generate import Gemini_Generate
from source.core.config import Settings

from source.model.generate_model import Gemini

class TrafficRAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.retriever = HybridRetriever(settings)
        # Reuse existing LLM logic (Ollama/Gemini)
        gemini_m = Gemini(settings)
        self.llm_engine = Gemini_Generate(gemini_m, settings)
        self.model = self.llm_engine._get_model(temperature=0.0)
        
        # Load prompts
        current_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(current_dir, '..', 'core', 'traffic_prompts.yaml')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompts_data = yaml.safe_load(f)

    def _get_prompt(self, section: str) -> ChatPromptTemplate:
        messages = self.prompts_data['prompts'][section]['messages']
        return ChatPromptTemplate.from_messages(
            [(msg['role'], msg['content']) for msg in messages]
        )

    def run(self, query: str):
        # 1. Query Rewrite
        rewrite_prompt = self._get_prompt('query_rewrite')
        rewrite_chain = rewrite_prompt | self.model | StrOutputParser()
        search_query = rewrite_chain.invoke({"query": query}).strip()
        print(f"Rewritten Query: {search_query}")
        
        # 2. Hybrid Retrieval
        retrieved_docs = self.retriever.search(search_query, top_k=5)
        
        # 3. Format Context
        context_parts = []
        for i, res in enumerate(retrieved_docs):
            chunk = res['chunk']
            meta = chunk['metadata']
            context_parts.append(
                f"[{i+1}] Tài liệu: {meta['ten_van_ban']}\n"
                f"Vị trí: {meta['chuong']}, {meta['dieu']}, Khoản {meta['khoan']}, Điểm {meta['diem']}\n"
                f"Nội dung: {chunk['content']}"
            )
        context_str = "\n\n".join(context_parts)
        
        # 4. Generate Answer
        qa_prompt = self._get_prompt('traffic_qa')
        qa_chain = qa_prompt | self.model | StrOutputParser()
        
        # Set temperature to 0.0 (already done in _get_model if config is updated)
        # Here we just invoke
        answer = qa_chain.invoke({
            "context": context_str,
            "query": query
        })
        
        return {
            "query": query,
            "search_query": search_query,
            "answer": answer,
            "references": retrieved_docs
        }

if __name__ == "__main__":
    from source.core.config import Settings
    settings = Settings()
    # Mock chunks must be indexed first
    pipeline = TrafficRAGPipeline(settings)
    # Important: index_data must have been called separately once
    # For testing, we assume indexing is done.
    
    # Ensure rank_bm25 is re-loaded
    pipeline.retriever.index_data("/media/pphong/D:/Do_An_Tot_Nghiep/GitHub1/Retrieval_Information_System_With_Agentic_RAG-main/traffic_rag/data/traffic_chunks.json")
    
    result = pipeline.run("Mức phạt nồng độ cồn xe máy?")
    print("\n=== ANSWER ===")
    print(result['answer'])
