import os
import sys
import yaml
import json
import time
import google.generativeai as genai

# Tự động thêm thư mục gốc dự án vào path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from source.core.config import Settings
from source.retrieval.hybrid_retriever import HybridRetriever
from rank_bm25 import BM25Okapi

class TrafficRAGPipeline:
    def __init__(self, settings: Settings, model_name: str = 'gemini-flash-latest'):
        self.settings = settings
        self.model_name = model_name

        # 1. Configure Gemini
        api_key = settings.api_key or os.getenv("API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        # 2. Initialize Retriever
        self.retriever = HybridRetriever(settings=settings, collection_name="Traffic_Law_Hybrid")
        self._load_data()

        # 3. Load Prompts
        current_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(current_dir, '..', 'core', 'traffic_prompts.yaml')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)['prompts']

    def _load_data(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        chunks_path = os.path.join(project_root, 'Data', 'chunks', 'traffic_chunks.json')

        if os.path.exists(chunks_path):
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.retriever.corpus_chunks = json.load(f)
                tokenized_corpus = [self.retriever._tokenize(c['content']) for c in self.retriever.corpus_chunks]
                self.retriever.bm25 = BM25Okapi(tokenized_corpus)
        else:
            print(f"⚠️ Warning: Chunks not found at {chunks_path}")

    def _call_gemini(self, prompt_name, **kwargs):
        prompt_data = self.prompts[prompt_name]
        full_prompt = ""
        for msg in prompt_data["messages"]:
            content = msg["content"].format(**kwargs)
            role = "SYSTEM" if msg["role"] == "system" else "USER"
            full_prompt += f"{role}:\n{content}\n\n"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(full_prompt)
                return response.text.strip()
            except Exception as e:
                if "429" in str(e):
                    import re
                    match = re.search(r'retry in (\d+)', str(e))
                    wait = int(match.group(1)) if match else 60
                    wait = min(wait + 5, 120)
                    print(f"   ⏳ Rate limit. Chờ {wait}s (lần {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Vượt quá số lần retry khi gọi '{prompt_name}'.")

    def run(self, query: str):
        # 1. Query Rewrite
        rewrite_key = 'query_generator' if 'query_generator' in self.prompts else 'query_rewrite'
        search_queries_raw = self._call_gemini(rewrite_key, history="Trống", query=query)
        search_queries = [q.strip() for q in search_queries_raw.split("\n") if q.strip()][:3]
        if not search_queries:
            search_queries = [query]

        # 2. Retrieval
        all_results = []
        seen_ids = set()
        for sq in search_queries:
            results = self.retriever.search(sq, top_k=3)
            for r in results:
                cid = r['chunk']['metadata'].get('id', r['chunk']['content'][:50])
                if cid not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(cid)

        top_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:5]

        # 3. Format Context
        context_str = ""
        for i, res in enumerate(top_results):
            chunk = res['chunk']
            m = chunk['metadata']
            context_str += f"[{i+1}] {m.get('ten_van_ban', 'Luật')} | {m.get('dieu', '?')}\nNội dung: {chunk['content']}\n\n"

        # 4. Generate Answer
        answer = self._call_gemini("traffic_qa", context=context_str, history="Trống", query=query)

        return {
            "query": query,
            "answer": answer,
            "references": top_results
        }

if __name__ == "__main__":
    s = Settings()
    p = TrafficRAGPipeline(s)
    print(p.run("Nồng độ cồn xe máy phạt bao nhiêu?")['answer'])
