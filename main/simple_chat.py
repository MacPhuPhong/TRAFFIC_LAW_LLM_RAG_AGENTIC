import os
import sys
import yaml
import json
from dotenv import load_dotenv

# Thêm thư mục gốc vào path để import 'source'
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Tự động tính toán đường dẫn tuyệt đối dự án
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PROJECT_ROOT = os.path.join(BASE_DIR, '..')

# Load environment variables từ file .env ở gốc dự án
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

from source.core.config import Settings
from source.retrieval.hybrid_retriever import HybridRetriever
import google.generativeai as genai

class AgenticTrafficChat:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.history = []
        
        # Cấu hình Gemini
        gemini_key = settings.api_key or os.getenv("API_KEY")
        genai.configure(api_key=gemini_key)
        self.model = genai.GenerativeModel('gemini-flash-latest')
        
        # Khởi tạo Retriever
        self.retriever = HybridRetriever(settings=settings, collection_name="Traffic_Law_Hybrid")
        self._load_bm25_data()
        
        # Tải Prompts từ YAML
        prompt_path = os.path.join(BASE_DIR, "source/core/traffic_prompts.yaml")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)["prompts"]

    def _load_bm25_data(self):
        chunk_path = os.path.join(BASE_DIR, "Data/chunks/traffic_chunks.json")
        if os.path.exists(chunk_path):
            with open(chunk_path, 'r', encoding='utf-8') as f:
                self.retriever.corpus_chunks = json.load(f)
                tokenized_corpus = [self.retriever._tokenize(c['content']) for c in self.retriever.corpus_chunks]
                from rank_bm25 import BM25Okapi
                self.retriever.bm25 = BM25Okapi(tokenized_corpus)
        else:
            print("⚠️ Cảnh báo: Không tìm thấy traffic_chunks.json. BM25 Search sẽ bị lỗi.")

    def _call_gemini(self, prompt_name, **kwargs):
        prompt_data = self.prompts[prompt_name]
        
        # Build prompt string from messages
        full_prompt = ""
        for msg in prompt_data["messages"]:
            content = msg["content"].format(**kwargs)
            if msg["role"] == "system":
                full_prompt += f"SYSTEM:\n{content}\n\n"
            else:
                full_prompt += f"USER:\n{content}\n\n"
        
        response = self.model.generate_content(full_prompt)
        return response.text.strip()

    def _format_history(self):
        if not self.history:
            return "Trống"
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.history[-4:]])

    def run_pipeline(self, user_query):
        # 1. Phân loại ý định (Classification)
        print("🧠 Đang phân tích ý định...")
        category = self._call_gemini("classify_query", query=user_query)
        
        if "0" in category:
            # Thông tin hệ thống / Chào hỏi
            answer = self._call_gemini("information_query", query=user_query)
            self.history.append({"role": "User", "content": user_query})
            self.history.append({"role": "AI", "content": answer})
            return answer
            
        elif "2" in category:
            # Không liên quan / Thô tục
            answer = self._call_gemini("invalid_query", query=user_query)
            return answer

        # 2. Câu hỏi Luật (Category 1)
        print("🔎 Đang sinh đa truy vấn tìm kiếm...")
        history_str = self._format_history()
        search_queries_str = self._call_gemini("query_generator", history=history_str, query=user_query)
        search_queries = [q.strip() for q in search_queries_str.split("\n") if q.strip()][:3]
        if not search_queries: search_queries = [user_query]

        # 3. Tìm kiếm và tổng hợp kết quả (Search & Aggregate)
        print(f"📡 Đang tìm kiếm dựa trên {len(search_queries)} hướng truy vấn...")
        all_results = []
        seen_chunks = set()
        
        for sq in search_queries:
            results = self.retriever.search(sq, top_k=3, alpha=0.6)
            for r in results:
                chunk_id = r['chunk']['metadata'].get('id', r['chunk']['content'][:50])
                if chunk_id not in seen_chunks:
                    all_results.append(r)
                    seen_chunks.add(chunk_id)
        
        # Sắp xếp lại theo điểm số cao nhất
        all_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:6]

        if not all_results:
            return "Tôi xin lỗi, thông tin này không có trong cơ sở dữ liệu luật hiện tại của tôi."

        # 4. Tạo ngữ cảnh và Trả lời (QA)
        context_str = ""
        for i, res in enumerate(all_results):
            chunk = res['chunk']
            m = chunk['metadata']
            context_str += f"[{i+1}] {m.get('ten_van_ban', 'Luật')} | {m.get('dieu', 'Điều ?')}\nNội dung: {chunk['content']}\n\n"

        print("⚖️ Đang nhờ 'Luật sư AI' tư vấn...")
        final_answer = self._call_gemini("traffic_qa", context=context_str, history=history_str, query=user_query)
        
        # Lưu vào lịch sử
        self.history.append({"role": "User", "content": user_query})
        self.history.append({"role": "AI", "content": final_answer})
        
        return final_answer

def main():
    settings = Settings()
    try:
        agent = AgenticTrafficChat(settings)
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Agent: {e}")
        return

    print("\n" + "="*50)
    print("🚦 HỆ THỐNG TRỢ LÝ LUẬT GIAO THÔNG (AGENTIC V2) 🚦")
    print("="*50)
    print("Gõ 'q' để thoát. Hệ thống hiện đã có 'Trí nhớ'!\n")

    while True:
        user_input = input("😎 User: ")
        if user_input.lower() in ['q', 'quit']:
            break
        if not user_input.strip():
            continue

        try:
            response = agent.run_pipeline(user_input)
            print(f"\n🚀 Trợ lý Luật:\n{response}\n")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()
