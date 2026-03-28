import os
import sys
from dotenv import load_dotenv

# Đảm bảo import được source
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load enviroment variables từ cấp thư mục cha (GitHub1/.env)
load_dotenv(dotenv_path="../.env")

from source.core.config import Settings
from source.retrieval.hybrid_retriever import HybridRetriever
import google.generativeai as genai

def chat():
    settings = Settings()
    
    gemini_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ LỖI: Không tìm thấy API_KEY trong file ../.env. Vui lòng thêm cấu hình API_KEY=AIzaSy... để tiếp tục.")
        return
        
    genai.configure(api_key=gemini_key)
    # Khởi tạo model AI
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    print("⏳ Khởi tạo Trợ lý AI và Kết nối Qdrant Database (Hybrid Search)...")
    try:
        retriever = HybridRetriever(settings=settings, collection_name="Traffic_Law_Hybrid")
        # Phải bắt buộc có data chunks để build cái BM25 index (Lexical search)
        chunk_path = "Data/chunks/traffic_chunks.json"
        if os.path.exists(chunk_path):
            import json
            with open(chunk_path, 'r', encoding='utf-8') as f:
                retriever.corpus_chunks = json.load(f)
                # Re-build BM25 trên RAM từ JSON
                tokenized_corpus = [retriever._tokenize(c['content']) for c in retriever.corpus_chunks]
                from rank_bm25 import BM25Okapi
                retriever.bm25 = BM25Okapi(tokenized_corpus)
        else:
            print("Cảnh báo: Không tìm thấy Data/chunks/traffic_chunks.json, tìm kiếm BM25 sẽ gặp lỗi.")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Retriever: {e}")
        return
        
    print("\n" + "="*50)
    print("🚦 HỆ THỐNG RAG TƯ VẤN LUẬT GIAO THÔNG VÀO HOẠT ĐỘNG 🚦")
    print("="*50)
    print("Gõ 'q' hoặc 'quit' để thoát.\n")
    
    while True:
        query = input("😎 User: ")
        if query.lower() in ['q', 'quit']:
            break
            
        if not query.strip():
            continue
            
        print("🔎 Đang rà soát thư viện Pháp luật...")
        results = retriever.search(query, top_k=5, alpha=0.6)
        
        if not results:
            print("🤖 AI: Hệ thống không tìm thấy điều luật nào khớp với câu hỏi của bạn.\n")
            continue
            
        # Tổ hợp Context RAG
        context_str = ""
        for i, res in enumerate(results):
            chunk = res['chunk']
            meta = chunk['metadata']
            score = res['score']
            context_str += f"[{i+1}] (Độ tin cậy: {score:.2f}) {meta.get('ten_van_ban', '')} | {meta.get('chuong', '')} | {meta.get('dieu', '')} | Khoản {meta.get('khoan', '')}\nNội dung: {chunk['content']}\n\n"
            
        # Để in ra màn hình cho vui mắt (Tracing)
        print(f"✅ Đã kéo về {len(results)} mảnh luật liên quan. Đang nhờ Gemini đúc kết câu trả lời...")
        
        prompt = f"""
Bạn là chuyên gia tư vấn Luật Giao thông xuất sắc. 
Hãy trả lời câu hỏi sau bằng Tiếng Việt dựa DUY NHẤT vào "Tài Liệu Pháp Lý" được cung cấp bên dưới.
Nguyên tắc:
1. Bạn tuyệt đối không được tự bịa ra thông tin.
2. Bắt buộc phải trích dẫn tên văn bản (ví dụ Nghị định 100) và số Điều, Khoản gốc từ tài liệu.
3. Nếu không có thông tin, hãy mạnh dạn nói rõ là tài liệu hiện tại không đề cập đến.

[Tài liệu Pháp Lý]
{context_str}

[Câu hỏi người dùng]
{query}
"""
        response = model.generate_content(prompt)
        print(f"\n🚀 AI_Luật_Sư:\n{response.text}\n")
        print("-" * 50)

if __name__ == "__main__":
    chat()
