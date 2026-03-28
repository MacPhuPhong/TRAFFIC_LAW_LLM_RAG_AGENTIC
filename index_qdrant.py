import os
import sys

# Đảm bảo import được source
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from source.core.config import Settings
from source.retrieval.hybrid_retriever import HybridRetriever

if __name__ == "__main__":
    print("Khởi tạo cấu hình và kết nối Qdrant...")
    settings = Settings()
    
    # Khởi tạo instance kết nối port 6334
    retriever = HybridRetriever(settings=settings, collection_name="Traffic_Law_Hybrid")
    
    chunk_path = "Data/chunks/traffic_chunks.json"
    if not os.path.exists(chunk_path):
        print(f"Không tìm thấy file {chunk_path}! Hãy chạy process_raw_pdfs.py trước.")
        sys.exit(1)
        
    print(f"Bắt đầu quy trình nạp dữ liệu từ {chunk_path}...")
    print("Ghi chú: Việc chuyển hóa 3575 chunks thành Vector có thể mất vài phút tùy cấu hình máy tính.")
    retriever.index_data(chunk_path)
    
    print("\n=> [THÀNH CÔNG] Đã nạp toàn bộ tri thức pháp luật vào Bộ não Qdrant!")
