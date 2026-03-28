import os
import json
import pdfplumber
import sys

# Thêm thư mục gốc vào path để import 'source'
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from source.chunking.hierarchical_splitter import HierarchicalLegalSplitter

def process_pdfs(data_raw_dir, data_cleaned_dir, output_json):
    all_chunks = []
    
    if not os.path.exists(data_raw_dir):
        print(f"Thư mục '{data_raw_dir}' không tồn tại.")
        return
        
    pdf_files = [f for f in os.listdir(data_raw_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"Không tìm thấy file PDF nào trong '{data_raw_dir}'.")
        return

    os.makedirs(data_cleaned_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    for filename in pdf_files:
        pdf_path = os.path.join(data_raw_dir, filename)
        base_name = os.path.splitext(filename)[0]
        cleaned_path = os.path.join(data_cleaned_dir, f"{base_name}.txt")
        
        print(f"Đang xử lý file: {filename}...")
        
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        
        # Làm sạch cơ bản: Xóa các dòng chỉ chứa số trang hoặc khoảng trắng thừa
        lines = full_text.splitlines()
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Bỏ qua những dòng quá ngắn chỉ là số trang
            if line.isdigit() and len(line) < 4: 
                continue
            if line:
                cleaned_lines.append(line)
                
        cleaned_text = "\n".join(cleaned_lines)

        # Lưu lại bản text sạch vào thư mục cleaned
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f" -> Đã lưu bản text sạch vào {cleaned_path}")

        # Cắt chunk theo cấu trúc Luật
        splitter = HierarchicalLegalSplitter(document_name=filename)
        chunks = splitter.split_text(cleaned_text)
        print(f" -> Đã băm {len(chunks)} chunks pháp lý cho file {filename}")
        
        all_chunks.extend(chunks)

    # Lưu kết quả chunks phục vụ VectorDB
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"\n=> HOÀN TẤT! Đã lưu tổng cộng {len(all_chunks)} chunks vào {output_json}")


if __name__ == "__main__":
    # Tự động tính toán đường dẫn tuyệt đối dựa trên vị trí file script
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    RAW_DIR = os.path.join(BASE_DIR, "Data/raw")
    CLEANED_DIR = os.path.join(BASE_DIR, "Data/cleaned")
    JSON_OUT = os.path.join(BASE_DIR, "Data/chunks/traffic_chunks.json")
    
    process_pdfs(RAW_DIR, CLEANED_DIR, JSON_OUT)
