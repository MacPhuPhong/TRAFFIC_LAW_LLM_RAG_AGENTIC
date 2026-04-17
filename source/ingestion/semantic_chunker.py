# -*- coding: utf-8 -*-
"""
semantic_chunker.py — Bước 3 & 4: Cắt nhỏ tài liệu và Gán Metadata
========================================================================
Nhiệm vụ:
  1. Đọc các file Markdown sạch từ Data/cleaned/...
  2. Bóc tách theo đơn vị "Điều".
  3. Gắn nhãn (Metadata): Tên văn bản, Trạng thái (active/repealed).
  4. Chèn thêm cảnh báo vào chunk nếu văn bản đã bị thay thế (ví dụ NĐ 100/2019).
  5. Đổ ra Data/cleaned/semantic_chunks/ theo định dạng chuẩn hóa.
"""

import os
import re
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEANED_DIRS = {
    "luat": BASE_DIR / "Data" / "cleaned" / "luat",
    "nghidinh": BASE_DIR / "Data" / "cleaned" / "nghidinh",
    "thongtu": BASE_DIR / "Data" / "cleaned" / "thongtu",
}
OUTPUT_DIR = BASE_DIR / "Data" / "semantic_chunks"

# ---------------------------------------------------------------------------
# Cấu hình Kiến thức Pháp lý (Quy trình kiểm soát sự thay thế)
# ---------------------------------------------------------------------------
METADATA_RULES = {
    "luat35_db_2024": {"title": "Luật Đường bộ 2024", "status": "active", "topic": "Luật nền tảng"},
    "luat36_ttatgt_2024": {"title": "Luật Trật tự ATGT đường bộ 2024", "status": "active", "topic": "Luật nền tảng"},
    "luat23_db_2008": {"title": "Luật Giao thông đường bộ 2008", "status": "repealed", "warning": "LƯU Ý: VĂN BẢN NÀY ĐÃ HẾT HIỆU LỰC. Thay thế hoàn toàn bởi Luật 35/2024 và Luật 36/2024 từ 01/01/2025."},
    "nd168_2024_XuPhat_TruDiem_DB_baibo_nd100": {"title": "Nghị định 168/2024/NĐ-CP (Xử phạt)", "status": "active", "topic": "Xử phạt vi phạm"},
    "nd336_2025_xu_phat_van_tai": {"title": "Nghị định 336/2025/NĐ-CP (Xử phạt vận tải)", "status": "active", "topic": "Xử phạt vi phạm"},
    "nd100_2019_xu_phat": {"title": "Nghị định 100/2019/NĐ-CP", "status": "partially_repealed", "warning": "LƯU Ý: Mảng đường bộ của văn bản này đã bị thay thế bởi NĐ 168/2024 và NĐ 336/2025."},
    "nd123_2021_xu_phat": {"title": "Nghị định 123/2021/NĐ-CP", "status": "partially_repealed", "warning": "LƯU Ý: Đã bị bãi bỏ một phần bởi NĐ 168/2024."},
    "nd10_2020_kinh_doanh_van_tai": {"title": "Nghị định 10/2020/NĐ-CP", "status": "active", "topic": "Kinh doanh vận tải"},
}

def get_file_metadata(filename: str):
    """Lấy bộ quy tắc metadata dựa trên tên file"""
    key = filename.replace(".md", "")
    return METADATA_RULES.get(key, {"title": key, "status": "active"})

# Regex bắt đầu "Điều X."
RE_ARTICLE = re.compile(r"^Điều\s+(\d+)[.:]\s*(.*)$", re.IGNORECASE)

def chunk_markdown_file(file_path: Path, output_subdir: Path, file_doc_type: str):
    """
    Tách file thành nhiều chunk dựa trên các Điều.
    """
    if not file_path.exists():
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    doc_meta = get_file_metadata(file_path.name)
    
    current_article_id = "0"
    current_article_title = "Giới thiệu chung"
    current_chunk_lines = []
    
    # Hàm lưu chunk
    def save_current_chunk():
        if not current_chunk_lines:
            return
            
        content = "".join(current_chunk_lines).strip()
        if not content:
            return
            
        # Thêm warning nếu có
        if "warning" in doc_meta:
            content = f"> {doc_meta['warning']}\n\n{content}"
            
        chunk_data = {
            "metadata": {
                "source_file": file_path.name,
                "document_title": doc_meta["title"],
                "document_type": file_doc_type,
                "article_number": current_article_id,
                "article_title": current_article_title,
                "status": doc_meta["status"],
                "topic": doc_meta.get("topic", "Chung")
            },
            "content": content
        }
        
        # Lưu file
        safe_name = f"dieu_{current_article_id}.md" if current_article_id != "0" else "gioi_thieu.md"
        out_file = output_subdir / safe_name
        
        with open(out_file, "w", encoding="utf-8") as out_f:
            out_f.write("---\n")
            yaml.dump(chunk_data["metadata"], out_f, allow_unicode=True, default_flow_style=False)
            out_f.write("---\n\n")
            out_f.write(chunk_data["content"])
            
    # Xử lý từng dòng
    for line in lines:
        match = RE_ARTICLE.match(line.strip())
        if match:
            # Lưu chunk hiện tại lại
            save_current_chunk()
            # Bắt đầu chunk mới
            current_article_id = match.group(1)
            current_article_title = match.group(2).strip()
            current_chunk_lines = [line]
        else:
            current_chunk_lines.append(line)
            
    # Lưu chunk cuối
    save_current_chunk()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for doc_type, dir_path in CLEANED_DIRS.items():
        if not dir_path.exists():
            continue
            
        # Duyệt tất cả file md
        for file_path in dir_path.rglob("*.md"):
            print(f"[{doc_type}] Chunking: {file_path.name}...")
            # Tạo thư mục gốc dựa theo tên file
            output_subdir = OUTPUT_DIR / doc_type / file_path.stem
            os.makedirs(output_subdir, exist_ok=True)
            chunk_markdown_file(file_path, output_subdir, file_doc_type=doc_type)
            
    print("HOÀN TẤT. Semantic chunking đã xong.")

if __name__ == "__main__":
    main()
