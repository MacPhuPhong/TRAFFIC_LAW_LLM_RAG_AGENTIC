# -*- coding: utf-8 -*-
"""
clean_phuluc_docx.py — Giai đoạn 2: Trích xuất nội dung từ Phụ lục (Docx)
========================================================================
Nhiệm vụ:
  1. Duyệt các file .docx trong Data/raw/phuluc/
  2. Đọc nội dung văn bản thường (paragraphs).
  3. Đọc dữ liệu bảng biểu (tables) và chuyển đổi sang dạng Markdown để AI hiểu cấu trúc.
  4. Lưu kết quả gộp lại vào file .txt trong Data/cleaned/phuluc/
"""

import os
from pathlib import Path
from docx import Document

# ---------------------------------------------------------------------------
# Cấu hình đường dẫn
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "Data" / "raw" / "phuluc"
CLEANED_DIR = BASE_DIR / "Data" / "cleaned" / "phuluc"

def table_to_markdown(table) -> str:
    """
    Chuyển đổi một bảng trong python-docx thành chuỗi Markdown.
    """
    md_lines = []
    for i, row in enumerate(table.rows):
        row_data = []
        for cell in row.cells:
            # Loại bỏ ký tự xuống dòng dư thừa trong ô
            text = cell.text.replace("\n", " ").strip()
            row_data.append(text)
            
        md_row = "| " + " | ".join(row_data) + " |"
        md_lines.append(md_row)
        
        # Thêm dòng phân cách sau dòng tiêu đề (header row)
        if i == 0:
            separator = "|" + "|".join(["---"] * len(row_data)) + "|"
            md_lines.append(separator)
            
    return "\n".join(md_lines)

def process_docx_files():
    """
    Quét và xử lý toàn bộ file Docx trong thư mục raw/phuluc.
    """
    docx_files = list(RAW_DIR.rglob("*.docx"))
    print(f"[INFO] Tìm thấy {len(docx_files)} file phụ lục (docx) cần xử lý.")
    
    for docx_path in docx_files:
        rel_path = docx_path.relative_to(RAW_DIR)
        output_dir = CLEANED_DIR / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"  -> Đang xử lý: {rel_path}...")
        
        try:
            doc = Document(docx_path)
            full_text = []
            
            # Đọc từng thành phần trong văn bản theo thứ tự khai báo trong file
            for element in doc.element.body:
                if element.tag.endswith('p'):  # Đoạn văn (Paragraph)
                    from docx.text.paragraph import Paragraph
                    para = Paragraph(element, doc)
                    if para.text.strip():
                        full_text.append(para.text.strip())
                        
                elif element.tag.endswith('tbl'):  # Bảng biểu (Table)
                    from docx.table import Table
                    tbl = Table(element, doc)
                    md_table = table_to_markdown(tbl)
                    full_text.append("\n" + md_table + "\n")

            # Lưu kết quả (.md)
            output_filename = docx_path.stem + ".md"
            output_path = output_dir / output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(full_text))
                
            print(f"   Hoàn tất! Đã lưu tại: Data/cleaned/phuluc/{rel_path.parent}/{output_filename}")
            
        except Exception as e:
            print(f"   Lỗi khi xử lý {rel_path}: {e}")

if __name__ == "__main__":
    process_docx_files()
