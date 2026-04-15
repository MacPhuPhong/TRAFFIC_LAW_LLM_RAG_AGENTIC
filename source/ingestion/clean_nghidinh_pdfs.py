# -*- coding: utf-8 -*-
"""
clean_nghidinh_pdfs.py — Giai đoạn 2: Trích xuất và làm sạch dữ liệu Nghị định (Nghị định 168, 336, 17)
==================================================================================================
Nhiệm vụ:
  1. Đọc file .pdf từ Data/raw/nghidinh/
  2. Loại bỏ nhiễu "about:blank" (header/footer) từ quá trình in web.
  3. Chuẩn hóa văn bản tiếng Việt sang NFC.
  4. Lưu kết quả ra file .txt trong Data/cleaned/nghidinh/
"""

import os
import re
import unicodedata
import pdfplumber
from pathlib import Path

# ---------------------------------------------------------------------------
# Cấu hình đường dẫn
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "Data" / "raw" / "nghidinh"
CLEANED_DIR = BASE_DIR / "Data" / "cleaned" / "nghidinh"

# ---------------------------------------------------------------------------
# Các biểu thức chính quy (Regex) để làm sạch
# ---------------------------------------------------------------------------

# Header dạng: "4/13/26, 5:57 PM about:blank"
RE_HEADER_NOISE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},.*about:blank$", re.MULTILINE)

# Footer dạng: "about:blank 52/52"
RE_FOOTER_NOISE = re.compile(r"^about:blank\s+\d+/\d+$", re.MULTILINE)

def clean_text(text: str) -> str:
    """
    Làm sạch text trích xuất từ PDF.
    """
    if not text:
        return ""
        
    # 1. Chuẩn hóa tiếng Việt (NFC)
    text = unicodedata.normalize("NFC", text)
    
    # 2. Xử lý từng dòng để lọc nhiễu
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Bỏ qua dòng trống
        if not line:
            continue
            
        # Kiểm tra xem có phải dòng nhiễu header/footer không
        if RE_HEADER_NOISE.match(line):
            continue
        if RE_FOOTER_NOISE.match(line):
            continue
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def table_to_markdown(table_data) -> str:
    """
    Chuyển đổi dữ liệu bảng (list of lists) thành chuỗi Markdown.
    """
    if not table_data:
        return ""
    
    # Lọc bỏ các dòng hoàn toàn rỗng
    table_data = [row for row in table_data if any(cell and cell.strip() for cell in row)]
    if not table_data:
        return ""

    md_lines = []
    for i, row in enumerate(table_data):
        # Làm sạch dữ liệu trong từng ô
        row_data = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
        
        md_row = "| " + " | ".join(row_data) + " |"
        md_lines.append(md_row)
        
        # Thêm dòng phân cách sau dòng tiêu đề
        if i == 0:
            separator = "|" + "|".join(["---"] * len(row_data)) + "|"
            md_lines.append(separator)
            
    return "\n".join(md_lines)

def process_pdfs():
    """
    Quét và xử lý toàn bộ file PDF trong thư mục raw/nghidinh.
    """
    os.makedirs(CLEANED_DIR, exist_ok=True)
    
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    print(f"[INFO] Tìm thấy {len(pdf_files)} file nghị định cần xử lý.")
    
    for pdf_path in pdf_files:
        print(f"  -> Đang xử lý: {pdf_path.name}...")
        
        full_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Lấy text toàn trang
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_page_text = clean_text(page_text)
                        if cleaned_page_text:
                            full_content.append(cleaned_page_text)
                    
                    # Trích xuất và định dạng bảng
                    extracted_tables = page.extract_tables()
                    for table in extracted_tables:
                        md_table = table_to_markdown(table)
                        if md_table:
                            full_content.append("\n\n" + md_table + "\n\n")
            
            # Lưu kết quả (.md)
            output_filename = pdf_path.stem + ".md"
            output_path = CLEANED_DIR / output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(full_content))
                
            print(f"   Hoàn tất! Đã lưu tại: Data/cleaned/nghidinh/{output_filename}")
            
        except Exception as e:
            print(f"   Lỗi khi xử lý {pdf_path.name}: {e}")

if __name__ == "__main__":
    process_pdfs()
