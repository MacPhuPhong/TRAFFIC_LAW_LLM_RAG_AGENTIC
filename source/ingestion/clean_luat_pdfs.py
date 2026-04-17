# -*- coding: utf-8 -*-
"""
clean_luat_pdfs.py — Giai đoạn 2: Trích xuất và làm sạch dữ liệu Luật (Luật 34, 35)
==============================================================================
Nhiệm vụ:
  1. Đọc file .pdf từ Data/raw/luat/
  2. Loại bỏ nhiễu "about:blank" (header/footer) từ quá trình in web.
  3. Chuẩn hóa văn bản tiếng Việt sang NFC.
  4. Lưu kết quả ra file .txt trong Data/cleaned/luat/
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
RAW_LUAT_DIR = BASE_DIR / "Data" / "raw" / "luat"
CLEANED_LUAT_DIR = BASE_DIR / "Data" / "cleaned" / "luat"

# ---------------------------------------------------------------------------
# Các biểu thức chính quy (Regex) để làm sạch
# ---------------------------------------------------------------------------

# Header dạng: "4/13/26, 5:35 PM about:blank"
RE_HEADER_NOISE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},.*about:blank$", re.MULTILINE)

# Footer dạng: "about:blank 52/52"
RE_FOOTER_NOISE = re.compile(r"^about:blank\s+\d+/\d+$", re.MULTILINE)

# Biểu mẫu rỗng: Cắt các nét đứt ". . ." hoặc "___" (từ 5 ký tự trở lên)
RE_FORM_DOTS = re.compile(r"(\.{5,}|_{5,})")
RE_CHECKBOX = re.compile(r"([☐☑\uf06f])")

def clean_text(text: str) -> str:
    """
    Làm sạch text trích xuất từ PDF: loại nhiễu, nối câu bị gãy, lọc biểu mẫu.
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
            
        # Lọc nét đứt biểu mẫu và Checkbox -> Schema Text
        line = RE_FORM_DOTS.sub(" [Cần điền thông tin] ", line)
        line = RE_CHECKBOX.sub(" [Lựa chọn] ", line).strip()
        
        if not line or line == "[Cần điền thông tin]" or line == "[Lựa chọn]":
            continue
            
        # Thêm dòng vào danh sách nếu là nội dung hợp lệ
        cleaned_lines.append(line)
        
    # 3. Nối các dòng bị ngắt giữa chừng
    merged_lines = []
    for i, line in enumerate(cleaned_lines):
        if not merged_lines:
            merged_lines.append(line)
            continue
            
        prev_line = merged_lines[-1]
        
        # Nếu dòng trước kết thúc bằng dấu tiếng Việt hoặc chữ cái, và dòng hiện tại bắt đầu bằng chữ thường
        if re.search(r'[a-zA-ZáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ,;]$', prev_line[-1:]) \
           and (line[0].islower() or line[0] in 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'):
            merged_lines[-1] = prev_line + " " + line
        else:
            merged_lines.append(line)
            
    return "\n".join(merged_lines)

def table_to_markdown(table_data) -> str:
    """
    Chuyển đổi dữ liệu bảng (list of lists) thành chuỗi Markdown, áp dụng Flattening cho merged cells.
    """
    if not table_data:
        return ""
    
    # Lọc bỏ các dòng hoàn toàn rỗng
    table_data = [row for row in table_data if any(cell and str(cell).strip() for cell in row)]
    if not table_data:
        return ""

    flattened_data = []
    # Khởi tạo ma trận lưu vết để nội suy (flatten merged cells)
    last_seen = [""] * len(table_data[0]) if table_data else []
    
    for row in table_data:
        new_row = []
        for j, cell in enumerate(row):
            cell_str = str(cell).replace("\n", " ").strip() if cell else ""
            
            # Khôi phục ngữ cảnh (flatten) nếu ô hiện tại bị rỗng do gộp theo chiều dọc
            if not cell_str and j < len(last_seen) and last_seen[j]:
                cell_str = last_seen[j]
            elif cell_str:
                if j < len(last_seen):
                    last_seen[j] = cell_str
                    
            new_row.append(cell_str)
        flattened_data.append(new_row)

    md_lines = []
    for i, row in enumerate(flattened_data):
        md_row = "| " + " | ".join(row) + " |"
        md_lines.append(md_row)
        
        # Thêm dòng phân cách sau dòng tiêu đề
        if i == 0:
            separator = "|" + "|".join(["---"] * len(row)) + "|"
            md_lines.append(separator)
            
    return "\n".join(md_lines)

def process_law_pdfs():
    """
    Quét và xử lý toàn bộ file PDF trong thư mục luật.
    """
    os.makedirs(CLEANED_LUAT_DIR, exist_ok=True)
    
    pdf_files = list(RAW_LUAT_DIR.glob("*.pdf"))
    print(f"[INFO] Tìm thấy {len(pdf_files)} file luật cần xử lý.")
    
    for pdf_path in pdf_files:
        print(f"  -> Đang xử lý: {pdf_path.name}...")
        
        full_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # 1. Thiết lập bảng khắt khe để tránh table giả
                    strict_table_settings = {
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "snap_tolerance": 3,
                        "join_tolerance": 3
                    }
                    
                    # Lấy text toàn trang
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_page_text = clean_text(page_text)
                        if cleaned_page_text:
                            full_content.append(cleaned_page_text)
                    
                    # 3. Trích xuất và định dạng bảng
                    extracted_tables = page.extract_tables(table_settings=strict_table_settings)
                    for table in extracted_tables:
                        md_table = table_to_markdown(table)
                        if md_table:
                            full_content.append("\n\n" + md_table + "\n\n")
            
            # Lưu kết quả (.md)
            output_filename = pdf_path.stem + ".md"
            output_path = CLEANED_LUAT_DIR / output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(full_content))
                
            print(f"   Hoàn tất! Đã lưu tại: Data/cleaned/luat/{output_filename}")
            
        except Exception as e:
            print(f"   Lỗi khi xử lý {pdf_path.name}: {e}")

if __name__ == "__main__":
    process_law_pdfs()
