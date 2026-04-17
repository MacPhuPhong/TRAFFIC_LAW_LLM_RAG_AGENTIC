# -*- coding: utf-8 -*-
"""
clean_thongtu_pdfs.py — Giai đoạn 2: Trích xuất và làm sạch dữ liệu Thông tư
========================================================================
Nhiệm vụ:
  1. Duyệt tất cả các thư mục con trong Data/raw/thongtu/
  2. Đọc các file .pdf.
  3. Loại bỏ nhiễu "about:blank" (header/footer) từ quá trình in web.
  4. Chuẩn hóa văn bản tiếng Việt sang NFC.
  5. Lưu kết quả ra file .txt trong Data/cleaned/thongtu/ theo cấu trúc thư mục con tương ứng.
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
RAW_DIR = BASE_DIR / "Data" / "raw" / "thongtu"
CLEANED_DIR = BASE_DIR / "Data" / "cleaned" / "thongtu"

# ---------------------------------------------------------------------------
# Các biểu thức chính quy (Regex) để làm sạch
# ---------------------------------------------------------------------------

# Header dạng: "4/13/26, 5:57 PM about:blank" hoặc tương tự
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
        
    text = unicodedata.normalize("NFC", text)
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if RE_HEADER_NOISE.match(line):
            continue
        if RE_FOOTER_NOISE.match(line):
            continue
            
        # Lọc nét đứt biểu mẫu và Checkbox -> Schema Text
        line = RE_FORM_DOTS.sub(" [Cần điền thông tin] ", line)
        line = RE_CHECKBOX.sub(" [Lựa chọn] ", line).strip()
        
        if not line or line == "[Cần điền thông tin]" or line == "[Lựa chọn]":
            continue
            
        cleaned_lines.append(line)
        
    # Nối các dòng bị ngắt giữa chừng
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

def process_pdfs():
    """
    Quét và xử lý toàn bộ file PDF trong thư mục raw/thongtu và các thư mục con.
    """
    pdf_files = list(RAW_DIR.rglob("*.pdf"))
    print(f"[INFO] Tìm thấy {len(pdf_files)} file thông tư cần xử lý.")
    
    for pdf_path in pdf_files:
        # Lấy đường dẫn tương đối để tạo cấu trúc thư mục tương ứng trong cleaned
        rel_path = pdf_path.relative_to(RAW_DIR)
        output_dir = CLEANED_DIR / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"  -> Đang xử lý: {rel_path}...")
        
        full_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Thiết lập cho bảng: Yêu cầu bảng phải có kẻ lưới thật sự (lines)
                    # Nếu file có bảng giả (text aligned), nó sẽ không trích xuất -> tránh làm hỏng text.
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
                    
                    # Trích xuất và định dạng bảng với setting khắt khe
                    extracted_tables = page.extract_tables(table_settings=strict_table_settings)
                    for table in extracted_tables:
                        md_table = table_to_markdown(table)
                        if md_table:
                            full_content.append("\n\n" + md_table + "\n\n")
            
            # Lưu kết quả (.md)
            output_filename = pdf_path.stem + ".md"
            output_path = output_dir / output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(full_content))
                
            print(f"   Hoàn tất! Đã lưu tại: Data/cleaned/thongtu/{rel_path.parent}/{output_filename}")
            
        except Exception as e:
            print(f"   Lỗi khi xử lý {rel_path}: {e}")

if __name__ == "__main__":
    process_pdfs()
