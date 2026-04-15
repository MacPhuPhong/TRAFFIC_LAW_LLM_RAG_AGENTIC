# -*- coding: utf-8 -*-
"""
metadata_parser.py — Giai đoạn 1: Phân loại và trích xuất Metadata cơ bản
=======================================================================
Mục tiêu:
  - Nhận diện loại văn bản (Luật, Nghị định, Thông tư, Quy chuẩn, Phụ lục).
  - Trích xuất mã hiệu văn bản từ tên file hoặc đường dẫn.
  - Xác định ngữ cảnh thư mục (vd: thuộc thông tư nào).
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Cấu hình quy tắc nhận diện
# ---------------------------------------------------------------------------

# Map tên thư mục sang thông tin định danh
FOLDER_METADATA_MAP = {
    "tt12": {"so_thong_tu": "12/2017/TT-BGTVT", "chu_de": "Đào tạo, sát hạch lái xe"},
    "tt13": {"so_thong_tu": "13/2025/TT-BCA", "chu_de": "Tuần tra, kiểm soát giao thông"},
    "tt155": {"so_thong_tu": "155/2025/TT-BTC", "chu_de": "Lệ phí đăng ký xe"},
    "tt48": {"so_thong_tu": "48/2024/TT-BGTVT", "chu_de": "Quy chuẩn an toàn kỹ thuật"},
    # Có thể bổ sung thêm các folder khác tại đây
}

# Quy tắc nhận diện loại văn bản dựa trên từ khóa trong tên file
DOC_TYPE_RULES = [
    (r"(?i)Luật|Luat", "luat"),
    (r"(?i)Nghị định|Nghi dinh|NĐ", "nghi_dinh"),
    (r"(?i)Thông tư|Thong tu|TT", "thong_tu"),
    (r"(?i)QCVN|TCVN", "quy_chuan"),
    (r"(?i)Phụ lục|Phu luc", "phu_luc"),
    (r"(?i)Mẫu số|Mau so", "mau_bieu"),
    (r"(?i)Biên bản|Bien ban|BB", "bien_ban"),
]

# Regex trích xuất số hiệu (ví dụ: 12/2017/TT-BGTVT)
RE_SO_HIEU = re.compile(r"(\d+/\d{4}/(?:TT|ND|QD|NQ|QCVN|TCVN)[-\w]*)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Hàm xử lý chính
# ---------------------------------------------------------------------------

def parse_metadata(file_path: str) -> Dict:
    """
    Phân tích một đường dẫn file để trích xuất metadata.
    """
    path = Path(file_path)
    filename = path.name
    stem = path.stem
    
    # 1. Xác định loại văn bản
    doc_type = "khac"
    for pattern, label in DOC_TYPE_RULES:
        if re.search(pattern, filename):
            doc_type = label
            break
            
    # 2. Trích xuất số hiệu từ tên file
    so_hieu = ""
    match_so_hieu = RE_SO_HIEU.search(filename)
    if match_so_hieu:
        so_hieu = match_so_hieu.group(1).upper()
        
    # 3. Tra cứu ngữ cảnh từ thư mục
    folder_context = {}
    for part in path.parts:
        part_lower = part.lower()
        if part_lower in FOLDER_METADATA_MAP:
            folder_context = FOLDER_METADATA_MAP[part_lower]
            folder_context["folder_name"] = part_lower
            break
            
    # 4. Trích xuất số phụ lục nếu là loại phu_luc (vd: từ file "Phu luc 01.docx")
    so_phu_luc = ""
    if doc_type == "phu_luc":
        match_pl = re.search(r"(?i)phụ lục\s*([\d\.\-]+)", stem)
        if match_pl:
            so_phu_luc = match_pl.group(1)

    # 5. Tổng hợp Metadata
    metadata = {
        "filename": filename,
        "ten_van_ban": stem,
        "loai_van_ban": doc_type,
        "so_hieu": so_hieu or folder_context.get("so_thong_tu", ""),
        "so_phu_luc": so_phu_luc,
        "chu_de": folder_context.get("chu_de", "Chưa phân loại"),
        "folder": folder_context.get("folder_name", "unknown"),
        "path_goc": str(path.absolute()),
        "format": path.suffix.lower().replace(".", ""),
    }
    
    return metadata

# --- Test nhanh ---
if __name__ == "__main__":
    test_paths = [
        "Data/raw/luat/luat34_ttatgt_2024.pdf",
        "Data/raw/phuluc/tt12/Phu luc.docx",
        "Data/raw/thongtu/banglai/TT12_2025_BCA_QuyTrinhCap_GPLX.pdf",
        "Data/raw/phuluc/tt48/01. QCVN 04-2024.docx"
    ]
    
    print(f"{'FILE':<40} | {'LOẠI':<15} | {'SỐ HIỆU':<20}")
    print("-" * 85)
    for p in test_paths:
        meta = parse_metadata(p)
        print(f"{meta['filename']:<40} | {meta['loai_van_ban']:<15} | {meta['so_hieu']:<20}")
