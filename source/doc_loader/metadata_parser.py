# -*- coding: utf-8 -*-
"""
metadata_parser.py — Parse metadata từ đường dẫn & tên file
=============================================================
Nhận diện:
  - Folder (tt12, tt13, tt35, tt48) → số thông tư
  - Tên file  → loại tài liệu, số phụ lục, số hiệu, năm ban hành
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping folder → thông tin thông tư
# ---------------------------------------------------------------------------
FOLDER_MAP: Dict[str, Dict] = {
    "tt12": {"so_thong_tu": "12", "loai": "thong_tu", "ten_thong_tu": "Thông tư 12"},
    "tt13": {"so_thong_tu": "13", "loai": "thong_tu", "ten_thong_tu": "Thông tư 13"},
    "tt35": {"so_thong_tu": "35", "loai": "thong_tu", "ten_thong_tu": "Thông tư 35"},
    "tt48": {"so_thong_tu": "48", "loai": "thong_tu", "ten_thong_tu": "Thông tư 48"},
}

# ---------------------------------------------------------------------------
# Patters nhận diện loại tài liệu (theo thứ tự ưu tiên — khớp trước sẽ thắng)
# ---------------------------------------------------------------------------
DOC_TYPE_PATTERNS = [
    # Quy chuẩn kỹ thuật quốc gia (QCVN) — ưu tiên cao nhất
    (r"(?i)QCVN", "quy_chuan_ky_thuat"),
    # Phụ lục tổng hợp
    (r"(?i)ph[uụ]\s*l[uụ]c|Phu\s*luc", "phu_luc"),
    # Biên bản các loại
    (r"(?i)bi[eê]n\s*b[aả]n|BB\b", "bien_ban"),
    # Mẫu biểu / form
    (r"(?i)m[aẫ]u\s*s[oố]|MAU\b|m[aẫ]u\b", "mau_bieu"),
    # Quyết định
    (r"(?i)quy[eế]t\s*[đd][iị]nh|QĐ\b", "quyet_dinh"),
    # Kế hoạch
    (r"(?i)k[eế]\s*ho[aạ]ch|KH\b", "ke_hoach"),
    # Báo cáo
    (r"(?i)b[áa]o\s*c[áa]o|BC\b", "bao_cao"),
    # Thông báo
    (r"(?i)th[ôo]ng\s*b[áa]o|TB\b", "thong_bao"),
    # Sơ đồ
    (r"(?i)s[oơ]\s*[đd][oồ]|SD\b", "so_do"),
    # Đề nghị / trưng cầu
    (r"(?i)[đd][eề]\s*ngh[iị]|tr[uư]ng\s*c[aầ]u", "de_nghi"),
    # Sổ theo dõi
    (r"(?i)s[oổ]\s*theo\s*d[oõ]i", "so_theo_doi"),
    # Bìa / cover
    (r"(?i)b[iì]a\b", "bia"),
]

# ---------------------------------------------------------------------------
# Pattern tách số phụ lục dạng "1.1", "1.2", "2.1" ở đầu tên file
# ---------------------------------------------------------------------------
RE_SO_PHU_LUC = re.compile(r"^(\d+\.\d+(?:\.\d+)?)[.\s]")

# Pattern số hiệu văn bản: "Mẫu số 01", "QCVN 04-2024", "01/2024/TT-BCA"
RE_SO_HIEU_MAU = re.compile(r"(?i)m[aẫ]u\s*s[oố]\s*(\d+[.\-\w]*)")
RE_SO_HIEU_QCVN = re.compile(r"(?i)QCVN\s*([\d\-]+(?:-\d{4})?)")
RE_SO_HIEU_TT = re.compile(r"(\d+/\d{4}/(?:TT|ND|QD|NQ)-\w+)")

# Pattern extract năm (4 chữ số 19xx hoặc 20xx)
RE_NAM = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Pattern số thứ tự đầu tên file: "01.", "1.", "1.1."
RE_STT = re.compile(r"^(\d+(?:\.\d+)*)[.\s]")


def _detect_doc_type(filename_no_ext: str) -> str:
    """Nhận diện loại tài liệu từ tên file."""
    for pattern, doc_type in DOC_TYPE_PATTERNS:
        if re.search(pattern, filename_no_ext):
            return doc_type
    return "khac"


def _extract_so_hieu(filename_no_ext: str) -> str:
    """Extract mã/số hiệu văn bản từ tên file."""
    # Mẫu số XX
    m = RE_SO_HIEU_MAU.search(filename_no_ext)
    if m:
        return f"Mẫu số {m.group(1)}"

    # QCVN XX-XXXX
    m = RE_SO_HIEU_QCVN.search(filename_no_ext)
    if m:
        return f"QCVN {m.group(1)}"

    # Số hiệu dạng XX/XXXX/TT-XXX
    m = RE_SO_HIEU_TT.search(filename_no_ext)
    if m:
        return m.group(1)

    return ""


def _extract_nam(filename_no_ext: str) -> str:
    """Extract năm ban hành từ tên file (nếu có)."""
    m = RE_NAM.search(filename_no_ext)
    return m.group(1) if m else ""


def _extract_so_phu_luc(filename_no_ext: str) -> str:
    """Extract số phụ lục (dạng 1.1, 1.2, ...) từ đầu tên file."""
    m = RE_SO_PHU_LUC.match(filename_no_ext)
    return m.group(1) if m else ""


def _extract_so_thu_tu(filename_no_ext: str) -> str:
    """Extract số thứ tự đứng đầu tên file (01, 1, 02, ...)."""
    # Ưu tiên số phụ lục phân cấp (1.1) trước
    phu_luc = _extract_so_phu_luc(filename_no_ext)
    if phu_luc:
        return phu_luc
    m = RE_STT.match(filename_no_ext)
    return m.group(1) if m else ""


def parse_metadata_from_path(file_path: str) -> Dict:
    """
    Parse metadata đầy đủ từ đường dẫn + tên file.

    Args:
        file_path: Đường dẫn tuyệt đối hoặc tương đối đến file .doc/.docx

    Returns:
        dict metadata với đầy đủ các trường mô tả tài liệu.
    """
    path = Path(file_path).resolve()
    filename = path.name                    # "1.2. Biên bản vụ việc.doc"
    stem = path.stem                       # "1.2. Biên bản vụ việc"
    suffix = path.suffix.lstrip(".").lower()  # "doc" hoặc "docx"

    # --- Xác định folder liên quan đến thông tư ---
    # Tìm phần tử trong parts khớp FOLDER_MAP
    folder_key = ""
    folder_info: Dict = {}
    for part in path.parts:
        key = part.lower()
        if key in FOLDER_MAP:
            folder_key = key
            folder_info = FOLDER_MAP[key]
            break

    # --- Extract các thông tin từ stem ---
    doc_type = _detect_doc_type(stem)
    so_hieu = _extract_so_hieu(stem)
    nam = _extract_nam(stem)
    so_phu_luc = _extract_so_phu_luc(stem)
    so_thu_tu = _extract_so_thu_tu(stem)

    metadata: Dict = {
        # --- Đường dẫn ---
        "source_path": str(path),
        "docx_path": "",                 # Sẽ được set sau khi convert
        # --- Folder / Thông tư ---
        "folder": folder_key,
        "so_thong_tu": folder_info.get("so_thong_tu", ""),
        "ten_thong_tu": folder_info.get("ten_thong_tu", ""),
        # --- Tên văn bản ---
        "filename": filename,
        "ten_van_ban": stem,            # Tên đầy đủ không có extension
        # --- Phân loại ---
        "loai_tai_lieu": doc_type,
        "so_hieu": so_hieu,
        "so_phu_luc": so_phu_luc,       # "1.1", "1.2", "2.1" ...
        "so_thu_tu": so_thu_tu,          # số thứ tự đứng đầu tên file
        "nam": nam,
        "format_goc": suffix,           # "doc" hoặc "docx"
        # --- Chunking (sẽ set khi chunking) ---
        "chunk_id": "",
        "chunk_index": 0,
        "total_chunks": 0,
        "chunk_strategy": "",
        # --- Thống kê (sẽ set sau extract) ---
        "word_count": 0,
        "has_tables": False,
    }

    logger.debug(
        f"Metadata parsed: {filename} → loai={doc_type} | "
        f"folder={folder_key} | so_hieu={so_hieu}"
    )
    return metadata
