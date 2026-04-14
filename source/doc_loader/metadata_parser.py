# -*- coding: utf-8 -*-
"""
metadata_parser.py — Parse metadata từ đường dẫn & tên file (Pro Edition)
===========================================================================
Nâng cấp:
  1. AUTO-TAGGING: Tự động gán hashtag từ nội dung + tên file
  2. CROSS-LINK DETECTION: Phát hiện tham chiếu QCVN/TCVN trong text
"""

import os
import re
import logging
import unicodedata
from pathlib import Path
from typing import Dict, List

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
# Patterns nhận diện loại tài liệu (theo thứ tự ưu tiên)
# ---------------------------------------------------------------------------
DOC_TYPE_PATTERNS = [
    (r"(?i)QCVN", "quy_chuan_ky_thuat"),
    (r"(?i)ph[uụ]\s*l[uụ]c|Phu\s*luc", "phu_luc"),
    (r"(?i)bi[eê]n\s*b[aả]n|BB\b", "bien_ban"),
    (r"(?i)m[aẫ]u\s*s[oố]|MAU\b|m[aẫ]u\b", "mau_bieu"),
    (r"(?i)quy[eế]t\s*[đd][iị]nh|QĐ\b", "quyet_dinh"),
    (r"(?i)k[eế]\s*ho[aạ]ch|KH\b", "ke_hoach"),
    (r"(?i)b[áa]o\s*c[áa]o|BC\b", "bao_cao"),
    (r"(?i)th[ôo]ng\s*b[áa]o|TB\b", "thong_bao"),
    (r"(?i)s[oơ]\s*[đd][oồ]|SD\b", "so_do"),
    (r"(?i)[đd][eề]\s*ngh[iị]|tr[uư]ng\s*c[aầ]u", "de_nghi"),
    (r"(?i)s[oổ]\s*theo\s*d[oõ]i", "so_theo_doi"),
    (r"(?i)b[iì]a\b", "bia"),
]

# ---------------------------------------------------------------------------
# AUTO-TAGGING: bộ từ khóa → hashtag
# ---------------------------------------------------------------------------
TAG_RULES: List[tuple] = [
    # Thiết bị / an toàn
    (r"(?i)mũ\s*bảo\s*hiểm|helmet|mu\s*bao\s*hiem", "#mũ_bảo_hiểm"),
    (r"(?i)dây\s*an\s*toàn|seat\s*belt|day\s*an\s*toan", "#dây_an_toàn"),
    (r"(?i)biển\s*số|bien\s*so|biển\s*kiểm\s*soát", "#biển_số_xe"),
    (r"(?i)đăng\s*ký\s*xe|dang\s*ky\s*xe", "#đăng_ký_xe"),
    # Tai nạn giao thông
    (r"(?i)tai\s*nạn|TNGT|tngt", "#tai_nạn_giao_thông"),
    (r"(?i)khám\s*nghiệm\s*hiện\s*trường|KNHT", "#khám_nghiệm_hiện_trường"),
    (r"(?i)khám\s*nghiệm\s*phương\s*tiện|KNPT", "#khám_nghiệm_phương_tiện"),
    (r"(?i)hiện\s*trường|hien\s*truong", "#hiện_trường"),
    # Quy trình / thủ tục
    (r"(?i)tuần\s*tra|tuan\s*tra|kiểm\s*soát", "#tuần_tra_kiểm_soát"),
    (r"(?i)xử\s*phạt|phat\s*vi\s*pham|vi\s*phạm", "#xử_phạt_vi_phạm"),
    (r"(?i)giám\s*định|giam\s*dinh", "#giám_định"),
    # Phương tiện
    (r"(?i)ô\s*tô|xe\s*ô\s*tô|motor\s*vehicle", "#ô_tô"),
    (r"(?i)xe\s*máy|mô\s*tô|xe\s*moto", "#xe_máy"),
    (r"(?i)xe\s*đạp\s*điện|xe\s*dap\s*dien", "#xe_đạp_điện"),
    (r"(?i)xe\s*ba\s*bánh|xe\s*ba\s*banh", "#xe_ba_bánh"),
    # Tiêu chuẩn / quy chuẩn
    (r"(?i)QCVN\s*\d+", "#quy_chuẩn_kỹ_thuật"),
    (r"(?i)TCVN\s*\d+", "#tiêu_chuẩn_kỹ_thuật"),
    (r"(?i)khí\s*thải|khi\s*thai|emission", "#khí_thải"),
    (r"(?i)an\s*toàn\s*kỹ\s*thuật|an\s*toan\s*ky\s*thuat", "#an_toàn_kỹ_thuật"),
    (r"(?i)kiểm\s*định|kiem\s*dinh", "#kiểm_định"),
    # Lái xe / bằng lái
    (r"(?i)giấy\s*phép\s*lái\s*xe|GPLX|bằng\s*lái", "#giấy_phép_lái_xe"),
    (r"(?i)sát\s*hạch|sat\s*hach", "#sát_hạch_lái_xe"),
    # Đường sắt
    (r"(?i)đường\s*sắt|đs\b|duong\s*sat", "#đường_sắt"),
]

# ---------------------------------------------------------------------------
# CROSS-LINK DETECTION: regex tìm tham chiếu QCVN/TCVN/Thông tư trong text
# ---------------------------------------------------------------------------
RE_CROSS_LINK = re.compile(
    r"\b(QCVN\s+[\d\-:\/]+|TCVN\s+[\d\-:\/]+|Thông\s+tư\s+số\s+[\d\/\-\w]+|"
    r"Nghị\s+định\s+số\s+[\d\/\-\w]+|Luật\s+[\w\s]{3,30}?\d{4})",
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# Helpers extract từ tên file
# ---------------------------------------------------------------------------
RE_SO_PHU_LUC = re.compile(r"^(\d+\.\d+(?:\.\d+)?)[.\s]")
RE_SO_HIEU_MAU = re.compile(r"(?i)m[aẫ]u\s*s[oố]\s*(\d+[.\-\w]*)")
RE_SO_HIEU_QCVN = re.compile(r"(?i)QCVN\s*([\d\-]+(?:-\d{4})?)")
RE_SO_HIEU_TT = re.compile(r"(\d+/\d{4}/(?:TT|ND|QD|NQ)-\w+)")
RE_NAM = re.compile(r"\b(19\d{2}|20\d{2})\b")
RE_STT = re.compile(r"^(\d+(?:\.\d+)*)[.\s]")


def _detect_doc_type(filename_no_ext: str) -> str:
    for pattern, doc_type in DOC_TYPE_PATTERNS:
        if re.search(pattern, filename_no_ext):
            return doc_type
    return "khac"


def _extract_so_hieu(filename_no_ext: str) -> str:
    m = RE_SO_HIEU_MAU.search(filename_no_ext)
    if m:
        return f"Mẫu số {m.group(1)}"
    m = RE_SO_HIEU_QCVN.search(filename_no_ext)
    if m:
        return f"QCVN {m.group(1)}"
    m = RE_SO_HIEU_TT.search(filename_no_ext)
    if m:
        return m.group(1)
    return ""


def _extract_nam(filename_no_ext: str) -> str:
    m = RE_NAM.search(filename_no_ext)
    return m.group(1) if m else ""


def _extract_so_phu_luc(filename_no_ext: str) -> str:
    m = RE_SO_PHU_LUC.match(filename_no_ext)
    return m.group(1) if m else ""


def _extract_so_thu_tu(filename_no_ext: str) -> str:
    phu_luc = _extract_so_phu_luc(filename_no_ext)
    if phu_luc:
        return phu_luc
    m = RE_STT.match(filename_no_ext)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# AUTO-TAGGING
# ---------------------------------------------------------------------------

def generate_tags(text: str, filename: str = "") -> List[str]:
    """
    Sinh danh sách hashtag dựa trên nội dung text và tên file.

    Args:
        text:     Nội dung văn bản đã extract
        filename: Tên file gốc (để quét thêm)

    Returns:
        list[str] — ví dụ: ["#tai_nạn_giao_thông", "#khám_nghiệm_hiện_trường"]
    """
    combined = (text[:3000] + " " + filename)  # Chỉ scan 3000 ký tự đầu + tên file
    tags_found: List[str] = []
    seen = set()

    for pattern, tag in TAG_RULES:
        if re.search(pattern, combined) and tag not in seen:
            tags_found.append(tag)
            seen.add(tag)

    return tags_found


# ---------------------------------------------------------------------------
# CROSS-LINK DETECTION
# ---------------------------------------------------------------------------

def detect_cross_links(text: str) -> List[str]:
    """
    Phát hiện các tham chiếu đến văn bản pháp lý khác trong nội dung.
    Ví dụ: ["QCVN 04:2021", "TCVN 5756:2017", "Thông tư số 13/2025/TT-BCA"]

    Args:
        text: Toàn bộ text đã extract (giới hạn 5000 ký tự đầu để nhanh)

    Returns:
        list[str] — các tham chiếu tìm được, đã deduplicate
    """
    matches = RE_CROSS_LINK.findall(text[:5000])
    seen = set()
    result: List[str] = []
    for m in matches:
        key = re.sub(r"\s+", " ", m.strip())
        if key not in seen:
            result.append(key)
            seen.add(key)
    return result


# ---------------------------------------------------------------------------
# MAIN: parse_metadata_from_path
# ---------------------------------------------------------------------------

def parse_metadata_from_path(file_path: str) -> Dict:
    """
    Parse metadata đầy đủ từ đường dẫn + tên file.

    Args:
        file_path: Đường dẫn tuyệt đối hoặc tương đối đến file .doc/.docx

    Returns:
        dict metadata với đầy đủ các trường.
    """
    path = Path(file_path).resolve()
    filename = path.name
    stem = path.stem
    suffix = path.suffix.lstrip(".").lower()

    # --- Xác định folder thông tư ---
    folder_key = ""
    folder_info: Dict = {}
    for part in path.parts:
        key = part.lower()
        if key in FOLDER_MAP:
            folder_key = key
            folder_info = FOLDER_MAP[key]
            break

    doc_type = _detect_doc_type(stem)
    so_hieu = _extract_so_hieu(stem)
    nam = _extract_nam(stem)
    so_phu_luc = _extract_so_phu_luc(stem)
    so_thu_tu = _extract_so_thu_tu(stem)

    metadata: Dict = {
        # --- Đường dẫn ---
        "source_path": str(path),
        "docx_path": "",
        # --- Folder / Thông tư ---
        "folder": folder_key,
        "so_thong_tu": folder_info.get("so_thong_tu", ""),
        "ten_thong_tu": folder_info.get("ten_thong_tu", ""),
        # --- Tên văn bản ---
        "filename": filename,
        "ten_van_ban": stem,
        # --- Phân loại ---
        "loai_tai_lieu": doc_type,
        "so_hieu": so_hieu,
        "so_phu_luc": so_phu_luc,
        "so_thu_tu": so_thu_tu,
        "nam": nam,
        "format_goc": suffix,
        # --- Pro Edition: Tags & Cross-links ---
        "tags": [],             # Sẽ được populate sau khi extract text
        "cross_links": [],      # Sẽ được populate sau khi extract text
        # --- Chunking (set khi chunking) ---
        "chunk_id": "",
        "chunk_index": 0,
        "total_chunks": 0,
        "chunk_strategy": "",
        "chunk_priority": "normal",
        # --- Thống kê ---
        "word_count": 0,
        "has_tables": False,
        "has_images": False,
        "image_paths": [],
    }

    logger.debug(
        f"Metadata parsed: {filename} → loai={doc_type} | "
        f"folder={folder_key} | so_hieu={so_hieu}"
    )
    return metadata
