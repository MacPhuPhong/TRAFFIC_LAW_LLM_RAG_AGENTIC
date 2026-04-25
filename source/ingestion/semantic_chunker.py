# -*- coding: utf-8 -*-
"""
semantic_chunker.py — Bước 3 & 4: HierarchicalLegalSplitter + Metadata đầy đủ
========================================================================
NÂNG CẤP v2.1:
  - Fix: Thêm effective_date vào metadata (bắt buộc theo prompt)
  - Fix: Level 2/3 chunking (Khoản → Điểm) khi Điều quá dài
  - Fix: Output path collision với thư mục con (dùng rel_path hash)
  - Fix: BASE_DIR Colab-compatible
  - Fix: OSError: File name too long (truncate 150 chars)
  - Fix: Regex chunking (1 capturing group)
  - Thêm: Token counting chính xác hơn (word-based estimate)
  - Thêm: chunk_id chuẩn hóa đầy đủ
  - Thêm: Statistics và deduplication
  - Thêm: Export toàn bộ chunks ra JSONL (dùng cho ChromaDB/Qdrant)
  - v2.1: Tích hợp Regex Refinement (nối từ bị ngắt trang PDF)
  - v2.1: Context Enrichment (chèn tiêu đề Điều/Khoản vào content)
"""

import os
import re
import yaml
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ---------------------------------------------------------------------------
# BASE_DIR
# ---------------------------------------------------------------------------
def get_base_dir() -> Path:
    colab_path = Path("/content/drive/MyDrive/project")
    if colab_path.exists():
        return colab_path
    try:
        return Path(__file__).resolve().parent.parent.parent
    except NameError:
        return Path(".").resolve().parent.parent

BASE_DIR = get_base_dir()
CLEANED_DIRS = {
    "luat":     BASE_DIR / "Data" / "cleaned" / "luat",
    "nghidinh": BASE_DIR / "Data" / "cleaned" / "nghidinh",
    "thongtu":  BASE_DIR / "Data" / "cleaned" / "thongtu",
}
OUTPUT_DIR  = BASE_DIR / "Data" / "semantic_chunks"
JSONL_PATH  = BASE_DIR / "Data" / "all_chunks.jsonl"   # export cho Qdrant/ChromaDB
LOG_DIR     = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"chunker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ---------------------------------------------------------------------------
# Metadata rules — đầy đủ theo prompt (bao gồm effective_date)
# ---------------------------------------------------------------------------
METADATA_RULES: dict[str, dict] = {
    # ── Luật ──────────────────────────────────────────────────────────────
    "luat35_db_2024": {
        "doc_id":         "35/2024/QH15",
        "title":          "Luật Đường bộ 2024",
        "issuer":         "Quốc hội",
        "status":         "active",
        "effective_date": "2025-01-01",
        "topic":          "Luật nền tảng",
    },
    "luat36_ttatgt_2024": {
        "doc_id":         "36/2024/QH15",
        "title":          "Luật Trật tự ATGT đường bộ 2024",
        "issuer":         "Quốc hội",
        "status":         "active",
        "effective_date": "2025-01-01",
        "topic":          "Luật nền tảng",
    },
    "luat23_db_2008_inactive": {
        "doc_id":         "23/2008/QH12",
        "title":          "Luật Giao thông đường bộ 2008 (ĐÃ HẾT HIỆU LỰC)",
        "issuer":         "Quốc hội",
        "status":         "repealed",
        "effective_date": "2009-07-01",
        "expiry_date":    "2024-12-31",
        "topic":          "Lịch sử pháp lý",
        "warning":        (
            "VĂN BẢN NÀY ĐÃ HẾT HIỆU LỰC từ 01/01/2025. "
            "Thay thế bởi Luật 35/2024/QH15 và Luật 36/2024/QH15."
        ),
    },

    # ── Nghị định ─────────────────────────────────────────────────────────
    "nd168_2024_XuPhat_TruDiem_DB_baibo_nd100": {
        "doc_id":         "168/2024/NĐ-CP",
        "title":          "Nghị định 168/2024/NĐ-CP (Xử phạt VPHC đường bộ)",
        "issuer":         "Chính phủ",
        "status":         "active",
        "effective_date": "2025-01-01",
        "topic":          "Xử phạt vi phạm giao thông",
    },
    "nd336_2025_xu_phat_van_tai": {
        "doc_id":         "336/2025/NĐ-CP",
        "title":          "Nghị định 336/2025/NĐ-CP (Xử phạt kinh doanh vận tải)",
        "issuer":         "Chính phủ",
        "status":         "active",
        "effective_date": "2025-07-01",   # cập nhật theo thực tế
        "topic":          "Xử phạt vi phạm giao thông",
    },
    "nd100_2019_xu_phat": {
        "doc_id":         "100/2019/NĐ-CP",
        "title":          "Nghị định 100/2019/NĐ-CP (ĐƯỜNG SẮT — mảng đường bộ đã bãi bỏ)",
        "issuer":         "Chính phủ",
        "status":         "partially_repealed",
        "effective_date": "2020-01-01",
        "topic":          "Xử phạt vi phạm giao thông",
        "warning":        (
            "Mảng ĐƯỜNG BỘ của NĐ này đã bị thay thế bởi NĐ 168/2024. "
            "Chunk này chỉ chứa nội dung đường sắt (nếu có)."
        ),
    },
    "nd123_2021_xu_phat": {
        "doc_id":         "123/2021/NĐ-CP",
        "title":          "Nghị định 123/2021/NĐ-CP (Đường thủy/hàng hải — mảng ĐB đã bãi bỏ)",
        "issuer":         "Chính phủ",
        "status":         "partially_repealed",
        "effective_date": "2022-01-01",
        "topic":          "Xử phạt vi phạm giao thông",
        "warning":        "Mảng đường bộ đã bị NĐ 168/2024 bãi bỏ. Chỉ còn hiệu lực với đường thủy/hàng hải.",
    },
    "nd10_2020_kinh_doanh_van_tai": {
        "doc_id":         "10/2020/NĐ-CP",
        "title":          "Nghị định 10/2020/NĐ-CP (Kinh doanh vận tải bằng xe ô tô)",
        "issuer":         "Chính phủ",
        "status":         "active",
        "effective_date": "2020-04-01",
        "topic":          "Kinh doanh vận tải",
    },

    # ── Thông tư ──────────────────────────────────────────────────────────
    # [Đăng ký xe]
    "tt79_2024_dang_ky_xe_GOC": {
        "doc_id":         "79/2024/TT-BCA",
        "title":          "Thông tư 79/2024/TT-BCA (Đăng ký xe)",
        "issuer":         "Bộ Công an",
        "status":         "active",
        "effective_date": "2025-01-01",
        "topic":          "Đăng ký phương tiện",
    },
    "tt51_2025_sua_doi_dang_ky_xe": {
        "doc_id":         "51/2025/TT-BCA",
        "title":          "Thông tư 51/2025/TT-BCA (Sửa đổi TT 79/2024)",
        "issuer":         "Bộ Công an",
        "status":         "active",
        "effective_date": "2025-06-01",
        "topic":          "Đăng ký phương tiện",
    },
    "tt155_2025_le_phi_dang_ky_xe": {
        "doc_id":         "155/2025/TT-BTC",
        "title":          "Thông tư 155/2025/TT-BTC (Lệ phí đăng ký xe)",
        "issuer":         "Bộ Tài chính",
        "status":         "active",
        "effective_date": "2025-12-31",
        "topic":          "Lệ phí",
    },

    # [Kiểm định xe cơ giới, khí thải]
    "TT47_Tram_KhiThai_XeMay": {
        "doc_id":         "47/2024/TT-BGTVT",
        "title":          "Thông tư 47/2024/TT-BGTVT (Trạm kiểm định khí thải xe máy)",
        "issuer":         "Bộ GTVT",
        "status":         "active",
        "effective_date": "2025-01-01",
        "topic":          "Kiểm định phương tiện",
    },
    "TT48_2024_BGTVT_QuyChuan_AnToan_KyThuat": {
        "doc_id":         "48/2024/TT-BGTVT",
        "title":          "Thông tư 48/2024/TT-BGTVT (Phân loại lỗi kiểm định)",
        "issuer":         "Bộ GTVT",
        "status":         "active",
        "effective_date": "2025-01-01",
        "topic":          "Kiểm định phương tiện",
    },
    "TT30_2024_BGTVT_DangKiem_Oto_DanSu": {
        "doc_id":         "30/2024/TT-BGTVT",
        "title":          "Thông tư 30/2024/TT-BGTVT (Đăng kiểm ô tô)",
        "issuer":         "Bộ GTVT",
        "status":         "active",
        "effective_date": "2024-08-12",
        "topic":          "Kiểm định phương tiện",
    },
    "TT46_2024_BGTVT_ThuTuc_KiemDinh_KhiThai_XeMay": {
        "doc_id":         "46/2024/TT-BGTVT",
        "title":          "Thông tư 46/2024/TT-BGTVT (Thủ tục kiểm định khí thải xe máy)",
        "issuer":         "Bộ GTVT",
        "status":         "active",
        "effective_date": "2024-11-15",
        "topic":          "Kiểm định phương tiện",
    },
    "TT70_TieuChuan_XeMoi": {
        "doc_id":         "70/2025/TT-BXD",
        "title":          "Thông tư 70/2025/TT-BXD (Tiêu chuẩn xe mới)",
        "issuer":         "Bộ Xây dựng",
        "status":         "active",
        "effective_date": "2025-12-31",
        "topic":          "Tiêu chuẩn",
    },
    "TT92_2025_KhiThai_XeMay": {
        "doc_id":         "92/2025/TT-BNNMT",
        "title":          "Thông tư 92/2025/TT-BNNMT (Khí thải xe máy)",
        "issuer":         "Bộ NNPTNT",
        "status":         "active",
        "effective_date": "2025-12-31",
        "topic":          "Khí thải",
    },

    # [Tuyển sinh, Đào tạo, Sát hạch]
    "TT12_2017_BGTVT": {
        "doc_id":         "12/2017/TT-BGTVT",
        "title":          "Thông tư 12/2017/TT-BGTVT (Đào tạo, sát hạch, cấp GPLX)",
        "issuer":         "Bộ GTVT",
        "status":         "active",
        "effective_date": "2017-04-15",
        "topic":          "Đào tạo GPLX",
    },
    "TT12_2025_BCA_QuyTrinhCap_GPLX": {
        "doc_id":         "12/2025/TT-BCA",
        "title":          "Thông tư 12/2025/TT-BCA (Quy trình cấp GPLX)",
        "issuer":         "Bộ Công an",
        "status":         "active",
        "effective_date": "2025-02-28",
        "topic":          "Đào tạo GPLX",
    },
    "TT35_2024_BGTVT_DaoTaoSatHach_GPLX": {
        "doc_id":         "35/2024/TT-BGTVT",
        "title":          "Thông tư 35/2024/TT-BGTVT (Đào tạo sát hạch GPLX)",
        "issuer":         "Bộ GTVT",
        "status":         "active",
        "effective_date": "2024-11-15",
        "topic":          "Đào tạo GPLX",
    },
    "TT36_2024_BYT_TieuChuanSucKhoe_LaiXe": {
        "doc_id":         "36/2024/TT-BYT",
        "title":          "Thông tư 36/2024/TT-BYT (Tiêu chuẩn sức khỏe lái xe)",
        "issuer":         "Bộ Y tế",
        "status":         "active",
        "effective_date": "2024-11-16",
        "topic":          "Y tế",
    },

    # [Tuần tra, Kiểm soát, Thiết bị nghiệp vụ]
    "TT13_2025_BCA_SuaDoi_TuanTra_GiaoThong": {
        "doc_id":         "13/2025/TT-BCA",
        "title":          "Thông tư 13/2025/TT-BCA (Tuần tra, kiểm soát)",
        "issuer":         "Bộ Công an",
        "status":         "active",
        "effective_date": "2025-02-28",
        "topic":          "Tuần tra",
    },
    "TT51_2022_BGTVT_HuongDan_ThietBi": {
        "doc_id":         "51/2022/TT-BGTVT",
        "title":          "Thông tư 51/2022/TT-BGTVT (Hướng dẫn thiết bị)",
        "issuer":         "Bộ GTVT",
        "status":         "active",
        "effective_date": "2022-12-30",
        "topic":          "Thiết bị",
    },
    "TT73_2024_BCA_TuanTra_KiemSoat_GOC": {
        "doc_id":         "73/2024/TT-BCA",
        "title":          "Thông tư 73/2024/TT-BCA (Tuần tra kiểm soát)",
        "issuer":         "Bộ Công an",
        "status":         "active",
        "effective_date": "2024-11-15",
        "topic":          "Tuần tra",
    },
    "TT72_2024_BCA_Quytrinh_Dieutra": {
        "doc_id":         "72/2024/TT-BCA",
        "title":          "Thông tư 72/2024/TT-BCA (Quy trình điều tra, giải quyết tai nạn giao thông đường bộ của CSGT)",
        "issuer":         "Bộ Công an",
        "status":         "active",
        "effective_date": "2025-01-01",
        "topic":          "Điều tra TNGT",
    },
    "nd135_2021_ChinhPhu_ThietBi_NghiepVu": {
        "doc_id":         "135/2021/NĐ-CP",
        "title":          "Nghị định 135/2021/NĐ-CP (Thiết bị nghiệp vụ)",
        "issuer":         "Chính phủ",
        "status":         "active",
        "effective_date": "2021-12-31",
        "topic":          "Thiết bị nghiệp vụ",
    },
}

def get_file_metadata(file_stem: str) -> dict:
    """Lấy metadata dựa trên tên file (không có extension)."""
    meta = METADATA_RULES.get(file_stem)
    if meta:
        return meta
    # Fallback thông minh: tìm key substring match
    for key, val in METADATA_RULES.items():
        if key.lower() in file_stem.lower() or file_stem.lower() in key.lower():
            logger.debug(f"Metadata fuzzy match: {file_stem} → {key}")
            return val
    logger.warning(f"Không tìm thấy metadata rule cho: {file_stem}")
    return {
        "doc_id":         file_stem,
        "title":          file_stem,
        "issuer":         "Không xác định",
        "status":         "active",
        "effective_date": "Không xác định",
        "topic":          "Chung",
    }

# ---------------------------------------------------------------------------
# Token counting (word-based, đủ cho ngưỡng split)
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    """Ước tính số token bằng cách đếm từ (1 từ ≈ 1.3 token cho tiếng Việt)."""
    words = len(text.split())
    return int(words * 1.3)


# ---------------------------------------------------------------------------
# Regex Refinement — Sửa lỗi ngắt dòng do PDF page break
# ---------------------------------------------------------------------------
def refine_content(text: str) -> str:
    """
    Sửa các lỗi phổ biến khi extract PDF:
      1. Từ bị ngắt bởi dấu gạch nối cuối dòng + xuống dòng
      2. Xuống dòng vô nghĩa giữa câu
      3. Dòng kẻ trang trí, dòng trống liên tiếp
    """
    # ── Bảo vệ bảng Markdown: tạm thay thế các dòng bảng bằng placeholder ──
    table_lines = []
    lines = text.split('\n')
    protected = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('|') or (re.match(r'^[-:|]+$', stripped) and '|' in stripped):
            table_lines.append((i, line))
            protected.append(f'__TABLE_LINE_{i}__')
        else:
            protected.append(line)
    text = '\n'.join(protected)

    # [TASK 1a] Nối từ bị ngắt: giữ dấu gạch nối, chỉ xóa xuống dòng vô nghĩa
    text = re.sub(r'(\s*-\s*)\n\s*', r'\1', text)

    # [TASK 1b] Nối dòng bị ngắt giữa câu
    text = re.sub(
        r'(?<=[a-zàáạảãắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ,;])'
        r'\n'
        r'(?=[a-zàáạảãắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ])',
        ' ', text
    )

    # [TASK 1c] Xóa dòng kẻ trang trí (5+ ký tự ---/===/___ liên tiếp)
    text = re.sub(r'^[-=_]{5,}\s*$', '', text, flags=re.MULTILINE)

    # [TASK 1d] Gộp dòng trống liên tiếp (>2 dòng → 1 dòng trống)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # ── Khôi phục dòng bảng Markdown ──
    lines_out = text.split('\n')
    for idx, original_line in table_lines:
        for j, line in enumerate(lines_out):
            if line.strip() == f'__TABLE_LINE_{idx}__':
                lines_out[j] = original_line
                break

    return '\n'.join(lines_out).strip()

# ---------------------------------------------------------------------------
# Form noise: các keyword chỉ ra biểu mẫu nội bộ (không hữu ích cho RAG)
# ---------------------------------------------------------------------------
FORM_NOISE_KEYWORDS = [
    'sổ theo dõi', 'biên bản phân công', 'báo cáo định kỳ',
    'sổ giao nhận', 'nhật ký', 'sổ đăng ký',
    'báo cáo thống kê', 'phiếu xuất kho',
]


# ---------------------------------------------------------------------------
# Legal Styling: Bold các mức phạt và hành vi vi phạm quan trọng
# ---------------------------------------------------------------------------
def apply_legal_styling(text: str) -> str:
    """Bold các mức phạt tiền và hình thức xử phạt bổ sung quan trọng."""
    # Bold mức phạt tiền:  "phạt tiền từ X đồng đến Y đồng"
    text = re.sub(
        r'(phạt tiền từ\s+[\d.,]+\s+đồng\s+đến\s+[\d.,]+\s+đồng)',
        r'**\1**', text, flags=re.IGNORECASE
    )
    # Bold tước quyền sử dụng GPLX
    text = re.sub(
        r'(tước quyền sử dụng[^.;]{5,80}tháng)',
        r'**\1**', text, flags=re.IGNORECASE
    )
    # Tránh bold lồng (** bên trong **)
    text = re.sub(r'\*{4,}', '**', text)
    return text


# ---------------------------------------------------------------------------
# Regex phát hiện cấu trúc pháp luật
# ---------------------------------------------------------------------------
# Điều: "Điều 5." / "Điều 5:" / "## Điều 5."
RE_DIEU   = re.compile(r"(?:##\s*)?Điều\s+(\d+)[.:]\s*(.*)", re.IGNORECASE)

# Khoản: dòng bắt đầu bằng số + dấu chấm + chữ hoa
# ví dụ: "1. Người điều khiển..."
# Chỉ capture 1 group (số khoản) để split chính xác
RE_KHOAN  = re.compile(r"^(\d+)\.(?=\s+[A-ZĐÀÁẠẢÃẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ])", re.MULTILINE)

# Điểm: "a) ..." / "b) ..." (chữ thường + ngoặc đơn)
# Chỉ capture 1 group (nhãn điểm)
RE_DIEM   = re.compile(r"^([a-z])\)(?=\s+)", re.MULTILINE)

# Ngưỡng split
THRESH_DIEU  = 600   # token — nếu Điều > ngưỡng này → split xuống Khoản
THRESH_KHOAN = 500   # token — nếu Khoản > ngưỡng này → split xuống Điểm
MIN_CHUNK    = 30    # token — chunk quá nhỏ thì bỏ

# ---------------------------------------------------------------------------
# HierarchicalLegalSplitter
# ---------------------------------------------------------------------------
class HierarchicalLegalSplitter:
    """
    Tách văn bản pháp luật theo 3 tầng phân cấp:
      Level 1 — Điều   (≤ 600 token)
      Level 2 — Khoản  (nếu Điều > 600 token)
      Level 3 — Điểm   (nếu Khoản > 500 token)
      Fallback         (văn bản xuôi, không có cấu trúc Điều/Khoản)
    """

    def split_into_articles(self, full_text: str) -> list[tuple[str, str, str]]:
        """
        Tách full text thành list of (dieu_num, dieu_title, dieu_content).
        """
        articles = []
        lines    = full_text.splitlines()

        current_num     = "0"
        current_title   = "Phần giới thiệu"
        current_lines   = []

        for line in lines:
            m = RE_DIEU.match(line.strip())
            if m:
                # Lưu Điều hiện tại
                if current_lines:
                    articles.append((current_num, current_title, "\n".join(current_lines)))
                current_num   = m.group(1)
                current_title = m.group(2).strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        # Lưu Điều cuối
        if current_lines:
            articles.append((current_num, current_title, "\n".join(current_lines)))

        return articles

    def split_article_into_khoans(self, text: str) -> list[tuple[str, str]]:
        """
        Tách nội dung một Điều thành list of (khoan_num, khoan_content).
        """
        parts = RE_KHOAN.split(text)
        khoans = []

        if len(parts) <= 1:
            # Không có Khoản rõ ràng — cả Điều là 1 block
            return [("0", text)]

        # parts[0] = text trước Khoản 1 (thường là tiêu đề Điều)
        if parts[0].strip():
            khoans.append(("0", parts[0].strip()))

        # Mỗi Khoản: parts[i] = khoan_num, parts[i+1] = content_between
        i = 1
        while i + 1 < len(parts):
            khoan_num = parts[i]
            content   = parts[i + 1]
            khoans.append((khoan_num, f"{khoan_num}.{content}"))
            i += 2

        return khoans if khoans else [("0", text)]

    def split_khoan_into_diems(self, text: str) -> list[tuple[str, str]]:
        """
        Tách nội dung một Khoản thành list of (diem_label, diem_content).
        """
        parts = RE_DIEM.split(text)
        diems = []

        if len(parts) <= 1:
            return [("", text)]

        if parts[0].strip():
            diems.append(("", parts[0].strip()))

        i = 1
        while i + 1 < len(parts):
            label   = parts[i]
            content = parts[i + 1]
            diems.append((label, f"{label}){content}"))
            i += 2

        return diems if diems else [("", text)]

    def chunk_document(
        self,
        full_text:   str,
        doc_meta:    dict,
        doc_type:    str,
        source_file: str,
    ) -> list[dict]:
        """
        Entry point chính: nhận full text → trả về list chunk dicts.
        """
        articles = self.split_into_articles(full_text)
        chunks   = []
        seen_hashes = set()  # Deduplication

        for dieu_num, dieu_title, dieu_content in articles:
            if count_tokens(dieu_content) < MIN_CHUNK:
                continue  # Quá ngắn — bỏ qua

            if count_tokens(dieu_content) <= THRESH_DIEU:
                # ── Level 1: Cả Điều là 1 chunk ──
                chunk = self._make_chunk(
                    text        = dieu_content,
                    doc_meta    = doc_meta,
                    doc_type    = doc_type,
                    source_file = source_file,
                    dieu_num    = dieu_num,
                    dieu_title  = dieu_title,
                    khoan_num   = None,
                    diem_label  = None,
                    level       = 1,
                )
                chunks.append(chunk)
            else:
                # ── Level 2: Split theo Khoản ──
                khoans = self.split_article_into_khoans(dieu_content)
                for khoan_num, khoan_content in khoans:
                    if count_tokens(khoan_content) < MIN_CHUNK:
                        continue

                    if count_tokens(khoan_content) <= THRESH_KHOAN:
                        chunk = self._make_chunk(
                            text        = khoan_content,
                            doc_meta    = doc_meta,
                            doc_type    = doc_type,
                            source_file = source_file,
                            dieu_num    = dieu_num,
                            dieu_title  = dieu_title,
                            khoan_num   = khoan_num if khoan_num != "0" else None,
                            diem_label  = None,
                            level       = 2,
                        )
                        chunks.append(chunk)
                    else:
                        # ── Level 3: Split theo Điểm ──
                        diems = self.split_khoan_into_diems(khoan_content)
                        for diem_label, diem_content in diems:
                            if count_tokens(diem_content) < MIN_CHUNK:
                                continue
                            chunk = self._make_chunk(
                                text        = diem_content,
                                doc_meta    = doc_meta,
                                doc_type    = doc_type,
                                source_file = source_file,
                                dieu_num    = dieu_num,
                                dieu_title  = dieu_title,
                                khoan_num   = khoan_num if khoan_num != "0" else None,
                                diem_label  = diem_label if diem_label else None,
                                level       = 3,
                            )
                            chunks.append(chunk)

        # [TASK 3c] Lọc biểu mẫu nhiễu nội bộ + Deduplication
        unique_chunks = []
        form_noise_removed = 0
        for chunk in chunks:
            # Kiểm tra biểu mẫu nhiễu nội bộ
            content_lower = chunk['content'].lower()
            if any(kw in content_lower for kw in FORM_NOISE_KEYWORDS):
                form_noise_removed += 1
                logger.debug(f"Form noise bị loại: {chunk['metadata']['chunk_id']}")
                continue

            # Deduplication dựa trên content hash
            h = hashlib.md5(chunk['content'].encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_chunks.append(chunk)
            else:
                logger.debug(f"Duplicate chunk bị loại: {chunk['metadata']['chunk_id']}")

        if form_noise_removed:
            logger.info(f"  → Lọc {form_noise_removed} chunk biểu mẫu nhiễu nội bộ")

        return unique_chunks

    def _make_chunk(
        self,
        text:        str,
        doc_meta:    dict,
        doc_type:    str,
        source_file: str,
        dieu_num:    str,
        dieu_title:  str,
        khoan_num:   str | None,
        diem_label:  str | None,
        level:       int,
    ) -> dict:
        """Tạo chunk dict với metadata đầy đủ + refinement + context enrichment."""
        # Tạo chunk_id chuẩn hóa
        doc_slug   = doc_meta.get("doc_id", source_file).replace("/", "_").replace("-", "_")
        khoan_part = f"_khoan{khoan_num}" if khoan_num else ""
        diem_part  = f"_diem{diem_label}" if diem_label else ""
        chunk_id   = f"{doc_slug}_dieu{dieu_num}{khoan_part}{diem_part}"

        # ── BƯỚC MỚI v2.1: Regex Refinement — sửa lỗi ngắt dòng PDF ──
        content = refine_content(text)

        # ── [TASK 4] Legal Styling: Bold mức phạt và hành vi vi phạm ──
        content = apply_legal_styling(content)

        # ── [TASK 2] Context Enrichment ──
        ten_van_ban = doc_meta.get("title", source_file)
        if dieu_num and dieu_num != "0":
            context_line = f"Văn bản: {ten_van_ban} | Điều {dieu_num}: {dieu_title}"
            if not content.startswith(context_line):
                content = f"{context_line}\n{content}"

        # Chèn warning vào đầu mỗi chunk nếu văn bản có cảnh báo
        if doc_meta.get("warning") and not content.startswith(">"):
            content = f"> ⚠️ {doc_meta['warning']}\n\n{content}"

        return {
            "metadata": {
                # Trường bắt buộc theo prompt
                "chunk_id":       chunk_id,
                "doc_id":         doc_meta.get("doc_id", source_file),
                "ten_van_ban":    doc_meta.get("title", source_file),
                "issuer":         doc_meta.get("issuer", ""),
                "document_type":  doc_type,
                "status":         doc_meta.get("status", "active"),
                "effective_date": doc_meta.get("effective_date", ""),
                "topic":          doc_meta.get("topic", "Chung"),

                # Trường pháp lý phân cấp
                "dieu":       int(dieu_num) if dieu_num.isdigit() else dieu_num,
                "dieu_title": dieu_title,
                "khoan":      khoan_num,
                "diem":       diem_label,
                "level":      level,

                # Trường kỹ thuật
                "source_file":   source_file,
                "token_estimate": count_tokens(content),
            },
            "content": content,
        }


# ---------------------------------------------------------------------------
# Lưu chunk ra file .md + YAML frontmatter
# ---------------------------------------------------------------------------
def save_chunk_as_md(chunk: dict, output_dir: Path):
    """Lưu 1 chunk ra file Markdown với YAML frontmatter."""
    meta     = chunk["metadata"]
    chunk_id = meta["chunk_id"]

    # Tên file từ chunk_id (giới hạn độ dài để tránh OSError: File name too long)
    # Hầu hết OS giới hạn 255 chars, ta để 150 cho an toàn
    safe_name = re.sub(r"[^\w\-]", "_", chunk_id)
    if len(safe_name) > 150:
        # Nếu quá dài, lấy 140 ký tự đầu + 8 ký tự hash của phần còn lại
        suffix = hashlib.md5(safe_name.encode()).hexdigest()[:8]
        safe_name = safe_name[:140] + "_" + suffix
    
    out_path = output_dir / f"{safe_name}.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("---\n")
        yaml.dump(meta, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        f.write("---\n\n")
        f.write(chunk["content"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splitter     = HierarchicalLegalSplitter()
    all_chunks   = []
    global_stats = {
        "total_files": 0,
        "total_chunks": 0,
        "duplicates_removed": 0,
        "by_doc_type": {},
        "by_level": {1: 0, 2: 0, 3: 0},
    }

    for doc_type, dir_path in CLEANED_DIRS.items():
        if not dir_path.exists():
            logger.warning(f"Thư mục không tồn tại: {dir_path}")
            continue

        md_files = list(dir_path.rglob("*.md"))
        # Bỏ qua file stats
        md_files = [f for f in md_files if not f.name.startswith("_")]

        logger.info(f"\n[{doc_type.upper()}] Tìm thấy {len(md_files)} file")
        global_stats["by_doc_type"][doc_type] = {"files": len(md_files), "chunks": 0}

        for file_path in tqdm(md_files, desc=f"Chunking {doc_type}", unit="file"):
            global_stats["total_files"] += 1

            doc_meta = get_file_metadata(file_path.stem)

            # ── FIX output path collision: dùng doc_type + relative path ──
            rel      = file_path.relative_to(dir_path)
            out_subdir = OUTPUT_DIR / doc_type / rel.parent / file_path.stem
            out_subdir.mkdir(parents=True, exist_ok=True)

            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            if not full_text.strip():
                logger.warning(f"  File rỗng: {file_path.name}")
                continue

            chunks = splitter.chunk_document(
                full_text   = full_text,
                doc_meta    = doc_meta,
                doc_type    = doc_type,
                source_file = file_path.name,
            )

            for chunk in chunks:
                save_chunk_as_md(chunk, out_subdir)
                all_chunks.append(chunk)
                lvl = chunk["metadata"]["level"]
                global_stats["by_level"][lvl] = global_stats["by_level"].get(lvl, 0) + 1

            global_stats["total_chunks"]                         += len(chunks)
            global_stats["by_doc_type"][doc_type]["chunks"]      += len(chunks)

            logger.info(
                f"  {file_path.name} → {len(chunks)} chunks "
                f"(L1:{sum(1 for c in chunks if c['metadata']['level']==1)} "
                f"L2:{sum(1 for c in chunks if c['metadata']['level']==2)} "
                f"L3:{sum(1 for c in chunks if c['metadata']['level']==3)})"
            )

    # ── Export tất cả chunks ra JSONL (dùng để load vào ChromaDB / Qdrant) ──
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"\n{'='*60}")
    logger.info(f"TỔNG KẾT CHUNKING:")
    logger.info(f"  Tổng file:          {global_stats['total_files']}")
    logger.info(f"  Tổng chunks:        {global_stats['total_chunks']}")
    logger.info(f"  Theo level:         L1={global_stats['by_level'].get(1,0)} | "
                f"L2={global_stats['by_level'].get(2,0)} | L3={global_stats['by_level'].get(3,0)}")
    for dt, s in global_stats["by_doc_type"].items():
        logger.info(f"  [{dt}]: {s['files']} files → {s['chunks']} chunks")
    logger.info(f"  JSONL export: {JSONL_PATH}")

    # Lưu stats JSON
    stats_path = OUTPUT_DIR / "_chunking_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2)

    logger.info("HOÀN TẤT.")
    return all_chunks


if __name__ == "__main__":
    main()