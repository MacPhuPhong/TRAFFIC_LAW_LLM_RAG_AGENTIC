# -*- coding: utf-8 -*-
"""
clean_thongtu_pdfs.py — Giai đoạn 2: Trích xuất và làm sạch dữ liệu Thông tư
========================================================================
NÂNG CẤP v2.0:
  - Fix: Duplicate text + table (bbox exclusion)
  - Fix: BASE_DIR Colab-compatible
  - Fix: Line-merging không nhầm heading
  - Thêm: Blacklist file
  - Thêm: Per-file strategy (TT30/46: bỏ mẫu biểu; TT51: merge vào TT79)
  - Thêm: Output path collision fix (dùng rel_path đầy đủ)
  - Thêm: Logging + Stats + tqdm
"""

import os
import re
import shutil
import unicodedata
import logging
import json
from pathlib import Path
from datetime import datetime

try:
    import pdfplumber
except ImportError:
    raise ImportError("Chạy: pip install pdfplumber")

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

BASE_DIR    = get_base_dir()
RAW_DIR     = BASE_DIR / "Data" / "raw"     / "thongtu"
CLEANED_DIR = BASE_DIR / "Data" / "cleaned" / "thongtu"
LOG_DIR     = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"clean_thongtu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
# Blacklist
# ---------------------------------------------------------------------------
EXCLUDED_FILES = {
    "nd17_2026_sua_doi_bo_sung.pdf",
    "nd81_2026_XuPhat_DuongSat.pdf",
    "TT130_2025_BTC_LePhi_CapBang.pdf",
}

# ---------------------------------------------------------------------------
# Per-file strategy
# ---------------------------------------------------------------------------
FILE_STRATEGIES = {
    # TT79/2024: Thông tư về đăng ký xe — giữ toàn bộ, là văn bản gốc
    "tt79_2024_GOC.pdf": {
        "keep_all": True,
        "convert_images_to_text": True,  # Ghi chú: ảnh biển số → mô tả văn bản
    },

    # TT51/2025: Sửa đổi TT79 — sau khi xử lý, MERGE vào output của TT79
    "tt51_2025.pdf": {
        "keep_all": True,
        "merge_into": "tt79_2024_GOC.md",
        "prepend_marker": "\n\n---\n## [SỬA ĐỔI BỔ SUNG — TT 51/2025]\n\n",
    },

    # TT12/2025 (BCA): Bỏ sổ nội bộ, chỉ giữ mẫu đơn học/đổi GPLX
    "TT12_2025_BCA_GPLX.pdf": {
        "keep_all": False,
        "include_section_keywords": [
            "mẫu đơn", "đề nghị", "giấy phép lái xe", "đổi gplx",
            "học lái xe", "thi sát hạch", "lệ phí"
        ],
        "exclude_section_keywords": [
            "sổ theo dõi nội bộ", "biên bản kiểm tra",
            "phiếu kiểm soát", "báo cáo định kỳ"
        ],
    },

    # TT35/2024 (BGTVT): Tương tự TT12
    "TT35_2024_BGTVT_GPLX.pdf": {
        "keep_all": False,
        "include_section_keywords": [
            "mẫu đơn", "đề nghị", "giấy phép lái xe",
            "học lái xe", "thi", "lệ phí", "hồ sơ"
        ],
        "exclude_section_keywords": [
            "sổ theo dõi", "biên bản kiểm tra nội bộ",
            "phiếu kiểm soát thiết bị"
        ],
    },

    # TT155/2025: Giữ bảng lệ phí, bỏ phụ lục không có số liệu
    "tt155_2025.pdf": {
        "keep_all": False,
        "include_section_keywords": [
            "lệ phí", "mức thu", "phí", "biểu phí", "thu nộp"
        ],
        "exclude_section_keywords": [
            "phụ lục mẫu", "biên bản", "quy trình nội bộ"
        ],
    },

    # TT30/2024 & TT46/2024: BỎ mẫu biểu kiểm tra thiết bị nội bộ
    "TT30_2024_KiemDinh.pdf": {
        "keep_all": False,
        "include_section_keywords": [
            "chu kỳ kiểm định", "phân loại lỗi", "tiêu chuẩn",
            "kiểm định an toàn", "điều kiện kỹ thuật"
        ],
        "exclude_section_keywords": [
            "mẫu biên bản kiểm tra thiết bị", "phiếu kiểm soát nội bộ",
            "sổ theo dõi bảo dưỡng"
        ],
    },
    "TT46_2024_KiemDinh.pdf": {
        "keep_all": False,
        "include_section_keywords": [
            "chu kỳ kiểm định", "phân loại lỗi", "MaD", "MiD", "DD",
            "tiêu chuẩn kỹ thuật"
        ],
        "exclude_section_keywords": [
            "mẫu biên bản", "phiếu kiểm soát nội bộ", "sổ theo dõi"
        ],
    },

    # TT47 & TT48/2024: Giữ nguyên bảng chu kỳ & phân loại lỗi
    "TT47_2024.pdf": {"keep_all": True},
    "TT48_2024.pdf": {"keep_all": True},

    # TT92/2025 & TT70: Khí thải — trích ngưỡng nồng độ
    "TT92_2025_KhiThai.pdf": {
        "keep_all": False,
        "include_section_keywords": [
            "nồng độ", "ngưỡng", "$$CO$$", "$$HC$$", "$$NOx$$",
            "tiêu chuẩn khí thải", "euro"
        ],
    },
    "TT70_KhiThai.pdf": {
        "keep_all": False,
        "include_section_keywords": [
            "nồng độ khí thải", "$$CO$$", "$$HC$$", "mức giới hạn", "tiêu chuẩn"
        ],
    },
}

# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------
RE_HEADER_NOISE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},.*about:blank$", re.MULTILINE)
RE_FOOTER_NOISE = re.compile(r"about:blank\s+\d+/\d+|Thư viện pháp luật|Mã tra cứu", re.IGNORECASE)
RE_PAGE_NUM     = re.compile(r"^(Trang\s+)?\d+(\s*/\s*\d+)?$", re.IGNORECASE)
RE_FORM_DOTS    = re.compile(r"(\.{5,}|_{5,})")
RE_CHECKBOX     = re.compile(r"([☐☑\uf06f])")
RE_IS_HEADING   = re.compile(
    r"^(Điều\s+\d+|Khoản\s+\d+|Chương\s+[IVXLCDM]+|Phần\s+[IVXLCDM]+|"
    r"\d+\.\s+[A-ZĐÀÁẠẢÃ]|[a-z]\)\s+)",
    re.IGNORECASE
)
RE_MERGE_END    = re.compile(
    r'[a-zA-ZáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ,;]$'
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def should_keep_block(text: str, strategy: dict) -> bool:
    if strategy.get("keep_all", True):
        return True
    text_lower = text.lower()
    inc = strategy.get("include_section_keywords", [])
    exc = strategy.get("exclude_section_keywords", [])
    if inc and not any(k in text_lower for k in inc):
        return False
    if exc and any(k in text_lower for k in exc):
        return False
    return True


def get_table_bboxes(page) -> list[tuple]:
    try:
        return [t.bbox for t in page.find_tables()]
    except Exception:
        return []


def is_inside_table(obj: dict, bboxes: list[tuple], tol: float = 2.0) -> bool:
    x0, top = obj.get("x0", 0), obj.get("top", 0)
    return any(
        tx0 - tol <= x0 <= tx1 + tol and ttop - tol <= top <= tbot + tol
        for tx0, ttop, tx1, tbot in bboxes
    )


def clean_text(raw_text: str, strategy: dict | None = None) -> str:
    if not raw_text:
        return ""
    text  = unicodedata.normalize("NFC", raw_text)
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:                        continue
        if RE_HEADER_NOISE.match(line):     continue
        if RE_FOOTER_NOISE.search(line):    continue
        if RE_PAGE_NUM.match(line):          continue
        line = RE_FORM_DOTS.sub(" [Cần điền thông tin] ", line)
        line = RE_CHECKBOX.sub(" [Lựa chọn] ", line).strip()
        if line in ("", "[Cần điền thông tin]", "[Lựa chọn]"):
            continue

        line = re.sub(r"^(Chương\s+[IVXLCDM]+.*|Phần\s+[IVXLCDM]+.*)$",
                      r"# \1", line, flags=re.IGNORECASE)
        line = re.sub(r"^(Điều\s+\d+[.:].*)",
                      lambda m: "## " + m.group(0).lstrip("# "),
                      line, flags=re.IGNORECASE)
        line = re.sub(r"(ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
                      r"**\1**", line, flags=re.IGNORECASE)
        line = re.sub(r"\b(CO2?|HC|NOx|PM2\.5|mg/l)\b", r"$$\1$$", line)
        cleaned_lines.append(line)

    merged_lines = []
    for line in cleaned_lines:
        if not merged_lines:
            merged_lines.append(line)
            continue
        prev = merged_lines[-1]
        if (RE_MERGE_END.search(prev[-1:])
                and (line[0].islower() or
                     line[0] in 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ')
                and not RE_IS_HEADING.match(line)):
            merged_lines[-1] = prev + " " + line
        else:
            merged_lines.append(line)

    result = "\n".join(merged_lines)
    if strategy and strategy.get("prepend_warning"):
        result = f"> {strategy['prepend_warning']}\n\n{result}"
    return result


def table_to_markdown(table_data: list) -> str:
    if not table_data:
        return ""
    table_data = [r for r in table_data if any(c and str(c).strip() for c in r)]
    if not table_data:
        return ""
    max_cols   = max(len(row) for row in table_data)
    table_data = [row + [""] * (max_cols - len(row)) for row in table_data]
    last_seen  = [""] * max_cols
    flattened  = []
    for row in table_data:
        new_row = []
        for j, cell in enumerate(row):
            cs = str(cell).replace("\n", " ").strip() if cell else ""
            if not cs and last_seen[j]:
                cs = last_seen[j]
            elif cs:
                last_seen[j] = cs
            new_row.append(cs)
        flattened.append(new_row)
    md_lines = []
    for i, row in enumerate(flattened):
        escaped = [c.replace("|", "\\|") for c in row]
        md_lines.append("| " + " | ".join(escaped) + " |")
        if i == 0:
            md_lines.append("|" + "|".join(["---"] * len(row)) + "|")
    return "\n".join(md_lines)


# ---------------------------------------------------------------------------
# Xử lý một file PDF
# ---------------------------------------------------------------------------
def process_single_pdf(pdf_path: Path, output_dir: Path, strategy: dict) -> dict:
    """Xử lý một file, trả về stats dict."""
    full_content = []
    page_count = table_count = char_count = filtered_blocks = 0

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            strict = {"vertical_strategy": "lines", "horizontal_strategy": "lines",
                      "snap_tolerance": 3, "join_tolerance": 3}

            bboxes = get_table_bboxes(page)
            if bboxes:
                page_text = page.filter(lambda o: not is_inside_table(o, bboxes)).extract_text()
            else:
                page_text = page.extract_text()

            if page_text:
                cleaned = clean_text(page_text, strategy)
                if cleaned:
                    if should_keep_block(cleaned, strategy):
                        full_content.append(cleaned)
                        char_count += len(cleaned)
                    else:
                        filtered_blocks += 1

            for table in page.extract_tables(table_settings=strict):
                md = table_to_markdown(table)
                if md:
                    full_content.append("\n\n" + md + "\n\n")
                    table_count += 1

    out_name = pdf_path.stem + ".md"
    out_path = output_dir / out_name

    # Nếu có merge_into: append vào file đích thay vì tạo file mới
    merge_target = strategy.get("merge_into")
    if merge_target:
        target_path = output_dir.parent / merge_target   # thư mục thongtu root
        if target_path.exists():
            marker = strategy.get("prepend_marker", "\n\n---\n## [SỬA ĐỔI BỔ SUNG]\n\n")
            with open(target_path, "a", encoding="utf-8") as f:
                f.write(marker)
                f.write("\n\n".join(full_content))
            logger.info(f"   ✓ Merged {pdf_path.name} → {target_path.name}")
        else:
            logger.warning(f"   ⚠ Không tìm thấy target merge: {target_path}, lưu riêng")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(full_content))
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_content))

    return {
        "pages": page_count, "tables": table_count,
        "chars": char_count, "filtered_blocks": filtered_blocks
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def process_pdfs():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # rglob để bắt cả thư mục con
    pdf_files = list(RAW_DIR.rglob("*.pdf"))
    logger.info(f"Tìm thấy {len(pdf_files)} file thông tư trong {RAW_DIR}")

    stats = {"total": len(pdf_files), "processed": 0, "skipped": 0, "errors": 0, "files": {}}

    for pdf_path in tqdm(pdf_files, desc="Xử lý Thông tư", unit="file"):
        if pdf_path.name in EXCLUDED_FILES:
            logger.info(f"  [SKIP] {pdf_path.name}")
            stats["skipped"] += 1
            continue

        # --- FIX output path collision: dùng rel_path đầy đủ ---
        rel_path   = pdf_path.relative_to(RAW_DIR)
        output_dir = CLEANED_DIR / rel_path.parent      # giữ nguyên cấu trúc thư mục con
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy = FILE_STRATEGIES.get(pdf_path.name, {"keep_all": True})
        logger.info(f"  -> {rel_path} | {'keep_all' if strategy.get('keep_all') else 'filtered'}")

        try:
            file_stats = process_single_pdf(pdf_path, output_dir, strategy)
            stats["processed"] += 1
            stats["files"][str(rel_path)] = file_stats
            logger.info(
                f"   ✓ {pdf_path.stem}.md | "
                f"{file_stats['pages']}tr | {file_stats['tables']}bảng | "
                f"{file_stats['chars']:,}ký tự | {file_stats['filtered_blocks']} blocks lọc"
            )
        except Exception as e:
            logger.error(f"   ✗ Lỗi {rel_path}: {e}", exc_info=True)
            stats["errors"] += 1

    logger.info("\n" + "="*60)
    logger.info(f"KẾT QUẢ: {stats['processed']} thành công | {stats['skipped']} bỏ qua | {stats['errors']} lỗi")
    stats_path = CLEANED_DIR / "_processing_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


if __name__ == "__main__":
    process_pdfs()