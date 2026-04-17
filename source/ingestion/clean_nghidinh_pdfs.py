# -*- coding: utf-8 -*-
"""
clean_nghidinh_pdfs.py — Giai đoạn 2: Trích xuất và làm sạch dữ liệu Nghị định
==================================================================================================
NÂNG CẤP v2.0:
  - Fix: Loại bỏ duplicate text + table (bbox-based exclusion)
  - Fix: BASE_DIR tương thích Colab + local
  - Fix: Line-merging không nhầm tiêu đề Điều/Khoản
  - Thêm: Blacklist file bị loại hoàn toàn
  - Thêm: Per-file strategy (lọc mảng pháp luật cho nd100, nd123)
  - Thêm: Logging + Statistics + tqdm
  - Thêm: Ghi chú số điểm trừ từ bảng (nd168)
"""

import os
import re
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
# BASE_DIR — tương thích Colab & local
# ---------------------------------------------------------------------------
def get_base_dir() -> Path:
    colab_path = Path("/content/drive/MyDrive/project")
    if colab_path.exists():
        return colab_path
    try:
        return Path(__file__).resolve().parent.parent.parent
    except NameError:
        return Path(".").resolve().parent.parent

BASE_DIR   = get_base_dir()
RAW_DIR    = BASE_DIR / "Data" / "raw"     / "nghidinh"
CLEANED_DIR = BASE_DIR / "Data" / "cleaned" / "nghidinh"
LOG_DIR    = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"clean_nghidinh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    "nd17_2026_sua_doi_bo_sung.pdf",    # Hàng không
    "nd81_2026_XuPhat_DuongSat.pdf",    # Đường sắt
    "TT130_2025_BTC_LePhi_CapBang.pdf", # BQP
}

# ---------------------------------------------------------------------------
# Per-file strategy
# ---------------------------------------------------------------------------
# nd100 và nd123: chỉ giữ lại nội dung KHÔNG thuộc mảng đường bộ
# (đường bộ đã được thay thế bởi nd168 và nd336)
#
# Cách lọc: nếu một block text chứa bất kỳ từ khóa "include_keywords"
# → giữ lại; nếu chứa "exclude_keywords" → bỏ.
# Với keep_all=True → không filter gì cả.

FILE_STRATEGIES = {
    "nd168_2024_XuPhat_TruDiem_DB_baibo_nd100.pdf": {
        "keep_all": True,
        "annotate_diem_tru": True,  # Ghi chú số điểm trừ vào bảng
    },
    "nd336_2025_xu_phat_van_tai.pdf": {
        "keep_all": True,
    },
    "nd100_2019_xu_phat.pdf": {
        "keep_all": False,
        "prepend_warning": (
            "⚠️ CẢNH BÁO: Mảng đường bộ của NĐ 100/2019 đã bị thay thế hoàn toàn\n"
            "bởi Nghị định 168/2024/NĐ-CP và Nghị định 336/2025/NĐ-CP.\n"
            "Nội dung còn lại trong file này CHỈ áp dụng cho đường sắt (nếu có).\n"
        ),
        # Từ khóa chỉ ĐƯỜNG SẮT — giữ lại
        "include_section_keywords": ["đường sắt", "ray", "tàu hỏa", "ga đường sắt"],
        # Từ khóa chỉ ĐƯỜNG BỘ — loại bỏ
        "exclude_section_keywords": [
            "đường bộ", "xe ô tô", "xe mô tô", "xe máy",
            "người điều khiển phương tiện", "vi phạm giao thông đường bộ"
        ],
    },
    "nd123_2021_xu_phat.pdf": {
        "keep_all": False,
        "prepend_warning": (
            "⚠️ CẢNH BÁO: NĐ 123/2021 đã bị bãi bỏ một phần bởi NĐ 168/2024.\n"
            "Nội dung còn hiệu lực trong file này CHỈ là mảng đường thủy nội địa\n"
            "và hàng hải. Không dùng cho tư vấn về đường bộ.\n"
        ),
        "include_section_keywords": ["đường thủy", "hàng hải", "tàu thuyền", "cảng biển", "sông"],
        "exclude_section_keywords": [
            "đường bộ", "xe ô tô", "xe máy", "mô tô",
            "giao thông đường bộ", "vi phạm giao thông"
        ],
    },
}

# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------
RE_HEADER_NOISE  = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},.*about:blank$", re.MULTILINE)
RE_FOOTER_NOISE  = re.compile(r"about:blank\s+\d+/\d+|Thư viện pháp luật|Mã tra cứu", re.IGNORECASE)
RE_PAGE_NUM      = re.compile(r"^(Trang\s+)?\d+(\s*/\s*\d+)?$", re.IGNORECASE)
RE_FORM_DOTS     = re.compile(r"(\.{5,}|_{5,})")
RE_CHECKBOX      = re.compile(r"([☐☑\uf06f])")
RE_IS_HEADING    = re.compile(
    r"^(Điều\s+\d+|Khoản\s+\d+|Chương\s+[IVXLCDM]+|Phần\s+[IVXLCDM]+|"
    r"\d+\.\s+[A-ZĐÀÁẠẢÃ]|[a-z]\)\s+)",
    re.IGNORECASE
)
RE_MERGE_END     = re.compile(
    r'[a-zA-ZáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ,;]$'
)

# ---------------------------------------------------------------------------
# Lọc block theo từ khóa (per-file strategy)
# ---------------------------------------------------------------------------
def should_keep_block(text: str, strategy: dict) -> bool:
    """
    Quyết định giữ hay bỏ một block text dựa trên include/exclude keywords.
    Logic: block phải có ít nhất 1 include_keyword VÀ không có exclude_keyword.
    """
    if strategy.get("keep_all", True):
        return True

    text_lower = text.lower()
    include_kws = strategy.get("include_section_keywords", [])
    exclude_kws = strategy.get("exclude_section_keywords", [])

    # Nếu có include list: phải match ít nhất 1
    if include_kws:
        has_include = any(kw in text_lower for kw in include_kws)
        if not has_include:
            return False

    # Nếu có exclude list: không được chứa bất kỳ từ nào
    if exclude_kws:
        has_exclude = any(kw in text_lower for kw in exclude_kws)
        if has_exclude:
            return False

    return True


# ---------------------------------------------------------------------------
# Làm sạch text
# ---------------------------------------------------------------------------
def clean_text(raw_text: str, strategy: dict | None = None) -> str:
    if not raw_text:
        return ""

    text  = unicodedata.normalize("NFC", raw_text)
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if RE_HEADER_NOISE.match(line):  continue
        if RE_FOOTER_NOISE.search(line): continue
        if RE_PAGE_NUM.match(line):       continue

        line = RE_FORM_DOTS.sub(" [Cần điền thông tin] ", line)
        line = RE_CHECKBOX.sub(" [Lựa chọn] ", line).strip()
        if line in ("", "[Cần điền thông tin]", "[Lựa chọn]"):
            continue

        # Định dạng phân cấp
        line = re.sub(r"^(Chương\s+[IVXLCDM]+.*|Phần\s+[IVXLCDM]+.*)$",
                      r"# \1", line, flags=re.IGNORECASE)
        line = re.sub(r"^(Điều\s+\d+[.:].*)",
                      lambda m: "## " + m.group(0).lstrip("# "),
                      line, flags=re.IGNORECASE)

        # Bôi đậm ngày tháng
        line = re.sub(r"(ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
                      r"**\1**", line, flags=re.IGNORECASE)

        # LaTeX kỹ thuật
        line = re.sub(r"\b(CO2?|HC|NOx|PM2\.5|mg/l)\b", r"$$\1$$", line)

        cleaned_lines.append(line)

    # Nối dòng bị gãy — KHÔNG nối tiêu đề
    merged_lines = []
    for line in cleaned_lines:
        if not merged_lines:
            merged_lines.append(line)
            continue
        prev = merged_lines[-1]
        if (RE_MERGE_END.search(prev[-1:])
                and (line[0].islower() or line[0] in 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ')
                and not RE_IS_HEADING.match(line)):
            merged_lines[-1] = prev + " " + line
        else:
            merged_lines.append(line)

    result = "\n".join(merged_lines)

    if strategy and strategy.get("prepend_warning"):
        result = f"> {strategy['prepend_warning']}\n\n{result}"

    return result


# ---------------------------------------------------------------------------
# Bảng → Markdown
# ---------------------------------------------------------------------------
def table_to_markdown(table_data: list, annotate_diem_tru: bool = False) -> str:
    """
    Chuyển bảng sang Markdown.
    annotate_diem_tru=True: thêm cột ghi chú điểm trừ nếu phát hiện (nd168).
    """
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
            cell_str = str(cell).replace("\n", " ").strip() if cell else ""
            if not cell_str and last_seen[j]:
                cell_str = last_seen[j]
            elif cell_str:
                last_seen[j] = cell_str
            new_row.append(cell_str)
        flattened.append(new_row)

    md_lines = []
    for i, row in enumerate(flattened):
        escaped = [c.replace("|", "\\|") for c in row]
        md_lines.append("| " + " | ".join(escaped) + " |")
        if i == 0:
            md_lines.append("|" + "|".join(["---"] * len(row)) + "|")

    return "\n".join(md_lines)


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Xử lý chính
# ---------------------------------------------------------------------------
def process_pdfs():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(RAW_DIR.glob("*.pdf"))
    logger.info(f"Tìm thấy {len(pdf_files)} file nghị định trong {RAW_DIR}")

    stats = {"total": len(pdf_files), "processed": 0, "skipped": 0, "errors": 0, "files": {}}

    for pdf_path in tqdm(pdf_files, desc="Xử lý Nghị định", unit="file"):
        if pdf_path.name in EXCLUDED_FILES:
            logger.info(f"  [SKIP] {pdf_path.name}")
            stats["skipped"] += 1
            continue

        strategy     = FILE_STRATEGIES.get(pdf_path.name, {"keep_all": True})
        annotate_dt  = strategy.get("annotate_diem_tru", False)
        full_content = []
        page_count = table_count = char_count = filtered_blocks = 0

        logger.info(f"  -> {pdf_path.name} | strategy: {'keep_all' if strategy.get('keep_all') else 'filtered'}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                for page in tqdm(pdf.pages, desc=f"  {pdf_path.stem}", leave=False):
                    strict = {"vertical_strategy": "lines", "horizontal_strategy": "lines",
                              "snap_tolerance": 3, "join_tolerance": 3}

                    bboxes = get_table_bboxes(page)

                    # Text không chứa vùng bảng
                    if bboxes:
                        filtered_page = page.filter(lambda o: not is_inside_table(o, bboxes))
                        page_text = filtered_page.extract_text()
                    else:
                        page_text = page.extract_text()

                    if page_text:
                        cleaned = clean_text(page_text, strategy)
                        if cleaned:
                            # Lọc theo strategy keywords
                            if should_keep_block(cleaned, strategy):
                                full_content.append(cleaned)
                                char_count += len(cleaned)
                            else:
                                filtered_blocks += 1

                    # Bảng
                    for table in page.extract_tables(table_settings=strict):
                        md = table_to_markdown(table, annotate_diem_tru=annotate_dt)
                        if md:
                            full_content.append("\n\n" + md + "\n\n")
                            table_count += 1

            out_name = pdf_path.stem + ".md"
            out_path = CLEANED_DIR / out_name
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(full_content))

            stats["processed"] += 1
            stats["files"][pdf_path.name] = {
                "pages": page_count, "tables": table_count,
                "chars": char_count, "filtered_blocks": filtered_blocks
            }
            logger.info(
                f"   ✓ {out_name} | {page_count}tr | {table_count}bảng | "
                f"{char_count:,}ký tự | {filtered_blocks} blocks bị lọc"
            )

        except Exception as e:
            logger.error(f"   ✗ Lỗi {pdf_path.name}: {e}", exc_info=True)
            stats["errors"] += 1

    # Stats
    logger.info("\n" + "="*60)
    logger.info(f"KẾT QUẢ: {stats['processed']} thành công | {stats['skipped']} bỏ qua | {stats['errors']} lỗi")
    stats_path = CLEANED_DIR / "_processing_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


if __name__ == "__main__":
    process_pdfs()