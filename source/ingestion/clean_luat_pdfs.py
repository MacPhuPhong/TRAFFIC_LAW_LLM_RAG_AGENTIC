# -*- coding: utf-8 -*-
"""
clean_luat_pdfs.py — Giai đoạn 2: Trích xuất và làm sạch dữ liệu Luật
==============================================================================
NÂNG CẤP v2.0:
  - Fix: Loại bỏ duplicate text + table (bbox-based exclusion)
  - Fix: BASE_DIR tương thích cả Colab lẫn local
  - Fix: Line-merging không nhầm tiêu đề Điều/Khoản
  - Thêm: Blacklist file bị loại hoàn toàn
  - Thêm: Per-file strategy (filter theo mảng pháp luật)
  - Thêm: Logging ra file + console
  - Thêm: Statistics sau khi xử lý
  - Thêm: tqdm progress bar
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
    # Fallback nếu chưa cài tqdm
    def tqdm(iterable, **kwargs):
        return iterable

# ---------------------------------------------------------------------------
# Cấu hình đường dẫn — tương thích Colab và local
# ---------------------------------------------------------------------------
def get_base_dir() -> Path:
    """Tự động phát hiện môi trường Colab hoặc local."""
    colab_path = Path("/content/drive/MyDrive/project")
    if colab_path.exists():
        return colab_path
    # Fallback: 3 cấp lên từ file hiện tại (nếu chạy local)
    try:
        return Path(__file__).resolve().parent.parent.parent
    except NameError:
        # __file__ không tồn tại trong notebook cell
        return Path(".").resolve().parent.parent

BASE_DIR = get_base_dir()
RAW_LUAT_DIR   = BASE_DIR / "Data" / "raw"    / "luat"
CLEANED_LUAT_DIR = BASE_DIR / "Data" / "cleaned" / "luat"
LOG_DIR        = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------------
def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"clean_luat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
# Blacklist — các file KHÔNG xử lý
# ---------------------------------------------------------------------------
EXCLUDED_FILES = {
    "nd17_2026_sua_doi_bo_sung.pdf",   # Hàng không — ngoài phạm vi
    "nd81_2026_XuPhat_DuongSat.pdf",   # Đường sắt — ngoài phạm vi
    "TT130_2025_BTC_LePhi_CapBang.pdf",# Bộ Quốc phòng — ngoài phạm vi
    "luat23_db_2008.pdf",              # Luật 2008 đã hết hiệu lực — dùng bản inactive riêng
}

# ---------------------------------------------------------------------------
# Per-file strategy — logic xử lý riêng từng văn bản
# ---------------------------------------------------------------------------
FILE_STRATEGIES = {
    # Luật 35 & 36 giữ toàn bộ
    "luat35_db_2024.pdf":     {"keep_all": True},
    "luat36_ttatgt_2024.pdf": {"keep_all": True},
    # Luật 2008: giữ nhưng gán inactive, chỉ dùng tra cứu lịch sử
    "luat23_db_2008_inactive.pdf": {
        "keep_all": True,
        "prepend_warning": (
            "⚠️ LƯU Ý: VĂN BẢN NÀY ĐÃ HẾT HIỆU LỰC KỂ TỪ 01/01/2025.\n"
            "Luật Giao thông đường bộ 2008 được thay thế hoàn toàn bởi:\n"
            "  • Luật Đường bộ 35/2024/QH15\n"
            "  • Luật Trật tự ATGT đường bộ 36/2024/QH15\n"
            "Chỉ sử dụng văn bản này cho mục đích tra cứu lịch sử pháp lý.\n"
        )
    },
}

# ---------------------------------------------------------------------------
# Regex làm sạch
# ---------------------------------------------------------------------------
RE_HEADER_NOISE  = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},.*about:blank$",     re.MULTILINE)
RE_FOOTER_NOISE  = re.compile(r"about:blank\s+\d+/\d+|Thư viện pháp luật|Mã tra cứu", re.IGNORECASE)
RE_PAGE_NUM      = re.compile(r"^(Trang\s+)?\d+(\s*/\s*\d+)?$",               re.IGNORECASE)
RE_FORM_DOTS     = re.compile(r"(\.{5,}|_{5,})")
RE_CHECKBOX      = re.compile(r"([☐☑\uf06f])")

# Phát hiện tiêu đề cấu trúc — KHÔNG được nối với dòng trên
RE_IS_HEADING    = re.compile(
    r"^(Điều\s+\d+|Khoản\s+\d+|Chương\s+[IVXLCDM]+|Phần\s+[IVXLCDM]+|"
    r"\d+\.\s+[A-ZĐÀÁẠẢÃ]|[a-z]\)\s+)",
    re.IGNORECASE
)

# Ký tự kết thúc câu — dòng kết thúc bằng những ký tự này mới được nối
RE_MERGE_ELIGIBLE_END = re.compile(
    r'[a-zA-ZáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ,;]$'
)

# ---------------------------------------------------------------------------
# Hàm làm sạch text
# ---------------------------------------------------------------------------
def clean_text(raw_text: str, strategy: dict | None = None) -> str:
    """
    Làm sạch text: loại nhiễu, định dạng phân cấp, nối dòng bị gãy.
    Không nối nhầm dòng tiêu đề Điều/Khoản vào dòng trước.
    """
    if not raw_text:
        return ""

    text = unicodedata.normalize("NFC", raw_text)
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # --- Lọc nhiễu ---
        if RE_HEADER_NOISE.match(line):   continue
        if RE_FOOTER_NOISE.search(line):  continue
        if RE_PAGE_NUM.match(line):        continue

        # --- Lọc biểu mẫu ---
        line = RE_FORM_DOTS.sub(" [Cần điền thông tin] ", line)
        line = RE_CHECKBOX.sub(" [Lựa chọn] ", line).strip()
        if line in ("", "[Cần điền thông tin]", "[Lựa chọn]"):
            continue

        # --- Định dạng phân cấp Markdown ---
        line = re.sub(
            r"^(Chương\s+[IVXLCDM]+.*|Phần\s+[IVXLCDM]+.*)$",
            r"# \1", line, flags=re.IGNORECASE
        )
        line = re.sub(
            r"^(Điều\s+\d+[.:].*|## Điều\s+\d+[.:].*)",
            lambda m: "## " + m.group(0).lstrip("# "),
            line, flags=re.IGNORECASE
        )

        # --- Bôi đậm ngày tháng ---
        line = re.sub(
            r"(ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
            r"**\1**", line, flags=re.IGNORECASE
        )

        # --- LaTeX cho công thức kỹ thuật ---
        line = re.sub(r"\b(CO2?|HC|NOx|PM2\.5)\b", r"$$\1$$", line)

        cleaned_lines.append(line)

    # --- Nối dòng bị gãy (cải tiến: không nối tiêu đề) ---
    merged_lines = []
    for line in cleaned_lines:
        if not merged_lines:
            merged_lines.append(line)
            continue

        prev = merged_lines[-1]
        prev_ends_mid_sentence = RE_MERGE_ELIGIBLE_END.search(prev[-1:]) if prev else False
        curr_is_heading        = RE_IS_HEADING.match(line)
        curr_starts_lowercase  = (
            line[0].islower() or
            line[0] in 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
        )

        if prev_ends_mid_sentence and curr_starts_lowercase and not curr_is_heading:
            merged_lines[-1] = prev + " " + line
        else:
            merged_lines.append(line)

    result = "\n".join(merged_lines)

    # Thêm warning nếu strategy yêu cầu
    if strategy and strategy.get("prepend_warning"):
        result = f"> {strategy['prepend_warning']}\n\n{result}"

    return result


# ---------------------------------------------------------------------------
# Chuyển bảng sang Markdown (với flatten merged cells)
# ---------------------------------------------------------------------------
def table_to_markdown(table_data: list) -> str:
    """Chuyển list-of-lists thành Markdown Table, flatten ô gộp chiều dọc."""
    if not table_data:
        return ""

    # Lọc hàng rỗng
    table_data = [
        row for row in table_data
        if any(cell and str(cell).strip() for cell in row)
    ]
    if not table_data:
        return ""

    # Đảm bảo tất cả hàng cùng độ rộng
    max_cols = max(len(row) for row in table_data)
    table_data = [row + [""] * (max_cols - len(row)) for row in table_data]

    last_seen = [""] * max_cols
    flattened = []

    for row in table_data:
        new_row = []
        for j, cell in enumerate(row):
            cell_str = str(cell).replace("\n", " ").strip() if cell else ""
            if not cell_str and last_seen[j]:
                cell_str = last_seen[j]   # flatten merged cell
            elif cell_str:
                last_seen[j] = cell_str
            new_row.append(cell_str)
        flattened.append(new_row)

    md_lines = []
    for i, row in enumerate(flattened):
        # Escape pipe trong nội dung ô
        escaped = [cell.replace("|", "\\|") for cell in row]
        md_lines.append("| " + " | ".join(escaped) + " |")
        if i == 0:
            md_lines.append("|" + "|".join(["---"] * len(row)) + "|")

    return "\n".join(md_lines)


# ---------------------------------------------------------------------------
# Lấy bbox của tất cả bảng trên một trang (để loại khỏi text extract)
# ---------------------------------------------------------------------------
def get_table_bboxes(page) -> list[tuple]:
    """Trả về list bbox (x0,top,x1,bottom) của tất cả bảng trên trang."""
    try:
        return [tbl.bbox for tbl in page.find_tables()]
    except Exception:
        return []


def is_inside_table(obj: dict, bboxes: list[tuple], tolerance: float = 2.0) -> bool:
    """Kiểm tra obj có nằm trong vùng bảng không."""
    x0 = obj.get("x0", 0)
    top = obj.get("top", 0)
    for (tx0, ttop, tx1, tbottom) in bboxes:
        if (tx0 - tolerance <= x0 and
                x0 <= tx1 + tolerance and
                ttop - tolerance <= top and
                top <= tbottom + tolerance):
            return True
    return False


# ---------------------------------------------------------------------------
# Hàm xử lý chính
# ---------------------------------------------------------------------------
def process_law_pdfs():
    CLEANED_LUAT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(RAW_LUAT_DIR.glob("*.pdf"))
    logger.info(f"Tìm thấy {len(pdf_files)} file luật trong {RAW_LUAT_DIR}")

    stats = {
        "total": len(pdf_files),
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "files": {}
    }

    for pdf_path in tqdm(pdf_files, desc="Xử lý file Luật", unit="file"):
        # --- Kiểm tra blacklist ---
        if pdf_path.name in EXCLUDED_FILES:
            logger.info(f"  [SKIP] {pdf_path.name} — nằm trong danh sách loại trừ")
            stats["skipped"] += 1
            continue

        strategy = FILE_STRATEGIES.get(pdf_path.name, {"keep_all": True})
        logger.info(f"  -> Đang xử lý: {pdf_path.name} | strategy: {strategy}")

        full_content = []
        page_count   = 0
        table_count  = 0
        char_count   = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)

                for page in tqdm(pdf.pages, desc=f"  Trang {pdf_path.stem}", leave=False):
                    strict_settings = {
                        "vertical_strategy":   "lines",
                        "horizontal_strategy": "lines",
                        "snap_tolerance":      3,
                        "join_tolerance":      3,
                    }

                    # --- FIX DUPLICATE: Lấy bboxes bảng trước ---
                    table_bboxes = get_table_bboxes(page)

                    # --- Trích text KHÔNG bao gồm vùng bảng ---
                    if table_bboxes:
                        filtered_page = page.filter(
                            lambda obj: not is_inside_table(obj, table_bboxes)
                        )
                        page_text = filtered_page.extract_text()
                    else:
                        page_text = page.extract_text()

                    if page_text:
                        cleaned = clean_text(page_text, strategy)
                        if cleaned:
                            full_content.append(cleaned)
                            char_count += len(cleaned)

                    # --- Trích bảng riêng ---
                    extracted_tables = page.extract_tables(table_settings=strict_settings)
                    for table in extracted_tables:
                        md_table = table_to_markdown(table)
                        if md_table:
                            full_content.append("\n\n" + md_table + "\n\n")
                            table_count += 1

            # --- Lưu kết quả ---
            output_filename = pdf_path.stem + ".md"
            output_path     = CLEANED_LUAT_DIR / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(full_content))

            stats["processed"] += 1
            stats["files"][pdf_path.name] = {
                "pages": page_count,
                "tables": table_count,
                "chars": char_count,
                "output": str(output_path)
            }
            logger.info(
                f"   ✓ Hoàn tất: {output_filename} "
                f"| {page_count} trang | {table_count} bảng | {char_count:,} ký tự"
            )

        except Exception as e:
            logger.error(f"   ✗ Lỗi khi xử lý {pdf_path.name}: {e}", exc_info=True)
            stats["errors"] += 1

    # --- In thống kê cuối ---
    logger.info("\n" + "="*60)
    logger.info(f"THỐNG KÊ XỬ LÝ LUẬT:")
    logger.info(f"  Tổng:        {stats['total']}")
    logger.info(f"  Thành công:  {stats['processed']}")
    logger.info(f"  Bỏ qua:      {stats['skipped']}")
    logger.info(f"  Lỗi:         {stats['errors']}")

    # Lưu stats ra JSON
    stats_path = CLEANED_LUAT_DIR / "_processing_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"  Stats đã lưu tại: {stats_path}")

    return stats


if __name__ == "__main__":
    process_law_pdfs()