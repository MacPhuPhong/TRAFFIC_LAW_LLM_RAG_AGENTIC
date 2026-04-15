# -*- coding: utf-8 -*-
"""
extractor.py — Extract text từ file .docx (Pro Edition)
=========================================================
Nâng cấp so với phiên bản cũ:
  1. IMAGE EXTRACTION: Trích xuất ảnh nhúng → lưu vào assets/ + Markdown placeholder
  2. TABLE FLATTENING: Xử lý merged cells → bảng Markdown phẳng chính xác
  3. SECTION PRIORITY: Nhận diện mục 3, 4, 5 trong QCVN/TCVN → đánh dấu priority=high
"""

import hashlib
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Regex nhận diện mục có độ ưu tiên cao (3, 4, 5) trong QCVN/TCVN
# ------------------------------------------------------------------
RE_HIGH_PRIORITY_SECTION = re.compile(
    r"^(?:mục|phần|chương)?\s*[345][\.\s]|"
    r"^3\.\s*(quy định|thuật ngữ|định nghĩa)|"
    r"^4\.\s*(phân loại|phạm vi|yêu cầu)|"
    r"^5\.\s*(yêu cầu kỹ thuật|tiêu chuẩn|chỉ tiêu)",
    re.IGNORECASE | re.UNICODE,
)

# Phần mở rộng ảnh phổ biến trong DOCX
IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/wmf": ".wmf",
    "image/emf": ".emf",
}


def _normalize_text(text: str) -> str:
    """Chuẩn hoá chuỗi về NFC (quan trọng cho tiếng Việt)."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text.strip())


# ==================================================================
# 1. IMAGE EXTRACTION
# ==================================================================

def _extract_images_from_docx(docx_path: str, assets_dir: str) -> Dict[str, str]:
    """
    Trích xuất tất cả ảnh nhúng trong DOCX ra thư mục assets/.
    Returns: dict mapping  relationship_id → absolute file path
    """
    os.makedirs(assets_dir, exist_ok=True)
    rId_to_path: Dict[str, str] = {}

    try:
        from docx import Document
        doc = Document(docx_path)
        part = doc.part

        for rId, rel in part.rels.items():
            if "image" not in rel.target_ref.lower():
                continue
            try:
                image_part = rel.target_part
                image_blob = image_part.blob
                content_type = image_part.content_type
                ext = IMAGE_EXTENSIONS.get(content_type, ".bin")
                # Tên file = MD5 đầu 8 ký tự để tránh trùng
                md5 = hashlib.md5(image_blob).hexdigest()[:8]
                filename = f"img_{md5}{ext}"
                img_path = os.path.join(assets_dir, filename)
                if not os.path.isfile(img_path):
                    with open(img_path, "wb") as f:
                        f.write(image_blob)
                rId_to_path[rId] = img_path
                logger.debug(f"Extracted image: {rId} → {img_path}")
            except Exception as e:
                logger.debug(f"Skip image {rId}: {e}")

    except Exception as e:
        logger.warning(f"Image extraction failed for {docx_path}: {e}")

    return rId_to_path


def _build_image_placeholder(img_path: str, caption: str = "") -> str:
    """Tạo placeholder Markdown cho ảnh. #Bổ_sung_mô_tả nếu chưa có caption."""
    cap = caption if caption else "Hình ảnh trong tài liệu"
    tag = " #Bổ_sung_mô_tả" if not caption else ""
    return f"\n![{cap}]({img_path}){tag}\n"


# ==================================================================
# 2. TABLE FLATTENING  (merged cells → flat Markdown)
# ==================================================================

def _table_to_flat_markdown(table: Any) -> str:
    """
    Chuyển bảng python-docx thành Markdown phẳng.
    Xử lý merged cells: python-docx's row.cells đã tự expand ô gộp,
    ta chỉ cần deduplicate giá trị liền kề giống nhau trong cùng một hàng.
    """
    num_rows = len(table.rows)
    if num_rows == 0:
        return ""

    grid: List[List[str]] = []
    for row in table.rows:
        raw = [_normalize_text(cell.text.replace("\n", " ")) for cell in row.cells]
        # Bỏ trùng liền kề (do merged cell bị repeat bởi python-docx)
        deduped: List[str] = []
        prev = object()
        for val in raw:
            if val != prev:
                deduped.append(val)
            prev = val
        grid.append(deduped)

    if not grid:
        return ""

    # Cân chỉnh số cột
    max_cols = max(len(r) for r in grid)
    normalized = [r + [""] * (max_cols - len(r)) for r in grid]

    lines: List[str] = []
    for i, row in enumerate(normalized):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(row)) + " |")

    return "\n[BẢNG]\n" + "\n".join(lines) + "\n[/BẢNG]\n"


# ==================================================================
# 3. SECTION PRIORITY DETECTION
# ==================================================================

def _check_section_priority(paragraph_text: str, current_heading: Optional[str]) -> str:
    """
    Trả về 'high' nếu đoạn thuộc mục 3, 4, 5 của QCVN/TCVN, else 'normal'.
    """
    for text in [paragraph_text, current_heading or ""]:
        if RE_HIGH_PRIORITY_SECTION.match(text.strip()):
            return "high"
    return "normal"


# ==================================================================
# 4. MAIN PUBLIC FUNCTION
# ==================================================================

def extract_text_from_docx(
    docx_path: str,
    assets_dir: Optional[str] = None,
) -> Dict:
    """
    Extract toàn bộ text từ file .docx — Pro Edition.

    Args:
        docx_path:  Đường dẫn đến file .docx
        assets_dir: Thư mục lưu ảnh trích xuất.
                    Nếu None → tự tạo '<docx_dir>/assets/<stem>/'

    Returns dict:
        full_text       (str)        : Toàn bộ text dạng Markdown
        sections        (list[dict]) : Danh sách section theo heading
        has_tables      (bool)
        has_images      (bool)
        word_count      (int)
        image_paths     (list[str])  : Đường dẫn ảnh đã trích xuất
        priority_chunks (list[str])  : Text của các chunk priority=high
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx chưa được cài. Chạy: pip install python-docx")

    docx_path = str(Path(docx_path).resolve())

    # Thư mục chứa ảnh
    if assets_dir is None:
        stem = Path(docx_path).stem
        assets_dir = str(Path(docx_path).parent / "assets" / stem)

    # ── Bước 1: Trích xuất ảnh ──────────────────────────────────────
    rId_to_img = _extract_images_from_docx(docx_path, assets_dir)
    image_paths = list(rId_to_img.values())

    # ── Bước 2: Mở document ─────────────────────────────────────────
    try:
        doc = Document(docx_path)
    except Exception as e:
        logger.error(f"Không thể mở file {docx_path}: {e}")
        return _empty_result()

    from docx.oxml.ns import qn

    body = doc.element.body
    full_parts: List[str] = []
    sections: List[Dict] = []
    has_tables = False
    priority_chunks: List[str] = []

    current_heading: Optional[str] = None
    current_heading_level: int = 0
    current_section_parts: List[str] = []
    current_section_priority: str = "normal"

    def _flush_section() -> None:
        nonlocal current_section_parts, current_section_priority
        if not current_section_parts:
            return
        sec_text = _normalize_text("\n".join(current_section_parts))
        sections.append({
            "heading": current_heading or "Mở đầu",
            "heading_level": current_heading_level,
            "text": sec_text,
            "priority": current_section_priority,
        })
        if current_section_priority == "high":
            priority_chunks.append(sec_text)
        current_section_parts.clear()
        current_section_priority = "normal"

    # ── Bước 3: Duyệt body theo thứ tự ─────────────────────────────
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        # ── Paragraph ──────────────────────────────────────────────
        if tag == "p":
            # Ghép text từ tất cả run
            para_text_parts: List[str] = []
            for run in child.findall(f".//{qn('w:r')}"):
                for t in run.findall(qn("w:t")):
                    if t.text:
                        para_text_parts.append(t.text)
            para_text = _normalize_text("".join(para_text_parts))

            if not para_text:
                # Đoạn trống — kiểm tra có ảnh nhúng không
                for blip in child.findall(f".//{qn('a:blip')}"):
                    embed = blip.get(qn("r:embed"), "")
                    if embed in rId_to_img:
                        ph = _build_image_placeholder(rId_to_img[embed])
                        current_section_parts.append(ph)
                        full_parts.append(ph)
                continue

            # Xác định heading level từ style
            heading_level: Optional[int] = None
            pPr = child.find(qn("w:pPr"))
            if pPr is not None:
                pStyle = pPr.find(qn("w:pStyle"))
                if pStyle is not None:
                    val = pStyle.get(qn("w:val"), "").lower()
                    if "heading" in val:
                        digits = "".join(filter(str.isdigit, val))
                        if digits:
                            heading_level = int(digits)

            if heading_level is not None:
                # Flush section cũ, bắt đầu section mới
                _flush_section()
                current_heading = para_text
                current_heading_level = heading_level
                current_section_priority = _check_section_priority(para_text, None)
                full_parts.append(f"\n{'#' * heading_level} {para_text}\n")
            else:
                prio = _check_section_priority(para_text, current_heading)
                if prio == "high":
                    current_section_priority = "high"
                current_section_parts.append(para_text)
                full_parts.append(para_text)

        # ── Table ──────────────────────────────────────────────────
        elif tag == "tbl":
            has_tables = True
            try:
                from docx.table import Table as DocxTable
                tbl_obj = DocxTable(child, doc)
                md_table = _table_to_flat_markdown(tbl_obj)
                if md_table:
                    current_section_parts.append(md_table)
                    full_parts.append(md_table)
            except Exception as te:
                logger.warning(f"Lỗi parse table: {te}")

    # Flush section cuối
    _flush_section()

    full_text = _normalize_text("\n".join(full_parts))
    word_count = len(full_text.split())
    high_prio = sum(1 for s in sections if s["priority"] == "high")

    logger.info(
        f"Extracted [Pro]: {Path(docx_path).name} | "
        f"{word_count} words | {len(sections)} sections | "
        f"tables={has_tables} | images={len(image_paths)} | high_priority={high_prio}"
    )

    return {
        "full_text": full_text,
        "sections": sections,
        "has_tables": has_tables,
        "has_images": bool(image_paths),
        "word_count": word_count,
        "image_paths": image_paths,
        "priority_chunks": priority_chunks,
    }


def _empty_result() -> Dict:
    return {
        "full_text": "", "sections": [], "has_tables": False,
        "has_images": False, "word_count": 0,
        "image_paths": [], "priority_chunks": [],
    }
