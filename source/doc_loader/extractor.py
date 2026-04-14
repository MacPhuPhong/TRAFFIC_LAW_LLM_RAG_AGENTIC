# -*- coding: utf-8 -*-
"""
extractor.py — Extract text từ file .docx
==========================================
Dùng python-docx để đọc paragraphs, headings và tables.
Bảng được convert sang dạng markdown-style text.
Hỗ trợ encoding UTF-8 và chuẩn hoá NFC cho tiếng Việt.
"""

import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Chuẩn hoá chuỗi về NFC (quan trọng cho tiếng Việt)."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text.strip())


def _get_paragraph_level(paragraph: Any) -> Optional[int]:
    """
    Trả về cấp độ heading (1-9) nếu paragraph là heading,
    hoặc None nếu là đoạn thường.
    """
    style_name = paragraph.style.name if paragraph.style else ""
    style_lower = style_name.lower()
    if style_lower.startswith("heading"):
        parts = style_lower.split()
        for part in parts:
            try:
                return int(part)
            except ValueError:
                continue
    return None


def _table_to_markdown(table: Any) -> str:
    """
    Chuyển đổi bảng python-docx thành chuỗi markdown-style.

    Output format:
        [BẢNG]
        | Cột 1 | Cột 2 | Cột 3 |
        | val1  | val2  | val3  |
        [/BẢNG]
    """
    rows_text: List[str] = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            cell_text = _normalize_text(cell.text.replace("\n", " "))
            cells.append(cell_text)
        rows_text.append("| " + " | ".join(cells) + " |")

    if not rows_text:
        return ""

    return "\n[BẢNG]\n" + "\n".join(rows_text) + "\n[/BẢNG]\n"


def extract_text_from_docx(docx_path: str) -> Dict:
    """
    Extract toàn bộ text từ file .docx.

    Args:
        docx_path: Đường dẫn đến file .docx

    Returns:
        dict với các keys:
          - full_text  (str):        Toàn bộ text đã ghép
          - sections   (list[dict]): Danh sách section theo heading
          - has_tables (bool):       File có chứa bảng không
          - word_count (int):        Số từ ước tính
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx chưa được cài. Chạy: pip install python-docx"
        )

    docx_path = str(Path(docx_path).resolve())

    try:
        doc = Document(docx_path)
    except Exception as e:
        logger.error(f"Không thể mở file {docx_path}: {e}")
        return {
            "full_text": "",
            "sections": [],
            "has_tables": False,
            "word_count": 0,
        }

    # ------------------------------------------------------------------
    # Lấy tập hợp XML elements theo thứ tự xuất hiện trong document body
    # (bao gồm cả paragraph và table xen kẽ nhau)
    # ------------------------------------------------------------------
    from docx.oxml.ns import qn

    body = doc.element.body
    full_parts: List[str] = []
    sections: List[Dict] = []
    has_tables = False

    current_heading: Optional[str] = None
    current_heading_level: int = 0
    current_section_parts: List[str] = []

    def _flush_section(heading: Optional[str], level: int, parts: List[str]) -> Dict:
        return {
            "heading": heading or "Mở đầu",
            "heading_level": level,
            "text": _normalize_text("\n".join(parts)),
        }

    # Duyệt qua từng element trong body
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "p":
            # -- Paragraph --
            # Tìm paragraph object tương ứng trong doc.paragraphs
            # bằng cách map qua XML element
            para_text_parts = []
            for run in child.findall(f".//{qn('w:r')}"):
                t_elements = run.findall(qn("w:t"))
                for t in t_elements:
                    if t.text:
                        para_text_parts.append(t.text)
            raw_para = "".join(para_text_parts)
            para_text = _normalize_text(raw_para)
            if not para_text:
                continue

            # Xác định style/heading
            pPr = child.find(qn("w:pPr"))
            heading_level: Optional[int] = None
            if pPr is not None:
                pStyle = pPr.find(qn("w:pStyle"))
                if pStyle is not None:
                    val = pStyle.get(qn("w:val"), "").lower()
                    if "heading" in val:
                        digits = "".join(filter(str.isdigit, val))
                        if digits:
                            heading_level = int(digits)

            if heading_level is not None:
                # Flush section trước
                if current_section_parts:
                    sections.append(
                        _flush_section(
                            current_heading, current_heading_level, current_section_parts
                        )
                    )
                    current_section_parts = []
                current_heading = para_text
                current_heading_level = heading_level
                full_parts.append(f"\n{'#' * heading_level} {para_text}\n")
            else:
                current_section_parts.append(para_text)
                full_parts.append(para_text)

        elif tag == "tbl":
            # -- Table --
            has_tables = True
            # Tìm table object tương ứng
            # Dùng python-docx's Table class
            from docx.table import Table as DocxTable

            try:
                tbl_obj = DocxTable(child, doc)
                md_table = _table_to_markdown(tbl_obj)
                if md_table:
                    current_section_parts.append(md_table)
                    full_parts.append(md_table)
            except Exception as te:
                logger.warning(f"Lỗi parse table: {te}")

    # Flush section cuối
    if current_section_parts:
        sections.append(
            _flush_section(current_heading, current_heading_level, current_section_parts)
        )

    full_text = _normalize_text("\n".join(full_parts))
    word_count = len(full_text.split())

    logger.info(
        f"Extracted: {docx_path} | "
        f"{word_count} words | {len(sections)} sections | tables={has_tables}"
    )

    return {
        "full_text": full_text,
        "sections": sections,
        "has_tables": has_tables,
        "word_count": word_count,
    }
