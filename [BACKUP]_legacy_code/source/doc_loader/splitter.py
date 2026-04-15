# -*- coding: utf-8 -*-
"""
splitter.py — Chunking thông minh theo loại tài liệu hành chính (Pro Edition)
===============================================================================
Nâng cấp:
  - Priority Chunking: chunk từ mục 3, 4, 5 của QCVN/TCVN có metadata chunk_priority='high'
  - Giữ toàn bộ logic cũ, chỉ thêm metadata priority vào từng chunk
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns split
# ---------------------------------------------------------------------------
RE_SECTION_HEADER = re.compile(
    r"(?m)(?=^[IVX]+\.\s|^\d+\.\s[A-ZĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẨẪ]"
    r"|^\b[A-ZĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẨẪ]{3,}\s*:)",
    re.UNICODE,
)
RE_DIEU = re.compile(r"(?=Điều\s+\d+[\.\:])", re.UNICODE)
RE_MAU_SO = re.compile(
    r"(?=M[aẫ]u\s+s[oố]\s*\d+[\-\.\s]?[A-Za-z0-9]*)",
    re.UNICODE | re.IGNORECASE,
)

# Regex nhận diện mục 3, 4, 5 ở đầu đoạn (cho priority chunking)
RE_HIGH_PRIORITY = re.compile(
    r"^(?:mục|phần|chương)?\s*[345][\.\s]|"
    r"^3\.\s*(quy định|thuật ngữ|định nghĩa)|"
    r"^4\.\s*(phân loại|phạm vi|yêu cầu)|"
    r"^5\.\s*(yêu cầu kỹ thuật|tiêu chuẩn|chỉ tiêu)",
    re.IGNORECASE | re.UNICODE,
)


def _assign_priority(text: str) -> str:
    """Kiểm tra text có thuộc mục ưu tiên (3, 4, 5) không."""
    first_line = text.strip().split("\n")[0]
    return "high" if RE_HIGH_PRIORITY.match(first_line) else "normal"


class AdminDocumentSplitter:
    """
    Chunking tài liệu hành chính theo loại tài liệu.
    Mỗi chunk là dict: {"text": str, "strategy": str, "priority": str}
    """

    def split(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Dispatch đến hàm split phù hợp dựa trên loại tài liệu.
        Mỗi chunk trả về kèm field 'priority': 'high' hoặc 'normal'.
        """
        doc_type = metadata.get("loai_tai_lieu", "khac")
        ten_van_ban = metadata.get("ten_van_ban", "")

        # --- Phụ lục tổng hợp lớn (Phu luc.doc) ---
        if doc_type == "phu_luc" or "phu luc" in ten_van_ban.lower():
            chunks = self._split_by_mau_so(text)
            if len(chunks) > 1:
                return chunks
            return self._split_with_overlap(text, chunk_size=1000, overlap=150)

        # --- Mẫu biểu đơn lẻ: giữ nguyên 1 chunk ---
        if doc_type == "mau_bieu":
            return [{"text": text.strip(), "strategy": "full_document", "priority": "normal"}]

        # --- Văn bản tường thuật: split theo section heading ---
        if doc_type in ("bien_ban", "bao_cao", "ke_hoach", "quyet_dinh", "thong_bao"):
            chunks = self._split_by_sections(text)
            if chunks:
                return chunks
            return self._split_with_overlap(text, chunk_size=600, overlap=80)

        # --- Quy chuẩn kỹ thuật: split theo Điều ---
        if doc_type == "quy_chuan_ky_thuat":
            return self._split_qcvn(text)

        # --- Tài liệu ngắn: giữ nguyên ---
        if doc_type in ("de_nghi", "so_theo_doi", "bia", "so_do"):
            return [{"text": text.strip(), "strategy": "full_document", "priority": "normal"}]

        # --- Default ---
        return self._split_with_overlap(text, chunk_size=600, overlap=80)

    # ------------------------------------------------------------------
    def _split_by_mau_so(self, text: str) -> List[Dict]:
        """Split phụ lục tổng hợp theo 'Mẫu số XX'."""
        parts = RE_MAU_SO.split(text)
        result = []
        for part in parts:
            part = part.strip()
            if len(part) > 50:
                result.append({
                    "text": part,
                    "strategy": "mau_so_section",
                    "priority": _assign_priority(part),
                })
        return result if result else [{"text": text.strip(), "strategy": "full_document", "priority": "normal"}]

    def _split_by_sections(self, text: str) -> List[Dict]:
        """Split theo section headers (I., II., 1., 2. ở đầu dòng)."""
        parts = RE_SECTION_HEADER.split(text)
        result = []
        for part in parts:
            part = part.strip()
            if len(part) > 50:
                result.append({
                    "text": part,
                    "strategy": "section_header",
                    "priority": _assign_priority(part),
                })
        return result

    def _split_qcvn(self, text: str) -> List[Dict]:
        """Split QCVN theo Điều, với sub-split nếu Điều quá dài."""
        parts = RE_DIEU.split(text)
        if len(parts) > 1:
            result = []
            for p in parts:
                p = p.strip()
                if len(p) < 50:
                    continue
                prio = _assign_priority(p)
                # Điều dài hơn 1200 từ → sub-split với overlap
                if len(p.split()) > 1200:
                    sub = self._split_with_overlap(p, chunk_size=900, overlap=120)
                    for s in sub:
                        s["priority"] = prio  # Kế thừa priority từ Điều cha
                        s["strategy"] = "dieu+overlap"
                    result.extend(sub)
                else:
                    result.append({"text": p, "strategy": "dieu", "priority": prio})
            return result
        return self._split_with_overlap(text, chunk_size=700, overlap=100)

    def _split_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Token-based split với sliding window và overlap."""
        words = text.split()
        if not words:
            return []
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i: i + chunk_size]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "strategy": "token_overlap",
                    "priority": _assign_priority(chunk_text),
                })
            i += chunk_size - overlap
        return chunks
