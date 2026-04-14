# -*- coding: utf-8 -*-
"""
splitter.py — Chunking thông minh theo loại tài liệu hành chính
================================================================
AdminDocumentSplitter xử lý các loại file phụ lục thông tư:
  - mau_bieu:           Giữ nguyên 1 chunk (form không nên cắt)
  - bien_ban/bao_cao/ke_hoach: Split theo section heading
  - quy_chuan_ky_thuat: Split theo Điều/Mục
  - phu_luc:           Split theo "Mẫu số XX" + overlap
  - default:           Split theo token với overlap

Lưu ý đặc biệt:
  - File "Phu luc.doc" (tt35): rất lớn (~112k ký tự),
    split theo mỗi "Mẫu số XX" là một chunk riêng.
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Các pattern dùng để split
# ---------------------------------------------------------------------------

# Nhận diện đầu section kiểu La Mã hoặc số Ả Rập ở đầu dòng,
# hoặc heading IN HOA từ 3 ký tự trở lên
RE_SECTION_HEADER = re.compile(
    r"(?m)(?=^[IVX]+\.\s|^\d+\.\s[A-ZĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẨẪ]"
    r"|^\b[A-ZĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẨẪ]{3,}\s*:)",
    re.UNICODE,
)

# Điều XX. hoặc Điều XX: — dùng cho QCVN
RE_DIEU = re.compile(r"(?=Điều\s+\d+[\.\:])", re.UNICODE)

# Mẫu số XX — dùng để split phụ lục tổng hợp lớn
RE_MAU_SO = re.compile(
    r"(?=M[aẫ]u\s+s[oố]\s*\d+[\-\.\s]?[A-Za-z0-9]*)",
    re.UNICODE | re.IGNORECASE,
)


class AdminDocumentSplitter:
    """
    Chunking tài liệu hành chính theo loại tài liệu.
    Mỗi chunk là dict: {"text": str, "strategy": str}
    """

    def split(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Dispatch đến hàm split phù hợp dựa trên loại tài liệu.

        Args:
            text:     Full text đã extract
            metadata: dict metadata từ parse_metadata_from_path()

        Returns:
            list[dict] — mỗi phần tử: {"text": str, "strategy": str}
        """
        doc_type = metadata.get("loai_tai_lieu", "khac")
        ten_van_ban = metadata.get("ten_van_ban", "")

        # --- Trường hợp đặc biệt: Phu luc.doc lớn (tt35) ---
        # Có thể chứa nhiều mẫu biểu gộp lại → split theo Mẫu số XX
        if doc_type == "phu_luc" or "phu luc" in ten_van_ban.lower():
            chunks = self._split_by_mau_so(text)
            if len(chunks) > 1:
                return chunks
            # Nếu không nhận diện được Mẫu số, dùng overlap lớn
            return self._split_with_overlap(text, chunk_size=1000, overlap=150)

        if doc_type == "mau_bieu":
            # Mẫu biểu đơn lẻ: giữ nguyên 1 chunk
            return [{"text": text.strip(), "strategy": "full_document"}]

        if doc_type in ("bien_ban", "bao_cao", "ke_hoach", "quyet_dinh", "thong_bao"):
            # Văn bản tường thuật: split theo section heading
            chunks = self._split_by_sections(text)
            if chunks:
                return chunks
            # Fallback nếu không có section
            return self._split_with_overlap(text, chunk_size=600, overlap=80)

        if doc_type == "quy_chuan_ky_thuat":
            return self._split_qcvn(text)

        if doc_type in ("de_nghi", "so_theo_doi", "bia", "so_do"):
            # Tài liệu ngắn / biểu mẫu phụ: giữ nguyên
            return [{"text": text.strip(), "strategy": "full_document"}]

        # --- Default ---
        return self._split_with_overlap(text, chunk_size=600, overlap=80)

    # ------------------------------------------------------------------
    # Internal split methods
    # ------------------------------------------------------------------

    def _split_by_mau_so(self, text: str) -> List[Dict]:
        """
        Split theo "Mẫu số XX" — dùng cho file phụ lục tổng hợp lớn.
        Mỗi mẫu biểu là một chunk riêng.
        """
        parts = RE_MAU_SO.split(text)
        result = []
        for part in parts:
            part = part.strip()
            if len(part) > 50:
                result.append({"text": part, "strategy": "mau_so_section"})
        logger.debug(f"_split_by_mau_so → {len(result)} chunks")
        return result if result else [{"text": text.strip(), "strategy": "full_document"}]

    def _split_by_sections(self, text: str) -> List[Dict]:
        """
        Split theo section headers (I., II., 1., 2. ở đầu dòng,
        hoặc cụm IN HOA).
        """
        parts = RE_SECTION_HEADER.split(text)
        result = []
        for part in parts:
            part = part.strip()
            if len(part) > 50:
                result.append({"text": part, "strategy": "section_header"})
        logger.debug(f"_split_by_sections → {len(result)} chunks")
        return result

    def _split_qcvn(self, text: str) -> List[Dict]:
        """
        Split QCVN theo Điều.
        Nếu không tìm thấy Điều → dùng overlap lớn.
        """
        parts = RE_DIEU.split(text)
        if len(parts) > 1:
            result = []
            for p in parts:
                p = p.strip()
                if len(p) > 50:
                    # Mỗi Điều có thể rất dài → tự split tiếp nếu cần
                    if len(p.split()) > 1200:
                        sub = self._split_with_overlap(p, chunk_size=900, overlap=120)
                        result.extend(
                            [{"text": s["text"], "strategy": "dieu+overlap"} for s in sub]
                        )
                    else:
                        result.append({"text": p, "strategy": "dieu"})
            logger.debug(f"_split_qcvn (by Điều) → {len(result)} chunks")
            return result

        logger.debug("_split_qcvn: no Điều found, using overlap")
        return self._split_with_overlap(text, chunk_size=700, overlap=100)

    def _split_with_overlap(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[Dict]:
        """
        Token-based split với sliding window và overlap.

        Args:
            chunk_size: Số từ mỗi chunk
            overlap:    Số từ overlap giữa 2 chunk liên tiếp
        """
        words = text.split()
        if not words:
            return []

        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "strategy": "token_overlap"})
            i += chunk_size - overlap

        logger.debug(
            f"_split_with_overlap (size={chunk_size}, overlap={overlap}) "
            f"→ {len(chunks)} chunks"
        )
        return chunks
