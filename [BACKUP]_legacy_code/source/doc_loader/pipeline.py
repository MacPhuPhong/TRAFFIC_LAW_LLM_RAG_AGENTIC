# -*- coding: utf-8 -*-
"""
pipeline.py — DocRAGPipeline (Pro Edition)
==========================================
Nâng cấp:
  1. ASSET MANAGEMENT: tự động tạo thư mục assets/ per-document cho ảnh
  2. TAGS + CROSS-LINKS: gọi generate_tags() và detect_cross_links() sau extract
  3. HEADER INJECTION: chèn tags + cross-links vào page_content chunk đầu tiên
  4. PRIORITY metadata: chunk_priority từ splitter + extractor
  5. PDF MODULE: xử lý file .pdf trong Data/raw/quychuan/
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from .converter import convert_doc_to_docx
from .extractor import extract_text_from_docx
from .metadata_parser import (
    parse_metadata_from_path,
    generate_tags,
    detect_cross_links,
)
from .splitter import AdminDocumentSplitter

logger = logging.getLogger(__name__)


class DocRAGPipeline:
    """
    Pipeline xử lý file .doc/.docx/.pdf → LangChain Documents (Pro Edition).

    Args:
        input_root:    Thư mục gốc chứa .doc/.docx (có thể có subfolder)
        output_dir:    Thư mục lưu chunks JSON
        converted_dir: Thư mục lưu file .docx đã convert
        assets_root:   Thư mục gốc chứa ảnh trích xuất
                       (mặc định: <converted_dir>/assets/)
    """

    def __init__(
        self,
        input_root: str,
        output_dir: str,
        converted_dir: str,
        assets_root: Optional[str] = None,
    ):
        self.input_root = str(Path(input_root).resolve())
        self.output_dir = str(Path(output_dir).resolve())
        self.converted_dir = str(Path(converted_dir).resolve())
        self.assets_root = (
            str(Path(assets_root).resolve())
            if assets_root
            else str(Path(converted_dir).resolve() / "assets")
        )
        self.splitter = AdminDocumentSplitter()
        self.failed_files: List[str] = []
        self.processed_files: List[str] = []

        for d in [self.output_dir, self.converted_dir, self.assets_root]:
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, save_json: bool = True) -> List:
        """
        Chạy toàn bộ pipeline.

        Args:
            save_json: Lưu chunks ra JSON per-file để debug/cache

        Returns:
            list[LangChain Document]
        """
        all_docs: List = []
        all_files = self._collect_files()
        logger.info(f"[PIPELINE] Tìm thấy {len(all_files)} files trong {self.input_root}")
        print(f"[INFO] Tìm thấy {len(all_files)} files")

        for file_path in tqdm(all_files, desc="Processing", unit="file"):
            docs = self._process_single_file(file_path, save_json=save_json)
            all_docs.extend(docs)

        print(f"\n[DONE] Processed:    {len(self.processed_files)} files")
        print(f"[WARN] Failed:       {len(self.failed_files)} files")
        print(f"[INFO] Total chunks: {len(all_docs)}")
        self._save_failed_log()
        return all_docs

    def run_single(self, file_path: str) -> List:
        """Xử lý 1 file duy nhất."""
        return self._process_single_file(file_path, save_json=False)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _collect_files(self) -> List[str]:
        """Quét đệ quy .doc, .docx (và .pdf nếu có) trong input_root."""
        files = []
        for root, _dirs, filenames in os.walk(self.input_root):
            for fname in sorted(filenames):
                lower = fname.lower()
                if lower.endswith((".doc", ".docx", ".pdf")):
                    files.append(os.path.join(root, fname))
        return sorted(files)

    def _process_single_file(self, file_path: str, save_json: bool = True) -> List:
        try:
            path = Path(file_path)
            suffix = path.suffix.lower()

            # ── Step 1: Convert / Route ───────────────────────────────
            if suffix == ".pdf":
                docx_path = None  # PDF xử lý riêng qua _extract_pdf
                extracted = _extract_pdf(file_path)
            elif suffix == ".doc":
                rel_folder = path.parent.relative_to(self.input_root)
                subfolder_out = os.path.join(self.converted_dir, str(rel_folder))
                os.makedirs(subfolder_out, exist_ok=True)
                docx_path = convert_doc_to_docx(file_path, subfolder_out)
                if docx_path is None:
                    logger.error(f"Convert thất bại: {file_path}")
                    self.failed_files.append(file_path)
                    return []
                assets_dir = self._assets_dir_for(docx_path)
                extracted = extract_text_from_docx(docx_path, assets_dir=assets_dir)
            else:
                # .docx trực tiếp
                docx_path = file_path
                assets_dir = self._assets_dir_for(docx_path)
                extracted = extract_text_from_docx(docx_path, assets_dir=assets_dir)

            if not extracted["full_text"].strip():
                logger.warning(f"Text rỗng: {file_path}")
                return []

            # ── Step 2: Metadata ──────────────────────────────────────
            metadata = parse_metadata_from_path(file_path)
            metadata["docx_path"] = docx_path or file_path
            metadata["word_count"] = extracted["word_count"]
            metadata["has_tables"] = extracted["has_tables"]
            metadata["has_images"] = extracted.get("has_images", False)
            metadata["image_paths"] = extracted.get("image_paths", [])

            # ── Step 3: Auto-tagging + Cross-link detection ───────────
            full_text = extracted["full_text"]
            metadata["tags"] = generate_tags(full_text, metadata["filename"])
            metadata["cross_links"] = detect_cross_links(full_text)

            # ── Step 4: Header để inject vào chunk đầu tiên ──────────
            header = _build_header(metadata)

            # ── Step 5: Chunking ──────────────────────────────────────
            chunks = self.splitter.split(full_text, metadata)
            if not chunks:
                logger.warning(f"Không có chunk: {file_path}")
                return []

            # ── Step 6: LangChain Documents ───────────────────────────
            docs = self._build_documents(chunks, metadata, header)
            if save_json:
                self._save_chunks_json(docs, metadata)

            self.processed_files.append(file_path)
            logger.info(
                f"OK: {path.name} → {len(docs)} chunks "
                f"(loai={metadata['loai_tai_lieu']} | "
                f"tags={len(metadata['tags'])} | "
                f"cross_links={len(metadata['cross_links'])})"
            )
            return docs

        except Exception as e:
            logger.exception(f"[ERROR] {file_path}: {e}")
            self.failed_files.append(file_path)
            return []

    def _assets_dir_for(self, docx_path: str) -> str:
        """Xác định thư mục assets riêng cho từng file."""
        stem = Path(docx_path).stem
        # Lấy relative path so với converted_dir để tổ chức đúng subfolder
        try:
            rel = Path(docx_path).parent.relative_to(self.converted_dir)
            return str(Path(self.assets_root) / rel / stem)
        except ValueError:
            return str(Path(self.assets_root) / stem)

    def _build_documents(
        self, chunks: List[Dict], base_metadata: Dict, header: str
    ) -> List:
        """Tạo danh sách LangChain Document objects."""
        try:
            from langchain.schema import Document
        except ImportError:
            try:
                from langchain_core.documents import Document
            except ImportError:
                raise ImportError(
                    "langchain hoặc langchain-core chưa được cài. "
                    "Chạy: pip install langchain-core"
                )

        docs = []
        total = len(chunks)
        folder = base_metadata.get("folder", "unknown")
        safe_name = _safe_id(base_metadata.get("ten_van_ban", "doc"))

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "").strip()
            if not chunk_text:
                continue

            # Inject header vào chunk đầu tiên
            if i == 0 and header:
                chunk_text = header + chunk_text

            chunk_meta = base_metadata.copy()
            chunk_meta.update({
                "chunk_index": i,
                "total_chunks": total,
                "chunk_id": f"{folder}_{safe_name}_chunk{i:04d}",
                "chunk_strategy": chunk.get("strategy", "unknown"),
                "chunk_priority": chunk.get("priority", "normal"),
            })
            # Chuyển list → string để tương thích ChromaDB/Qdrant
            chunk_meta["tags"] = " ".join(chunk_meta.get("tags", []))
            chunk_meta["cross_links"] = " | ".join(chunk_meta.get("cross_links", []))
            chunk_meta["image_paths"] = " | ".join(chunk_meta.get("image_paths", []))

            docs.append(Document(page_content=chunk_text, metadata=chunk_meta))

        return docs

    def _save_chunks_json(self, docs: List, metadata: Dict) -> None:
        folder = metadata.get("folder", "unknown")
        safe_name = _safe_id(metadata.get("ten_van_ban", "doc"))
        json_path = os.path.join(self.output_dir, f"{folder}_{safe_name}.json")
        try:
            serializable = [
                {
                    "page_content": d.page_content,
                    "metadata": _make_serializable(d.metadata),
                }
                for d in docs
            ]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Không thể lưu JSON {json_path}: {e}")

    def _save_failed_log(self) -> None:
        if not self.failed_files:
            return
        log_path = os.path.join(self.output_dir, "failed_files.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                for fp in self.failed_files:
                    f.write(fp + "\n")
            print(f"[LOG] Danh sách file thất bại: {log_path}")
        except Exception as e:
            logger.warning(f"Không thể ghi failed_files.log: {e}")


# ==================================================================
# PDF MODULE (bonus — dùng cho Data/raw/quychuan/*.pdf)
# ==================================================================

def _extract_pdf(pdf_path: str) -> Dict:
    """
    Extract text từ file PDF bằng pypdf.
    Trả về dict tương thích với extract_text_from_docx().
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf chưa được cài. Chạy: pip install pypdf")

    try:
        reader = PdfReader(pdf_path)
        pages_text: List[str] = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                pages_text.append("")

        full_text = "\n".join(pages_text)
        word_count = len(full_text.split())

        logger.info(
            f"Extracted [PDF]: {Path(pdf_path).name} | "
            f"{word_count} words | {len(reader.pages)} pages"
        )

        return {
            "full_text": full_text,
            "sections": [],  # PDF không có heading structure
            "has_tables": False,
            "has_images": False,
            "word_count": word_count,
            "image_paths": [],
            "priority_chunks": [],
        }

    except Exception as e:
        logger.error(f"PDF extraction failed for {pdf_path}: {e}")
        return {
            "full_text": "", "sections": [], "has_tables": False,
            "has_images": False, "word_count": 0,
            "image_paths": [], "priority_chunks": [],
        }


# ==================================================================
# HELPERS
# ==================================================================

def _build_header(metadata: Dict) -> str:
    """
    Xây dựng chuỗi header để chèn vào đầu chunk đầu tiên.
    Chứa thông tin Tags, Cross-links để RAG retriever biết ngữ cảnh.
    """
    lines: List[str] = []
    tags = metadata.get("tags", [])
    if isinstance(tags, str):
        tags = tags.split()
    cross_links = metadata.get("cross_links", [])
    if isinstance(cross_links, str):
        cross_links = [x for x in cross_links.split(" | ") if x]

    ten = metadata.get("ten_van_ban", "")
    loai = metadata.get("loai_tai_lieu", "")
    so_thong_tu = metadata.get("so_thong_tu", "")
    so_hieu = metadata.get("so_hieu", "")

    if ten:
        lines.append(f"[TÀI LIỆU: {ten}]")
    if so_thong_tu:
        lines.append(f"[THÔNG TƯ: {so_thong_tu}]")
    if so_hieu:
        lines.append(f"[SỐ HIỆU: {so_hieu}]")
    if loai:
        lines.append(f"[LOẠI: {loai}]")
    if tags:
        lines.append(f"[TAGS: {' '.join(tags)}]")
    if cross_links:
        lines.append(f"[LIÊN KẾT: {'; '.join(cross_links[:5])}]")

    return "\n".join(lines) + "\n---\n" if lines else ""


def _safe_id(text: str) -> str:
    """Chuyển tên file thành ID an toàn (chỉ chữ, số, _)."""
    import re as _re
    import unicodedata as _uni
    text = _uni.normalize("NFC", text)
    nfd = _uni.normalize("NFD", text)
    ascii_text = "".join(c for c in nfd if _uni.category(c) != "Mn")
    safe = _re.sub(r"[^\w\d]", "_", ascii_text)
    safe = _re.sub(r"_+", "_", safe).strip("_")
    return safe[:80]


def _make_serializable(obj):
    """Đảm bảo dict metadata có thể JSON-serialize."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)
