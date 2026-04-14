# -*- coding: utf-8 -*-
"""
pipeline.py — DocRAGPipeline: Pipeline tổng hợp xử lý .doc/.docx → LangChain Documents
========================================================================================
Quy trình:
  1. Quét tất cả .doc/.docx trong thư mục input
  2. Convert .doc → .docx (LibreOffice headless / fallback)
  3. Extract text từ .docx
  4. Parse metadata từ đường dẫn & tên file
  5. Chunk thông minh theo loại tài liệu
  6. Trả về list[LangChain Document] sẵn sàng để index

Đường dẫn trong dự án:
  input_root     = traffic_rag/Data/raw/phuluc/
  converted_dir  = traffic_rag/Data/converted_docx/
  output_dir     = traffic_rag/Data/chunks/phuluc/
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from .converter import convert_doc_to_docx
from .extractor import extract_text_from_docx
from .metadata_parser import parse_metadata_from_path
from .splitter import AdminDocumentSplitter

logger = logging.getLogger(__name__)


class DocRAGPipeline:
    """
    Pipeline xử lý file .doc/.docx cho hệ thống RAG phụ lục thông tư.

    Args:
        input_root:    Thư mục gốc chứa .doc/.docx (có thể có subfolder)
        output_dir:    Thư mục lưu chunks JSON (dùng cho debug/cache)
        converted_dir: Thư mục lưu file .docx đã convert từ .doc
    """

    def __init__(
        self,
        input_root: str,
        output_dir: str,
        converted_dir: str,
    ):
        self.input_root = str(Path(input_root).resolve())
        self.output_dir = str(Path(output_dir).resolve())
        self.converted_dir = str(Path(converted_dir).resolve())
        self.splitter = AdminDocumentSplitter()

        self.failed_files: List[str] = []
        self.processed_files: List[str] = []

        # Tạo thư mục output nếu chưa có
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.converted_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, save_json: bool = True) -> List:
        """
        Chạy toàn bộ pipeline.

        Args:
            save_json: Nếu True, lưu chunks ra file JSON theo từng file gốc
                       (tiện cho debug và re-index mà không cần chạy lại)

        Returns:
            list[LangChain Document]
        """
        all_docs: List = []

        # 1. Quét file
        all_files = self._collect_files()
        logger.info(f"[PIPELINE] Tìm thấy {len(all_files)} files trong {self.input_root}")
        print(f"[INFO] Tìm thấy {len(all_files)} files")

        # 2. Xử lý từng file
        for file_path in tqdm(all_files, desc="Processing docs", unit="file"):
            docs = self._process_single_file(file_path, save_json=save_json)
            all_docs.extend(docs)

        # 3. Báo cáo kết quả
        print(f"\n[DONE] Processed:   {len(self.processed_files)} files")
        print(f"[WARN] Failed:      {len(self.failed_files)} files")
        print(f"[INFO] Total chunks: {len(all_docs)}")

        # Lưu danh sách file thất bại
        self._save_failed_log()

        return all_docs

    def run_single(self, file_path: str) -> List:
        """Xử lý 1 file duy nhất — tiện cho test/debug."""
        return self._process_single_file(file_path, save_json=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_files(self) -> List[str]:
        """Quét đệ quy tất cả .doc và .docx trong input_root."""
        files = []
        for root, _dirs, filenames in os.walk(self.input_root):
            for fname in sorted(filenames):
                if fname.lower().endswith((".doc", ".docx")):
                    full_path = os.path.join(root, fname)
                    files.append(full_path)
        return sorted(files)

    def _process_single_file(
        self, file_path: str, save_json: bool = True
    ) -> List:
        """
        Xử lý một file .doc/.docx.

        Returns:
            list[LangChain Document]  hoặc [] nếu thất bại
        """
        try:
            path = Path(file_path)

            # ── Step 1: Convert .doc → .docx ──────────────────────────
            if path.suffix.lower() == ".doc":
                # Đặt output_dir theo subfolder để tránh đụng tên file
                relative_folder = path.parent.relative_to(self.input_root)
                subfolder_out = os.path.join(self.converted_dir, str(relative_folder))
                os.makedirs(subfolder_out, exist_ok=True)

                docx_path = convert_doc_to_docx(file_path, subfolder_out)
                if docx_path is None:
                    logger.error(f"Convert thất bại: {file_path}")
                    self.failed_files.append(file_path)
                    return []
            else:
                docx_path = file_path

            # ── Step 2: Extract text ───────────────────────────────────
            extracted = extract_text_from_docx(docx_path)
            if not extracted["full_text"].strip():
                logger.warning(f"Text rỗng (empty): {file_path}")
                return []

            # ── Step 3: Parse metadata ─────────────────────────────────
            metadata = parse_metadata_from_path(file_path)
            metadata["docx_path"] = docx_path
            metadata["word_count"] = extracted["word_count"]
            metadata["has_tables"] = extracted["has_tables"]

            # ── Step 4: Chunking ───────────────────────────────────────
            chunks = self.splitter.split(extracted["full_text"], metadata)
            if not chunks:
                logger.warning(f"Không có chunk nào: {file_path}")
                return []

            # ── Step 5: Tạo LangChain Document objects ─────────────────
            docs = self._build_documents(chunks, metadata)

            # ── Step 6: Lưu JSON (debug/cache) ────────────────────────
            if save_json:
                self._save_chunks_json(docs, metadata)

            self.processed_files.append(file_path)
            logger.info(
                f"OK: {path.name} → {len(docs)} chunks "
                f"(loai={metadata['loai_tai_lieu']})"
            )
            return docs

        except Exception as e:
            logger.exception(f"[ERROR] Lỗi không xác định: {file_path}: {e}")
            self.failed_files.append(file_path)
            return []

    def _build_documents(self, chunks: List[Dict], base_metadata: Dict) -> List:
        """
        Tạo danh sách LangChain Document objects từ danh sách chunk.
        Import LangChain Document lazily để tránh lỗi nếu chưa cài.
        """
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
        # Làm sạch tên file để dùng trong chunk_id (loại bỏ ký tự đặc biệt)
        safe_name = _safe_id(base_metadata.get("ten_van_ban", "doc"))

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "").strip()
            if not chunk_text:
                continue

            chunk_meta = base_metadata.copy()
            chunk_meta.update(
                {
                    "chunk_index": i,
                    "total_chunks": total,
                    "chunk_id": f"{folder}_{safe_name}_chunk{i:04d}",
                    "chunk_strategy": chunk.get("strategy", "unknown"),
                }
            )

            docs.append(Document(page_content=chunk_text, metadata=chunk_meta))

        return docs

    def _save_chunks_json(self, docs: List, metadata: Dict) -> None:
        """Lưu chunks ra file JSON để debug hoặc re-index không cần convert lại."""
        folder = metadata.get("folder", "unknown")
        safe_name = _safe_id(metadata.get("ten_van_ban", "doc"))
        json_filename = f"{folder}_{safe_name}.json"
        json_path = os.path.join(self.output_dir, json_filename)

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
            logger.debug(f"Saved JSON: {json_path}")
        except Exception as e:
            logger.warning(f"Không thể lưu JSON {json_path}: {e}")

    def _save_failed_log(self) -> None:
        """Lưu danh sách file thất bại ra failed_files.log."""
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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_id(text: str) -> str:
    """Chuyển chuỗi tên file thành ID an toàn (chỉ giữ chữ, số, dấu gạch dưới)."""
    import re
    import unicodedata

    # Chuẩn hoá NFC trước
    text = unicodedata.normalize("NFC", text)
    # Loại bỏ dấu tiếng Việt (decompose rồi loại combining marks)
    nfd = unicodedata.normalize("NFD", text)
    ascii_text = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # Thay thế ký tự không phải chữ/số bằng '_'
    safe = re.sub(r"[^\w\d]", "_", ascii_text)
    # Thu gọn nhiều dấu _ liên tiếp
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe[:80]  # Giới hạn độ dài


def _make_serializable(obj):
    """Đảm bảo dict metadata có thể serialize sang JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)):
        return obj
    return str(obj)
