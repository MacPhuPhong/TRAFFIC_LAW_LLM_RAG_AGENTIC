# -*- coding: utf-8 -*-
"""
doc_loader — Pipeline xử lý file .doc/.docx cho hệ thống RAG
=============================================================
Module này bao gồm:
  - converter:       Convert .doc → .docx (LibreOffice headless)
  - extractor:       Extract text từ .docx (python-docx)
  - metadata_parser: Parse metadata từ đường dẫn + tên file
  - splitter:        Chunking thông minh theo loại tài liệu
  - pipeline:        Pipeline tổng hợp (DocRAGPipeline)
"""

from .converter import convert_doc_to_docx
from .extractor import extract_text_from_docx
from .metadata_parser import parse_metadata_from_path
from .splitter import AdminDocumentSplitter
from .pipeline import DocRAGPipeline

__all__ = [
    "convert_doc_to_docx",
    "extract_text_from_docx",
    "parse_metadata_from_path",
    "AdminDocumentSplitter",
    "DocRAGPipeline",
]
