# -*- coding: utf-8 -*-
"""
converter.py — Convert .doc (legacy binary) → .docx
=====================================================
Dùng LibreOffice headless làm primary method.
Fallback sang docx2txt nếu LibreOffice không khả dụng.
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Danh sách đường dẫn LibreOffice có thể có trên các hệ điều hành
SOFFICE_PATHS = [
    "soffice",
    "/usr/bin/soffice",
    "/usr/lib/libreoffice/program/soffice",
    "/usr/local/bin/soffice",
    "/snap/bin/libreoffice.soffice",
    "C:/Program Files/LibreOffice/program/soffice.exe",
    "C:/Program Files (x86)/LibreOffice/program/soffice.exe",
]

CONVERT_TIMEOUT = 120  # seconds — tăng lên cho file lớn


def _find_soffice() -> Optional[str]:
    """Tìm đường dẫn soffice khả dụng trên hệ thống."""
    for path in SOFFICE_PATHS:
        if shutil.which(path):
            return path
    return None


def _convert_with_libreoffice(
    doc_path: str, output_dir: str, soffice_bin: str
) -> Optional[str]:
    """
    Convert .doc → .docx bằng LibreOffice headless.
    
    Returns:
        Đường dẫn file .docx mới, hoặc None nếu thất bại.
    """
    try:
        result = subprocess.run(
            [
                soffice_bin,
                "--headless",
                "--convert-to", "docx",
                "--outdir", output_dir,
                str(doc_path),
            ],
            capture_output=True,
            text=True,
            timeout=CONVERT_TIMEOUT,
        )

        if result.returncode != 0:
            logger.error(
                f"LibreOffice convert failed ({result.returncode}): "
                f"{result.stderr.strip()}"
            )
            return None

        # Xác định tên file output
        stem = Path(doc_path).stem
        output_path = os.path.join(output_dir, f"{stem}.docx")

        if os.path.isfile(output_path):
            logger.info(f"Converted (LibreOffice): {doc_path} → {output_path}")
            return output_path
        else:
            logger.error(
                f"LibreOffice reported success but output file not found: {output_path}"
            )
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"LibreOffice timeout ({CONVERT_TIMEOUT}s): {doc_path}")
        return None
    except FileNotFoundError:
        logger.error(f"LibreOffice binary not found: {soffice_bin}")
        return None
    except Exception as e:
        logger.error(f"LibreOffice unexpected error: {e}")
        return None


def _convert_with_docx2txt(doc_path: str, output_dir: str) -> Optional[str]:
    """
    Fallback: dùng docx2txt để extract text từ .doc.
    Lưu ý: docx2txt chỉ extract text, KHÔNG convert sang .docx thực sự.
    Ta sẽ tạo một .docx mới chứa text đã extract.
    """
    try:
        import docx2txt
        from docx import Document as DocxDocument

        # docx2txt thực ra chỉ hỗ trợ .docx, nên fallback này
        # chỉ hoạt động nếu file .doc thực chất là .docx bị đổi extension
        text = docx2txt.process(doc_path)
        if not text or not text.strip():
            logger.warning(f"docx2txt returned empty text: {doc_path}")
            return None

        # Tạo file .docx mới chứa text
        stem = Path(doc_path).stem
        output_path = os.path.join(output_dir, f"{stem}.docx")

        doc = DocxDocument()
        for paragraph in text.split("\n"):
            if paragraph.strip():
                doc.add_paragraph(paragraph)
        doc.save(output_path)

        logger.info(f"Converted (docx2txt fallback): {doc_path} → {output_path}")
        return output_path

    except ImportError:
        logger.warning("docx2txt not installed. Skipping fallback.")
        return None
    except Exception as e:
        logger.warning(f"docx2txt fallback failed for {doc_path}: {e}")
        return None


def convert_doc_to_docx(
    doc_path: str,
    output_dir: str,
    force: bool = False,
) -> Optional[str]:
    """
    Convert file .doc sang .docx.

    Args:
        doc_path:   Đường dẫn tuyệt đối đến file .doc
        output_dir: Thư mục chứa file .docx output
        force:      Nếu True, convert lại kể cả khi file đã tồn tại

    Returns:
        Đường dẫn file .docx đã convert, hoặc None nếu thất bại.
    """
    doc_path = str(Path(doc_path).resolve())
    os.makedirs(output_dir, exist_ok=True)

    # --- Cache: skip nếu đã convert trước đó ---
    stem = Path(doc_path).stem
    output_path = os.path.join(output_dir, f"{stem}.docx")

    if not force and os.path.isfile(output_path):
        logger.info(f"Skipping (cached): {output_path}")
        return output_path

    # --- Method 1: LibreOffice headless ---
    soffice = _find_soffice()
    if soffice:
        result = _convert_with_libreoffice(doc_path, output_dir, soffice)
        if result:
            return result
        logger.warning("LibreOffice convert failed, trying fallback...")

    # --- Method 2: docx2txt fallback ---
    result = _convert_with_docx2txt(doc_path, output_dir)
    if result:
        return result

    # --- Tất cả methods đều thất bại ---
    logger.error(f"ALL conversion methods failed for: {doc_path}")
    return None
