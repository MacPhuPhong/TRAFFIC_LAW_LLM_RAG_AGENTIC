#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_phuluc_pipeline.py
======================
Script chạy nhanh pipeline xử lý phụ lục .doc/.docx.
Dùng để test và xem output trước khi index vào vector DB.

Cách chạy:
    cd /media/pphong/D:/Do_An_Tot_Nghiep/GitHub1/traffic_rag
    source /home/pphong/venv/LLM_Agentic/bin/activate
    python run_phuluc_pipeline.py
"""

import sys
import os
import logging
from pathlib import Path
from collections import Counter

# --- Thêm thư mục gốc vào PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Cấu hình đường dẫn
# ---------------------------------------------------------------------------
INPUT_ROOT    = PROJECT_ROOT / "Data" / "raw" / "phuluc"
CONVERTED_DIR = PROJECT_ROOT / "Data" / "converted_docx" / "phuluc"
OUTPUT_DIR    = PROJECT_ROOT / "Data" / "chunks" / "phuluc"

print("=" * 65)
print("  PHULUC DOC-LOADER PIPELINE")
print("=" * 65)
print(f"  Input  : {INPUT_ROOT}")
print(f"  Convert: {CONVERTED_DIR}")
print(f"  Chunks : {OUTPUT_DIR}")
print("=" * 65)

# ---------------------------------------------------------------------------
# BƯỚC 0 — Kiểm tra LibreOffice
# ---------------------------------------------------------------------------
import shutil
soffice = shutil.which("soffice")
print(f"\n[CHECK] LibreOffice : {'✅ ' + soffice if soffice else '⚠️  Không tìm thấy — sẽ dùng fallback'}")

# ---------------------------------------------------------------------------
# BƯỚC 1 — Test metadata parser trước (không cần convert)
# ---------------------------------------------------------------------------
print("\n" + "─" * 65)
print("  BƯỚC 1 — Test Metadata Parser")
print("─" * 65)

from source.doc_loader.metadata_parser import parse_metadata_from_path

test_files = [
    str(INPUT_ROOT / "tt13" / "1.2. Biên bản vụ việc.doc"),
    str(INPUT_ROOT / "tt13" / "1.6. Kế hoạch xác minh vụ TNGT A4.doc"),
    str(INPUT_ROOT / "tt13" / "1.5. QĐ phân công cán bộ xác minh.doc"),
    str(INPUT_ROOT / "tt13" / "1.18. Mẫu số 01 phiếu chuyển kết quả từ TBKTNV.docx"),
    str(INPUT_ROOT / "tt48" / "01. QCVN 04-2024.doc"),
    str(INPUT_ROOT / "tt35" / "Phu luc.doc"),
]

existing_test = [f for f in test_files if os.path.isfile(f)]
for fp in existing_test:
    meta = parse_metadata_from_path(fp)
    print(
        f"  [{meta['folder']:4s}] {meta['filename'][:45]:45s}"
        f"  → loai={meta['loai_tai_lieu']:20s}"
        f"  so_hieu={meta['so_hieu'] or '-':15s}"
        f"  phu_luc={meta['so_phu_luc'] or '-'}"
    )

# ---------------------------------------------------------------------------
# BƯỚC 2 — Chạy full pipeline (chỉ 1 subfolder để test nhanh)
# ---------------------------------------------------------------------------
print("\n" + "─" * 65)
print("  BƯỚC 2 — Chạy Pipeline (tt13 trước để test)")
print("─" * 65)

from source.doc_loader.pipeline import DocRAGPipeline

# ── Test với tt13 trước (27 files — đa dạng nhất) ──
pipeline_tt13 = DocRAGPipeline(
    input_root    = str(INPUT_ROOT / "tt13"),
    output_dir    = str(OUTPUT_DIR),
    converted_dir = str(CONVERTED_DIR),
)

docs_tt13 = pipeline_tt13.run(save_json=True)

# ---------------------------------------------------------------------------
# BƯỚC 3 — Hiển thị kết quả chi tiết
# ---------------------------------------------------------------------------
print("\n" + "─" * 65)
print("  BƯỚC 3 — Kết Quả Chi Tiết")
print("─" * 65)

if docs_tt13:
    # Thống kê theo loại
    by_type = Counter(d.metadata.get("loai_tai_lieu") for d in docs_tt13)
    by_strategy = Counter(d.metadata.get("chunk_strategy") for d in docs_tt13)

    print(f"\n  Tổng chunks tạo ra: {len(docs_tt13)}")

    print("\n  Chunks theo loại tài liệu:")
    for k, v in by_type.most_common():
        bar = "█" * min(v, 30)
        print(f"    {k:25s}: {v:3d}  {bar}")

    print("\n  Chunks theo strategy:")
    for k, v in by_strategy.most_common():
        print(f"    {k:25s}: {v:3d}")

    # Xem mẫu 3 document đầu tiên
    print("\n" + "─" * 65)
    print("  MẪU 3 DOCUMENT ĐẦU TIÊN")
    print("─" * 65)
    for i, doc in enumerate(docs_tt13[:3]):
        m = doc.metadata
        print(f"\n  ── Document #{i+1} ──")
        print(f"    file        : {m.get('filename', '')}")
        print(f"    loai        : {m.get('loai_tai_lieu', '')}")
        print(f"    folder      : {m.get('folder', '')}")
        print(f"    chunk_id    : {m.get('chunk_id', '')}")
        print(f"    chunk_index : {m.get('chunk_index', 0)} / {m.get('total_chunks', 0)}")
        print(f"    strategy    : {m.get('chunk_strategy', '')}")
        print(f"    word_count  : {m.get('word_count', 0)}")
        print(f"    has_tables  : {m.get('has_tables', False)}")
        content_preview = doc.page_content[:200].replace("\n", " ")
        print(f"    content     : {content_preview}...")
else:
    print("  ⚠️  Không có document nào được tạo ra. Kiểm tra lại lỗi ở trên.")

# ---------------------------------------------------------------------------
# BƯỚC 4 — Hỏi có chạy ALL không
# ---------------------------------------------------------------------------
print("\n" + "─" * 65)
answer = input("  Chạy pipeline cho TẤT CẢ folder (tt12+tt13+tt35+tt48)? [y/N]: ").strip().lower()

if answer == "y":
    print("\n  Đang chạy toàn bộ phuluc pipeline...")
    pipeline_all = DocRAGPipeline(
        input_root    = str(INPUT_ROOT),
        output_dir    = str(OUTPUT_DIR),
        converted_dir = str(CONVERTED_DIR),
    )
    all_docs = pipeline_all.run(save_json=True)

    print(f"\n  ✅ HOÀN THÀNH! Tổng {len(all_docs)} LangChain Documents sẵn sàng index.")

    # Ghi tóm tắt ra file
    summary_path = OUTPUT_DIR / "pipeline_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total documents: {len(all_docs)}\n")
        f.write(f"Processed files: {len(pipeline_all.processed_files)}\n")
        f.write(f"Failed files   : {len(pipeline_all.failed_files)}\n\n")
        by_t = Counter(d.metadata.get("loai_tai_lieu") for d in all_docs)
        f.write("Chunks by type:\n")
        for k, v in by_t.most_common():
            f.write(f"  {k:25s}: {v}\n")
    print(f"  📄 Tóm tắt lưu tại: {summary_path}")
else:
    print("  Dừng lại. Chạy lại và nhập 'y' khi muốn index toàn bộ.")

print("\n" + "=" * 65)
print("  DONE")
print("=" * 65)
