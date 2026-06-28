# -*- coding: utf-8 -*-
"""Fixed-size structure-blind chunker (RQ2 ablation baseline).

Splits text into ~512-token windows with 50-token overlap, ignoring
Điều/Khoản/Điểm boundaries. Post-pass regex extracts dieu_primary/khoan_primary
from the last header that appears at-or-before each chunk's start offset,
so citation scoring still has a best-effort structural hint.

Output schema matches `semantic_chunker.py` so `indexer.py` can re-use the
same loader. Collection target: `traffic_law_fixed512` (see indexer_fixed512.py).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from source.ingestion.semantic_chunker import (
    CLEANED_DIRS,
    METADATA_RULES,
    RE_DIEU,
    RE_KHOAN,
    get_file_metadata,
    count_tokens,
)

logger = logging.getLogger(__name__)

# Word-based approximation of 512 tokens with 50-token overlap.
# count_tokens multiplies words by 1.3 — invert for window sizing.
WORDS_PER_CHUNK = int(512 / 1.3)   # ~394
WORDS_OVERLAP   = int(50 / 1.3)    # ~38


def _build_header_map(text: str):
    """Return sorted (char_offset, kind, value) for every Điều/Khoản header."""
    headers = []
    for m in RE_DIEU.finditer(text):
        headers.append((m.start(), "dieu", m.group(1)))
    for m in RE_KHOAN.finditer(text):
        headers.append((m.start(), "khoan", m.group(1)))
    headers.sort()
    return headers


def _last_header_before(headers, char_offset, kind):
    last = None
    for off, k, v in headers:
        if off > char_offset:
            break
        if k == kind:
            last = v
    return last


class FixedSizeChunker:
    def chunk_document(
        self,
        full_text: str,
        doc_meta: dict,
        doc_type: str,
        source_file: str,
    ) -> list[dict]:
        words = full_text.split()
        if not words:
            return []

        # Build a word-offset → char-offset map so we can look up headers.
        char_offsets = []
        cursor = 0
        for w in words:
            idx = full_text.find(w, cursor)
            char_offsets.append(idx if idx >= 0 else cursor)
            cursor = (idx if idx >= 0 else cursor) + len(w)

        headers = _build_header_map(full_text)

        doc_slug = doc_meta.get("doc_id", source_file).replace("/", "_").replace("-", "_")
        out: list[dict] = []
        seen = set()

        start = 0
        idx = 0
        while start < len(words):
            end = min(start + WORDS_PER_CHUNK, len(words))
            chunk_text = " ".join(words[start:end])

            h = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
            if h in seen:
                start = end - WORDS_OVERLAP if end != len(words) else len(words)
                idx += 1
                continue
            seen.add(h)

            char_start = char_offsets[start] if start < len(char_offsets) else 0
            dieu_primary = _last_header_before(headers, char_start, "dieu")
            khoan_primary = _last_header_before(headers, char_start, "khoan")

            chunk = {
                "metadata": {
                    "chunk_id": f"fixed_{doc_slug}_{idx:04d}",
                    "doc_id": doc_meta.get("doc_id", source_file),
                    "ten_van_ban": doc_meta.get("title", source_file),
                    "issuer": doc_meta.get("issuer", ""),
                    "document_type": doc_type,
                    "status": doc_meta.get("status", "active"),
                    "effective_date": doc_meta.get("effective_date", ""),
                    "topic": doc_meta.get("topic", "Chung"),
                    "dieu": int(dieu_primary) if dieu_primary and dieu_primary.isdigit() else 0,
                    "dieu_title": "",
                    "khoan": khoan_primary,
                    "diem": None,
                    "level": 0,  # sentinel: structure-blind
                    "source_file": source_file,
                    "token_estimate": count_tokens(chunk_text),
                    "chunk_strategy": "fixed_512_o50",
                },
                "content": chunk_text,
            }
            out.append(chunk)

            if end == len(words):
                break
            start = end - WORDS_OVERLAP
            idx += 1

        return out


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    out_path = Path(__file__).resolve().parent / "data" / "all_chunks_fixed512.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunker = FixedSizeChunker()
    total_chunks = 0
    files = 0

    with out_path.open("w", encoding="utf-8") as fo:
        for doc_type, dir_path in CLEANED_DIRS.items():
            if not dir_path.exists():
                logger.warning("Missing: %s", dir_path)
                continue
            for fp in sorted(dir_path.rglob("*.md")):
                if fp.name.startswith("_"):
                    continue
                files += 1
                meta = get_file_metadata(fp.stem)
                text = fp.read_text(encoding="utf-8")
                if not text.strip():
                    continue
                chunks = chunker.chunk_document(text, meta, doc_type, fp.name)
                for c in chunks:
                    fo.write(json.dumps(c, ensure_ascii=False) + "\n")
                total_chunks += len(chunks)
                logger.info("  %s → %d chunks", fp.name, len(chunks))

    logger.info("Processed %d files, wrote %d chunks → %s", files, total_chunks, out_path)


if __name__ == "__main__":
    main()
