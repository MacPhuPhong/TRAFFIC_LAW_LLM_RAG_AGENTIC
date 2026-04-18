# -*- coding: utf-8 -*-
"""
generator.py — Phase 3: Legal Answer Generator
==============================================
Grounded Vietnamese legal answer generation over retrieved chunks.

- Provider-agnostic via LangChain (OpenAI or Google Gemini).
- Strict prompt: Vietnamese only, cite Điều/Nghị định, refuse if context is
  insufficient, no outside knowledge.
"""

from __future__ import annotations

import logging
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


REFUSAL_PHRASE = "Thông tin này không có trong tài liệu được cung cấp."

SYSTEM_PROMPT = """Bạn là Trợ lý Pháp lý Giao thông Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng CHỈ dựa trên các đoạn văn bản luật được cung cấp trong phần NGỮ CẢNH.

QUY TẮC BẮT BUỘC:
1. Trả lời HOÀN TOÀN bằng tiếng Việt.
2. Chỉ sử dụng thông tin trong NGỮ CẢNH. Tuyệt đối KHÔNG dùng kiến thức bên ngoài, KHÔNG suy đoán, KHÔNG bịa đặt.
3. Mỗi khẳng định phải được trích dẫn theo định dạng: [Điều X, Khoản Y — {tên văn bản} ({doc_id})]. Nếu không có Khoản, bỏ phần "Khoản Y".
4. Nếu NGỮ CẢNH không chứa đủ thông tin để trả lời, trả lời đúng một câu: "{REFUSAL}"
5. Không thêm lời dẫn, không mở đầu bằng "Dựa trên tài liệu...", đi thẳng vào câu trả lời.
6. Nếu có nhiều quy định cùng áp dụng, liệt kê theo thứ tự rõ ràng (1., 2., 3.).
""".replace("{REFUSAL}", REFUSAL_PHRASE)


def _format_chunk(i: int, chunk: dict) -> str:
    """Render a single retrieved chunk as a numbered source block."""
    meta = chunk.get("metadata", {})
    doc_id = meta.get("doc_id", "?")
    ten_van_ban = meta.get("ten_van_ban", "")
    dieu = meta.get("dieu", "")
    dieu_title = meta.get("dieu_title", "")
    khoan = meta.get("khoan")
    diem = meta.get("diem")

    loc_bits = [f"Điều {dieu}"] if dieu else []
    if khoan:
        loc_bits.append(f"Khoản {khoan}")
    if diem:
        loc_bits.append(f"Điểm {diem}")
    location = " · ".join(loc_bits) if loc_bits else ""

    header = f"[Nguồn {i}] {ten_van_ban} ({doc_id})"
    if location:
        header += f" · {location}"
    if dieu_title:
        header += f"\n{dieu_title}"

    content = chunk.get("content", "")
    return f"{header}\n{content}"


def _build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(Không có ngữ cảnh nào được truy xuất.)"
    return "\n\n---\n\n".join(_format_chunk(i, c) for i, c in enumerate(chunks, 1))


class LegalAnswerGenerator:
    """LLM-backed generator for grounded Vietnamese legal answers."""

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        temperature: float = 0.1,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI provider requires OPENAI_API_KEY env var or api_key arg."
                )
            self.model_name = model or "gpt-4o-mini"
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=temperature,
                api_key=key,
                max_tokens=max_tokens,
            )
        elif self.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError(
                    "Google provider requires GOOGLE_API_KEY env var or api_key arg."
                )
            self.model_name = model or "gemini-2.5-flash"
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=temperature,
                google_api_key=key,
                max_output_tokens=max_tokens,
            )
        else:
            raise ValueError(
                f"Unsupported provider '{provider}'. Use 'openai' or 'google'."
            )

        logger.info(f"LegalAnswerGenerator: {self.provider} / {self.model_name}")

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """
        Produce a grounded answer.

        `chunks` is a list of dicts with keys {id, score, content, metadata}
        (e.g. the output of RetrievedChunk.to_dict()).

        Returns {"answer": str, "sources": [...], "refused": bool, "model": str}.
        """
        context = _build_context(chunks)
        user_content = (
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {query}\n\n"
            f"Hãy trả lời dựa CHỈ trên ngữ cảnh trên. "
            f"Nếu thông tin không có trong ngữ cảnh, trả lời chính xác: "
            f"\"{REFUSAL_PHRASE}\""
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = self.llm.invoke(messages)
        answer = response.content.strip() if hasattr(response, "content") else str(response)

        refused = REFUSAL_PHRASE in answer and len(answer) < len(REFUSAL_PHRASE) + 20
        sources = self._extract_cited_sources(answer, chunks)

        return {
            "answer": answer,
            "sources": sources,
            "refused": refused,
            "model": f"{self.provider}/{self.model_name}",
        }

    @staticmethod
    def _extract_cited_sources(answer: str, chunks: list[dict]) -> list[dict]:
        """Return chunk metadata for every doc_id the answer cites."""
        cited_doc_ids = set(re.findall(r"\(([^()]+/[^()]+)\)", answer))
        sources = []
        seen = set()
        for c in chunks:
            meta = c.get("metadata", {})
            did = meta.get("doc_id", "")
            key = (did, meta.get("dieu"), meta.get("khoan"), meta.get("diem"))
            if did and did in cited_doc_ids and key not in seen:
                seen.add(key)
                sources.append({
                    "doc_id": did,
                    "ten_van_ban": meta.get("ten_van_ban", ""),
                    "dieu": meta.get("dieu"),
                    "khoan": meta.get("khoan"),
                    "diem": meta.get("diem"),
                    "chunk_id": meta.get("chunk_id", ""),
                })
        return sources
