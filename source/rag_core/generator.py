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
6. ĐỊNH DẠNG LIỆT KÊ — BẮT BUỘC TUÂN THỦ:
   MỖI hành vi / mỗi điểm a), b), c)... phải nằm trên DÒNG RIÊNG BIỆT.
   TUYỆT ĐỐI KHÔNG gộp nhiều hành vi trên cùng một dòng hay một đoạn văn.

   ❌ SAI (KHÔNG BAO GIỜ viết thế này):
   "1. Các hành vi: a) Buông cả hai tay khi điều khiển xe; b) Điều khiển xe chạy bằng một bánh; c) Tái phạm lạng lách [Điều 7, Khoản 11 — NĐ 168]."

   ✅ ĐÚNG (LUÔN LUÔN viết thế này):
   "1. Buông cả hai tay khi đang điều khiển xe; dùng chân điều khiển xe [Điều 7, Khoản 11, Điểm a — NĐ 168/2024/NĐ-CP]

   2. Điều khiển xe chạy bằng một bánh đối với xe hai bánh [Điều 7, Khoản 11, Điểm b — NĐ 168/2024/NĐ-CP]

   3. Tái phạm hành vi lạng lách, đánh võng [Điều 7, Khoản 11, Điểm c — NĐ 168/2024/NĐ-CP]"

   QUY TẮC CHI TIẾT:
   - Mỗi mục bắt đầu bằng số thứ tự "1. ", "2. ", "3. "...
   - Mỗi mục chứa ĐÚNG MỘT hành vi/quy định + ĐÚNG MỘT trích dẫn [Điều/Khoản/Điểm] ở cuối.
   - Giữa các mục PHẢI có một dòng trống.
   - Nếu cùng Khoản có nhiều Điểm (a, b, c...), mỗi Điểm là MỘT mục riêng biệt.

QUY TẮC PHÂN BIỆT NGỮ CẢNH:
7. Về "kinh doanh vận tải": Ưu tiên chunk có đối tượng khớp với câu hỏi. Nếu câu hỏi rõ ràng về xe KINH DOANH vận tải thì ưu tiên Nghị định 10/2020/NĐ-CP; nếu câu hỏi về cá nhân/hộ gia đình/không kinh doanh thì ưu tiên các chunk khác và chỉ trích dẫn 10/2020/NĐ-CP khi thật sự liên quan.
8. Về loại phương tiện trong NĐ 168/2024/NĐ-CP: Điều 6 (ô tô), Điều 7 (xe mô tô, xe gắn máy), Điều 8 (xe máy chuyên dùng), Điều 9 (xe đạp, xe thô sơ). Ưu tiên Điều khớp với phương tiện trong câu hỏi; nếu câu hỏi không nêu rõ loại xe, chọn Điều phù hợp nhất và NÊU RÕ loại xe trong câu trả lời.
9. Một số chunk được đánh dấu "[Ngữ cảnh bổ sung — cùng Điều]": đó là các khoản/điểm cùng Điều với chunk chính, thường chứa thông tin bổ sung như mức phạt tiền hoặc số điểm bị trừ. Được phép dùng các chunk này để hoàn chỉnh câu trả lời (ví dụ: chunk chính có mức phạt, chunk bổ sung có số điểm trừ → ghép lại thành câu trả lời đầy đủ).
10. CROSS-REFERENCE TRONG NĐ 168/2024/NĐ-CP — BẮT BUỘC khi câu hỏi yêu cầu cả phạt tiền VÀ trừ điểm:
    - Các Điều 6, 7, 8, 9 có cấu trúc: các Khoản đầu liệt kê mức **phạt tiền** cho từng hành vi (kèm điểm a, b, c...); các Khoản cuối (thường K13–K16) liệt kê **số điểm trừ GPLX** bằng cách tham chiếu ngược tới "điểm X khoản Y Điều này".
    - QUY TRÌNH 3 BƯỚC khi trả lời:
      (1) Tìm trong ngữ cảnh chunk có **mức phạt tiền** khớp hành vi → ghi nhận (khoản Y, điểm X).
      (2) Tìm trong ngữ cảnh chunk có cụm "trừ điểm giấy phép lái xe" → đây là bảng tham chiếu. Kiểm tra bảng có liệt kê (khoản Y, điểm X) không. Nếu có → đọc số điểm trừ tương ứng.
      (3) Ghép: "Phạt tiền ... đồng + trừ N điểm GPLX".
    - VÍ DỤ CỤ THỂ: Câu hỏi "vượt đèn đỏ ô tô bị phạt và trừ mấy điểm?".
      Chunk [Điều 6 Khoản 9]: "Phạt tiền từ 18 đến 20 triệu đồng... c) không chấp hành hiệu lệnh của đèn tín hiệu giao thông" → (Khoản 9, điểm c).
      Chunk [Điều 6 Khoản 16] (sibling): "b) điểm b, c, d khoản 9 bị trừ 04 điểm GPLX" → (khoản 9 điểm c) khớp → trừ 4 điểm.
      Trả lời: "Phạt tiền 18.000.000–20.000.000 đồng + trừ 04 điểm GPLX [Điều 6, Khoản 9, Điểm c — NĐ 168/2024/NĐ-CP] [Điều 6, Khoản 16, Điểm b — NĐ 168/2024/NĐ-CP]".
    - Nếu bảng trừ điểm KHÔNG liệt kê (khoản Y, điểm X) → nói "không bị trừ điểm GPLX", KHÔNG từ chối.
11. QUAN TRỌNG — KHÔNG được từ chối nếu ngữ cảnh có ÍT NHẤT một phần thông tin liên quan. Nếu tìm được mức phạt tiền nhưng không tìm được số điểm trừ (hoặc ngược lại), trả lời phần tìm được và ghi rõ một câu ngắn về phần thiếu. Chỉ dùng câu từ chối ở Quy tắc 4 khi ngữ cảnh KHÔNG có bất kỳ chunk nào liên quan đến câu hỏi.
12. TỔNG HỢP (Summarization): Với các câu hỏi yêu cầu liệt kê (Trường hợp nào, Các hành vi...), hãy rà soát TOÀN BỘ ngữ cảnh để trích xuất các ví dụ tiêu biểu và tổng hợp thành một danh sách đầy đủ nhất có thể dựa trên tài liệu.
13. GIẢI THÍCH DỄ HIỂU — QUY TẮC QUAN TRỌNG NHẤT:
    TUYỆT ĐỐI KHÔNG BAO GIỜ chỉ trích dẫn mã Điều/Khoản/Điểm mà không giải thích nội dung.
    Người dùng là CÔNG DÂN BÌNH THƯỜNG, không phải luật sư — họ cần biết hành vi cụ thể.

    ❌ SAI (KHÔNG BAO GIỜ viết thế này):
    "Tạm giữ phương tiện đối với hành vi quy định tại điểm a khoản 4 Điều 13"

    ✅ ĐÚNG (LUÔN LUÔN viết thế này):
    "Tạm giữ phương tiện khi: Điều khiển xe không có giấy đăng ký xe hoặc giấy đăng ký xe đã hết hạn [Điều 13, Khoản 4, Điểm a — NĐ 168/2024/NĐ-CP]"

    QUY TẮC:
    - Nếu trong NGỮ CẢNH có chunk chứa NỘI DUNG CHI TIẾT của hành vi → PHẢI mô tả hành vi đó bằng ngôn ngữ rõ ràng.
    - Nếu chunk chỉ chứa tham chiếu chéo (ví dụ: "hành vi quy định tại điểm X khoản Y Điều Z") và NGỮ CẢNH KHÔNG có nội dung chi tiết của Điều Z đó → vẫn phải ghi rõ: "Hành vi quy định tại [Điều Z, Khoản Y, Điểm X] — (chi tiết xem tại Điều Z)" thay vì chỉ ghi mã số.
    - Ưu tiên tuyệt đối: MÔ TẢ HÀNH VI BẰNG NGÔN NGỮ TỰ NHIÊN trước, rồi mới trích dẫn [Điều/Khoản] ở cuối.
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
    is_sibling = bool(meta.get("is_sibling"))

    loc_bits = [f"Điều {dieu}"] if dieu else []
    if khoan:
        loc_bits.append(f"Khoản {khoan}")
    if diem:
        loc_bits.append(f"Điểm {diem}")
    location = " · ".join(loc_bits) if loc_bits else ""

    tag = "Ngữ cảnh bổ sung — cùng Điều" if is_sibling else f"Nguồn {i}"
    header = f"[{tag}] {ten_van_ban} ({doc_id})"
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
            self.model_name = model or "gemini-1.5-flash"
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
            f"Hãy trả lời dựa CHỈ trên ngữ cảnh trên, tuân thủ các quy tắc ở hệ thống."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = self.llm.invoke(messages)
        answer = response.content.strip() if hasattr(response, "content") else str(response)

        answer_trim = answer.strip().rstrip(".")
        refused = answer_trim == REFUSAL_PHRASE.rstrip(".")
        sources = self._extract_cited_sources(answer, chunks)

        return {
            "answer": answer,
            "sources": sources,
            "refused": refused,
            "model": f"{self.provider}/{self.model_name}",
        }

    @staticmethod
    def _extract_cited_sources(answer: str, chunks: list[dict]) -> list[dict]:
        """Return chunk metadata for every doc_id the answer cites.

        Recognises three citation formats the model tends to emit:
          1. `(168/2024/NĐ-CP)` — parentheses per the prompted format.
          2. `[168/2024/NĐ-CP]` — square brackets (LLM often substitutes).
          3. Naked `168/2024/NĐ-CP` inside free text.
        """
        bracketed = re.findall(
            r"[\(\[]\s*([^()\[\]]+?/[^()\[\]]+?)\s*[\)\]]", answer
        )
        cited_doc_ids = set(bracketed)
        for m in re.finditer(r"\b(\d+/\d{4}/[A-ZĐ\-]+)\b", answer):
            cited_doc_ids.add(m.group(1))

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
