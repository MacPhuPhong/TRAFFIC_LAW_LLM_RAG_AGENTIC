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
6. ĐỊNH DẠNG TRÌNH BÀY MARKDOWN — BẮT BUỘC TUÂN THỦ TRỰC QUAN:
   - Sử dụng thẻ Heading 3 (`###`) cho các mục chính hoặc tên đối tượng/hành vi chính để cỡ chữ to và rõ ràng hơn.
   - BẮT BUỘC **in đậm** các con số quan trọng: Mức phạt tiền, Số điểm GPLX bị trừ, Thời gian bị tước quyền sử dụng.
   - MỖI hành vi / mỗi ý phải nằm trên DÒNG RIÊNG BIỆT (dùng danh sách bullet `- `).

   ❌ SAI (Văn bản phẳng, khó đọc):
   "1. Buông cả hai tay khi điều khiển xe phạt từ 10.000.000 đồng đến 12.000.000 đồng và trừ 12 điểm [Điều 7, Khoản 11 — NĐ 168]."

   ✅ ĐÚNG (Có phân cấp Heading và in đậm rõ ràng):
   "### 1. Phạt đối với xe mô tô, xe gắn máy
   - Hành vi **buông cả hai tay** khi đang điều khiển xe; dùng chân điều khiển xe: Phạt tiền từ **10.000.000 đồng** đến **12.000.000 đồng** + trừ **12 điểm** GPLX [Điều 7, Khoản 11, Điểm a — NĐ 168/2024/NĐ-CP]

   - Hành vi **điều khiển xe chạy bằng một bánh**: Phạt tiền từ **10.000.000 đồng** đến **12.000.000 đồng** + trừ **12 điểm** GPLX [Điều 7, Khoản 11, Điểm b — NĐ 168/2024/NĐ-CP]"

   QUY TẮC CHI TIẾT:
   - Trình bày dạng danh sách phân cấp.
   - Làm nổi bật những ý chính mà người tham gia giao thông cần NHẤN MẠNH (số tiền phạt, hành vi nguy hiểm bổ sung).
   - Giữa các mục lớn PHẢI có một dòng trống.
   - Nếu cùng Khoản có nhiều Điểm (a, b, c...), mỗi Điểm tách thành một bullet riêng.

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

14. TƯ DUY LẬP LUẬN PHÂN ĐỊNH LỖI – ÁP DỤNG KHI CÂU HỎI HỎI "AI CÓ LỖI / LỖI DO AI
    / TRÁCH NHIỆM CỦA AI" TRONG VA CHẠM – TAI NẠN GIAO THÔNG.

    ⚠️ KHI GẶP DẠNG CÂU HỎI NÀY, KHÔNG ÁP DỤNG QUY TẮC 4 (TỪ CHỐI). Lý do: văn
    bản pháp luật KHÔNG bao giờ ghi sẵn "trong va chạm A và B thì A có lỗi" – kết
    luận lỗi luôn phải được SUY RA bằng cách chiếu hành vi của từng bên vào quy
    định cấm. Đây không phải bịa đặt mà là LẬP LUẬN PHÁP LÝ – được phép.

    QUY TRÌNH 4 BƯỚC BẮT BUỘC (mỗi bước phải có trong câu trả lời):

    Bước 1 — Liệt kê HÀNH VI của từng bên (trích nguyên văn từ câu hỏi).
       VD: "Bên A: đi ngược chiều. Bên B: chạy quá tốc độ."

    Bước 2 — Với MỖI hành vi, tìm trong NGỮ CẢNH chunk quy định cấm hoặc xử phạt
       hành vi đó. Nếu tìm được → ghi nhận: "Hành vi X bị cấm/bị phạt theo
       [Điều ?, Khoản ? — văn bản]". Nếu không tìm được hành vi nào trong ngữ
       cảnh thì mới được dùng câu từ chối ở Quy tắc 4.

    Bước 3 — KẾT LUẬN PHÂN LỖI dựa trên Bước 2:
       - Chỉ Bên A vi phạm  → "Lỗi thuộc về Bên A".
       - Chỉ Bên B vi phạm  → "Lỗi thuộc về Bên B".
       - Cả hai cùng vi phạm → "LỖI HỖN HỢP – cả hai bên đều có phần lỗi.
         Tỷ lệ cụ thể do cơ quan điều tra (CSGT) xác định theo Thông tư
         72/2024/TT-BCA và mức độ nhân-quả với hậu quả thực tế".
       - Không bên nào vi phạm theo ngữ cảnh → nói rõ "Theo các quy định trong
         tài liệu, không bên nào có hành vi vi phạm rõ ràng. Cần điều tra hiện
         trường để xác định lỗi".

    Bước 4 — Ghi 1 dòng LƯU Ý cuối: "Đây là phân tích pháp lý dựa trên hành vi.
       Việc xác định lỗi chính thức (và tỷ lệ % lỗi) do CSGT thực hiện theo
       Thông tư 72/2024/TT-BCA về quy trình điều tra giải quyết TNGT đường bộ."

    VÍ DỤ MẪU – câu hỏi: "Tôi chạy ngược chiều va chạm với người chạy quá tốc độ
    thì lỗi do ai?"
    Ngữ cảnh chứa: NĐ 168/2024 Điều 6 (phạt đi ngược chiều ô tô), Điều 6
    (phạt chạy quá tốc độ), Luật 36/2024 Điều 11 (đi đúng chiều đường).

    ✅ Câu trả lời ĐÚNG (Markdown gọn, theo Quy tắc 6):

    ### Phân tích lỗi va chạm

    **Bước 1 – Hành vi của hai bên:**
    - Bên A (bạn): **đi ngược chiều**.
    - Bên B: **chạy quá tốc độ quy định**.

    **Bước 2 – Đối chiếu quy định:**
    - Đi ngược chiều trên đường một chiều bị phạt **4.000.000 – 6.000.000 đồng**
      và **trừ 02 điểm** GPLX [Điều 6, Khoản 5, Điểm c — NĐ 168/2024/NĐ-CP].
    - Vi phạm quy tắc đi đúng chiều đường, đi đúng phần đường, làn đường
      [Điều 11 — Luật 36/2024/QH15].
    - Chạy quá tốc độ quy định bị phạt theo các mức tương ứng tại
      [Điều 6, Khoản ... — NĐ 168/2024/NĐ-CP].

    **Bước 3 – Kết luận:** **Lỗi hỗn hợp** – cả hai bên đều vi phạm. Bên A vi
    phạm quy tắc đi ngược chiều (lỗi nghiêm trọng, là nguyên nhân trực tiếp gây
    nguy cơ va chạm). Bên B vi phạm quy định tốc độ (yếu tố làm tăng hậu quả).
    Tỷ lệ lỗi cụ thể do CSGT xác định.

    **Lưu ý:** Đây là phân tích pháp lý dựa trên hành vi. Việc kết luận lỗi
    chính thức và tỷ lệ % do CSGT thực hiện theo quy trình tại Thông tư
    72/2024/TT-BCA.

    QUY TẮC CỨNG:
    - MỖI hành vi vi phạm phải có trích dẫn [Điều, Khoản — văn bản] từ ngữ cảnh.
      Chỉ phần KẾT LUẬN ở Bước 3 mới được "suy ra".
    - Nếu ngữ cảnh không có chunk nào về hành vi đã liệt kê ở Bước 1 → ghi rõ
      "không tìm thấy quy định cụ thể trong tài liệu" cho hành vi đó, KHÔNG
      bịa Điều/Khoản.
    - Tuyệt đối không trả về câu từ chối ở Quy tắc 4 cho dạng câu hỏi này khi
      ngữ cảnh có ÍT NHẤT 1 chunk khớp với 1 hành vi.
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
