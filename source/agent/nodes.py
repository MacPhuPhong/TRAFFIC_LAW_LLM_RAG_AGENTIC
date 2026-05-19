# -*- coding: utf-8 -*-
"""
nodes.py — LangGraph node callables for the agentic RAG workflow.

Each node receives AgentState and returns a partial-state dict that
LangGraph merges back into the global state.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .state import AgentState, Category, Intent, VehicleType


logger = logging.getLogger(__name__)


# --- Prompts -----------------------------------------------------------------

CHIT_CHAT_SYSTEM_PROMPT = """Bạn là Trợ lý Luật Giao thông Việt Nam thân thiện.
Trả lời ngắn gọn 1-2 câu, thân mật, bằng tiếng Việt. Gợi ý người dùng có thể
hỏi về luật giao thông (mức phạt, quy định đăng ký/đăng kiểm, v.v.) nếu phù hợp.
KHÔNG trích dẫn Điều/Khoản luật ở đây; không bịa thông tin pháp lý cụ thể."""


WEB_SEARCH_SYSTEM_PROMPT = """Bạn là chuyên gia tổng hợp thông tin pháp lý từ Internet.
Tổng hợp câu trả lời dựa trên các kết quả web cung cấp.

QUY TẮC BẮT BUỘC:
1. Ghi LUÔN Ở ĐẦU CÂU dòng cảnh báo: "⚠️ Lưu ý: Thông tin tra cứu từ Internet mở, hãy cẩn trọng."
2. Nếu kết quả web không đủ để trả lời, nói dứt khoát: "Xin lỗi, không có thông tin online xác thực."
3. Trích xuất và liệt kê URLs nguồn ở cuối câu trả lời dưới mục "Nguồn:".
4. Trả lời bằng tiếng Việt, ngắn gọn và rõ ràng."""


OUT_OF_SCOPE_MESSAGE = (
    "Xin lỗi, tôi chỉ tư vấn về Luật Giao thông Việt Nam và không giải quyết các "
    "vấn đề ngoài phạm vi này. Bạn có thể hỏi tôi về mức phạt vi phạm, quy định "
    "đăng ký xe, đăng kiểm, cấp giấy phép lái xe, v.v."
)


class AnalyzerOutput(BaseModel):
    """Consolidated output for the pre-processing analyzer.

    Drives 3 downstream gates:
      - `category`: top-level routing (legal_rag / chit_chat / web / oos)
      - `intent`:   fine-grained type WITHIN legal_rag — penalty / fault /
                    procedure / definition / list / mixed. Downstream uses
                    intent to (a) demand specific shape in chunks via
                    Confidence Judge, (b) activate Rule 14 fault analysis,
                    (c) adjust top_k and judging threshold for list queries.
      - `expanded_queries`: 1-5 retrieval seeds, one per legal frame.
    """
    category: Category = Field(description="Top-level intent (legal_rag, chit_chat, etc.)")
    intent: Intent = Field(
        default="penalty",
        description=(
            "Fine-grained intent inside legal_rag: 'penalty' (asks fine amount), "
            "'fault' (who is at fault in collision), 'procedure' (registration, "
            "inspection, licensing process), 'definition' (what is X?), 'list' "
            "(enumerate cases / behaviors), 'mixed' (multiple intents). "
            "For non-legal_rag categories, use the closest fit or 'definition'."
        ),
    )
    vehicle_type: VehicleType = Field(
        default="any",
        description=(
            "Phương tiện chính trong câu hỏi: 'o_to' (ô tô con/tải/khách → Điều 6 "
            "NĐ 168), 'mo_to' (xe mô tô/gắn máy/xe máy điện → Điều 7), "
            "'chuyen_dung' (xe máy chuyên dùng, máy kéo → Điều 8), 'xe_dap' "
            "(xe đạp/xe thô sơ → Điều 9). Mặc định 'any' khi câu hỏi không nêu "
            "loại xe rõ ràng. KHÔNG suy đoán; chỉ dùng từ ngữ trong câu hỏi + "
            "lịch sử hội thoại."
        ),
    )
    standalone_query: str = Field(description="Self-contained rewrite of the query using history.")
    expanded_queries: list[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=5,
        description=(
            "1-5 reformulations of the query, each targeting ONE legal frame. "
            "For ambiguous / short queries (e.g. 'vượt tín hiệu đường sắt'), "
            "return ≥2 elements — one per plausible interpretation."
        ),
    )

ANALYZER_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích truy vấn Luật Giao thông Việt Nam.
Nhiệm vụ: Đọc lịch sử hội thoại + câu hỏi mới, thực hiện 5 việc trong 1 lần gọi:

1. PHÂN LOẠI (Category):
   - 'legal_rag': Hỏi về luật, mức phạt, đăng kiểm, đăng ký xe, quy định giao thông VN,
     **HOẶC** câu hỏi phân định lỗi/trách nhiệm trong va chạm – tai nạn giao thông
     (kể cả khi câu hỏi mô tả tình huống chứ không hỏi trực tiếp "luật nói gì").
   - 'chit_chat': Xã giao, chào hỏi, hỏi "bạn là ai".
   - 'web_legal_search': CHỈ dùng cho:
       (a) Luật giao thông NƯỚC NGOÀI (Mỹ, Nhật, Thái…) — phải nêu rõ tên nước.
       (b) Tin tức giao thông CỰC KỲ MỚI có ngày cụ thể (vd. "hôm qua",
           "tuần này", một sự kiện cụ thể) chưa thể có trong văn bản.
     ❌ KHÔNG bao giờ dùng cho:
       - Câu hỏi về hành vi vi phạm tại VN (vượt, dừng, đỗ, tốc độ,
         nồng độ cồn, không có gương, không đội mũ…) dù ngắn hay không.
       - Câu hỏi nêu Điều/Khoản/Nghị định/Thông tư.
       - Câu hỏi mô tả tình huống tai nạn hoặc va chạm tại VN.
       - Câu hỏi so sánh quy định giữa các tỉnh/thành VN.
       - Câu hỏi có thuật ngữ pháp lý VN (giấy phép lái xe, đăng kiểm,
         đăng ký xe, vi phạm hành chính…).
     Nếu nghi ngờ → DEFAULT 'legal_rag' (corpus có Luật 36/2024, NĐ
     168/2024, TT 72/2024 + nhiều văn bản khác — đủ phủ phần lớn câu hỏi).
   - 'out_of_scope': Hoàn toàn không liên quan giao thông (nấu ăn, code, y tế…).

   ⚠️ NGUYÊN TẮC VÀNG:
   - Bất kỳ câu hỏi nào CHỨA tình huống thực tế trên đường VN (va chạm, đụng xe,
     tai nạn, ai sai, lỗi do ai, bồi thường, phân định lỗi…) ĐỀU thuộc 'legal_rag'.
   - Câu hỏi so sánh quy định giữa các tỉnh/thành VN (Hà Nội, HCM, Đà Nẵng…) VẪN
     là 'legal_rag'. Luật giao thông VN áp dụng THỐNG NHẤT TOÀN QUỐC; nếu user nêu
     "tỉnh A khác tỉnh B" thì RAG phải đối chiếu corpus để bác/xác nhận.
   - Câu hỏi về VẬT THỂ NGOÀI GIAO THÔNG văng/rơi xuống đường (bóng thể thao, hàng
     hoá, súc vật…) gây va chạm → 'legal_rag'.

2. INTENT (chỉ áp dụng khi category = legal_rag) — phân loại LOẠI THÔNG TIN user hỏi:
   - 'penalty':    hỏi mức phạt tiền / điểm trừ GPLX cho hành vi cụ thể.
                   Trigger: "phạt bao nhiêu", "mức phạt", "bị phạt", "trừ mấy điểm".
   - 'fault':      phân định lỗi / trách nhiệm trong va chạm – tai nạn.
                   Trigger: "lỗi do ai", "ai có lỗi", "trách nhiệm", "ai sai".
   - 'procedure':  thủ tục, quy trình, hồ sơ, đăng ký, đăng kiểm, cấp GPLX.
                   Trigger: "thủ tục", "quy trình", "hồ sơ", "đăng ký xe", "đăng kiểm",
                   "cấp giấy phép", "thi sát hạch".
   - 'definition': định nghĩa khái niệm / loại hạng / quy tắc tổng quát.
                   Trigger: "là gì", "bao gồm những gì", "có mấy loại", "khái niệm",
                   "quy tắc khi…".
   - 'list':       liệt kê các trường hợp / hành vi / điều kiện.
                   Trigger: "khi nào", "các trường hợp", "những hành vi", "danh sách",
                   "liệt kê".
   - 'mixed':      có ≥2 intent trong câu (vd: tai nạn + hỏi mức phạt = fault + penalty).
                   Trigger: câu có CẢ keyword phân lỗi (lỗi, trách nhiệm) VÀ keyword
                   mức phạt (phạt, bao nhiêu).
   Nếu là chit_chat / web / out_of_scope → chọn intent gần nhất (thường 'definition').

3. LOẠI PHƯƠNG TIỆN (vehicle_type) — chỉ dùng từ ngữ trong câu hỏi + history:
   - 'o_to':         câu hỏi có "ô tô", "xe ô tô", "xe con", "xe tải", "xe khách",
                     "xe bán tải", "ô tô con" → Điều 6 NĐ 168.
   - 'mo_to':        câu hỏi có "xe máy", "mô tô", "xe gắn máy", "xe máy điện",
                     "moto", "xe 2 bánh" → Điều 7 NĐ 168.
   - 'chuyen_dung':  câu hỏi có "máy chuyên dùng", "xe ba bánh chuyên dùng",
                     "máy kéo" → Điều 8 NĐ 168.
   - 'xe_dap':       câu hỏi có "xe đạp", "xe thô sơ" → Điều 9 NĐ 168.
   - 'any':          câu hỏi KHÔNG nêu rõ loại xe (vd "vượt đèn đỏ bị phạt
                     bao nhiêu") → mặc định, generator sẽ liệt kê đa-loại-xe.
   ⚠️ TUYỆT ĐỐI KHÔNG SUY ĐOÁN. Câu "tôi vi phạm" không nêu xe → 'any'.
   Nếu lịch sử hội thoại có nhắc loại xe → có thể carry over.

4. CHUẨN HOÁ (standalone_query): Viết lại thành câu độc lập, giải tham chiếu từ
   lịch sử nếu có ("xe đó" → "xe ô tô con", "vậy thì sao" → câu đầy đủ trước đó).

5. MỞ RỘNG (expanded_queries) — LIST 1-5 PHẦN TỬ, MỖI FRAME 1 PHẦN TỬ:

   Retriever chạy hybrid search RIÊNG cho từng phần tử rồi RRF-fuse. Vì vậy:
   - MỖI phần tử = 1 frame pháp lý độc lập, KEYWORD TÁCH RỜI.
   - KHÔNG nhồi keyword của frame A vào phần tử của frame B.
   - Luôn bổ sung "Điều X Khoản Y Nghị định 168/2024" nếu biết — kích hoạt
     structural lookup downstream.

   - 1 phần tử: câu đơn-nghĩa đủ context (loại xe + hành vi + điều kiện).
   - 2-4 phần tử: câu đa-nghĩa hoặc tình huống tai nạn (1 hành vi của mỗi bên
     = 1 phần tử + 1 phần tử "phân định lỗi Luật 36/2024" ở cuối).

   ⚠️ KHÔNG cần "dịch" từ ngữ đời thường sang thuật ngữ pháp lý — hệ thống có
   nhánh HyDE riêng tự dùng vocabulary pháp lý để bridge gap. Bạn chỉ cần
   giữ trung thực ý câu hỏi + tách frame nếu đa-nghĩa.

FEW-SHOT:

Ví dụ 1 — penalty đơn-nghĩa, không nêu loại xe:
  Q: "đi ngược chiều thì bị phạt bao nhiêu?"
  → category: legal_rag · intent: penalty · vehicle_type: any
  → standalone_query: "Đi ngược chiều bị phạt bao nhiêu tiền?"
  → expanded_queries: [
      "mức phạt đi ngược chiều xe ô tô xe máy đường một chiều
       Điều 6 Điều 7 Nghị định 168/2024 trừ điểm GPLX"
    ]

Ví dụ 2 — fault (tai nạn 2 hành vi, hỏi LỖI):
  Q: "tôi chạy ngược chiều va chạm với người chạy quá tốc độ thì lỗi do ai"
  → category: legal_rag · intent: fault · vehicle_type: any
  → standalone_query: "Va chạm giữa người ngược chiều và người quá tốc độ, lỗi ai?"
  → expanded_queries: [
      "mức phạt đi ngược chiều Điều 6 Điều 7 Nghị định 168/2024",
      "mức phạt chạy quá tốc độ Điều 6 Khoản 5 Khoản 6 Nghị định 168/2024",
      "phân định lỗi va chạm trách nhiệm hỗn hợp Luật 36/2024 Thông tư 72/2024"
    ]

Ví dụ 3 — penalty đa-nghĩa (câu ngắn, 2 frame, tách keyword):
  Q: "vượt tín hiệu đường sắt"
  → category: legal_rag · intent: penalty · vehicle_type: any
  → standalone_query: "Vượt tín hiệu tại đường sắt bị phạt như thế nào?"
  → expanded_queries: [
      "mức phạt vượt đèn đỏ không chấp hành hiệu lệnh đèn tín hiệu giao thông
       Điều 6 Khoản 9 Điều 7 Khoản 7 Nghị định 168/2024 trừ điểm GPLX",
      "vượt qua đường ngang đường sắt khi đèn nhấp nháy chuông kêu rào chắn
       đóng Điều 6 Khoản 4 Điểm d Điều 7 Khoản 2 Điểm e Nghị định 168/2024"
    ]
  Lưu ý: phần tử 1 KHÔNG có "đường sắt"; phần tử 2 KHÔNG có "đèn tín hiệu
  giao thông" — keyword tách rời để BM25 không kéo lệch.

Ví dụ 4 — out-of-scope:
  Q: "công thức nấu phở bò"  → category: out_of_scope · intent: definition
                              · vehicle_type: any
  → standalone_query / expanded_queries: [câu gốc]

Ví dụ 5 — MIXED (tai nạn + hỏi MỨC PHẠT) — không nêu xe:
  Q: "tránh xe tải tông phải trẻ em mức phạt bao nhiêu"
  → category: legal_rag · intent: mixed · vehicle_type: any
     (user không nêu rõ MÌNH đi xe gì; chỉ nói "xe tải" là phương tiện khác.
      Generator sẽ liệt kê đa-loại-xe cho người gây va chạm.)
  → standalone_query: "Khi tránh xe tải dẫn đến va chạm với trẻ em đi bộ qua
     đường, người lái bị phạt bao nhiêu và xử lý thế nào?"
  → expanded_queries: [
      "mức phạt người điều khiển xe gây tai nạn giao thông làm thiệt hại về
       người Điều 6 Khoản 10 Điều 7 Khoản 10 Nghị định 168/2024 trừ điểm GPLX",
      "quy tắc giao thông nhường đường người đi bộ trẻ em qua đường
       Điều 11 Điều 30 Luật trật tự an toàn giao thông đường bộ 36/2024",
      "phân định lỗi va chạm tình thế cấp thiết bồi thường thiệt hại
       quy trình điều tra Thông tư 72/2024 TT-BCA"
    ]
  Lưu ý: mixed = penalty + fault → 3 phần tử (penalty + quy tắc + procedure).
  KHÔNG được bỏ phần penalty dù câu hỏi có cụm "tai nạn".

Ví dụ 6 — COMPOUND (2 hành vi cùng lúc, có nêu xe rõ):
  Q: "băng qua đường ray xe lửa khi có tín hiệu đèn đỏ và chở 3 người trên xe máy"
  → category: legal_rag · intent: penalty · vehicle_type: mo_to
     ("xe máy" trong câu → Điều 7 NĐ 168)
  → standalone_query: "Trên xe máy, vừa vượt tín hiệu đèn tại đường sắt vừa chở
     3 người thì bị phạt bao nhiêu?"
  → expanded_queries: [
      "mức phạt vượt qua đường ngang đường sắt khi đèn nhấp nháy
       Điều 7 Khoản 2 Điểm e Nghị định 168/2024 xe mô tô",
      "mức phạt vượt đèn đỏ không chấp hành hiệu lệnh đèn tín hiệu giao thông
       xe mô tô Điều 7 Khoản 7 Điểm c Nghị định 168/2024 trừ điểm GPLX",
      "mức phạt chở quá số người quy định xe mô tô xe gắn máy chở 3
       Điều 7 Khoản 2 Điểm g Điều 7 Khoản 3 Nghị định 168/2024"
    ]
  Lưu ý: compound = mỗi hành vi 1 phần tử (có thể đa-nghĩa thêm cho 1 hành vi).

QUY TẮC CHUNG:
- Trả về JSON đúng schema; expanded_queries LUÔN là list.
- Giữ nguyên loại phương tiện (ô tô / xe máy / xe tải / xe đạp).
- KHÔNG return 'web_legal_search' cho tai nạn / tình huống tại VN.
- Câu hỏi so sánh địa danh VN (HN, HCM, ĐN…) → vẫn là 'legal_rag' (luật áp
  dụng toàn quốc).
"""


# --- Node factories ----------------------------------------------------------

CONTEXTUALIZE_HISTORY_TURNS = 6


def _format_history_for_prompt(history: list[dict]) -> str:
    lines = []
    for msg in history[-CONTEXTUALIZE_HISTORY_TURNS:]:
        role = msg.get("role", "?")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        label = "user" if role == "user" else "assistant"
        lines.append(f"  {label}: {content}")
        
        # New: If there are sources, list them so the LLM can resolve "Điều số X"
        sources = msg.get("sources") or []
        if role == "assistant" and sources:
            source_labels = []
            for i, s in enumerate(sources, 1):
                parts = []
                if s.get("dieu"): parts.append(f"Điều {s['dieu']}")
                if s.get("khoan"): parts.append(f"Khoan {s['khoan']}")
                if s.get("ten_van_ban"): parts.append(s['ten_van_ban'])
                label_str = " · ".join(parts) or s.get("doc_id", "Nguồn")
                source_labels.append(f"[{i}] {label_str}")
            lines.append(f"    (Nguồn đã trích dẫn: {', '.join(source_labels)})")
            
    return "\n".join(lines) if lines else "(chưa có)"


def make_analyzer_node(llm) -> Callable[[AgentState], dict]:

    """Targeted optimization: consolidates Contextualize, Router, and Expansion
    into one LLM call to solve API timeout issues.
    """
    structured = llm.with_structured_output(AnalyzerOutput)

    def analyzer_node(state: AgentState) -> dict:
        raw = state.get("query", "")
        history = state.get("chat_history") or []
        history_text = _format_history_for_prompt(history)

        try:
            result = structured.invoke(
                [
                    SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
                    HumanMessage(content=f"Lịch sử hội thoại:\n{history_text}\n\nCâu hỏi mới nhất: {raw}"),
                ]
            )

            if not result or not hasattr(result, "category"):
                logger.warning("Analyzer returned empty or malformed result; using defaults")
                return {
                    "category": "legal_rag",
                    "intent": "penalty",
                    "vehicle_type": "any",
                    "query": raw,
                    "expanded_query": raw,
                    "expanded_queries": [raw],
                    "raw_query": raw,
                }

            queries = getattr(result, "expanded_queries", None) or [raw]
            queries = [q.strip() for q in queries if q and q.strip()]
            if not queries:
                queries = [raw]

            return {
                "category": getattr(result, "category", "legal_rag"),
                "intent": getattr(result, "intent", "penalty"),
                "vehicle_type": getattr(result, "vehicle_type", "any"),
                "query": getattr(result, "standalone_query", raw),
                # Joined form: kept for API back-compat + `_is_broad_query` regex pass.
                "expanded_query": " | ".join(queries),
                # Primary input for multi-retrieval in legal_rag_node.
                "expanded_queries": queries,
                "raw_query": raw,
            }
        except Exception as exc:
            logger.error("Analyzer failed (%s); falling back to raw defaults", exc)
            return {
                "category": "legal_rag",
                "intent": "penalty",
                "vehicle_type": "any",
                "query": raw,
                "expanded_query": raw,
                "expanded_queries": [raw],
                "raw_query": raw,
            }

    return analyzer_node


def route_by_category(state: AgentState) -> str:
    """Edge condition after analyzer_node."""
    return state.get("category", "out_of_scope")




# Listing-type queries ("khi nào", "các trường hợp", "liệt kê", ...) need a
# wider retrieval window than pinpoint queries. Regex is enough — these phrases
# are distinctive in Vietnamese and avoid an extra LLM hop.
BROAD_QUERY_PATTERNS: tuple[str, ...] = (
    r"\bkhi nào\b",
    r"\btrường hợp\b",
    r"\bcác trường hợp\b",
    r"\bnhững trường hợp\b",
    r"\bliệt kê\b",
    r"\bcác lỗi\b",
    r"\blỗi nào\b",
    r"\bnhững lỗi\b",
    r"\bcác hành vi\b",
    r"\bdanh sách\b",
    r"\bđiều kiện\b",
    r"\bhạng nào\b",
    r"\bcác hạng\b",
    r"\bnhững hạng\b",
    r"\bcác loại\b",
    r"\bnhững loại\b",
    r"\bcó mấy\b",
    r"\bgồm những gì\b",
    r"\bbao gồm\b",
)

LEGAL_RAG_TOP_K_DEFAULT = 15
LEGAL_RAG_TOP_K_BROAD = 25

# --- v3: HyDE (Hypothetical Document Embeddings) + Confidence Judge ---------
# Bridges user vocabulary ↔ legal corpus vocabulary without hand-written
# synonym rules: an LLM sketches a plausible legal answer; we embed THAT
# (not the raw query) and let dense retrieval find chunks in the same
# semantic neighbourhood. See doc/so_do_he_thong.md §12.
HYDE_SYSTEM_PROMPT = (
    "Bạn là chuyên gia Luật Giao thông Việt Nam. Viết 2-3 CÂU văn phong "
    "pháp lý chính thống trả lời thử câu hỏi sau (không cần đúng 100% — "
    "mục đích là để hệ thống tra cứu corpus đối chiếu). Dùng đúng thuật ngữ "
    "trong Luật 36/2024, Nghị định 168/2024, Thông tư 72/2024 (vd: 'súc vật' "
    "thay vì 'động vật'; 'người đi bộ' thay vì 'trẻ em qua đường'; 'va chạm' "
    "thay vì 'tông'). KHÔNG mở đầu kiểu 'Theo tôi'; viết thẳng vào câu trả lời.\n\n"
    "VÍ DỤ MẪU:\n"
    "Q: \"tôi cho động vật lưu thông trên đường\"\n"
    "A: \"Hành vi dẫn dắt súc vật, gia súc, gia cầm trên đường bộ không bảo "
    "đảm an toàn giao thông bị xử phạt theo Điều 9 Nghị định 168/2024/NĐ-CP. "
    "Người điều khiển phải tuân thủ quy định tại Điều 32 Luật trật tự an toàn "
    "giao thông đường bộ 2024 về việc dẫn dắt vật nuôi.\""
)


def _generate_hyde(llm, query: str) -> str:
    """Single short LLM call producing a hypothetical legal answer.

    Used purely as an embedding seed for additional retrieval — never shown
    to the user. Failures fall back to the raw query (safe default).
    """
    try:
        resp = llm.invoke(
            [
                SystemMessage(content=HYDE_SYSTEM_PROMPT),
                HumanMessage(content=f"Câu hỏi: {query}"),
            ]
        )
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = (text or "").strip()
        return text or query
    except Exception as exc:
        logger.warning("HyDE generation failed (%s); falling back to raw query", exc)
        return query


# Confidence-based routing — replaces "generator refuse → fallback web" path
# for queries where retrieval clearly missed (vague / out-of-corpus). Score-
# based, intent-aware: different intents demand different chunk shapes.
CONFIDENCE_MIN_CHUNKS = 3
# RRF-fused score = Σ 1/(60+rank). Top-1 in 1 list ≈ 0.0164; in 2 lists ≈
# 0.0328. Floor 0.010 admits genuine hits while rejecting truly weak matches.
CONFIDENCE_MIN_TOP_SCORE = 0.010
# Confidence score threshold — see _judge_retrieval_confidence below.
CONFIDENCE_SCORE_THRESHOLD = 2.0

# Regex helpers for intent-aware chunk-shape checks.
_MONEY_RX = re.compile(
    r"(\d{1,3}(?:\.\d{3})+\s*đồng|\d+\s*(?:triệu|nghìn)\s*đồng?|\d+\.\d+\s*đồng)",
    re.IGNORECASE,
)
_LEGAL_REF_RX = re.compile(r"(?:Điều|Khoản|Điểm)\s+[\d\w]", re.IGNORECASE)
_RULE_RX = re.compile(
    r"(cấm|không được|phải|nhường đường|quy tắc|đi đúng|tuân thủ)",
    re.IGNORECASE,
)
_PROCEDURE_RX = re.compile(
    r"(thủ tục|quy trình|hồ sơ|trình tự|thẩm quyền|cấp giấy|cấp phép)",
    re.IGNORECASE,
)


def _query_content_tokens(query: str) -> set[str]:
    """Words ≥ 3 chars from query, lowercased, stripped of stopwords."""
    text = re.sub(r"[^\w\s]", " ", (query or "").lower())
    return {
        t for t in text.split()
        if len(t) >= 3 and t not in {
            "bao", "nhiêu", "thế", "nào", "khi", "nếu", "thì", "đối",
            "với", "của", "các", "cho", "được", "trong", "mức", "bị",
        }
    }


def _judge_retrieval_confidence(
    chunks, query: str, intent: str = "penalty"
) -> tuple[bool, list[str], float]:
    """Score-based, intent-aware confidence check.

    Returns (ok, reasons, score).

    Base score:
      +1.0 if ≥ CONFIDENCE_MIN_CHUNKS primary chunks
      +1.0 if top chunk's RRF score ≥ floor
      +1.0 if query content tokens overlap top-3 chunks (≥1 token)

    Intent-specific bonuses/penalties on TOP-3 chunks combined:
      penalty   → +2.0 if money amount regex hits, -1.5 if not.
                  +0.5 if any "Điều/Khoản/Điểm" reference present.
      fault     → +1.5 if rule keywords present (cấm/nhường đường/phải…).
      procedure → +1.5 if procedure keywords present (thủ tục/hồ sơ…).
      definition / list / mixed → no penalty (broader).

    Threshold: score ≥ CONFIDENCE_SCORE_THRESHOLD → pass.
    """
    reasons: list[str] = []
    score = 0.0

    primary = [c for c in chunks if not getattr(c, "metadata", {}).get("is_sibling")]
    if len(primary) < CONFIDENCE_MIN_CHUNKS:
        reasons.append(f"too_few_primary_chunks={len(primary)}")
        return False, reasons, 0.0
    score += 1.0
    reasons.append(f"chunks={len(primary)}")

    top_score = getattr(primary[0], "score", 0.0) or 0.0
    if top_score >= CONFIDENCE_MIN_TOP_SCORE:
        score += 1.0
        reasons.append(f"top_score={top_score:.3f}")
    else:
        reasons.append(f"top_score_low={top_score:.3f}")

    q_tokens = _query_content_tokens(query)
    if q_tokens:
        overlap_count = sum(
            1 for c in primary[:3]
            if q_tokens & _query_content_tokens(getattr(c, "content", ""))
        )
        if overlap_count >= 1:
            score += 1.0
            reasons.append(f"overlap_top3={overlap_count}")
        else:
            reasons.append("no_overlap_top3")

    # Intent-specific shape checks (run on top-3 combined text)
    top3_text = " ".join(getattr(c, "content", "") for c in primary[:3])
    if intent == "penalty":
        if _MONEY_RX.search(top3_text):
            score += 2.0
            reasons.append("has_money")
        else:
            score -= 1.5
            reasons.append("NO_money_for_penalty")
        if _LEGAL_REF_RX.search(top3_text):
            score += 0.5
            reasons.append("has_legal_ref")
    elif intent == "fault":
        if _RULE_RX.search(top3_text):
            score += 1.5
            reasons.append("has_rule")
    elif intent == "procedure":
        if _PROCEDURE_RX.search(top3_text):
            score += 1.5
            reasons.append("has_procedure")
    elif intent == "mixed":
        # accept either penalty OR rule signal
        if _MONEY_RX.search(top3_text) or _RULE_RX.search(top3_text):
            score += 1.5
            reasons.append("mixed:has_signal")

    ok = score >= CONFIDENCE_SCORE_THRESHOLD
    return ok, reasons, score


# L2 parent enrichment thresholds — mitigate attention hijacking:
#   - Skip pull if L2 chunk is > MAX_L2_CHARS (likely lists 10+ Điểm).
#   - Skip pull unless ≥ MIN_L3_PER_KHOAN L3 children were retrieved (signal
#     that the Khoản is genuinely relevant, not just one stray L3 hit).
MAX_L2_CHARS_FOR_PARENT_PULL = 1500
MIN_L3_PER_KHOAN_FOR_PARENT_PULL = 1


def _attach_parent_l2(retriever, chunks: list, max_l2_chars: int = MAX_L2_CHARS_FOR_PARENT_PULL):
    """Pull L2 Khoản parent chunks for L3 Điểm chunks already in top-K.

    L3 chunks describe specific hành vi but the money amount typically lives
    in the L2 parent (e.g. "Phạt tiền từ 400-600K đối với người điều khiển xe...
    a) ... g) Chở 02 người ..."). Without the parent, the generator can't cite
    the fine — see Ảnh 2 under-listing bug.

    Mitigation against attention hijack:
      - Skip L2 chunks longer than `max_l2_chars` — those list 10+ Điểm and
        risk distracting the generator into summarising the entire Khoản.
      - Only pull L2 if ≥ MIN_L3_PER_KHOAN_FOR_PARENT_PULL L3 children of that
        Khoản are already in the top-K (signal of genuine relevance).
    """
    # Count L3 chunks per (doc_id, dieu, khoan)
    l3_count: dict = {}
    seen_ids = set()
    for c in chunks:
        seen_ids.add(c.id)
        meta = getattr(c, "metadata", {}) or {}
        if meta.get("level") != 3:
            continue
        key = (meta.get("doc_id"), meta.get("dieu"), meta.get("khoan"))
        if all(key):
            l3_count[key] = l3_count.get(key, 0) + 1

    pulled = 0
    extras = []
    for key, count in l3_count.items():
        if count < MIN_L3_PER_KHOAN_FOR_PARENT_PULL:
            continue
        doc_id, dieu, khoan = key
        try:
            parents = retriever.get_chunks_by_location(
                doc_id=doc_id, dieu=int(dieu), khoan=khoan,
            )
        except Exception as exc:
            logger.warning("parent-pull failed for %s: %s", key, exc)
            continue
        for p in parents:
            if p.id in seen_ids:
                continue
            p_meta = getattr(p, "metadata", {}) or {}
            if p_meta.get("level") != 2:
                continue
            content = getattr(p, "content", "") or ""
            if len(content) > max_l2_chars:
                logger.info(
                    "parent-pull: skipping L2 (Đ%s K%s) — too long (%d chars > %d)",
                    dieu, khoan, len(content), max_l2_chars,
                )
                continue
            # Mark as sibling so diversify_by_location won't count it against cap
            p_meta = {**p_meta, "is_sibling": True}
            p.metadata = p_meta
            extras.append(p)
            seen_ids.add(p.id)
            pulled += 1

    if pulled:
        logger.info("parent-pull added %d L2 chunks", pulled)
    return chunks + extras


def _has_multi_frame(chunks, intent: str = "penalty") -> bool:
    """Detect when retrieval pulled ≥2 distinct (doc_id, dieu, khoan) buckets,
    indicating user query maps to multiple legal interpretations. Used to
    flag the generator so it produces "### Trường hợp 1/2/..." headings.

    Only meaningful for intents where multiple cases are expected (penalty
    or mixed). For procedure/definition/list, single-frame is the norm.
    Siblings are ignored — they're enrichment, not separate cases.
    """
    if intent not in ("penalty", "mixed"):
        return False
    keys: set = set()
    for c in chunks:
        meta = getattr(c, "metadata", {}) or {}
        if meta.get("is_sibling"):
            continue
        doc_id = meta.get("doc_id")
        dieu = meta.get("dieu")
        khoan = meta.get("khoan")
        if doc_id and dieu and khoan:
            keys.add((doc_id, str(dieu), str(khoan)))
    return len(keys) >= 2


LOW_CONFIDENCE_CLARIFICATION = (
    "Câu hỏi của bạn chưa khớp rõ với quy định nào trong tài liệu được tra "
    "cứu (Luật 36/2024, Nghị định 168/2024, Thông tư 72/2024). Bạn có thể "
    "nói rõ thêm để mình tra chính xác hơn:\n\n"
    "- **Hành vi vi phạm cụ thể** (vd: vượt đèn đỏ, đỗ xe trên cầu, đi sai "
    "làn, chở quá tải, không đội mũ bảo hiểm…)\n"
    "- **Loại phương tiện** (ô tô / mô tô / xe máy / xe tải / xe máy chuyên dùng)\n"
    "- **Tình huống cụ thể** nếu có (đường cao tốc, khu dân cư, ban đêm…)\n\n"
    "Ví dụ câu hỏi rõ:\n"
    "- *Ô tô vượt đèn đỏ tại ngã tư bị phạt bao nhiêu?*\n"
    "- *Đỗ xe máy trên cầu bị xử lý thế nào?*\n"
    "- *Chở 3 người trên xe máy bị phạt bao nhiêu tiền?*"
)

# Multi-Query RAG (L2): when analyzer returns N expansions, each pass pulls
# top-`PER_QUERY_K` chunks; the N lists are then RRF-fused to `top_k` total.
# Tuned so N=3 gives ~24 candidate chunks before fusion (≈ original top_k×1.6).
PER_QUERY_K_MIN = 8
RRF_K_CONSTANT = 60

# Diversity cap (L3): each (doc_id, dieu, khoan) bucket holds at most this many
# chunks in the final top-K. Forces context to span multiple Khoản so the
# generator can list every relevant penalty case instead of silent-picking one.
MAX_CHUNKS_PER_KHOAN = 3


def _is_broad_query(*texts: str) -> bool:
    blob = " ".join(t for t in texts if t).lower()
    return any(re.search(p, blob) for p in BROAD_QUERY_PATTERNS)


def _rrf_fuse_lists(lists_of_chunks, top_k: int, rrf_k: int = RRF_K_CONSTANT):
    """Reciprocal Rank Fusion over N ranked lists of RetrievedChunk.

    Each chunk's RRF score = Σ over lists where it appears: 1 / (rrf_k + rank).
    The chunk with the highest fused score is preferred. Ties broken by the
    earliest list position (i.e. retrieval order preserved for determinism).
    """
    scores: dict = {}
    first_seen: dict = {}
    representatives: dict = {}
    for hits in lists_of_chunks:
        for rank, c in enumerate(hits, start=1):
            cid = c.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in first_seen:
                first_seen[cid] = rank
                representatives[cid] = c
    ordered = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], first_seen.get(kv[0], 1_000_000)),
    )
    return [representatives[cid] for cid, _ in ordered[:top_k]]


def _diversify_by_location(chunks, target_k: int, max_per_khoan: int = MAX_CHUNKS_PER_KHOAN):
    """Cap chunks per (doc_id, dieu, khoan) bucket — forces top-K to span
    multiple Khoản. Preserves the input order (which is already RRF-ranked).

    Siblings (is_sibling=True) are NOT counted against the cap: they always
    pass through, since they exist precisely to complete a primary chunk.
    """
    seen: dict = {}
    primary, sibling = [], []
    for c in chunks:
        if getattr(c, "metadata", {}).get("is_sibling"):
            sibling.append(c)
            continue
        key = (
            c.metadata.get("doc_id", ""),
            c.metadata.get("dieu", ""),
            c.metadata.get("khoan", ""),
        )
        if seen.get(key, 0) >= max_per_khoan:
            continue
        seen[key] = seen.get(key, 0) + 1
        primary.append(c)
        if len(primary) >= target_k:
            break
    return primary + sibling


# Cross-reference pattern: catches "điểm h khoản 14 Điều 32" and its variants.
# Similarity search dilutes pin-point references — resolve structurally via
# retriever.get_chunks_by_location instead.
_REF_DIEM = r"(?:điểm|diem)\s+([a-zđ])"
_REF_KHOAN = r"(?:khoản|khoan)\s+(\d+)"
_REF_DIEU = r"(?:điều|dieu)\s+(\d+)"
DEFAULT_REF_DOC_ID = "168/2024/NĐ-CP"


def _extract_legal_references(*texts: str) -> list[dict]:
    """Return a list of {doc_id, dieu, khoan, diem} dicts for each explicit
    legal reference found. Defaults doc_id to NĐ 168/2024/NĐ-CP when the user
    does not name a document (the dominant corpus)."""
    blob = " ".join(t for t in texts if t).lower()
    refs: list[dict] = []
    seen: set[tuple] = set()

    # Pattern 1: "điểm X khoản Y" (optionally with "Điều Z" nearby).
    for m in re.finditer(
        rf"{_REF_DIEM}\s+{_REF_KHOAN}(?:[^.]{{0,40}}?{_REF_DIEU})?",
        blob,
    ):
        diem, khoan, dieu = m.group(1), m.group(2), m.group(3)
        if not dieu:
            nearby = re.search(_REF_DIEU, blob)
            dieu = nearby.group(1) if nearby else None
        if not dieu:
            continue
        key = (DEFAULT_REF_DOC_ID, int(dieu), khoan, diem)
        if key not in seen:
            seen.add(key)
            refs.append({"doc_id": DEFAULT_REF_DOC_ID, "dieu": int(dieu),
                         "khoan": khoan, "diem": diem})

    # Pattern 2: "khoản Y Điều Z" without điểm (pulls the whole Khoản).
    for m in re.finditer(rf"{_REF_KHOAN}[^.]{{0,40}}?{_REF_DIEU}", blob):
        khoan, dieu = m.group(1), m.group(2)
        key = (DEFAULT_REF_DOC_ID, int(dieu), khoan, None)
        if key not in seen:
            seen.add(key)
            refs.append({"doc_id": DEFAULT_REF_DOC_ID, "dieu": int(dieu),
                         "khoan": khoan, "diem": None})

    return refs


def make_legal_rag_node(retriever, generator, llm=None) -> Callable[[AgentState], dict]:
    """Build the legal_rag node closure.

    `llm` is the analyzer's LangChain chat model — reused for the HyDE
    hypothetical-answer pass. If None, HyDE is skipped (graceful degrade
    to plain Multi-Query Retrieval).
    """
    def legal_rag_node(state: AgentState) -> dict:
        query = state["query"]

        # L1 input: list of N rewrites from analyzer. Fall back to joined string
        # (legacy field) or the standalone query if analyzer returned nothing.
        queries: list[str] = (
            state.get("expanded_queries")
            or ([state["expanded_query"]] if state.get("expanded_query") else [query])
        )
        queries = [q for q in queries if q]
        joined = state.get("expanded_query") or " | ".join(queries)

        top_k = (
            LEGAL_RAG_TOP_K_BROAD
            if _is_broad_query(state.get("raw_query", ""), query, joined)
            else LEGAL_RAG_TOP_K_DEFAULT
        )

        # v3 — HyDE branch: ask LLM to sketch a hypothetical legal answer; that
        # text becomes ONE MORE retrieval seed. Bridges user vocabulary to
        # corpus vocabulary (e.g. "động vật" → answer uses "súc vật" →
        # embedding hits the right chunks).
        #
        # v4-mitigation: SKIP HyDE for intent=mixed. Reason: HyDE amplifies the
        # DOMINANT semantic of a query — for compound queries (vd "chở 3 người
        # + đường sắt"), the hypothetical text leans one frame and starves the
        # other from retrieval. Multi-query rewrite already handles compound;
        # extra HyDE causes drift. See Ảnh 2 under-listing bug.
        intent = state.get("intent") or "penalty"
        hyde_text = ""
        if llm is not None and intent != "mixed":
            hyde_text = _generate_hyde(llm, query)
            logger.info("HyDE: %d chars (intent=%s)", len(hyde_text), intent)
        elif intent == "mixed":
            logger.info("HyDE skipped (intent=mixed — multi-query covers it)")

        # L2: per-query retrieval budget.
        n_pulls = len(queries) + (1 if hyde_text else 0)
        per_query_k = max(PER_QUERY_K_MIN, top_k // max(n_pulls, 1) + 2)

        all_hits = []
        try:
            for q in queries:
                hits = retriever.get_relevant_chunks(q, top_k=per_query_k)
                if hits:
                    all_hits.append(hits)
            if hyde_text:
                hits = retriever.get_relevant_chunks(hyde_text, top_k=per_query_k)
                if hits:
                    all_hits.append(hits)
        except Exception as exc:
            logger.exception("Retriever failed: %s", exc)
            return {
                "chunks": [],
                "answer": "",
                "sources": [],
                "refused": False,
                "model_info": "",
                "error": f"retriever_error: {exc}",
            }

        # L2: fuse N+1 ranked lists. Single-list case reduces to identity.
        if len(all_hits) <= 1:
            chunks = all_hits[0] if all_hits else []
        else:
            chunks = _rrf_fuse_lists(all_hits, top_k=top_k * 2)  # over-pull for diversify

        # Structural pass: if the query quotes an explicit Điều/Khoản/Điểm
        # reference (common in follow-ups like "điểm h khoản 14 Điều 32 là gì"),
        # pull those chunks directly — similarity search often buries them.
        refs = _extract_legal_references(
            state.get("raw_query", ""), query, joined
        )
        seen_ids = {c.id for c in chunks}
        if refs:
            added = 0
            for ref in refs:
                try:
                    extra = retriever.get_chunks_by_location(
                        doc_id=ref["doc_id"], dieu=ref["dieu"],
                        khoan=ref.get("khoan"), diem=ref.get("diem"),
                    )
                except Exception as exc:
                    logger.warning("get_chunks_by_location failed for %s: %s", ref, exc)
                    continue
                for c in extra:
                    if c.id in seen_ids:
                        continue
                    seen_ids.add(c.id)
                    chunks.append(c)
                    added += 1
            logger.info("Reference-pass added %d chunks (refs=%s)", added, refs)

        # --- Cross-reference resolution pass (v5.7) --------------------------
        # Retrieved chunks often contain cross-references like:
        #   "hành vi quy định tại điểm a khoản 4 Điều 13"
        # Only scan the PRIMARY chunks (top-5) for cross-references to avoid 100+ lookups
        # on broad queries, which causes context bloat and generation speed issues.
        primary_texts = " ".join(c.content for c in chunks[:5] if hasattr(c, "content"))
        xrefs = _extract_legal_references(primary_texts)
        if xrefs:
            xref_added = 0
            # Limit list of refs to investigate to avoid bloat
            xrefs = xrefs[:15] 

            # Only dieu-level refs that are NOT already in our chunks' articles
            present_dieus = {
                (c.metadata.get("doc_id", DEFAULT_REF_DOC_ID),
                 c.metadata.get("dieu"))
                for c in chunks if hasattr(c, "metadata")
            }
            for ref in xrefs:
                # Skip if we already have chunks from that exact article
                ref_key = (ref["doc_id"], ref["dieu"])
                if ref_key in present_dieus:
                    continue
                try:
                    extra = retriever.get_chunks_by_location(
                        doc_id=ref["doc_id"], dieu=ref["dieu"],
                        khoan=ref.get("khoan"), diem=ref.get("diem"),
                    )
                except Exception as exc:
                    logger.warning("xref-pass failed for %s: %s", ref, exc)
                    continue
                for c in extra:
                    if c.id in seen_ids:
                        continue
                    seen_ids.add(c.id)
                    chunks.append(c)
                    xref_added += 1
                # Safety cap: don't bloat context beyond usefulness
                if xref_added >= 30:
                    break
            if xref_added:
                logger.info("Cross-ref pass added %d chunks (xrefs=%d found)",
                            xref_added, len(xrefs))

        # L3 — Diversity reranking: cap chunks per (Điều, Khoản) so the
        # generator's context spans multiple Khoản. Skipped when only one
        # retrieval list was produced (no fusion happened).
        if len(all_hits) > 1:
            chunks = _diversify_by_location(chunks, target_k=top_k)

        # v4+ — Parent L2 enrichment: when top-K has L3 Điểm chunks but
        # missing their L2 Khoản parent (which contains the money amount),
        # auto-pull the parent. Critical for penalty/mixed intent where the
        # generator needs to cite a specific số tiền — see Ảnh 2 under-list bug.
        # Mitigation: skip pull when L2 chunk is >1500 chars (lists many Điểm)
        # to avoid attention hijack on Flash Lite.
        if intent in ("penalty", "mixed"):
            chunks = _attach_parent_l2(retriever, chunks)

        # v3+ — Intent-aware Confidence Judge (deterministic, score-based).
        # Different intents demand different shapes in retrieved chunks:
        #   penalty → must contain money amount
        #   fault   → must contain rule keywords
        #   procedure → must contain procedure keywords
        # If score < threshold → clarification template (no fallback to web).
        ok, reasons, conf_score = _judge_retrieval_confidence(chunks, query, intent)

        # Deterministic multi-frame detection: ≥2 distinct (Điều, Khoản)
        # buckets → flag generator to use "### Trường hợp 1/2" headings.
        # Only meaningful for penalty / mixed intents.
        multi_frame = _has_multi_frame(chunks, intent)

        khoan_set = sorted({
            f"Đ{c.metadata.get('dieu')}K{c.metadata.get('khoan')}"
            for c in chunks
            if c.metadata.get("dieu") and c.metadata.get("khoan")
        })
        vehicle_type = state.get("vehicle_type") or "any"
        logger.info(
            "RAG turn: n_queries=%d hyde=%s per_query_k=%d total_chunks=%d "
            "intent=%s vehicle=%s conf_score=%.2f conf_ok=%s reasons=%s "
            "multi_frame=%s khoan_set=%s",
            len(queries), bool(hyde_text), per_query_k, len(chunks),
            intent, vehicle_type, conf_score, ok, reasons, multi_frame, khoan_set[:20],
        )
        if not ok:
            # Keep chunks non-empty so legal_fallback_router routes to END,
            # not web_search. The chunks WERE retrieved — they just didn't
            # pass relevance bar for this intent. Surfacing clarification
            # template avoids burning tokens on a hallucinated answer AND
            # avoids Tavily call on a query we know is too vague.
            return {
                "chunks": [c.to_dict() for c in chunks],
                "answer": LOW_CONFIDENCE_CLARIFICATION,
                "sources": [],
                "refused": False,
                "multi_frame": False,
                "model_info": f"low-confidence:{','.join(reasons)}",
            }

        chunk_dicts = [c.to_dict() for c in chunks]

        try:
            result = generator.generate(
                query, chunk_dicts,
                intent=intent, multi_frame=multi_frame,
                vehicle_type=vehicle_type,
            )
        except Exception as exc:
            logger.exception("Generator failed: %s", exc)
            return {
                "chunks": chunk_dicts,
                "answer": "",
                "sources": [],
                "refused": False,
                "model_info": "",
                "error": f"generator_error: {exc}",
            }

        return {
            "chunks": chunk_dicts,
            "answer": result["answer"],
            "sources": result["sources"],
            "refused": bool(result.get("refused")),
            "multi_frame": multi_frame,
            "model_info": result.get("model", ""),
        }

    return legal_rag_node


def legal_fallback_router(state: AgentState) -> str:
    """Edge condition after legal_rag_node.

    Fall back to web_search ONLY when the generator genuinely could not answer
    from the local corpus (no chunks, or generator returned refused=True).
    Infra errors (retriever/generator exceptions) short-circuit to END so the
    user sees the real cause instead of a misleading web fallback."""
    if state.get("error"):
        logger.info("legal_rag error -> end (no web fallback for infra errors)")
        return "end"
    if state.get("refused") or not state.get("chunks"):
        logger.info("legal_rag refused/empty -> falling back to web_search")
        return "web_search"
    return "end"


def make_chit_chat_node(llm) -> Callable[[AgentState], dict]:
    def chit_chat_node(state: AgentState) -> dict:
        query = state["query"]
        try:
            resp = llm.invoke(
                [
                    SystemMessage(content=CHIT_CHAT_SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ]
            )
            answer = resp.content.strip() if hasattr(resp, "content") else str(resp)
        except Exception as exc:
            logger.exception("chit_chat LLM call failed: %s", exc)
            answer = (
                "Xin chào! Tôi là Trợ lý Luật Giao thông Việt Nam. "
                "Bạn có thể hỏi tôi về mức phạt, quy định đăng ký xe, đăng kiểm, v.v."
            )
        return {
            "answer": answer,
            "sources": [],
            "refused": False,
        }

    return chit_chat_node


def out_of_scope_node(state: AgentState) -> dict:
    return {
        "answer": OUT_OF_SCOPE_MESSAGE,
        "sources": [],
        "refused": False,
    }


WEB_SEARCH_LEGAL_PREFIX = "Luật giao thông Việt Nam"


def _build_web_search_query(state: AgentState) -> str:
    """Pick the best query for Tavily and anchor it to Vietnamese traffic law.

    Prefers the expanded legal-terminology query; falls back to raw. Always
    prepends a legal-domain prefix so the search stays on-topic even when the
    user input is ambiguous (e.g. a follow-up like "đã lỡ bỏ chạy rồi").
    """
    base = (state.get("expanded_query") or state.get("query") or "").strip()
    if not base:
        return WEB_SEARCH_LEGAL_PREFIX
    if WEB_SEARCH_LEGAL_PREFIX.lower() in base.lower():
        return base
    return f"{WEB_SEARCH_LEGAL_PREFIX}: {base}"


WEB_WARNING_PREFIX = "⚠️ Lưu ý: Thông tin tra cứu từ Internet mở, hãy cẩn trọng."


def make_web_search_node(tavily_tool, llm) -> Callable[[AgentState], dict]:
    """Produce a DRAFT answer (stored as `draft_answer`, not `answer`).

    The graph will then pause at `web_finalize` so a human reviewer can
    approve/edit/reject the draft before it becomes the final `answer` shown
    to the user. Does NOT overwrite `category` — the router's original decision
    (legal_rag vs web_legal_search) is preserved so the UI can distinguish
    "planned web search" vs "fell back from RAG".
    """

    def web_search_node(state: AgentState) -> dict:
        original_query = state.get("raw_query") or state.get("query", "")
        search_query = _build_web_search_query(state)
        logger.info("Web search query: %r", search_query)
        try:
            resp = tavily_tool.search(search_query)
        except Exception as exc:
            logger.exception("Tavily search failed: %s", exc)
            return {
                "draft_answer": (
                    f"{WEB_WARNING_PREFIX}\n"
                    "Xin lỗi, không có thông tin online xác thực (dịch vụ tìm kiếm lỗi)."
                ),
                "sources": [],
                "web_results": [],
                "warning_prefix": WEB_WARNING_PREFIX,
                "refused": False,
                "error": f"tavily_error: {exc}",
            }

        results = resp.get("results", []) if isinstance(resp, dict) else []

        if not results:
            return {
                "draft_answer": (
                    f"{WEB_WARNING_PREFIX}\n"
                    "Xin lỗi, không có thông tin online xác thực về chủ đề này "
                    "trên các nguồn luật Việt Nam."
                ),
                "sources": [],
                "web_results": [],
                "warning_prefix": WEB_WARNING_PREFIX,
                "refused": False,
            }

        web_context = "\n\n".join(
            f"[{i}] {r.get('title','(no title)')}\n{r.get('url','')}\n{r.get('content','')}"
            for i, r in enumerate(results, 1)
        )

        try:
            synth = llm.invoke(
                [
                    SystemMessage(content=WEB_SEARCH_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"Web Context:\n{web_context}\n\n"
                            f"Query: {original_query}"
                        )
                    ),
                ]
            )
            draft = synth.content.strip() if hasattr(synth, "content") else str(synth)
        except Exception as exc:
            logger.exception("Web-search synth LLM call failed: %s", exc)
            draft = (
                f"{WEB_WARNING_PREFIX}\n"
                "Xin lỗi, không tổng hợp được kết quả online lúc này."
            )

        sources = [
            {"url": r.get("url", ""), "title": r.get("title", "")}
            for r in results
            if r.get("url")
        ]

        return {
            "draft_answer": draft,
            "sources": sources,
            "web_results": results,
            "warning_prefix": WEB_WARNING_PREFIX,
            "refused": False,
            "requires_approval": True,
        }

    return web_search_node


def web_finalize_node(state: AgentState) -> dict:
    """Commit the reviewed draft as the final answer.

    The graph is compiled with `interrupt_before=["web_finalize"]`, so this
    node only runs AFTER a human reviewer has resumed the thread (either
    approving the draft as-is or supplying an override via /resume).
    Moves `draft_answer` -> `answer` and clears `requires_approval`.
    """
    draft = state.get("draft_answer") or state.get("answer") or ""
    return {
        "answer": draft,
        "requires_approval": False,
    }
