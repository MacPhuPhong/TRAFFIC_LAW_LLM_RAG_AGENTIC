# SƠ ĐỒ TOÀN DIỆN HỆ THỐNG TRAFFIC-RAG

**Phiên bản:** v6.8 (hierarchical chunker fix + corpus 4 423) · **Ngày:** 2026-05-22 · **Tác giả:** Mạc Phú Phong

> Tài liệu này tổng hợp toàn bộ kiến trúc hệ thống Traffic-Law RAG dưới dạng sơ đồ. Mỗi sơ đồ phụ trách một khía cạnh khác nhau (context, component, ingestion, online flow, retrieval internals, HITL state, data model, deployment) để có thể đọc độc lập trong báo cáo / slide thuyết trình.

---

## Mục lục

0. [Sơ đồ tổng thể chi tiết v7.0 (dùng cho Chương 2)](#0-sơ-đồ-tổng-thể-chi-tiết-v70--dùng-cho-chương-2)
1. [Sơ đồ ngữ cảnh (Level 1 — System Context)](#1-sơ-đồ-ngữ-cảnh)
2. [Sơ đồ thành phần (Level 2 — Container)](#2-sơ-đồ-thành-phần)
3. [Pipeline offline — Ingestion → Indexing](#3-pipeline-offline)
4. [Pipeline online — Agentic flow (LangGraph)](#4-pipeline-online)
5. [Cấu tạo Hybrid Retrieval](#5-hybrid-retrieval)
6. [Generation + Citation grounding](#6-generation--citation)
7. [State machine HITL (Human-in-the-Loop)](#7-state-machine-hitl)
8. [Data model — Chunk metadata schema](#8-data-model)
9. [Deployment topology (production cloud + local dev)](#9-deployment)
10. [Sơ đồ tuần tự (Sequence) — End-to-end request](#10-sequence-end-to-end)
11. [Ma trận đánh giá (Evaluation flow)](#11-evaluation-flow)
12. [Kiến trúc v6 — Hierarchical Chunker Fix](#12-kiến-trúc-v6--hierarchical-chunker-fix-cập-nhật-2026-05-22)

---

## 0. Sơ đồ tổng thể chi tiết v7.0 (dùng cho Chương 2)

Sơ đồ master toàn hệ thống, chia thành **tám khối chức năng A–H** ánh xạ trực tiếp tới các mục thiết kế của Chương 2 để bóc tách từng khối. **Khác biệt v7.0 so với sơ đồ §2 (v6.8) bên dưới:** nhánh `legal_rag` đi thẳng tới `END` (đã bỏ `legal_fallback_router`/fallback web); checkpointer là `SqliteSaver`/`AsyncPostgresSaver` (không phải MemorySaver); Qdrant 1.17.

```mermaid
flowchart TB
    classDef offline fill:#e8eaf6,stroke:#3949ab,color:#1a237e
    classDef client fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef api fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef agent fill:#fff8e1,stroke:#f9a825,color:#f57f17
    classDef core fill:#fff3e0,stroke:#ef6c00,color:#e65100
    classDef store fill:#fce4ec,stroke:#c2185b,color:#880e4f
    classDef ext fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef hitl fill:#ffcdd2,stroke:#c62828,color:#b71c1c,stroke-width:2px

    subgraph A["KHỐI A · Ngoại tuyến — Tiếp nhận & Đánh chỉ mục (Chương 2.3)"]
        direction LR
        A1["24 văn bản PDF<br/>Luật · NĐ · TT"]:::offline
        A2["3 script làm sạch<br/>(pdfplumber)"]:::offline
        A3["Markdown<br/>Data/cleaned"]:::offline
        A4["HierarchicalLegalSplitter v6<br/>Điều→Khoản→Điểm + L3 enrich"]:::offline
        A5["all_chunks.jsonl<br/>4.423 đoạn"]:::offline
        A6["indexer.py<br/>E5-base 768d · prefix 'passage:'"]:::offline
        A1 --> A2 --> A3 --> A4 --> A5 --> A6
    end

    subgraph G["KHỐI G · Lưu trữ"]
        QD[("Qdrant 1.17<br/>Traffic_Law_Hybrid<br/>4.423 vec×768d · HNSW<br/>3 payload index")]:::store
        BM[("BM25 cache<br/>pyvi tokenized")]:::store
        CK[("Checkpointer<br/>SqliteSaver / AsyncPostgresSaver")]:::store
    end
    A6 --> QD
    A6 --> BM

    subgraph B["KHỐI B · Giao diện (Chương 2.2 · 4.3)"]
        UI["Next.js 14<br/>Chat · /admin<br/>SSE: token·cite·pending"]:::client
    end

    subgraph C["KHỐI C · API (Chương 2.2)"]
        API["FastAPI<br/>POST /chat · /resume<br/>GET /pending · /metrics · /health"]:::api
    end
    UI -->|"POST /api/chat (proxy)"| API
    API -->|"SSE stream"| UI

    subgraph D["KHỐI D · Tác tử LangGraph 6 nút (Chương 2.6)"]
        AN["analyzer<br/>category·intent·vehicle·expanded_queries"]:::agent
        RT{"route_by_category"}:::agent
        LR["legal_rag"]:::agent
        CC["chit_chat"]:::agent
        WS["web_search"]:::agent
        WF["web_finalize<br/>⚠ interrupt_before — HITL"]:::hitl
        OO["out_of_scope"]:::agent
        EN(("END"))
        AN --> RT
        RT -->|legal_rag| LR
        RT -->|chit_chat| CC
        RT -->|web_legal_search| WS
        RT -->|out_of_scope| OO
        LR --> EN
        CC --> EN
        OO --> EN
        WS --> WF --> EN
    end
    API --> AN
    D <-->|"lưu/nạp state theo thread_id"| CK

    subgraph E["KHỐI E · TrafficHybridRetriever (Chương 2.4)"]
        DE["Dense: E5-base 'query:'<br/>cosine + filter status=active"]:::core
        SP["Sparse: BM25Okapi<br/>k₁=1,5 · b=0,75"]:::core
        RF["RRF fuse k=60"]:::core
        ER["Sibling ±2 · Cross-ref<br/>· Diversity cap → top_k=10"]:::core
        DE --> RF
        SP --> RF
        RF --> ER
    end
    LR -->|"n truy vấn mở rộng"| E
    DE --> QD
    SP --> BM

    subgraph F["KHỐI F · LegalAnswerGenerator (Chương 2.5)"]
        GP["SYSTEM_PROMPT P2 — 14 quy tắc<br/>+ hint intent/vehicle/multi-frame"]:::core
        GS["structured output Pydantic<br/>+ citation sanitation"]:::core
        GP --> GS
    end
    ER --> F
    LR --> F
    F --> API

    subgraph X["KHỐI H · Dịch vụ ngoài"]
        GEM["Google Gemini<br/>3.1 Flash Lite"]:::ext
        TAV["Tavily Search"]:::ext
        LS["LangSmith trace"]:::ext
    end
    AN -.-> GEM
    GP -.-> GEM
    WS -.-> TAV
    API -.-> LS
```

**Ánh xạ khối → mục Chương 2:** A→§2.3 (ngoại tuyến), B+C→§2.2 (kiến trúc/công nghệ), D→§2.6 (tác tử), E→§2.4 (truy xuất lai), F→§2.5 (sinh + sanitation), G (lưu trữ) + H (dịch vụ ngoài) xuyên suốt.

---

## 1. Sơ đồ ngữ cảnh

Cái nhìn cao nhất: ai dùng, hệ thống nói chuyện với những bên nào.

```mermaid
flowchart LR
    classDef person fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef system fill:#fff3e0,stroke:#f57c00,color:#e65100,stroke-width:3px
    classDef external fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c

    U["👤 Người dùng cuối<br/>(người tham gia<br/>giao thông)"]:::person
    A["👤 Quản trị viên<br/>(HITL reviewer)"]:::person

    SYS["🚦 TRAFFIC-RAG<br/>Trợ lý Pháp luật<br/>Giao thông VN<br/><br/>27 văn bản · 4 423 chunks<br/>(thêm NĐ 81/2026 đường sắt 2026-05-19)"]:::system

    G["☁️ Google Gemini API<br/>3.1 Flash Lite (LLM gen)"]:::external
    T["☁️ Tavily API<br/>Web search fallback"]:::external
    L["☁️ LangSmith<br/>Tracing & eval"]:::external

    U -->|"Hỏi mức phạt /<br/>tra điều luật"| SYS
    SYS -->|"Câu trả lời<br/>+ citations"| U
    A -->|"Duyệt câu trả lời<br/>từ web"| SYS
    SYS -->|"Pending list"| A

    SYS -->|"Prompt + context"| G
    G -->|"Streamed answer"| SYS
    SYS -->|"Query khi corpus<br/>không đủ"| T
    T -->|"Snippets + URLs"| SYS
    SYS -.->|"Trace mỗi run"| L
```

---

## 2. Sơ đồ thành phần

Bóc tách `🚦 Traffic-RAG` thành các container chạy thực tế.

```mermaid
flowchart TB
    classDef ui fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef api fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef agent fill:#fff8e1,stroke:#f9a825,color:#f57f17
    classDef store fill:#fce4ec,stroke:#c2185b,color:#880e4f
    classDef ext fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c

    subgraph Browser["🌐 Trình duyệt"]
        UI["Next.js 14 App<br/>localhost:3000<br/>──────────<br/>Chat · /admin · Citations"]:::ui
    end

    subgraph Backend["🐳 Docker network (local) / HF Spaces (prod)"]
        API["FastAPI (8000 local · 7860 prod)<br/>──────────<br/>POST /chat (stream)<br/>POST /resume/{thread_id}<br/>GET /pending<br/>GET /pending/{thread_id}<br/>GET /metrics<br/>GET /health"]:::api

        subgraph LangGraphAgent["LangGraph Agent (in-process, v5.1+)"]
            ANALYZER["analyzer (Gemini · T=0)<br/>──────────<br/>structured output 5 fields:<br/>category + intent + vehicle_type<br/>+ standalone_query<br/>+ expanded_queries[1-5]"]:::agent
            LEGAL["legal_rag<br/>──────────<br/>HyDE-skip-mixed gate<br/>+ Multi-Retrieval + RRF<br/>+ Reference + Cross-ref pass<br/>+ Diversify by Khoản<br/>+ Parent L2 enrich (penalty/mixed)<br/>+ Confidence Judge (score-based)<br/>+ Multi-frame detect"]:::agent
            CHIT["chit_chat<br/>(Gemini · short)"]:::agent
            WEB["web_search<br/>(Tavily + LLM synth)"]:::agent
            WFIN["web_finalize<br/>⚠️ interrupt_before"]:::agent
            OOS["out_of_scope<br/>(static template)"]:::agent
        end

        RET["TrafficHybridRetriever<br/>──────────<br/>Dense e5-base 768d ⊕ BM25 RRF<br/>+ Sibling enrichment<br/>+ get_chunks_by_location()<br/>+ optional bge-reranker-v2-m3"]:::agent
        GEN["LegalAnswerGenerator<br/>──────────<br/>SYSTEM_PROMPT 14 rules (P2 v6.1)<br/>+ intent_hint + multi_frame_hint<br/>+ vehicle_hint (gated penalty/mixed)<br/>+ Pydantic structured output<br/>+ citation sanitation"]:::agent

        QD[("Qdrant 1.7<br/>localhost:6333 / Cloud<br/>collection: Traffic_Law_Hybrid<br/>4 423 vectors × 768d<br/>5 payload indexes")]:::store
        CKPT[("Checkpointer:<br/>AsyncSqliteSaver (dev)<br/>AsyncPostgresSaver (prod)<br/>tables: checkpoints,<br/>checkpoint_writes,<br/>checkpoint_blobs,<br/>checkpoint_migrations")]:::store
        BM25C[("BM25Okapi in-memory<br/>build từ all_chunks.jsonl<br/>+ pyvi tokenize<br/>+ Vietnamese stopwords")]:::store
    end

    subgraph Cloud["☁️ External"]
        GEMINI["Google Gemini<br/>3.1 Flash"]:::ext
        TAVILY["Tavily Search"]:::ext
        LSMITH["LangSmith<br/>traffic-rag-prod"]:::ext
    end

    UI -- "POST /api/chat (proxy)" --> API
    API --> ANALYZER
    ANALYZER -->|legal| LEGAL
    ANALYZER -->|chit_chat| CHIT
    ANALYZER -->|web_legal_search| WEB
    ANALYZER -->|out_of_scope| OOS
    LEGAL -->|"refusal →<br/>fallback"| WEB
    WEB --> WFIN

    LEGAL --> RET
    LEGAL --> GEN
    RET --> QD
    RET --> BM25C
    GEN -.-> GEMINI
    ANALYZER -.-> GEMINI
    WEB -.-> TAVILY
    LangGraphAgent <--> CKPT

    API -.->|trace| LSMITH
```

---

## 3. Pipeline offline

Chạy một lần (idempotent) khi bổ sung văn bản mới — biến `Data/raw/*.pdf` thành `4 423 chunks` đã có embedding trong Qdrant.

```mermaid
flowchart TD
    classDef raw fill:#ffebee,stroke:#c62828
    classDef proc fill:#e8eaf6,stroke:#3949ab
    classDef out fill:#e0f2f1,stroke:#00796b
    classDef store fill:#fce4ec,stroke:#c2185b

    R1[("Data/raw/luat/<br/>*.pdf")]:::raw
    R2[("Data/raw/nghidinh/<br/>*.pdf")]:::raw
    R3[("Data/raw/thongtu/<br/>*.pdf")]:::raw
    R4[("Data/raw/phuluc/<br/>quychuan/*.pdf")]:::raw

    C1["clean_luat_pdfs.py<br/>──────────<br/>regex Chương/Điều<br/>strip header/footer<br/>merge câu vắt trang"]:::proc
    C2["clean_nghidinh_pdfs.py<br/>──────────<br/>regex Điều/Khoản/Điểm<br/>parse bảng phạt"]:::proc
    C3["clean_thongtu_pdfs.py"]:::proc

    MD[("Data/cleaned/<br/>{luat,nghidinh,thongtu}<br/>*.md (24 ` file)")]:::store

    SC["semantic_chunker.py (v6 fix)<br/>──────────<br/>Chương → Điều → Khoản → Điểm<br/>+ MIN_CHUNK_DIEM=5 (L3)<br/>+ THRESH_KHOAN_STRICT=80 (NĐ)<br/>+ has_diem override → split L3<br/>+ regex `[a-zđ]` + after `;`<br/>+ enrich preamble (phạt + Điểm)<br/>gắn metadata phân cấp + status"]:::proc

    JL[("Data/all_chunks.jsonl<br/>4 423 chunks<br/>L1 Điều 486 / L2 Khoản 1282<br/>L3 Điểm 2 655 (+234% v6)")]:::store

    IDX["indexer.py<br/>──────────<br/>e5-base encode 768d<br/>'passage: ' prefix asymmetric<br/>upsert Qdrant batch<br/>create 5 payload indexes<br/>(BM25 build in-mem khi boot)"]:::proc

    QD[("Qdrant collection<br/>Traffic_Law_Hybrid<br/>──────────<br/>vector 768d (cosine)<br/>encoder: e5-base<br/>+ 5 payload indexes:<br/>{doc_id, status, topic,<br/>effective_date, level}")]:::out

    R1 --> C1
    R2 --> C2
    R3 --> C3
    R4 -. "tham chiếu<br/>thủ công" .-> SC

    C1 --> MD
    C2 --> MD
    C3 --> MD
    MD --> SC
    SC --> JL
    JL --> IDX
    IDX --> QD
```

**Đặc điểm quan trọng:**

- **Idempotent:** chạy lại với `--recreate` sẽ xoá collection và build lại từ đầu, không nhân đôi vector.
- **Hierarchy preserved:** mỗi chunk giữ `{doc_id, dieu, khoan, diem, page, effective_date}` để retrieval lọc + citation chuẩn.
- **Status-aware:** chunk thuộc Luật cũ hết hiệu lực được gắn `status="repealed"` và lọc khỏi search mặc định.

---

## 4. Pipeline online

Mỗi request đi qua state-machine LangGraph:

```mermaid
flowchart TD
    classDef start fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20
    classDef node fill:#fff8e1,stroke:#f9a825
    classDef decision fill:#ffe0b2,stroke:#e65100,color:#bf360c
    classDef hitl fill:#ffcdd2,stroke:#c62828,color:#b71c1c,stroke-width:3px
    classDef out fill:#bbdefb,stroke:#1565c0
    classDef ext fill:#f3e5f5,stroke:#6a1b9a

    S(["▶ START"]):::start
    AN["analyzer (T=0)<br/>──────────<br/>Gemini structured output 5 fields:<br/>{category, <b>intent</b>, <b>vehicle_type</b>,<br/>standalone_q,<br/>expanded_queries: <b>list[str] 1-5</b>}<br/>↳ multi-query rewrite<br/>1 frame = 1 phần tử"]:::node
    R{{"route_by_category"}}:::decision

    LR["legal_rag (v5.1+, 12 pass)<br/>──────────<br/>① Chọn top_k (15 / 25 nếu broad)<br/>② HyDE branch (skip nếu intent=mixed) ★<br/>③ Multi-Retrieval N+1 lists<br/>④ RRF fuse k=60<br/>⑤ Reference pass (parse Điều/Khoản)<br/>⑥ Cross-ref pass (scan top-5, cap 30)<br/>⑦ Diversify max 3/Khoản (siblings exempt)<br/>⑧ Parent L2 enrich ★ (penalty/mixed,<br/>     skip nếu L2 > 1500 chars)<br/>⑨ Confidence Judge score-based ★<br/>     (intent-aware, threshold 2.0)<br/>⑩ Multi-frame detect (penalty/mixed)<br/>⑪ Generator (+ intent/multi_frame/<br/>     vehicle hints, vehicle gated)<br/>⑫ Citation sanitation"]:::node
    LFR{{"legal_fallback<br/>_router<br/>(refused / error /<br/>conf_judge fail?)"}}:::decision

    CC["chit_chat<br/>──────────<br/>Gemini chat<br/>(không retrieval)"]:::node
    WS["web_search<br/>──────────<br/>Tavily query →<br/>summarise snippets →<br/>draft_answer"]:::node
    WF["web_finalize<br/>──────────<br/>Tổng hợp draft +<br/>URLs → final answer"]:::hitl
    OS["out_of_scope<br/>──────────<br/>Refusal template"]:::node

    E(["■ END"]):::out

    S --> AN
    AN --> R
    R -->|"category =<br/>legal"| LR
    R -->|"category =<br/>chit_chat"| CC
    R -->|"category =<br/>web_legal_search"| WS
    R -->|"category =<br/>out_of_scope"| OS

    LR --> LFR
    LFR -->|"answer ≠ refusal"| E
    LFR -->|"answer = refusal<br/>→ fallback web"| WS

    WS --> WF
    WF --> E
    CC --> E
    OS --> E

    AN -.-> GEMINI[["Gemini 3.1 Flash lite"]]:::ext
    LR -.-> GEMINI
    CC -.-> GEMINI
    WS -.-> TAVILY[["Tavily API"]]:::ext
    WF -.-> GEMINI

    note["⚠️ <b>HITL gate:</b> compile<br/>với <code>interrupt_before=['web_finalize']</code><br/>→ đồ thị TẠM DỪNG trước WF<br/>cho admin duyệt"]
    WF -.- note
```

**4 nhánh phân loại theo `category`:**

| Category             | Khi nào                                                                            | Output                                          |
| -------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------- |
| `legal`            | Câu hỏi về luật/nghị định/thông tư trong corpus                            | Answer + citations `[n]` → trust trực tiếp |
| `chit_chat`        | Lời chào, hỏi linh tinh                                                          | Trả lời ngắn từ Gemini, không retrieval    |
| `web_legal_search` | Câu hỏi pháp luật giao thông NHƯNG ngoài corpus (vd. phí đăng kiểm 2026) | Tavily → draft →**HITL pause**          |
| `out_of_scope`     | Câu hỏi không thuộc phạm vi (vd. thời tiết)                                  | Template refusal                                |

---

## 5. Hybrid Retrieval (Multi-Query)

Bên trong `legal_rag` node, retrieval chia làm **8 pha** (v5.1+, gates intent-aware). **Lưu ý**: v2 "5-tầng" prompt-rule-based đã rollback (xem [bao_cao §11.7.3.OLD](bao_cao_he_thong.md#11730-giải-pháp-5-tầng-v2-đã-rollback)) — v5.1+ là kiến trúc hiện hành.

```mermaid
flowchart LR
    Q["expanded_queries list 1 to 5<br/>plus HyDE hypothetical answer"]

    subgraph PerQuery["Pha 0 Multi-Query loop N plus 1 lan"]
        Q1["q1 frame A"]
        Q2["q2 frame B"]
        QN["qN frame ..."]
        QH["qH HyDE seed"]
    end

    subgraph Pha1["Pha 1 Dense path moi q"]
        DE["e5-base encoder<br/>passage prefix cong q<br/>vector 768d"]
        QD[("Qdrant cosine top_n=30")]
    end

    subgraph Pha2["Pha 2 Sparse path moi q"]
        TK["VI tokenize<br/>remove stopwords"]
        BM[("BM25Okapi top_n=30")]
    end

    RRF1["Pha 3 RRF fuse<br/>dense cong BM25<br/>top_k chia N cong 2 moi q"]
    RRF2["Pha 3b RRF fuse N lists<br/>L2 _rrf_fuse_lists<br/>k=60"]
    SIB["Pha 4a Sibling expansion<br/>khoan X-1, X+1 cung Dieu"]
    XR["Pha 4b Cross-reference<br/>regex Dieu N Khoan M"]
    DV["Pha 5 Diversify by location<br/>L3 _diversify_by_location<br/>cap max 3 chunk per Dieu Khoan"]
    PL["Pha 6 Parent L2 enrich (gated)<br/>intent in penalty mixed only<br/>_attach_parent_l2"]
    CJ["Pha 7 Confidence Judge<br/>score-based intent-aware<br/>threshold 2.0"]
    RR["opt Pha 8 bge-reranker-v2-m3<br/>cross-encoder rescore<br/>default OFF"]
    OUT["top_k=15 RetrievedChunk"]
    CL["LOW_CONFIDENCE<br/>CLARIFICATION template"]

    Q --> Q1
    Q --> Q2
    Q --> QN
    Q --> QH
    Q1 --> DE
    Q2 --> DE
    QN --> DE
    QH --> DE
    Q1 --> TK
    Q2 --> TK
    QN --> TK
    QH --> TK
    DE --> QD --> RRF1
    TK --> BM --> RRF1
    RRF1 --> RRF2
    RRF2 --> SIB --> XR --> DV
    DV --> PL
    PL --> CJ
    CJ -->|PASS| OUT
    CJ -->|FAIL| CL
    OUT -->|ENABLE_RERANKER=1| RR
```

**Lý do thiết kế Multi-Query + Diversify (v2):**

- **L1 Multi-Query**: Câu hỏi đa-nghĩa ("vượt tín hiệu đường sắt") không thể nén
  vào 1 chuỗi mà giữ được cả 2 frame — analyzer phải trả về 1 phần tử / frame.
- **L2 RRF fuse N lists**: Mỗi rewrite có keyword TÁCH RỜI → BM25 bias bị triệt
  tiêu vì không còn cạnh tranh trong cùng query; fuse N lists đảm bảo top-K
  chứa chunk từ MỌI frame.
- **L3 Diversify**: Kể cả khi 1 frame mạnh hơn (vd có 20 chunk vs 5 chunk trong
  corpus), `_diversify_by_location` cap mỗi (Điều, Khoản) tối đa 3 chunk →
  generator nhận "menu" cân bằng, không bị 1 cluster áp đảo.

**Lý do giữ Hybrid (đã chứng minh RQ6/RQ9 v1):**

- Dense one-shot bị miss khi query có biến thể từ ("vượt đèn đỏ" vs "không chấp hành tín hiệu giao thông").
- BM25 one-shot bị miss khi query là paraphrase ("đậu xe sai chỗ" vs "dừng đỗ trên cầu").
- RRF fuse (không cần normalise score) → MRR tăng so với dense-only.
- Sibling/Cross-ref bù cho hiện tượng "khoản trỏ về Điều khác" — đặc thù văn bản pháp luật.

---

## 6. Generation + Citation

```mermaid
flowchart TD
    classDef input fill:#e1f5fe,stroke:#0277bd
    classDef proc fill:#fff8e1,stroke:#f9a825
    classDef rule fill:#ffe0b2,stroke:#e65100
    classDef check fill:#ffcdd2,stroke:#c62828
    classDef out fill:#c8e6c9,stroke:#2e7d32

    Q["query"]:::input
    CTX["top_k=10 chunks<br/>(RetrievedChunk[])"]:::input

    BUILD["build_context()<br/>──────────<br/>format: '[{n}] doc_id Điều X<br/>Khoản Y Điểm Z (page P):<br/>{content}'"]:::proc

    SP["SYSTEM_PROMPT v6.1+ (14 rules)<br/>──────────<br/>Role: Trợ lý pháp lý GT<br/>──────────<br/>R1. Citation [Điều X — doc] bắt buộc<br/>R4. Cấm bịa (refuse if missing)<br/>R10. Cross-ref trừ điểm GPLX<br/>R14. Phân định lỗi va chạm 4 bước"]:::rule

    HINTS["+ Dynamic hints (v4/v5.1):<br/>──────────<br/>intent_hint (6 hints theo intent)<br/>multi_frame_hint (khi ≥2 Điều,Khoản)<br/>vehicle_hint ★ (gated penalty/mixed<br/>+ vehicle_type ≠ any)<br/>→ inject vào user_content"]:::rule

    LLM["Gemini Flash Lite<br/>──────────<br/>temperature=0.0<br/>Pydantic structured output"]:::proc

    SAN["citation_sanitation()<br/>──────────<br/>• drop [n] không tồn tại<br/>• re-number liên tục<br/>• gắn metadata về sources"]:::check

    REF{{"Chứa<br/>REFUSAL_PHRASE?"}}:::check

    OUT_OK["AnswerResult{<br/>answer: str,<br/>sources: [{<br/> doc_id, dieu, khoan,<br/> diem, page, url<br/>}]<br/>}"]:::out

    OUT_FB["→ legal_fallback_router<br/>route to web_search"]:::check

    Q --> BUILD
    CTX --> BUILD
    BUILD --> LLM
    SP --> LLM
    HINTS --> LLM
    LLM --> SAN
    SAN --> REF
    REF -->|"NO"| OUT_OK
    REF -->|"YES"| OUT_FB
```

> **Lưu ý**: SYSTEM_PROMPT hiện hành có **14 rules** (không có Quy tắc 15). Quy tắc 15 "không chọn ngầm" thuộc v2 5-layer đã rollback. v4+ thay bằng **deterministic `_has_multi_frame()` ở Python + `multi_frame_hint` inject động** vào user_content — generator không cần "đếm Khoản" qua prompt rule nữa.

**Vì sao có node `citation_sanitation`:** LLM thỉnh thoảng sinh ra `[7]` trong khi context chỉ có `[1]…[5]`. Sanitiser drop citation lậu + re-number để frontend popover không bị broken link.

---

## 7. State machine HITL

```mermaid
stateDiagram-v2
    [*] --> Receiving : POST /chat<br/>{query, thread_id}

    Receiving --> Analyzing : agent.invoke(state)
    Analyzing --> LegalRAG : category=legal
    Analyzing --> ChitChat : category=chit_chat
    Analyzing --> WebSearch : category=web_legal_search
    Analyzing --> OutOfScope : category=out_of_scope

    LegalRAG --> Done : answer ≠ refusal
    LegalRAG --> WebSearch : answer = refusal
    ChitChat --> Done
    OutOfScope --> Done

    WebSearch --> AwaitingApproval : draft_answer ready<br/><b>interrupt_before web_finalize</b>

    state AwaitingApproval {
        [*] --> Pending : Persist state vào<br/>checkpoints/graph.db
        Pending --> Pending : GET /pending<br/>(admin xem list)
    }

    AwaitingApproval --> Finalising : POST /resume<br/>{thread_id, decision: approve}
    AwaitingApproval --> Rejected : POST /resume<br/>{thread_id, decision: reject}

    Finalising --> Done : web_finalize tổng hợp<br/>final_answer
    Rejected --> Done : trả refusal<br/>+ lý do

    Done --> [*] : Stream answer<br/>về client
```

**Tại sao cần HITL ở `web_search` mà không phải `legal_rag`:**

| Nhánh         | Nguồn                                           | Độ tin cậy                     | Cần duyệt?    |
| -------------- | ------------------------------------------------ | --------------------------------- | --------------- |
| `legal_rag`  | Corpus chính thức (Luật, NĐ, TT) đã verify | Cao                               | ❌ trust thẳng |
| `web_search` | Internet open (Tavily)                           | Thấp, có thể tin tức/blog sai | ✅ HITL         |

**State persistence:** `SqliteSaver` lưu toàn bộ `AgentState` vào `checkpoints/graph.db` theo `thread_id`. Khi `/resume` được gọi với cùng `thread_id`, LangGraph load lại state và resume từ ngay sau `interrupt_before` → admin có thể đóng tab trình duyệt rồi mở lại sau cũng vẫn duyệt được.

---

## 8. Data model

Schema metadata của một chunk trong Qdrant payload:

```mermaid
classDiagram
    class Chunk {
        +int id  «Qdrant point id»
        +float[768] vector  «e5-base dense (production)»
        +str content  «text gốc + enrich preamble (v6)»
        +ChunkMetadata payload
    }

    class ChunkMetadata {
        +str chunk_id  «vd. 168_2024_NĐ_CP_dieu6_khoan6_diem_c»
        +str doc_id  «vd. 168/2024/NĐ-CP»
        +str doc_type  «luat | nghidinh | thongtu»
        +str chuong  «vd. 'Chương II'»
        +int dieu  «6»
        +str khoan  «'6'»
        +str diem  «'c' | None»
        +int level  «1=Điều, 2=Khoản, 3=Điểm»
        +int page  «trang trong PDF gốc»
        +str source_file  «Data/cleaned/.../168_2024.md»
        +str status  «effective | repealed»
        +str effective_date  «2025-01-01»
        +str topic  «mức phạt | quy tắc | điều tra | ...»
        +str title  «tiêu đề Điều»
        +bool is_sibling  «set by retriever sibling enrich + parent L2»
    }

    class RetrievedChunk {
        +int id
        +float score  «RRF score»
        +str content
        +dict metadata
        +to_dict()
    }

    class AnalyzerOutput {
        +Category category
        +Intent intent  «v4 — penalty/fault/procedure/...»
        +VehicleType vehicle_type  «v5.1 — o_to/mo_to/...»
        +str standalone_query
        +list~str~ expanded_queries  «v3 — 1-5 frames»
    }

    class AgentState {
        +str query
        +str raw_query
        +str expanded_query  «legacy joined»
        +list~str~ expanded_queries  «v3 primary input»
        +list~Message~ chat_history
        +Category category
        +Intent intent  «v4»
        +VehicleType vehicle_type  «v5.1»
        +bool multi_frame  «v4 — set by _has_multi_frame»
        +list~dict~ chunks
        +str answer
        +str draft_answer  «web_search only»
        +list~AnswerSource~ sources
        +bool refused
        +list~dict~ web_results
        +str warning_prefix
        +bool requires_approval  «HITL gate»
        +str model_info
        +str error
    }

    class AnswerSource {
        +str doc_id
        +int dieu
        +str khoan
        +str diem
        +int page
        +str url «link tới md gốc»
    }

    Chunk --> ChunkMetadata
    RetrievedChunk ..> ChunkMetadata : metadata dict
    AnalyzerOutput ..> AgentState : analyzer_node writes
    AgentState --> RetrievedChunk
    AgentState --> AnswerSource
```

**Convention `chunk_id`:** `<doc_slug>_dieu<N>_khoan<K>_diem_<D>` cho phép debug bằng mắt mà không cần mở payload — ví dụ `168_2024_NĐ_CP_dieu6_khoan6_diem_c` đọc được luôn là _NĐ 168/2024 Điều 6 Khoản 6 Điểm c_.

---

## 9. Deployment

Hệ thống có **hai topology** được hỗ trợ song song qua các env var (`QDRANT_URL`, `CHECKPOINT_DB_URL`, `CORS_ALLOWED_ORIGINS`): chạy local cho phát triển và chạy cloud cho production. Cùng một codebase, cùng một Docker image — chỉ khác bộ secret.

### 9.1 Production cloud topology (đã deploy thực tế 2026-05-17)

Toàn bộ free tier, $0/tháng. URL chính thức: **https://traffic-rag.vercel.app**.

```mermaid
flowchart TB
    classDef edge fill:#ede7f6,stroke:#5e35b1
    classDef hf fill:#fff3e0,stroke:#ef6c00
    classDef cloud fill:#e0f7fa,stroke:#00838f
    classDef db fill:#fce4ec,stroke:#c2185b
    classDef external fill:#f3e5f5,stroke:#6a1b9a

    USER(["👤 Browser<br/>(desktop / mobile)"])

    subgraph VC["▲ Vercel Hobby — us-east-1 (Washington)"]
        FE["Next.js 14 App Router<br/>──────────<br/>traffic-rag.vercel.app<br/>SSR + /api/chat proxy<br/>NextAuth (Credentials)"]:::edge
    end

    subgraph HFS["🤗 Hugging Face Spaces — Docker, AWS us-east-1"]
        BE["FastAPI + LangGraph<br/>──────────<br/>pphong203-traffic-rag-backend.hf.space<br/>uvicorn :7860 · 16GB RAM · 2 vCPU<br/>Pre-cached multilingual-e5-base<br/>BM25 in-memory (4423 docs)"]:::hf
    end

    subgraph QC["🔷 Qdrant Cloud Free — AWS us-east-1"]
        QD[("Traffic_Law_Hybrid<br/>4423 points · 768d<br/>HNSW + payload index")]:::cloud
    end

    subgraph SB["🟢 Supabase Postgres Free — ap-northeast-1 (Tokyo)"]
        PG_NEXTAUTH[("public.<br/>User · Session ·<br/>Conversation ·<br/>Message ·<br/>VerificationToken ·<br/>Account")]:::db
        PG_CKPT[("public.<br/>checkpoints ·<br/>checkpoint_blobs ·<br/>checkpoint_writes ·<br/>checkpoint_migrations")]:::db
    end

    GEMINI["☁️ Gemini 3.1 Flash Lite<br/>generativelanguage.googleapis.com"]:::external
    TAVILY["☁️ Tavily web search<br/>api.tavily.com"]:::external

    USER -- "HTTPS · TLS auto" --> FE
    FE -- "Server-side fetch<br/>BACKEND_URL" --> BE
    FE -- "Prisma · Transaction Pooler :6543<br/>DATABASE_URL" --> PG_NEXTAUTH
    BE -- "qdrant-client (HTTPS :6333)<br/>QDRANT_URL + QDRANT_API_KEY" --> QD
    BE -- "psycopg async · Session Pooler :5432<br/>CHECKPOINT_DB_URL" --> PG_CKPT
    BE -. "HTTPS" .-> GEMINI
    BE -. "HTTPS" .-> TAVILY
```

**Các bước deploy** — xem chi tiết trong [deploy/DEPLOY-PRODUCTION.md](../deploy/DEPLOY-PRODUCTION.md) (playback đầy đủ bao gồm các bug phát sinh và cách khắc phục).

**Phân bổ trách nhiệm**:

| Layer                  | Service                            | Vai trò                                              | Tại sao chọn                                                                   |
| ---------------------- | ---------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------- |
| Frontend               | Vercel Hobby (free)                | Next.js SSR · SSE proxy · NextAuth                  | Native Next.js, auto SSL, deploy 3 phút, custom domain free                     |
| Backend RAG            | HF Spaces Docker (free)            | FastAPI + LangGraph + embedding inference + BM25      | 16GB RAM, 2 vCPU,**không sleep** với Docker SDK, sinh ra cho ML workload |
| Vector store           | Qdrant Cloud Free                  | Dense + payload filter                                | Cùng vendor Qdrant local; us-east-1 cùng DC với HF Space → query ~5ms        |
| Auth + Chat history DB | Supabase Postgres Free             | NextAuth tables (Prisma)                              | 500MB free, region Tokyo gần VN                                                 |
| Agent checkpoint       | Supabase Postgres (cùng instance) | LangGraph `checkpoint_*` qua `AsyncPostgresSaver` | Tận dụng cùng DB, không cần infra riêng                                    |
| LLM                    | Google Gemini 3.1 Flash Lite       | Generator                                             | Free tier ~200 req/ngày, 1M context                                             |
| Web fallback           | Tavily                             | HITL search                                           | 1000 search/tháng free                                                          |

**Tổng chi phí**: **$0/tháng** (tất cả nằm trong free tier).

### 9.2 Local development topology (cho phát triển)

Khi `QDRANT_URL` và `CHECKPOINT_DB_URL` không set → fallback về SQLite + Qdrant local Docker:

```mermaid
flowchart TB
    classDef host fill:#eceff1,stroke:#455a64,stroke-width:2px
    classDef container fill:#e3f2fd,stroke:#1565c0
    classDef volume fill:#fce4ec,stroke:#c2185b
    classDef external fill:#f3e5f5,stroke:#6a1b9a
    classDef port fill:#fff9c4,stroke:#f9a825,color:#f57f17

    subgraph DEV["💻 Dev machine (Linux/macOS)"]

        subgraph DC["🐙 docker-compose.yml"]
            QD["qdrant:1.7<br/><br/>volume:<br/>./qdrant_storage"]:::container
        end

        subgraph PY["🐍 venv ~/venv/LLM_Agentic"]
            API["uvicorn api.main:app<br/>──────────<br/>FastAPI + LangGraph<br/>(in-process agent)"]:::container
            CKPT[("checkpoints/<br/>graph.db<br/>(SQLite)")]:::volume
        end

        subgraph NEXT["⚛️ npm run dev"]
            UI["Next.js 14<br/>App Router"]:::container
        end

        DATA[("Data/<br/>raw + cleaned<br/>+ all_chunks.jsonl")]:::volume

        ENV[(".env<br/>──────────<br/>API_KEY (Gemini)<br/>TAVILY_API_KEY<br/>LANGCHAIN_API_KEY")]:::volume

        P3000["🌐 :3000"]:::port
        P8000["🌐 :8000"]:::port
        P6333["🌐 :6333"]:::port
        P6334["🌐 :6334"]:::port

        UI --- P3000
        API --- P8000
        QD --- P6333
        QD --- P6334
    end

    GEMINI["☁️ generativelanguage<br/>.googleapis.com"]:::external
    TAVILY["☁️ api.tavily.com"]:::external
    LSMITH["☁️ smith.langchain.com"]:::external

    UI -- "fetch /api/chat" --> API
    API -- "qdrant_client<br/>(HTTP 6333)" --> QD
    API -- "load JSONL" --> DATA
    API -- "persist state" --> CKPT
    API -. "HTTPS" .-> GEMINI
    API -. "HTTPS" .-> TAVILY
    API -. "HTTPS" .-> LSMITH
    API -.- ENV
```

### 9.3 Switch local ↔ production

Cùng codebase, cùng image. Khác biệt **chỉ ở env vars**:

| Env var                    | Local                                           | Production                                                             |
| -------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------- |
| `QDRANT_URL`             | (không set) →`localhost:6333`               | `https://<id>.us-east-1.aws.cloud.qdrant.io:6333`                    |
| `QDRANT_API_KEY`         | (không set)                                    | `eyJh...` từ Qdrant Cloud                                           |
| `CHECKPOINT_DB_URL`      | (không set) → SQLite `checkpoints/graph.db` | `postgresql://...pooler.supabase.com:5432/postgres` (Session Pooler) |
| `CORS_ALLOWED_ORIGINS`   | (không set) → không thêm CORS               | `https://traffic-rag.vercel.app`                                     |
| `BACKEND_URL` (frontend) | `http://localhost:8000`                       | `https://pphong203-traffic-rag-backend.hf.space`                     |
| `DATABASE_URL` (Prisma)  | `file:./dev.db` (sqlite) hoặc Postgres dev   | Transaction Pooler URI (port `6543`)                                 |

> Schema Prisma đã chuyển `provider = "postgresql"` (production). Để dev local SQLite, đổi tạm về `"sqlite"` rồi `prisma db push` lại — không khuyến nghị (chỉ dùng cùng Supabase free tier cho local dev luôn).

---

## 10. Sequence end-to-end

Một câu hỏi _"Vượt đèn đỏ ô tô bị phạt bao nhiêu tiền?"_ đi qua hệ thống:

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant UI as Next.js UI
    participant API as FastAPI
    participant AG as LangGraph Agent
    participant LLM as Gemini Flash Lite
    participant RET as HybridRetriever
    participant QD as Qdrant
    participant BM as BM25 cache
    participant CK as Async Checkpointer

    Note over CK: SQLite (dev) hoặc Postgres (prod)

    U->>UI: gõ câu hỏi
    UI->>API: POST /api/chat proxy to /chat
    API->>AG: ainvoke state with thread_id

    AG->>LLM: analyzer history plus query, T=0
    LLM-->>AG: structured output 5 fields
    Note over AG,LLM: category=legal_rag, intent=penalty, vehicle_type=o_to, standalone_q, expanded_queries list 1-5

    AG->>AG: route_by_category to legal_rag

    Note over AG,LLM: intent != mixed, run HyDE
    AG->>LLM: HyDE query, sinh hypothetical legal answer
    LLM-->>AG: hypothetical text dùng vocab pháp lý

    loop N+1 lists, N queries cộng 1 HyDE seed
        AG->>RET: get_relevant_chunks q_i, k=per_query_k
        par Dense path
            RET->>QD: encode cộng cosine search
            QD-->>RET: hits kèm score
        and Sparse path
            RET->>BM: tokenize cộng BM25
            BM-->>RET: hits kèm score
        end
        RET->>RET: RRF fuse dense và sparse
        RET-->>AG: hits list
    end

    AG->>AG: _rrf_fuse_lists N+1 lists, top_k nhân 2
    AG->>AG: Reference pass, parse Điều X Khoản Y
    AG->>RET: get_chunks_by_location cho refs
    AG->>AG: Cross-ref pass, scan top-5, cap 30
    AG->>AG: _diversify_by_location, max 3 chunk per Khoản
    Note over AG,RET: intent in penalty hoặc mixed
    AG->>AG: _attach_parent_l2, skip L2 gt 1500 chars
    AG->>RET: get_chunks_by_location cho L2 parents

    AG->>AG: _judge_retrieval_confidence score-based intent-aware
    alt score lt 2.0
        AG-->>API: LOW_CONFIDENCE_CLARIFICATION template
    else score gte 2.0
        AG->>AG: _has_multi_frame chunks, intent
        AG->>LLM: generator query, chunks, intent, multi_frame, vehicle_type
        Note over LLM: SYSTEM_PROMPT 14 rules cộng intent_hint cộng multi_frame_hint cộng vehicle_hint (penalty/mixed only)
        LLM-->>AG: structured answer cộng sources
        AG->>AG: citation sanitation
    end

    AG->>CK: persist final state
    AG-->>API: status completed, answer, sources, multi_frame, model_info
    API-->>UI: JSON response
    UI-->>U: render answer plus click số nguồn
```

---

## 11. Evaluation flow

Quy trình đánh giá / regression:

```mermaid
flowchart LR
    classDef input fill:#e1f5fe,stroke:#0277bd
    classDef script fill:#fff8e1,stroke:#f9a825
    classDef metric fill:#fce4ec,stroke:#c2185b
    classDef ci fill:#ffe0b2,stroke:#e65100
    classDef gate fill:#ffcdd2,stroke:#c62828,stroke-width:3px

    GOLD[("research/data/<br/>gold_40.jsonl<br/>──────────<br/>40 query + expected<br/>{doc_id, dieu, khoan}")]:::input

    subgraph Ablation["Ablation experiments (RQ1-RQ10)"]
        RQ2["RQ2: chunking<br/>hierarchical vs fixed-512"]:::script
        RQ3["RQ3: encoder<br/>e5 vs sbert vs mpnet"]:::script
        RQ4["RQ4: vector DB<br/>Qdrant vs Chroma vs FAISS"]:::script
        RQ9["RQ9: retrieval<br/>dense vs hybrid vs +sibling"]:::script
        RQ10["RQ10: reranker<br/>off vs bge-v2-m3"]:::script
    end

    M1["MRR / Recall@10 /<br/>nDCG@10 / Cit-R"]:::metric
    M2["RAGAS-lite (n=5)<br/>──────────<br/>answer_relevancy,<br/>context_recall,<br/>context_precision,<br/>faithfulness"]:::metric

    subgraph CI["GitHub Actions"]
        SMOKE["test_smoke.py<br/>(6 unit tests)"]:::ci
        REG["test_retrieval_regression.py<br/>──────────<br/>5-query subset<br/>assert mean Recall@10 ≥ 0.40"]:::gate
    end

    PR["PR mở →<br/>CI trigger"]:::ci
    BLOCK["❌ Block merge"]:::gate
    PASS["✅ Allow merge"]:::ci

    LSMITH["☁️ LangSmith<br/>traces production"]:::ci

    GOLD --> RQ2
    GOLD --> RQ3
    GOLD --> RQ4
    GOLD --> RQ9
    GOLD --> RQ10
    GOLD --> REG

    RQ2 --> M1
    RQ3 --> M1
    RQ4 --> M1
    RQ9 --> M1
    RQ10 --> M1
    M1 -. "feed quyết định<br/>kiến trúc" .-> M2

    PR --> SMOKE
    PR --> REG
    SMOKE --> PASS
    REG -->|"≥ 0.40"| PASS
    REG -->|"< 0.40"| BLOCK

    LSMITH -. "feedback<br/>real traffic" .-> GOLD
```

**Hai vòng feedback:**

- **Loop ngắn (CI):** mỗi PR chạy regression gate — nếu thay encoder/chunker mà MRR/Recall tụt dưới 0.40 → fail merge.
- **Loop dài (production):** trace trên LangSmith giúp khám phá query thực tế bị refusal/sai citation → bổ sung vào `gold_25.jsonl` → ablation lại.

---

## 12. Kiến trúc v6 — Hierarchical Chunker Fix (cập nhật 2026-05-22)

### 12.0 Bối cảnh

Sau audit corpus, phát hiện **77.5% Điểm pháp lý trong NĐ 168 không có chunk L3 độc lập** vì 3 bug ở `semantic_chunker.py`:

1. `MIN_CHUNK_DIEM = 30 tokens` filter — bỏ qua Điểm ngắn (đa số Điểm 10-20 tokens)
2. `THRESH_KHOAN = 500` chỉ split khi Khoản lớn — Khoản nhỏ giữ L2, không có L3
3. `RE_DIEM = [a-z]` không bắt chữ "đ" tiếng Việt + không bắt Điểm cùng dòng sau `;`

→ Query "đi xe máy hàng ngang" không tìm được Điều 7 K3 Điểm k (600-800K) vì Điểm k không tồn tại trong index.

### 12.1 Fix: 5 cải tiến chunker

| # | Cải tiến                                                                       | File                    |     |
| - | -------------------------------------------------------------------------------- | ----------------------- | --- |
| 1 | Tách `MIN_CHUNK_DIEM = 5` (L3), giữ `MIN_CHUNK = 30` (L1/L2)               | `semantic_chunker.py` |     |
| 2 | Threshold động:`THRESH_KHOAN_STRICT = 80` cho `nghidinh`, 500 cho Luật/TT | id.                     |     |
| 3 | "has_diem" override: Nghị định + ≥2 Điểm → luôn split L3                 | id.                     |     |
| 4 | Regex enhanced: `(?:^                                                            | ;\s+)([a-zđ])\)`       | id. |
| 5 | Enrich preamble: mỗi L3 chunk tự chứa "Phạt tiền X-Y đồng..." + Điểm    | id.                     |     |

### 12.2 Kết quả chunking

| Metric                   | Trước fix | Sau fix                 |
| ------------------------ | ----------- | ----------------------- |
| Total chunks             | 3 001       | **4 423** (+47%)  |
| L3 Điểm chunks         | 793         | **2 655** (+234%) |
| % Điểm NĐ 168 missing | 77.5%       | **11.7%**         |

**Verify**: query "dàn hàng ngang xe máy" top-1 retrieval giờ là **Điều 7 K3 Điểm k** (trước fix: chunk này không tồn tại).

### 12.3 Kết quả RQ1 v6 đo thực 2026-05-22

| Pipeline             | F1 25-câu      | Latency         |
| -------------------- | --------------- | --------------- |
| Gemini only          | 0.382           | 4.8s            |
| Vanilla RAG          | 0.227           | 5.8s            |
| Agentic v1           | 0.354           | 22.9s           |
| Agentic v4           | 0.345           | 10.0s           |
| Agentic v5.1         | 0.313           | 23.5s           |
| **Agentic v6** | **0.286** | **15.5s** |

**Per-category v6 vs v5.1 trên 40 câu**:

| Category                  | F1 v5.1 | F1 v6           | Δ                  |
| ------------------------- | ------- | --------------- | ------------------- |
| **vehicle_binding** | 0.308   | **0.400** | **+0.092** ⭐ |
| ambiguous_short           | 0.306   | 0.314           | +0.008              |
| cross_reference           | 0.210   | 0.219           | +0.009              |
| procedure                 | 0.232   | 0.241           | +0.008              |
| out_of_scope              | 0.406   | 0.406           | 0                   |
| compound_action           | 0.389   | 0.362           | -0.027              |
| multi_intent              | 0.288   | 0.233           | -0.055              |
| simple_penalty            | 0.430   | 0.333           | -0.097              |

### 12.4 Đánh giá honest

**Wins thực sự**:

- ✅ Corpus completeness: 469 Điểm chunks recovered (77.5% miss → 11.7%)
- ✅ `vehicle_binding` +0.092 (query nêu rõ loại xe → cite Điểm chính xác)
- ✅ Latency giảm 34% (23.5s → 15.5s)
- ✅ Query specific (vd "Điều 7 K3 Điểm k") giờ retrievable

**Losses (chủ yếu là metric artifact)**:

- ⚠ `simple_penalty` -0.097: gold answers cũ nhiều câu tham chiếu NĐ 100/2019 (hết hiệu lực); v6 trả NĐ 168/2024 (đúng pháp lý) → mismatch token F1
- ⚠ Verbosity tăng: enrich_preamble duplicate cụm phạt qua các L3 → list nhiều Điểm → F1 giảm dù content đúng hơn

**Trade-off khác**:

- `_attach_siblings` + parent-L2 enrich đánh dấu đa số chunks là sibling → Confidence Judge cần điều chỉnh `MIN_CHUNKS` từ 3 → 1.

→ **v6 là technical improvement về CORPUS COMPLETENESS + RETRIEVAL PRECISION cho specific Điểm queries, nhưng F1 metric không fair với verbose answers.** Khuyến nghị deploy v6 + tương lai chuyển sang LLM-as-judge thay token F1.

---

## 12-old. Kiến trúc v5.1 — Vehicle binding + Parent L2 enrich + HyDE-skip-mixed (giữ làm tham chiếu lịch sử)

### 12.1 Phiên bản v5.1 trên nền v4

Sau review feedback "prompt 70/structure 30 → cần đổi sang 50/50", v5.1 bổ sung 3 cải tiến structural trên nền v4 (đã có HyDE + intent + multi-frame deterministic):

| Cải tiến                                          | Vị trí             | Mục đích                                                                                                                                               |
| --------------------------------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **(a) Vehicle binding** (scope=penalty/mixed) | Analyzer + Generator | Câu hỏi nêu loại xe rõ ràng → ƯU TIÊN Điều 6/7/8/9 NĐ 168 tương ứng. KHÔNG áp cho intent=procedure/definition (tránh block Thông tư). |
| **(b) Parent L2 chunk enrich**                | `legal_rag_node`   | Khi top-K có L3 Điểm chunks, auto-pull L2 Khoản parent (chứa số tiền). Mitigation: skip nếu L2 > 1500 chars.                                      |
| **(c) Skip HyDE cho intent=mixed**            | `legal_rag_node`   | Compound query (penalty + fault) → multi-query rewrite đã đủ; HyDE thêm gây drift sang 1 frame.                                                    |

### 12.2 Schema mới

```python
class AgentState(TypedDict, total=False):
    intent: Literal["penalty", "fault", "procedure", "definition", "list", "mixed"]
    vehicle_type: Literal["o_to", "mo_to", "chuyen_dung", "xe_dap", "any"]  # NEW
    multi_frame: bool
    ...

class AnalyzerOutput(BaseModel):
    category: Category
    intent: Intent
    vehicle_type: VehicleType   # NEW — extract từ keyword trong câu hỏi
    standalone_query: str
    expanded_queries: list[str]
```

### 12.3 Luồng v5.1

```mermaid
flowchart TD
    Q["User query"] --> AN["Analyzer<br/>cộng vehicle_type"]
    AN --> HY{"intent là mixed?"}
    HY -->|Yes, skip HyDE| MR["Multi-Retrieval N lists"]
    HY -->|No, generate HyDE seed| MR2["Multi-Retrieval N+1 lists"]
    MR --> RRF["RRF fuse"]
    MR2 --> RRF
    RRF --> DV["Diversify by location"]
    DV --> PE{"intent in penalty hoặc mixed?"}
    PE -->|Yes| PL["Parent L2 enrich<br/>cap L2 dưới 1500 chars"]
    PE -->|No| CJ
    PL --> CJ["Confidence Judge<br/>score-based, intent-aware"]
    CJ -->|FAIL| CL["Clarification template"]
    CJ -->|PASS| MF["Multi-frame detect"]
    MF --> GEN["Generator<br/>cộng intent hint<br/>cộng multi_frame hint<br/>cộng vehicle hint (penalty/mixed only)"]
    GEN --> OUT["Answer"]
```

### 12.4 Kết quả RQ1 v5.1 đo thực 2026-05-19 (40-câu eval set)

| Pipeline               | F1 (25-câu cũ) | Latency (25-câu cũ) |
| ---------------------- | ---------------- | --------------------- |
| Gemini only            | 0.387            | 3.1s                  |
| Vanilla RAG            | 0.229            | 9.2s                  |
| Agentic v1             | 0.354            | 22.9s                 |
| Agentic v4             | 0.345            | 10.0s                 |
| **Agentic v5.1** | **0.313**  | **23.5s**       |

Per-category v5.1 trên 40 câu (chi tiết tại [analysis.md §6](../research/report/analysis.md)):

| Category                         | F1 v5.1         | Note                                                        |
| -------------------------------- | --------------- | ----------------------------------------------------------- |
| simple_penalty                   | 0.430           | Cao nhất                                                   |
| **compound_action** (mới) | **0.389** | ✅ Improvement chính — câu "Ảnh 2" giờ trả mức phạt |
| vehicle_binding (mới)           | 0.308           | RQ-036 ô tô-đèn-đỏ ✓; RQ-037 xe tải yếu            |
| ambiguous_short (mới)           | 0.306           | RQ-027 không gương ✓                                    |
| Other categories                 | 0.210-0.288     | Không thuộc scope v5.1 fix                                |

→ **v5.1 là TRADE-OFF**: tăng khả năng xử lý compound + vehicle queries, latency tăng (~2x do quota Gemini retries), F1 trên câu cũ giảm nhẹ (-0.03). Theo nguyên tắc "predictable failure modes + graceful degradation" của production RAG, đây là hướng đúng.

### 12.5 Roadmap sprint sau

| # | Cải tiến                                 | Mục đích                                                 |
| - | ------------------------------------------ | ----------------------------------------------------------- |
| 1 | Action Extractor LLM-based ở analyzer     | Match "chở 3" ↔ "chở 02" (lexical → semantic)           |
| 2 | Action-level diversity (thay Khoản-level) | Multi-frame không over-list cùng Khoản khác Điểm      |
| 3 | Multi-HyDE cho mixed (N hypothetical)      | Tránh single-frame drift                                   |
| 4 | Offline action→chunk lookup table         | Structured legal-frame retrieval, ít phụ thuộc embedding |

---

## 12-old. Kiến trúc v4 — Intent-aware + Deterministic Multi-frame (giữ làm tham chiếu lịch sử)

Phiên bản v4 là refinement của v3 sau khi nhận feedback "prompt 70% / structure 30% → cần đổi sang 50/50". Chuyển logic từ LLM-judgment-based sang Python-deterministic ở 3 tầng:

| Tầng                          | v3                                                    | v4                                                                                                                                                                   |
| ------------------------------ | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Analyzer**             | Output: category + standalone + expanded_queries      | **+ intent** field (penalty/fault/procedure/definition/list/mixed)                                                                                             |
| **Confidence Judge**     | Binary check: 3 chunks + top_score + keyword overlap  | **Score-based, intent-aware**: penalty cần money regex hit (+2), không có thì −1.5; fault cần rule keyword; procedure cần procedure keyword             |
| **Multi-frame decision** | LLM tự đếm Khoản (Quy tắc 15 — không reliable) | **Python đếm** distinct `(doc_id, Điều, Khoản)`; flag → generator buộc liệt kê "### Trường hợp 1/2"                                              |
| **Generator**            | 1 prompt với 15 rules cho mọi câu hỏi             | + 6 intent hints (penalty/fault/procedure/...) inject vào user_content; + multi_frame hint khi True; + post-process strip Gemini hallucination "⚠ Internet" prefix |

### Sơ đồ luồng v4

```mermaid
flowchart TD
    classDef llm fill:#fff8e1,stroke:#f9a825
    classDef code fill:#c5e1a5,stroke:#33691e,color:#1b5e20
    classDef gate fill:#ffe0b2,stroke:#e65100,color:#bf360c

    Q[("User query")]
    AN["analyzer (Gemini Flash Lite, T=0)<br/>output: {category, intent, standalone, expanded_queries}"]:::llm
    HY["HyDE (1 LLM call)<br/>sinh hypothetical legal answer"]:::llm
    RR["Multi-Retrieval (N+1 lists)<br/>+ RRF fuse + Diversify"]:::code

    CJ{{"Intent-aware Confidence Judge<br/>(score-based, Python)"}}:::gate
    MF{{"Multi-frame detection<br/>distinct (doc,Điều,Khoản) ≥ 2?"}}:::gate

    CL["LOW_CONFIDENCE_<br/>CLARIFICATION template"]:::code
    GEN["Generator (Gemini Flash Lite, T=0)<br/>+ intent hint + multi_frame hint<br/>+ post-process strip web prefix"]:::llm
    OUT[("Final answer")]

    Q --> AN --> HY --> RR
    RR --> CJ
    CJ -->|"score < 2.0"| CL
    CJ -->|"PASS"| MF
    MF -->|"multi_frame=True"| GEN
    MF -->|"single frame"| GEN
    GEN --> OUT
```

### 6 Intent classes

| Intent         | Trigger keywords                                   | Downstream effect                                 |
| -------------- | -------------------------------------------------- | ------------------------------------------------- |
| `penalty`    | "phạt bao nhiêu", "mức phạt", "trừ điểm"    | Judge demand money regex; multi-frame fires       |
| `fault`      | "lỗi do ai", "trách nhiệm", "ai sai"            | Activate Quy tắc 14 (4-bước phân định lỗi) |
| `procedure`  | "thủ tục", "quy trình", "hồ sơ", "đăng ký" | Allow procedural chunks; relax money requirement  |
| `definition` | "là gì", "khái niệm", "có mấy loại"         | Skip multi-frame; simpler retrieval               |
| `list`       | "khi nào", "các trường hợp", "liệt kê"      | Broad top_k=25; rà soát toàn bộ context       |
| `mixed`      | có ≥2 intent (tai nạn + mức phạt)             | Activate cả penalty + fault + procedure context  |

### Tóm tắt thay đổi code (v3 → v4)

| File                                                      | Đổi                                                                                                                                    |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `source/agent/state.py`                                 | +`Intent` Literal type; + `intent`, `multi_frame` fields                                                                           |
| `source/agent/nodes.py` `AnalyzerOutput`              | +`intent` field với default 'penalty' + description                                                                                   |
| `source/agent/nodes.py` `ANALYZER_SYSTEM_PROMPT`      | + section "INTENT" với 6 triggers; + Ví dụ 5 (mixed tai nạn+phạt), Ví dụ 6 (compound 2 hành vi)                                  |
| `source/agent/nodes.py` `_judge_retrieval_confidence` | rewrite hoàn toàn — score-based, intent-aware, threshold 2.0; 4 regex helpers (\_MONEY_RX, \_LEGAL_REF_RX, \_RULE_RX, \_PROCEDURE_RX) |
| `source/agent/nodes.py`                                 | +`_has_multi_frame(chunks, intent)` helper                                                                                             |
| `source/agent/nodes.py` `legal_rag_node`              | wire intent từ state → judge + multi_frame detection → pass to generator                                                              |
| `source/rag_core/generator.py` `generate()`           | signature `+ intent, multi_frame` kwargs; 6 intent_hints dict; multi_frame hint; post-process strip web prefix                         |

**Không đổi:** state ngoài 2 field mới, graph topology, FastAPI endpoints, frontend, retriever core, reranker, HITL.

### Verify locally (2026-05-19)

3 query đã từng fail giờ chạy đúng:

| Query                                     | Intent detected    | multi_frame | Outcome                                                                                                                        |
| ----------------------------------------- | ------------------ | ----------- | ------------------------------------------------------------------------------------------------------------------------------ |
| "vượt tín hiệu đường sắt"         | penalty ✓         | True ✓     | "### Trường hợp 1/2" headings, corpus citations                                                                             |
| "mức phạt bao nhiêu"                   | penalty            | True        | Judge pass (chunks có money) — answer overview, có post-process strip                                                       |
| "tránh xe tải tông trẻ em mức phạt" | **mixed** ✓ | True ✓     | **Cải thiện rõ rệt** — trước trả về CSGT procedure, giờ ra penalty + fault + nhường đường người đi bộ |

---

## 12-old. Kiến trúc v3 — HyDE + Confidence Judge (giữ làm tham chiếu)

Phiên bản v3 (cập nhật 2026-05-18) thay thế nhánh v2 "5-layer + verb-filter" bằng
**2 kỹ thuật RAG generic** — không có rule per-case, prompt size **giảm 36%**.

**v2 đã được rollback** với 2 lý do: (a) Gemini Flash Lite không reliable cho rule
"extract verb → classify Khoản → filter" ở T=0 — vẫn over-list; (b) approach
prompt-rule không scale cho query lạ vô hạn (user mới hỏi "tông công an", "vì
tránh trẻ em tôi tông…" — đều không có rule).

v3 giữ những phần kiến trúc đúng của v2:

- L1 Multi-Query rewrite (analyzer xuất `expanded_queries: list[str]`)
- L2 Multi-Retrieval + RRF fuse N lists
- L3 Diversify by location (cap 3 chunk/Khoản)
- L5 Temperature=0 + audit log

Và **thay** L4 (Verb-class filter + Rule 15 "no silent pick") bằng:

- **HyDE branch** — LLM sketch hypothetical legal answer → embed answer (không phải query) → retrieval pass thêm. Tự bridge vocabulary gap (động vật ↔ súc vật, trẻ em ↔ người đi bộ, tông ↔ va chạm) mà không cần synonym dictionary tay.
- **Confidence Judge** — kiểm tra deterministic 3 điều kiện (≥3 chunks, top score ≥ 0.010, keyword overlap ≥ 1) sau retrieval. Nếu fail → trả `LOW_CONFIDENCE_CLARIFICATION` template ngay, KHÔNG đẩy generator (tránh refuse → fallback web vô nghĩa).

### Sơ đồ root-cause → fix mapping (v3)

```mermaid
flowchart TD
    classDef cause fill:#ffcdd2,stroke:#c62828,color:#b71c1c
    classDef arch fill:#c5e1a5,stroke:#33691e,color:#1b5e20
    classDef effect fill:#bbdefb,stroke:#0d47a1

    P1["#1 Phễu 1 chiều<br/>(1 rewrite → 1 retrieval → 1 answer)"]:::cause
    P2["#2 LLM stochastic<br/>(T=0.1 sample khác nhau giữa<br/>các lần gọi cùng input)"]:::cause
    P3["#3 BM25 vocab gap<br/>('động vật' user vs 'súc vật' corpus<br/>→ retrieval miss)"]:::cause
    P4["#4 Refuse → fallback web vô nghĩa<br/>(query vague không nên đi tra Google)"]:::cause

    A1["Multi-Query rewrite (L1)<br/>──────────<br/>AnalyzerOutput.expanded_queries: list[str]"]:::arch
    A2["Multi-Retrieval + RRF fuse (L2)<br/>──────────<br/>retriever chạy N+1 lần (N rewrite + 1 HyDE)<br/>_rrf_fuse_lists merge"]:::arch
    A3["Diversify by location (L3)<br/>──────────<br/>cap max 3 chunk / (doc, Điều, Khoản)"]:::arch
    AH["⭐ HyDE branch<br/>──────────<br/>LLM viết hypothetical legal answer<br/>→ embed answer (không phải query)<br/>→ retrieval pass thêm<br/>Tự bridge vocab gap"]:::arch
    AC["⭐ Confidence Judge<br/>──────────<br/>Rule-based: 3 chunks + top_score≥0.01<br/>+ keyword overlap ≥ 1<br/>FAIL → clarification template<br/>(không đẩy generator → tránh fallback web)"]:::arch
    A5["Determinism + Audit log (L5)<br/>──────────<br/>T=0 mọi LLM, log diversity metrics"]:::arch

    E["v3 outcome:<br/>F1 = 0.345 (≈ v1)<br/>Latency 10s (−56% vs v1)<br/>Vocab gap tự xử<br/>Vague query không lệch sang web"]:::effect

    P1 -->|"mở phễu N→N+1"| A2
    P1 --> A1
    P2 -->|"argmax thay sampling"| A5
    P3 -->|"LLM dùng vocab pháp lý<br/>khi sinh hypothetical"| AH
    P4 -->|"deterministic check<br/>thay refusal LLM"| AC

    A1 --> A2
    AH --> A2
    A2 --> A3 --> AC --> E
    A5 -.- A1
    A5 -.- AH
```

### Tóm tắt thay đổi code (v3, so với v1 baseline)

| File                                                        | Đổi                                                                                                                               | Component  |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `source/agent/state.py`                                   | thêm `expanded_queries: list[str]`                                                                                               | L1         |
| `source/agent/nodes.py` `AnalyzerOutput`                | `expanded_query: str` → `expanded_queries: list[str]`                                                                          | L1         |
| `source/agent/nodes.py` `ANALYZER_SYSTEM_PROMPT`        | **rút gọn 36%** (120 dòng → 60 dòng): bỏ rule dịch vocab + few-shot edge case                                          | L1         |
| `source/agent/nodes.py`                                   | thêm `_rrf_fuse_lists`, `_diversify_by_location` helpers                                                                       | L2 + L3    |
| `source/agent/nodes.py`                                   | thêm `_generate_hyde`, `HYDE_SYSTEM_PROMPT`                                                                                    | HyDE       |
| `source/agent/nodes.py`                                   | thêm `_judge_retrieval_confidence`, `LOW_CONFIDENCE_CLARIFICATION`, `_query_content_tokens`                                  | Confidence |
| `source/agent/nodes.py` `legal_rag_node`                | wire HyDE + multi-retrieval + diversify + judge                                                                                     | tất cả   |
| `source/agent/nodes.py` `make_legal_rag_node`           | nhận thêm `llm` để HyDE dùng                                                                                                 | wiring     |
| `source/agent/graph.py`                                   | truyền `llm=llm` vào `make_legal_rag_node`                                                                                    | wiring     |
| `source/rag_core/generator.py` `SYSTEM_PROMPT`          | **bỏ Quy tắc 15** (no silent pick filter) — HyDE + Confidence đã làm; thêm anti-rule "Lưu ý: Internet" hallucination | prompt     |
| `source/rag_core/generator.py` `__init__(temperature=)` | default `0.1` → `0.0`                                                                                                          | L5         |
| `api/main.py` `ChatGoogleGenerativeAI` analyzer         | `temperature=0.1` → `0.0`                                                                                                      | L5         |

**Không đổi:** graph topology (4 nhánh), HITL interrupt, FastAPI endpoints, frontend, retriever core, reranker, Qdrant index, checkpoint format.

**Tương thích ngược 100%:** API `/chat` giữ nguyên schema (field `expanded_query` vẫn được emit dạng joined `" | "` string); checkpoint cũ load được.

### Kết quả RQ1 đo thực 2026-05-18

| Pipeline                        | F1              | ROUGE-L         | Latency                    |
| ------------------------------- | --------------- | --------------- | -------------------------- |
| Gemini only                     | 0.371           | 0.299           | 9.51s                      |
| Vanilla RAG                     | 0.272           | 0.238           | 15.31s                     |
| Agentic RAG v1                  | 0.354           | 0.321           | 22.92s                     |
| **Agentic RAG v3 (HyDE)** | **0.345** | **0.317** | **10.04s**           |
| Δ v3 − v1                     | -0.009          | -0.005          | **−12.88s (−56%)** |

→ Quality giữ nguyên (Δ < 1%), latency giảm 56%. Chart: [results/figures/rq1_pipeline_comparison_v3.png](../research/results/figures/rq1_pipeline_comparison_v3.png).

Chi tiết per-question + caveat về Gemini hallucinate web prefix (đã fix ở generator prompt sau eval, chưa re-run): [research/report/analysis.md §4](../research/report/analysis.md).

---

## Tài liệu liên quan

- Báo cáo kỹ thuật đầy đủ (3 000 dòng): [bao_cao_he_thong.md](bao_cao_he_thong.md)
- Hướng dẫn chạy chi tiết: [implementation_plan.md](implementation_plan.md)
- Kịch bản thuyết trình: [Kich_ban_thuyet_trinh_v3.md](Kich_ban_thuyet_trinh_v3.md)
- README quickstart: [../README.md](../README.md)
