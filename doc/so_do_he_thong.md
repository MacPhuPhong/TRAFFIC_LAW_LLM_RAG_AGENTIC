# SƠ ĐỒ TOÀN DIỆN HỆ THỐNG TRAFFIC-RAG

**Phiên bản:** v6.3 · **Ngày:** 2026-05-07 · **Tác giả:** Mạc Phú Phong

> Tài liệu này tổng hợp toàn bộ kiến trúc hệ thống Traffic-Law RAG dưới dạng sơ đồ. Mỗi sơ đồ phụ trách một khía cạnh khác nhau (context, component, ingestion, online flow, retrieval internals, HITL state, data model, deployment) để có thể đọc độc lập trong báo cáo / slide thuyết trình.

---

## Mục lục

1. [Sơ đồ ngữ cảnh (Level 1 — System Context)](#1-sơ-đồ-ngữ-cảnh)
2. [Sơ đồ thành phần (Level 2 — Container)](#2-sơ-đồ-thành-phần)
3. [Pipeline offline — Ingestion → Indexing](#3-pipeline-offline)
4. [Pipeline online — Agentic flow (LangGraph)](#4-pipeline-online)
5. [Cấu tạo Hybrid Retrieval](#5-hybrid-retrieval)
6. [Generation + Citation grounding](#6-generation--citation)
7. [State machine HITL (Human-in-the-Loop)](#7-state-machine-hitl)
8. [Data model — Chunk metadata schema](#8-data-model)
9. [Deployment topology](#9-deployment)
10. [Sơ đồ tuần tự (Sequence) — End-to-end request](#10-sequence-end-to-end)
11. [Ma trận đánh giá (Evaluation flow)](#11-evaluation-flow)

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

    SYS["🚦 TRAFFIC-RAG<br/>Trợ lý Pháp luật<br/>Giao thông VN<br/><br/>26 văn bản · 2 818 chunks"]:::system

    G["☁️ Google Gemini API<br/>2.5 Flash (LLM gen)"]:::external
    T["☁️ Tavily API<br/>Web search fallback"]:::external
    L["☁️ LangSmith<br/>Tracing &amp; eval"]:::external

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

    subgraph Backend["🐳 Docker network"]
        API["FastAPI (8000)<br/>──────────<br/>POST /chat (stream)<br/>POST /resume<br/>GET /pending<br/>GET /metrics<br/>GET /health"]:::api

        subgraph LangGraphAgent["LangGraph Agent (in-process)"]
            ANALYZER["analyzer<br/>(Gemini structured)"]:::agent
            LEGAL["legal_rag"]:::agent
            CHIT["chit_chat"]:::agent
            WEB["web_search"]:::agent
            WFIN["web_finalize<br/>⚠️ interrupt_before"]:::agent
            OOS["out_of_scope"]:::agent
        end

        RET["TrafficHybridRetriever<br/>──────────<br/>Dense E5 ⊕ BM25<br/>+ Sibling ±2 + Cross-ref"]:::agent
        GEN["LegalAnswerGenerator<br/>──────────<br/>SYSTEM_PROMPT v6.1<br/>+ citation sanitation"]:::agent

        QD[("Qdrant 1.7<br/>(6333)<br/>collection:<br/>Traffic_Law_Hybrid<br/>2 818 vectors")]:::store
        CKPT[("SQLite<br/>checkpoints/graph.db<br/>(LangGraph state)")]:::store
        BM25C[("BM25 cache<br/>all_chunks.jsonl")]:::store
    end

    subgraph Cloud["☁️ External"]
        GEMINI["Google Gemini<br/>2.5 Flash"]:::ext
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

Chạy một lần (idempotent) khi bổ sung văn bản mới — biến `Data/raw/*.pdf` thành `2 818 chunks` đã có embedding trong Qdrant.

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

    MD[("Data/cleaned/<br/>{luat,nghidinh,thongtu}<br/>*.md (23 file)")]:::store

    SC["semantic_chunker.py<br/>──────────<br/>Chương → Điều → Khoản → Điểm<br/>gắn metadata phân cấp<br/>+ status (effective/repealed)"]:::proc

    JL[("Data/all_chunks.jsonl<br/>2 818 chunks")]:::store

    IDX["indexer.py<br/>──────────<br/>e5-small encode<br/>'passage: ' prefix<br/>upsert Qdrant<br/>build BM25 cache"]:::proc

    QD[("Qdrant collection<br/>Traffic_Law_Hybrid<br/>──────────<br/>vector 384d (cosine)<br/>+ payload index<br/>{doc_id, dieu, khoan,<br/>diem, status, ...}")]:::out

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
    AN["analyzer<br/>──────────<br/>Gemini structured output:<br/>{category, standalone_q,<br/>expanded_q}"]:::node
    R{{"route_by_category"}}:::decision

    LR["legal_rag<br/>──────────<br/>retriever.search(top_k=10)<br/>+ generator.answer()"]:::node
    LFR{{"legal_fallback<br/>_router<br/>(refusal?)"}}:::decision

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

    AN -.-> GEMINI[["Gemini 2.5 Flash"]]:::ext
    LR -.-> GEMINI
    CC -.-> GEMINI
    WS -.-> TAVILY[["Tavily API"]]:::ext
    WF -.-> GEMINI

    note["⚠️ <b>HITL gate:</b> compile<br/>với <code>interrupt_before=['web_finalize']</code><br/>→ đồ thị TẠM DỪNG trước WF<br/>cho admin duyệt"]
    WF -.- note
```

**4 nhánh phân loại theo `category`:**

| Category | Khi nào | Output |
|---|---|---|
| `legal` | Câu hỏi về luật/nghị định/thông tư trong corpus | Answer + citations `[n]` → trust trực tiếp |
| `chit_chat` | Lời chào, hỏi linh tinh | Trả lời ngắn từ Gemini, không retrieval |
| `web_legal_search` | Câu hỏi pháp luật giao thông NHƯNG ngoài corpus (vd. phí đăng kiểm 2026) | Tavily → draft → **HITL pause** |
| `out_of_scope` | Câu hỏi không thuộc phạm vi (vd. thời tiết) | Template refusal |

---

## 5. Hybrid Retrieval

Bên trong `legal_rag` node, bước retrieve chia làm 4 pha:

```mermaid
flowchart LR
    classDef input fill:#e1f5fe,stroke:#0277bd
    classDef path fill:#f1f8e9,stroke:#558b2f
    classDef fuse fill:#fff3e0,stroke:#ef6c00
    classDef enrich fill:#fce4ec,stroke:#ad1457
    classDef output fill:#e0f2f1,stroke:#00695c

    Q["query<br/>(expanded_q từ analyzer)"]:::input

    subgraph Pha1["① Dense path"]
        DE["e5-small encoder<br/>'query: ' + text<br/>→ 384d vector"]:::path
        QD[("Qdrant<br/>cosine search<br/>top_n=30")]:::path
    end

    subgraph Pha2["② Sparse path"]
        TK["pyvi tokenize<br/>+ remove VI stopwords"]:::path
        BM[("BM25Okapi<br/>(rank_bm25)<br/>top_n=30")]:::path
    end

    RRF["Pha ③: RRF fusion<br/>──────────<br/>score = Σ 1/(k + rank)<br/>k=60 (default)<br/>→ top_n=30"]:::fuse

    SIB["Pha ④a: Sibling expansion<br/>──────────<br/>với mỗi chunk hit khoản X,<br/>kéo thêm khoản X-1, X-2,<br/>X+1, X+2 cùng Điều"]:::enrich

    XR["Pha ④b: Cross-reference<br/>──────────<br/>regex 'Điều N',<br/>'Khoản M Điều N'<br/>→ kéo chunk được tham chiếu"]:::enrich

    RR["(opt) Pha ⑤:<br/>bge-reranker-v2-m3<br/>──────────<br/>cross-encoder rescore<br/>⚠️ ×78 latency CPU<br/>default OFF"]:::enrich

    OUT["top_k=10 RetrievedChunk<br/>──────────<br/>{id, score, content,<br/>metadata: doc_id, dieu,<br/>khoan, diem, page, status}"]:::output

    Q --> DE --> QD
    Q --> TK --> BM
    QD --> RRF
    BM --> RRF
    RRF --> SIB
    SIB --> XR
    XR -->|"ENABLE_RERANKER=1"| RR --> OUT
    XR -->|"default"| OUT
```

**Lý do chọn hybrid (RQ9):**
- Dense one-shot bị miss khi query có biến thể từ ("vượt đèn đỏ" vs "không chấp hành tín hiệu giao thông").
- BM25 one-shot bị miss khi query là paraphrase ("đậu xe sai chỗ" vs "dừng đỗ trên cầu").
- RRF fuse 2 bảng xếp hạng (không cần normalise score) → MRR tăng so với dense-only.
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

    SP["SYSTEM_PROMPT v6.1<br/>──────────<br/>Role: Trợ lý pháp lý GT<br/>──────────<br/>12 quy tắc, gồm:<br/>R1. Citation [n] bắt buộc<br/>R4. Cấm bịa (refuse if missing)<br/>R11. Multi-intent splitting<br/>R12. Override R4 cho<br/>'phân định lỗi'"]:::rule

    LLM["Gemini 2.5 Flash<br/>──────────<br/>stream=True<br/>temperature=0.0"]:::proc

    SAN["citation_sanitation()<br/>──────────<br/>• drop [n] không tồn tại<br/>• re-number liên tục<br/>• gắn metadata về sources"]:::check

    REF{{"Chứa<br/>REFUSAL_PHRASE?"}}:::check

    OUT_OK["AnswerResult{<br/>answer: str,<br/>sources: [{<br/> doc_id, dieu, khoan,<br/> diem, page, url<br/>}]<br/>}"]:::out

    OUT_FB["→ legal_fallback_router<br/>route to web_search"]:::check

    Q --> BUILD
    CTX --> BUILD
    BUILD --> LLM
    SP --> LLM
    LLM --> SAN
    SAN --> REF
    REF -->|"NO"| OUT_OK
    REF -->|"YES"| OUT_FB
```

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

| Nhánh | Nguồn | Độ tin cậy | Cần duyệt? |
|---|---|---|---|
| `legal_rag` | Corpus chính thức (Luật, NĐ, TT) đã verify | Cao | ❌ trust thẳng |
| `web_search` | Internet open (Tavily) | Thấp, có thể tin tức/blog sai | ✅ HITL |

**State persistence:** `SqliteSaver` lưu toàn bộ `AgentState` vào `checkpoints/graph.db` theo `thread_id`. Khi `/resume` được gọi với cùng `thread_id`, LangGraph load lại state và resume từ ngay sau `interrupt_before` → admin có thể đóng tab trình duyệt rồi mở lại sau cũng vẫn duyệt được.

---

## 8. Data model

Schema metadata của một chunk trong Qdrant payload:

```mermaid
classDiagram
    class Chunk {
        +int id  «Qdrant point id»
        +float[384] vector  «e5-small dense»
        +str content  «text gốc»
        +ChunkMetadata payload
    }

    class ChunkMetadata {
        +str chunk_id  «vd. 168_2024_NĐ_CP_dieu6_khoan6_diem_c»
        +str doc_id  «vd. 168/2024/NĐ-CP»
        +str doc_type  «luat | nghidinh | thongtu»
        +str chuong  «vd. 'Chương II'»
        +int dieu  «6»
        +int khoan  «6»
        +str diem  «'c' | None»
        +int page  «trang trong PDF gốc»
        +str source_file  «Data/cleaned/.../168_2024.md»
        +str status  «effective | repealed»
        +str effective_date  «2025-01-01»
        +str title  «tiêu đề Điều»
    }

    class RetrievedChunk {
        +int id
        +float score  «RRF score»
        +str content
        +dict metadata
        +to_dict()
    }

    class AnswerSource {
        +str doc_id
        +int dieu
        +int khoan
        +str diem
        +int page
        +str url «link tới md gốc»
    }

    class AgentState {
        +str query
        +list~Message~ chat_history
        +str standalone_q
        +str expanded_q
        +str category
        +list~RetrievedChunk~ retrieved
        +str draft_answer
        +list~dict~ web_results
        +str answer
        +list~AnswerSource~ sources
    }

    Chunk --> ChunkMetadata
    RetrievedChunk ..> ChunkMetadata : metadata dict
    AgentState --> RetrievedChunk
    AgentState --> AnswerSource
```

**Convention `chunk_id`:** `<doc_slug>_dieu<N>_khoan<K>_diem_<D>` cho phép debug bằng mắt mà không cần mở payload — ví dụ `168_2024_NĐ_CP_dieu6_khoan6_diem_c` đọc được luôn là *NĐ 168/2024 Điều 6 Khoản 6 Điểm c*.

---

## 9. Deployment

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

**Lưu ý môi trường:**
- File `.env` đặt ở **gốc repo** (`GitHub1/.env`), không trong `traffic_rag/`.
- `requirements.txt` cũng ở gốc, không trong `traffic_rag/`.
- Qdrant chạy trong Docker; Backend + Frontend chạy ngoài host (dev mode `--reload`).
- Production có thể dockerize FastAPI bằng cùng `docker-compose.yml` (đã có service definition, bật khi cần).

---

## 10. Sequence end-to-end

Một câu hỏi *"Vượt đèn đỏ ô tô bị phạt bao nhiêu tiền?"* đi qua hệ thống:

```mermaid
sequenceDiagram
    autonumber
    actor U as 👤 User
    participant UI as Next.js UI
    participant API as FastAPI
    participant AG as LangGraph Agent
    participant LLM as Gemini 2.5 Flash
    participant RET as HybridRetriever
    participant QD as Qdrant
    participant BM as BM25 cache
    participant CK as SQLite checkpointer

    U->>UI: gõ câu hỏi
    UI->>API: POST /chat {query, thread_id}
    API->>AG: invoke(state, config{thread_id})

    AG->>LLM: analyzer(query) → structured
    LLM-->>AG: {category: "legal",<br/>standalone_q, expanded_q}

    AG->>AG: route_by_category → legal_rag

    AG->>RET: search(expanded_q, top_k=10)
    par Dense path
        RET->>QD: encode + cosine search top_n=30
        QD-->>RET: 30 hits
    and Sparse path
        RET->>BM: tokenize + BM25 top_n=30
        BM-->>RET: 30 hits
    end
    RET->>RET: RRF fuse → top_n=30
    RET->>RET: sibling ±2 + cross-ref expand
    RET-->>AG: top_k=10 RetrievedChunk[]

    AG->>LLM: generator(query, context, SYSTEM_PROMPT)
    LLM-->>AG: stream "Phạt 18-20 tr [1][2]..."

    AG->>AG: citation_sanitation
    AG->>AG: not refusal → END

    AG->>CK: persist final state
    AG-->>API: {answer, sources[]}
    API-->>UI: SSE stream answer + sources
    UI-->>U: render answer + click [1] xem nguồn
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

    GOLD[("research/data/<br/>gold_25.jsonl<br/>──────────<br/>25 query + expected<br/>{doc_id, dieu, khoan}")]:::input

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

## Tài liệu liên quan

- Báo cáo kỹ thuật đầy đủ (3 000 dòng): [bao_cao_he_thong.md](bao_cao_he_thong.md)
- Hướng dẫn chạy chi tiết: [implementation_plan.md](implementation_plan.md)
- Kịch bản thuyết trình: [Kich_ban_thuyet_trinh_v3.md](Kich_ban_thuyet_trinh_v3.md)
- README quickstart: [../README.md](../README.md)
