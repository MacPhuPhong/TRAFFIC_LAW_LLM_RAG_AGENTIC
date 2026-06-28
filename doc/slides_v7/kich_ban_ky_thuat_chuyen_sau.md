# Kịch bản kỹ thuật CHUYÊN SÂU — để học & bảo vệ ĐATN

> Đề tài: *Xây dựng hệ thống hỏi đáp luật giao thông đường bộ Việt Nam ứng dụng LLM kết hợp RAG*.
> Tài liệu này giải thích **từng kỹ thuật** ở mức chi tiết nhất: vấn đề → cơ chế (kèm công thức) →
> tham số thật trong code → bằng chứng số liệu → **lời nói mẫu khi trình bày** → **câu phản biện**.
> Dùng kèm `Slide_BaoCao_MacPhuPhong_30phut.pptx` và `kich_ban_thuyet_trinh_30phut.md`.
> Nguồn: `BaoCao_MacPhuPhong_3.pdf` + code thật trong `source/` (đọc tới từng dòng).

## ⚠️ LƯU Ý QUAN TRỌNG — vài chỗ CODE khác BÁO CÁO (chốt cách nói trước hội đồng)
Khi bảo vệ, **nói theo con số trong quyển báo cáo** (đó là tài liệu đã nộp). Nhưng bạn nên BIẾT
sự thật trong code để không bị bắt bẻ:

| Chi tiết | Quyển báo cáo | Code thật | Nên nói khi bảo vệ |
|---|---|---|---|
| Ngưỡng tách Điều (THRESH_DIEU) | **1.000 token** | `THRESH_DIEU = 600` | Nói **600** (con số thật) hoặc thống nhất sửa quyển; ngưỡng nhỏ hơn → tách sớm hơn, chi tiết hơn |
| Số quy tắc lệnh hướng dẫn | **14 quy tắc** | **15 quy tắc** (thêm "đa-hành-vi") | Nói "14 quy tắc cốt lõi (+1 quy tắc đa-hành-vi bổ sung trong production)" |
| Số ứng viên mỗi nhánh truy xuất | "top-2k" | `candidates_per_retriever = 30` | Nói "mỗi nhánh lấy ~30 ứng viên" |
| Model LLM mặc định trong code | Gemini 3.1 Flash Lite | default code = gemini-1.5-flash / gpt-4o-mini, **runtime truyền Gemini 3.1** | Nói Gemini 3.1 Flash Lite (đúng cấu hình chạy thật) |
| Pipeline truy xuất | "5 bước" | thực tế ~10 tầng (thêm HyDE, Parent-L2, Confidence Judge) | Nói 5 bước chính + "có thêm vài bước tối ưu" |

→ Đề xuất: thống nhất `THRESH_DIEU` và "14/15 quy tắc" giữa code và quyển trước buổi bảo vệ. Phần
dưới ghi **cả hai** để bạn chủ động.

---
---

# PHẦN A — QUY TRÌNH NGOẠI TUYẾN (offline)

## 1. Làm sạch văn bản PDF → Markdown  ·  *Mục 2.3.1 · `source/ingestion/clean_*.py` + `semantic_chunker.py:refine_content`*

**Vấn đề:** Văn bản luật phát hành dạng PDF: bảng mức phạt, header/footer lặp mỗi trang, và **câu bị
ngắt giữa chừng** khi sang trang. Nếu để nguyên, chunk sẽ dính rác và biên câu sai → truy xuất & trích
dẫn sai theo.

**Cơ chế (4 thao tác chính trong `refine_content`):**
1. **Bảo vệ bảng Markdown:** dòng bắt đầu `|` được thay placeholder `__TABLE_LINE_i__` trước khi xử lý,
   khôi phục sau — để regex nối câu không phá vỡ bảng phạt.
2. **Nối từ bị gạch nối cuối dòng:** `re.sub(r'(\s*-\s*)\n\s*', r'\1', text)`.
3. **Nối câu bị ngắt giữa chừng:** regex bắt ký tự thường/dấu phẩy ở cuối dòng + ký tự thường đầu dòng
   tiếp theo → thay `\n` bằng dấu cách (dùng lookbehind/lookahead trên trọn bộ ký tự tiếng Việt có dấu).
4. **Xóa dòng kẻ trang trí** (`-----`, `=====`) và **gộp >2 dòng trống** thành 1.

**Hai thao tác làm giàu thêm khi tạo chunk (`_make_chunk`):**
- **Legal styling:** tự **bold** cụm "phạt tiền từ X đồng đến Y đồng" và "tước quyền sử dụng… tháng"
  → giúp người đọc + giúp LLM "nhìn" số tiền.
- **Context enrichment:** chèn dòng `Văn bản: {tên} | Điều {n}: {tiêu đề}` vào đầu mỗi chunk; chèn
  cảnh báo `⚠️` nếu văn bản có `warning` (vd NĐ 100/2019 mảng đường bộ đã bãi bỏ).

**Lọc hiệu lực (trường `status`):** mỗi văn bản gắn `active / repealed / partially_repealed`. Bộ lọc
loại 2 văn bản hết hiệu lực (Luật GTĐB 2008 = `repealed`; NĐ 100/2019 mảng đường bộ = `partially_repealed`)
→ còn **24 văn bản** vào index. (Cũng lọc 8 keyword "biểu mẫu nội bộ" như *sổ theo dõi, nhật ký…* và
**khử trùng lặp** bằng MD5 hash của content.)

**Nói khi trình bày (~40s):** *"Văn bản luật là PDF layout phức tạp. Em viết ba chương trình làm sạch
riêng cho luật/nghị định/thông tư: bảo toàn bảng phạt, nối câu bị ngắt trang, bold mức phạt, và chèn
ngữ cảnh Điều vào đầu mỗi đoạn. Quan trọng nhất là trường status — chỉ văn bản còn hiệu lực mới được
index, nên hệ thống không bao giờ trả về điều luật đã bị bãi bỏ."*

**Phản biện:**
- *Q: Sao không dùng OCR/thư viện parse PDF có sẵn?* → pdfplumber đã trích text tốt; vấn đề còn lại là
  **cấu trúc pháp lý** (page-break, bảng) cần luật làm sạch riêng theo loại văn bản, không có lib chung.
- *Q: Làm sạch sai thì sao?* → mỗi lần chạy ghi `_processing_stats.json` (số trang/bảng/block lọc) để
  kiểm tra; và cổng CI chặn nếu chunk hỏng làm tụt Recall@10.

---

## 2. ⭐ PHÂN ĐOẠN PHÂN CẤP (HierarchicalLegalSplitter)  ·  *Mục 2.3.2 · `semantic_chunker.py` — ĐÓNG GÓP CỐT LÕI #1*

**Vấn đề:** Cách phổ biến trong RAG tiếng Việt là **cắt cố định 512 token + overlap**. Nó **cắt ngang
ranh giới Điều–Khoản–Điểm**: ghép nửa khoản này với nửa khoản kia, tách mức phạt khỏi hành vi. Hệ quả:
khó truy xuất đúng đơn vị điều khoản, càng khó trích dẫn đúng tới Điểm.

**Ý tưởng:** Phân đoạn **bám đúng cấu trúc pháp lý 3 cấp** Điều → Khoản → Điểm. Khi đó **biên đoạn TRÙNG
biên trích dẫn** → lấy đúng đoạn là tự có đúng Khoản/Điểm để trích dẫn.

**Nhận diện cấu trúc — 4 mẫu regex thật:**
```
L1 Điều:  (?:##\s*)?Điều\s+(\d+)[.:]\s*(.*)
L2 Khoản: ^(\d+)\.(?=\s+[A-ZĐÀÁ…])           # số + chấm + chữ HOA
L3 Điểm:  (?:^|;\s+)([a-zđ])\)(?=\s+)         # 'a)'…'đ)' kể cả sau dấu ';'
```

**Thuật toán 3 cấp + ngưỡng động (code thật):**
- Đếm token ≈ `số_từ × 1.3` (ước lượng cho tiếng Việt).
- **L1 — Điều:** nếu Điều ≤ `THRESH_DIEU` (code = **600**, quyển ghi 1.000) → giữ cả Điều là 1 đoạn.
- **L2 — Khoản:** nếu Điều dài hơn → tách theo Khoản; ngưỡng tách tiếp xuống Điểm:
  `THRESH_KHOAN_STRICT = 80` cho **nghị định**, `THRESH_KHOAN_DEFAULT = 500` cho luật/thông tư.
- **L3 — Điểm:** tách thành Điểm độc lập khi:
  `count(khoản) > ngưỡng` **HOẶC** `(là nghị định AND có ≥ 2 Điểm)` ← điều kiện ngữ nghĩa, không chỉ
  theo độ dài.
- Ngưỡng giữ đoạn tối thiểu: `MIN_CHUNK = 30` token cho L1/L2, nhưng **`MIN_CHUNK_DIEM = 5`** cho L3 —
  vì 1 Điểm pháp lý thường rất ngắn (1 hành vi ≈ 10–20 token) nhưng vẫn là mục tiêu truy xuất độc lập.

**5 cải tiến (so với cách phân đoạn thông thường):**
| # | Hạn chế cách thường | Cải tiến |
|---|---|---|
| 1 | Ngưỡng độ dài nuốt mất Điểm ngắn | `MIN_CHUNK_DIEM=5`, **không lọc Điểm ngắn** |
| 2 | Một ngưỡng cho mọi văn bản | **Hạ ngưỡng riêng cho nghị định** (`STRICT=80`) |
| 3 | Không ép tách tới Điểm | **NĐ có ≥2 Điểm → luôn tách L3** (`has_multi_diem`) |
| 4 | Regex bỏ sót `đ)` và Điểm sau `;` | Mở rộng mẫu bắt cả hai |
| 5 | Điểm bị tách rời khung phạt của Khoản | **Làm giàu preamble:** mỗi đoạn Điểm tự **prepend** câu mở đầu Khoản |

**Cải tiến #5 (quan trọng nhất) — `split_khoan_into_diems(enrich_preamble=True)`:** Điểm thường chỉ mô
tả hành vi (vd "k) Dàn hàng ngang từ 03 xe…"), KHÔNG có số tiền. Code lấy **preamble** (cụm "Phạt tiền
từ X đến Y đồng đối với…" trước Điểm đầu) **ghép vào đầu MỖI đoạn Điểm** → mỗi đoạn tự chứa đủ
`{mức phạt + hành vi}`, truy xuất Điểm-level mới chính xác.

**Kết quả (bằng chứng):** trước khi tách tới Điểm, **77,5% Điểm trong NĐ 168 (469/605) không có đoạn
riêng** → nhiều mức phạt không truy xuất được. Sau cải tiến: còn **11,7%**. Phân bố **4.423 đoạn**:
L1 Điều 486 (11%) · L2 Khoản 1.282 (29%) · **L3 Điểm 2.655 (60%)**.

**Mỗi đoạn gắn metadata đầy đủ** (`chunk_id, doc_id, ten_van_ban, document_type, status, dieu,
dieu_title, khoan, diem, level, token_estimate`) → phục vụ lọc + trích dẫn tới cấp Điểm.

**Nói khi trình bày (~1.5 phút):** *"Đây là đóng góp cốt lõi. Thay vì cắt cố định 512 token làm gãy
ranh giới điều khoản, em tách đúng theo cấu trúc pháp lý ba cấp Điều–Khoản–Điểm bằng bốn mẫu regex và
ngưỡng động. Năm cải tiến, mà then chốt là 'làm giàu preamble': mỗi đoạn Điểm tự mang theo câu 'Phạt
tiền từ X đến Y đồng…' của Khoản cha, nên đoạn tự chứa đủ mức phạt và hành vi. Nhờ vậy tỉ lệ Điểm
của Nghị định 168 không có đoạn riêng giảm từ 77,5% xuống 11,7%. Cái hay là biên đoạn trùng biên trích
dẫn — lấy đúng đoạn là tự có đúng khoản, điểm."*

**Phản biện:**
- *Q: Tại sao không dùng overlap như RAG thường?* → Overlap chỉ giảm gãy câu chứ không tạo biên = đơn
  vị trích dẫn; thực nghiệm (Mục 3.3) cho thấy phân cấp hơn cố định **×2,4 về Citation-Recall**.
- *Q: Vì sao MIN_CHUNK_DIEM chỉ 5?* → vì Điểm pháp lý ngắn nhưng là target truy xuất; lọc 30 token làm
  **mất ~70% Điểm** của NĐ 168.
- *Q: F1 token gần hoà, vậy phân cấp có thật sự tốt hơn?* → F1 token chỉ đo trùng n-gram bề mặt; trong
  pháp lý **độ chính xác trích dẫn (Citation-Recall) mới là thước đo thực chất** — và nó ×2,4.

---

## 3. Mô hình nhúng E5-base (dense embedding)  ·  *Mục 1.4.3 & 2.3.3 · `indexer.py`, `retriever.py`*

**Vấn đề:** Cần biến văn bản thành vector sao cho câu hỏi và đoạn luật **cùng nghĩa thì gần nhau** —
kể cả khi dùng từ khác ("vượt đèn đỏ" ≈ "không chấp hành hiệu lệnh đèn tín hiệu").

**Mô hình:** `intfloat/multilingual-e5-base` — nền **XLM-RoBERTa**, 12 tầng, hidden=768, max_seq=512,
~278M tham số. Huấn luyện 2 giai đoạn: (1) weak-supervised trên web đa ngôn ngữ (CC100/Wikipedia, 94
ngôn ngữ); (2) **đối chiếu InfoNCE trên 1 tỷ cặp truy vấn–đoạn**:

$$\mathcal{L}=-\log\frac{e^{\,\mathrm{sim}(q,p^{+})/\tau}}{\sum_i e^{\,\mathrm{sim}(q,p_i)/\tau}},\quad \tau=0{,}01$$

(mẫu âm khó lấy từ BM25). Vector cuối = **mean-pooling có mặt nạ** qua các token + **chuẩn hóa L2** →
cosine rút gọn thành tích vô hướng (tốt cho HNSW).

**Tiền tố BẮT BUỘC của họ E5 (vấn đề kỹ thuật #1):** thêm `"passage: "` khi nhúng ĐOẠN, `"query: "`
khi nhúng TRUY VẤN. Code: `self.model.encode(self._query_prefix + query, normalize_embeddings=True)`.
**Dùng sai tiền tố → giảm ~30% Recall@10.**

**Bằng chứng chọn model (Mục 3.4, chỉ nhánh dày đặc):** e5-base **R@10=0,424 · MRR=0,277** (cao nhất 4
mô hình). e5-small 0,367 (rẻ/nhanh, dự phòng). mpnet 0,195 (max_seq=128 cắt khoản dài). sbert 0,267
(không InfoNCE, huấn luyện chưng cất). → Chọn e5-base: chất lượng cao nhất, giữ 768d khớp collection.

**Nói khi trình bày (~50s):** *"Em dùng E5-base đa ngôn ngữ, nền XLM-RoBERTa, được huấn luyện đối chiếu
InfoNCE trên một tỷ cặp nên bắt rất tốt tương đồng ngữ nghĩa. Một lưu ý đặc thù của họ E5 là phải thêm
tiền tố 'passage:' và 'query:' — dùng sai làm giảm 30% recall. Em chọn e5-base vì recall@10 cao nhất
trong bốn mô hình so sánh."*

**Phản biện:** *Q: Sao không fine-tune trên luật VN?* → cần dữ liệu cặp gán nhãn lớn (tốn kém); e5-base
zero-shot đã đạt mức chấp nhận được; fine-tune là hướng phát triển. *Q: 768 vs 384 chiều?* → 768 (base)
hơn 384 (small) ~5,7 điểm R@10 và giữ nguyên collection nên không phải re-index.

---

## 4. Đánh chỉ mục Qdrant (HNSW) + cache BM25  ·  *Mục 2.3.3 / 3.5 · `indexer.py`, `retriever.py`*

**Cấu hình collection `Traffic_Law_Hybrid`:** 4.423 điểm × 768 chiều, khoảng cách **Cosine**, chỉ mục
**HNSW** `m=16, ef_construct=100` (đủ ổn định ở quy mô này). Batch nhúng 64 đoạn, batch upsert 100 điểm.
Tổng ~26 MB.

**HNSW là gì (1 câu):** đồ thị "thế giới nhỏ" phân tầng cho tìm **lân cận gần nhất xấp xỉ (ANN)** —
truy vấn nhảy từ tầng thưa xuống tầng dày, độ phức tạp ~O(log N) thay vì O(N) quét tuyến tính.

**3 chỉ mục payload (KEYWORD)** trên `doc_id, status, topic` → cho phép **lọc ngay tại tầng DB**. Quan
trọng nhất: `status="active"` loại đoạn hết hiệu lực trước khi tính tương đồng.

**Vì sao chọn Qdrant (Mục 3.5):** so 3 backend trên cùng 4.423 vector × 3.500 truy vấn — Qdrant/Chroma
R@10=0,424, FAISS 0,395 (IVF-Flat xấp xỉ). FAISS nhanh ×56 (0,08ms vs 4,44ms) **nhưng** không lọc
payload, không bền vững, không update online. Chênh vài mili-giây **không đáng** so với LLM mất 3–30s
→ giữ Qdrant vì **payload filter + HTTP API + update online**.

**Cache BM25:** toàn bộ đoạn được tách từ bằng **pyvi** (xử lý từ ghép như `vượt_đèn_đỏ`) rồi lưu lại;
khởi động lại nạp < 0,5s thay vì dựng lại ~8s.

**Nói khi trình bày (~30s):** *"4.423 vector được index vào Qdrant với HNSW và khoảng cách cosine. Em
tạo ba chỉ mục payload để lọc nhanh — đặc biệt lọc status active ngay tại DB. Em chọn Qdrant dù FAISS
nhanh hơn, vì điều quyết định không phải vài mili-giây mà là khả năng lọc payload và mở rộng."*

**Phản biện:** *Q: 4ms hay 0,08ms quan trọng gì?* → không, vì LLM mới là nút cổ chai (3–30s); recall và
khả năng lọc/scale mới quyết định.

---
---

# PHẦN B — QUY TRÌNH TRỰC TUYẾN: TRUY XUẤT

## 5. Mở rộng truy vấn (Query Expansion / Rewrite)  ·  *Mục 2.4.1 / 3.7.2 · `nodes.py:make_analyzer_node`*

**Vấn đề:** Truy vấn người dùng thường **ngắn/mơ hồ** ("đèn vàng có vượt được không?") hoặc **đa ý định**
(vừa hỏi phạt vừa hỏi lỗi). Truy xuất thẳng từ câu gốc → recall thấp.

**Cơ chế:** Nút `analyzer` gọi LLM với **structured output** (Pydantic `AnalyzerOutput`) trả về 5 trường:
- `category`: legal_rag / chit_chat / web_legal_search / out_of_scope (để định tuyến).
- `intent`: penalty / fault / procedure / definition / list / **mixed** (điều khiển hình dạng câu trả lời).
- `vehicle_type`: o_to / mo_to / chuyen_dung / xe_dap / any (gắn đúng Điều 6/7/8/9 NĐ 168).
- `standalone_query`: **giải tham chiếu** theo lịch sử hội thoại → câu tự đứng vững ("Còn xe tải thì
  sao?" → "Xe tải vượt đèn đỏ phạt bao nhiêu?").
- `expanded_queries`: **list 1–5 biến thể**, mỗi "khung pháp lý" (frame) một phần tử. Ví dụ "vượt đèn
  đỏ xe máy" → ["vi phạm tín hiệu đèn giao thông mô tô", "Điều 7 NĐ 168 xe máy vượt đèn đỏ"].

**Bằng chứng (Mục 3.7.2):** bật mở rộng nâng **R@10 từ 0,324 → 0,581 (+26 điểm %)** — **mức tăng lớn
nhất toàn bộ thực nghiệm**; nDCG@10 +14 điểm; độ trễ chỉ +7% (thêm 1 lời gọi LLM). Đảo chiều so kho cũ:
khi đoạn còn dài, mở rộng tạo nhiễu; với đoạn cấp Điểm nhỏ & chính xác, mở rộng rất hữu ích.

**Nói khi trình bày (~1 phút):** *"Trước khi truy xuất, nút analyzer phân tích câu hỏi và trả về cấu
trúc gồm: phân loại nhánh, loại ý định, loại phương tiện, câu hỏi đã giải tham chiếu, và một đến năm
truy vấn mở rộng — mỗi khung pháp lý một truy vấn. Đây là kỹ thuật tác động lớn nhất: recall@10 tăng
từ 0,324 lên 0,581, cộng 26 điểm phần trăm, mà độ trễ chỉ tăng 7%."*

**Phản biện:** *Q: Mở rộng nhiều có tạo nhiễu?* → có giới hạn 1–5 biến thể theo số khung; sau đó **RRF**
hợp nhất nên các đoạn xuất hiện ở nhiều truy vấn được thưởng, nhiễu bị chìm. *Q: Vì sao dùng LLM mà không
dùng từ điển đồng nghĩa?* → câu pháp lý cần hiểu ngữ cảnh + giải tham chiếu hội thoại, từ điển không làm
được.

---

## 6. Truy xuất dày đặc (Dense retrieval)  ·  *Mục 2.4.2 · `retriever.py:get_relevant_chunks`*

**Cơ chế:** nhúng truy vấn (prefix `query:`, chuẩn hóa L2) → `client.search` trên Qdrant với
`query_filter` (status=active mặc định) → trả `candidates_per_retriever = 30` đoạn gần nhất theo cosine
qua HNSW. Bắt **tương đồng ngữ nghĩa** (bất đối xứng từ vựng).

**Chi tiết tinh:** có **auto-rewrite phía truy vấn** — nếu câu chứa "không kinh doanh vận tải" thì tự
thêm `exclude_topics=["Kinh doanh vận tải"]` và loại NĐ 10/2020, tránh nó lấn át kết quả.

---

## 7. Truy xuất thưa BM25 (Sparse retrieval)  ·  *Mục 1.4.4 / 2.4.2 · `retriever.py:_build_bm25_corpus`*

**Vấn đề:** Dense yếu khi truy vấn chứa **mã chính xác** ("Điều 7", "NĐ 168/2024"). BM25 (từ khóa) bù lại.

**Công thức đầy đủ (BM25 Okapi):**

$$\text{score}(D,Q)=\sum_{t\in Q}\text{IDF}(t)\cdot\frac{f(t,D)\,(k_1+1)}{f(t,D)+k_1\!\left(1-b+b\frac{|D|}{\text{avgDL}}\right)}$$
$$\text{IDF}(t)=\log\!\left(\frac{N-\text{df}(t)+0{,}5}{\text{df}(t)+0{,}5}+1\right)$$

- `f(t,D)`: tần suất từ t trong đoạn D · `|D|`: độ dài đoạn · `avgDL`: độ dài trung bình ·
  `N=4.423`: tổng đoạn · `df(t)`: số đoạn chứa t.
- **`k₁=1,5`** điều tiết "thưởng" khi từ lặp lại (bão hòa); **`b=0,75`** điều tiết "phạt" đoạn dài.

**Tách từ tiếng Việt (`_tokenize_vi`):** lowercase → bỏ ký tự không phải chữ → bỏ **stopword** (28 từ:
là, và, của, cho…) → bỏ token độ dài 1. (Báo cáo nêu **pyvi** xử lý từ ghép — dùng ở bước build cache.)

**Đồng bộ ID:** dòng BM25 căn theo **đúng thứ tự điểm Qdrant** (thứ tự dòng JSONL) nên không cần lớp
ánh xạ ID. BM25 cũng áp **cùng bộ lọc status** (qua `allowed_ids`).

**Nói khi trình bày (~30s):** *"BM25 chấm điểm theo tần suất từ với k1=1,5, b=0,75. Nó mạnh khi truy vấn
có mã chính xác như 'Điều 7' hay 'NĐ 168' — đúng chỗ embedding ngữ nghĩa hay bỏ lỡ. Em tách từ tiếng
Việt bằng pyvi để xử lý từ ghép như 'vượt_đèn_đỏ'."*

**Phản biện:** *Q: k₁, b chọn sao?* → giá trị chuẩn công nghiệp (Robertson); corpus nhỏ nên không cần
tinh chỉnh. *Q: BM25 với dense, cái nào quan trọng hơn?* → bổ sung nhau; hybrid hơn từng cái đơn lẻ.

---

## 8. ⭐ Hợp nhất xếp hạng RRF (Reciprocal Rank Fusion)  ·  *Mục 1.4.6 / 2.4.3 · `retriever.py:_rrf_fuse` + `nodes.py:_rrf_fuse_lists`*

**Vấn đề:** Dense trả điểm cosine (0–1), BM25 trả điểm không chuẩn hóa (vài chục). **Không thể cộng
thẳng**. Và còn nhiều danh sách (mỗi truy vấn mở rộng × 2 nhánh).

**Công thức RRF:**

$$\text{RRF}(d)=\sum_{\ell\in L}\frac{1}{k+\text{rank}_\ell(d)},\qquad k=60$$

Chỉ dùng **thứ hạng** (rank), không dùng điểm thô → **không cần chuẩn hóa**. Đoạn xuất hiện ở nhiều
danh sách / thứ hạng cao → điểm tổng cao.

**Code thật:** `scores[id] += 1.0 / (60 + rank)` với `rank` bắt đầu từ 1; sắp giảm dần; ties phá theo
"xuất hiện sớm nhất". Hằng số **k=60** (Cormack 2009) ổn định cho kho < 10.000 tài liệu — k lớn làm
giảm ảnh hưởng của top-1 từng danh sách.

**Áp dụng 2 tầng:** (a) trong 1 truy vấn: fuse dense + BM25; (b) đa-truy-vấn: fuse **N+1 danh sách**
(N biến thể + 1 HyDE) → over-pull `top_k×2` để còn dư cho bước đa dạng hóa.

**Ví dụ số:** đoạn đứng top-1 ở 1 danh sách ≈ 1/61 ≈ 0,0164; đứng top-1 ở **2** danh sách ≈ 0,0328 →
gấp đôi, nên đoạn được nhiều truy vấn đồng thuận sẽ nổi lên.

**Nói khi trình bày (~40s):** *"Để gộp kết quả từ nhánh ngữ nghĩa và nhánh từ khóa — vốn có thang điểm
khác nhau — em dùng RRF: chỉ dựa vào thứ hạng, điểm bằng tổng của một trên k cộng rank, với k bằng 60.
Ưu điểm là không cần chuẩn hóa điểm, và đoạn nào được nhiều truy vấn đồng thuận sẽ tự nổi lên."*

**Phản biện:** *Q: Sao không normalize điểm rồi cộng có trọng số?* → normalize giữa hai hệ rất nhạy
tham số và corpus; RRF không tham số (ngoài k=60 mặc định) nên ổn định, đỡ tinh chỉnh.

---

## 9. HyDE — Hypothetical Document Embeddings *(kỹ thuật nâng cao trong code, ngoài "5 bước" của quyển)*  ·  *`nodes.py:_generate_hyde`*

**Cơ chế:** trước khi truy xuất, nhờ LLM **viết nháp một câu trả lời pháp lý giả định**; văn bản nháp
này thành **một truy vấn nữa**. Lý do: bắc cầu từ vựng người dùng → từ vựng văn bản luật (vd người
dùng nói "động vật", câu trả lời luật dùng "súc vật" → nhúng trúng đoạn đúng). **Bỏ qua HyDE khi
intent=mixed** (vì khuếch đại 1 khung, bỏ đói khung kia — đa-truy-vấn đã lo).

**Nói khi trình bày (tùy chọn):** *"Em còn thêm HyDE: LLM phác một câu trả lời giả định làm truy vấn bổ
sung, giúp bắc cầu giữa cách nói của dân và thuật ngữ luật."* (Chỉ nói nếu hội đồng hỏi sâu hoặc còn
thời gian — không có trong quyển.)

---

## 10. Làm giàu ngữ cảnh lân cận (Sibling Enrichment) — 2 tầng  ·  *Mục 2.4.3 · `retriever.py:_attach_siblings`*

**Vấn đề:** Câu hỏi đa ý định cần nhiều khoản liền kề. Vd Điều 7 NĐ 168: Khoản 4 ghi **mức phạt tiền**,
nhưng **trừ điểm GPLX** nằm ở **Khoản 11** (tham chiếu ngược). Lấy đúng Khoản 4 vẫn thiếu.

**Cơ chế 2 tầng (code thật):**
- **Tầng (a) — completion clause:** với mỗi `(doc_id, Điều)` xuất hiện, luôn kéo các đoạn "hoàn chỉnh
  hình phạt" — nhận diện bằng cụm như *"còn bị trừ điểm giấy phép"*, *"hình thức xử phạt bổ sung"*,
  *"tước quyền sử dụng giấy phép"*, *"tịch thu phương tiện"*. (Thường ~3 đoạn/Điều trong NĐ 168.)
- **Tầng (b) — Khoản lân cận:** với mỗi đoạn có `khoan` là số nguyên N, kéo thêm Khoản
  **N±1, N±2** (`max_neighbors=2`) cùng Điều. Xử lý ca biên ngưỡng (vd query ">0,4mg" rơi vào K9 nhưng
  retriever lấy K8 chứa "0,25–0,4mg" → K9 lân cận lấp chỗ trống).

Các đoạn lân cận giữ `score=0.0` và đánh dấu **`is_sibling=True`** (để bước đa dạng hóa KHÔNG tính vào
hạn mức, và để prompt gắn nhãn "[Ngữ cảnh bổ sung — cùng Điều]").

**Nói khi trình bày (~40s):** *"Làm giàu lân cận đặc trị Nghị định 168: mức phạt tiền và mức trừ điểm
nằm ở hai khoản khác nhau. Với mỗi đoạn lấy được, em kéo thêm các khoản liền kề cộng trừ hai, và luôn
kéo các đoạn 'trừ điểm GPLX / xử phạt bổ sung' cùng Điều — để câu trả lời đủ cả tiền phạt lẫn điểm trừ."*

**Phản biện:** *Q: Kéo thêm có làm loãng ngữ cảnh?* → có giới hạn ±2 và đánh dấu sibling; bước **diversity
cap** + chỉ giữ 10 đoạn cuối kiểm soát độ loãng; RAGAS context_precision 0,885 chứng tỏ ngữ cảnh vẫn sạch.

---

## 11. ⭐ Giải tham chiếu chéo (Cross-Reference Resolution)  ·  *Mục 2.4.3 · `nodes.py:_extract_legal_references` + `retriever.py:get_chunks_by_location`*

**Vấn đề:** Đoạn luật hay dẫn chiếu: *"hành vi quy định tại điểm a khoản 4 Điều 13"*. Tìm kiếm tương
đồng **làm loãng** các tham chiếu chính xác này (số điều/khoản không mang nhiều tín hiệu ngữ nghĩa).

**Cơ chế (giải quyết bằng CẤU TRÚC, không bằng tương đồng):**
- **Quét regex** trên (i) câu hỏi và (ii) nội dung 5 đoạn đầu, bắt 2 mẫu:
  `điểm X khoản Y [Điều Z]` và `khoản Y Điều Z`. Mặc định `doc_id = 168/2024/NĐ-CP` nếu không nêu văn bản.
- Với mỗi tham chiếu, gọi **`get_chunks_by_location(doc_id, dieu, khoan, diem)`** — **tra cứu trực tiếp
  theo metadata** (không similarity), kéo đúng đoạn được dẫn vào ngữ cảnh.
- Có **chặn an toàn**: bỏ tham chiếu tới Điều đã có sẵn trong ngữ cảnh; giới hạn 15 tham chiếu, tối đa
  thêm 30 đoạn (tránh phình ngữ cảnh).

**Nói khi trình bày (~40s):** *"Văn bản luật đầy tham chiếu chéo kiểu 'theo quy định tại khoản X Điều
Y'. Tìm kiếm tương đồng hay bỏ lỡ chúng, nên em giải bằng cấu trúc: quét regex tìm các tham chiếu rồi
tra cứu trực tiếp theo metadata để kéo đúng đoạn được dẫn vào ngữ cảnh."*

**Phản biện:** *Q: Sao không để LLM tự suy?* → LLM không có đoạn đó trong ngữ cảnh thì sẽ **bịa** (ảo
giác); kéo đúng đoạn thật vào mới chống ảo giác. *Q: Mặc định 168 có sai không?* → vì là corpus chủ
đạo; nếu câu nêu văn bản khác thì regex bắt được Điều + ta dùng doc_id đó.

---

## 12. Làm giàu Khoản cha (Parent-L2 Enrichment) *(nâng cao, code)*  ·  *`nodes.py:_attach_parent_l2`*

**Cơ chế:** khi top-K có **đoạn L3 (Điểm)** nhưng thiếu **Khoản cha L2** (nơi chứa số tiền), tự kéo
Khoản cha vào — quan trọng cho intent **penalty/mixed** cần trích số tiền cụ thể. Bỏ qua nếu Khoản cha
> 1500 ký tự (tránh "hijack" sự chú ý của Flash Lite). Đoạn cha cũng đánh dấu sibling.

---

## 13. Giới hạn đa dạng nguồn (Diversity Cap)  ·  *Mục 2.4.3 · `nodes.py:_diversify_by_location`*

**Vấn đề:** sau RRF, nhiều đoạn cùng một Khoản có thể chiếm hết top-K → ngữ cảnh hẹp, generator chỉ
thấy 1 trường hợp.

**Cơ chế:** mỗi "rổ" `(doc_id, dieu, khoan)` giữ **tối đa `MAX_CHUNKS_PER_KHOAN = 3`** đoạn; cắt lấy
`target_k` (=10). **Đoạn sibling KHÔNG bị tính hạn mức** (luôn được giữ vì chúng tồn tại để hoàn chỉnh
đoạn chính). Giữ nguyên thứ tự đã xếp theo RRF.

**Nói khi trình bày (~20s):** *"Cuối cùng em áp giới hạn đa dạng — tối đa 3 đoạn từ cùng một khoản — để
ngữ cảnh trải trên nhiều khoản, giúp câu trả lời liệt kê được mọi trường hợp thay vì kẹt vào một."*

---

## 14. Confidence Judge — bộ thẩm định độ tin cậy truy xuất *(nâng cao, code)*  ·  *`nodes.py:_judge_retrieval_confidence`*

**Cơ chế:** chấm điểm **xác định, theo thang điểm RRF + đặc trưng theo intent** (vd penalty đòi top-đoạn
có số tiền; fault/procedure có tiêu chí khác). Nếu độ tin cậy thấp → **từ chối an toàn** hoặc **fallback
sang web_search**. Đây là một phần của "kiểm soát ảo giác theo chiều sâu" (đóng góp #3): thà từ chối
còn hơn trả lời sai.

---

## 15. Xếp hạng lại (Reranker — cross-encoder)  ·  *Mục 2.4.4 / 3.7.3 · `reranker.py`*

**Vấn đề:** Truy xuất 2 nhánh là **bi-encoder** (nhúng truy vấn và đoạn ĐỘC LẬP rồi so cosine) → nhanh
nhưng kém tinh. Cross-encoder đọc **đồng thời (truy vấn, đoạn)** nên đánh giá liên quan chính xác hơn.

**Cơ chế:** `BAAI/bge-reranker-v2-m3` (đa ngôn ngữ, ~568M tham số, max_length=512). Nhận **30 đoạn** sau
RRF, chấm điểm từng cặp `(query, chunk)` qua mô hình, sắp lại, giữ **top-10**. Nạp lười (chỉ load khi
bật).

**Bằng chứng đánh đổi (Mục 3.7.3):** **R@10 0,324 → 0,481 (+48,5%)** — lôi được khoản đúng ở hạng 11–20
lên top-10. **NHƯNG độ trễ ×156 trên CPU** (708ms → ~110 giây). → **Tắt mặc định** (`ENABLE_RERANKER`),
chỉ bật khi có GPU (dự kiến ~2–3s) hoặc dùng `bge-reranker-base` (278M, ~½ độ trễ, ~70% lợi ích).

**Nói khi trình bày (~30s):** *"Bộ xếp hạng lại là cross-encoder đọc đồng thời cặp truy vấn–đoạn nên
chính xác hơn, kéo recall@10 lên 0,481, tăng 48,5%. Nhưng cái giá là độ trễ tăng 156 lần trên CPU, nên
em tắt mặc định và chỉ bật khi có GPU."*

**Phản biện:** *Q: Bi-encoder vs cross-encoder khác gì?* → bi-encoder nhúng độc lập (cache được, nhanh,
dùng cho recall rộng); cross-encoder đọc chung cặp (chậm, dùng cho precision top nhỏ). *Q: Sao không
dùng luôn?* → 110s/câu không chấp nhận real-time; là cải thiện tiềm năng khi có GPU.

---
---

# PHẦN C — SINH CÂU TRẢ LỜI

## 16. ⭐ Lệnh hướng dẫn 14(/15) quy tắc (Prompt Engineering — P2)  ·  *Mục 2.5.1 / 3.6 · `generator.py:SYSTEM_PROMPT`*

**Vấn đề:** LLM dễ **ảo giác pháp lý** — bịa nguồn, nhầm mức phạt. Cần ràng buộc chặt ngay trong khâu sinh.

**Cấu trúc prompt:** phần **vai trò** ("Trợ lý Pháp lý Giao thông Việt Nam, chỉ dựa NGỮ CẢNH") + **14
quy tắc bắt buộc** (code production có thêm quy tắc 15 "đa-hành-vi"). Nhóm quy tắc tiêu biểu:
- **Chống ảo giác:** (1) tiếng Việt; (2) **chỉ dùng NGỮ CẢNH**, không kiến thức ngoài; (4) thiếu thông
  tin → trả đúng câu từ chối *"Thông tin này không có trong tài liệu được cung cấp."*; (11) **không từ
  chối thừa** — có một phần thì trả phần đó + ghi chú phần thiếu.
- **Buộc trích dẫn & số liệu:** (2/3) mọi mức phạt/điểm kèm `[Điều X, Khoản Y, Điểm Z — doc_id]`;
  (4) **copy nguyên văn** "từ X đến Y đồng", không tự nghĩ số.
- **Định dạng:** (6) câu đa ý → tách bullet, **bold** số tiền/điểm/thời gian, heading `###` mỗi nhóm.
- **Phân loại đúng:** (7) xe kinh doanh vận tải → NĐ 10/2020; (8) **Điều 6 ô tô · 7 mô tô · 8 chuyên
  dùng · 9 thô sơ** trong NĐ 168.
- **Tham chiếu chéo:** (9) được dùng đoạn "[Ngữ cảnh bổ sung — cùng Điều]"; (10) **quy trình 3 bước**
  ghép phạt tiền (Khoản đầu) với trừ điểm (Khoản cuối K13–K16) qua tham chiếu.
- **Dễ hiểu & lập luận:** (12) câu liệt kê → rà toàn bộ ngữ cảnh; (13) **mô tả hành vi bằng lời trước,
  trích điều khoản sau** (đối tượng là dân thường, không phải luật sư); (14) **phân định lỗi 4 bước**
  (liệt kê hành vi → đối chiếu quy định → kết luận lỗi → lưu ý CSGT theo TT 72/2024).

**"Gợi ý động" tiêm theo câu hỏi (code thật, ngoài 14 quy tắc tĩnh):**
- `intent_hint`: 6 mẫu (penalty/fault/procedure/definition/list/mixed) — ép đúng hình dạng câu trả lời.
- `vehicle_hint`: nếu vehicle_type ≠ any **và** intent ∈ {penalty, mixed} → ép **chỉ trích đúng Điều
  6/7/8/9** theo loại xe (chống lỗi "trích Điều 6 ô tô cho câu hỏi xe máy"). *Lưu ý: chỉ áp cho
  penalty/mixed — vì procedure/definition cần Điều ngoài NĐ 168 (TT 79 đăng ký, TT 35 kiểm định…).*
- `multi_frame_hint`: nếu ngữ cảnh có ≥2 (Điều, Khoản) khác nhau → ép liệt kê "### Trường hợp 1/2…".

**Bằng chứng (Mục 3.6):** 3 biến thể — không hướng dẫn / chỉ vai trò / **đầy đủ quy tắc**. Đầy đủ quy
tắc đạt **Citation-F1 0,279 = ×8,7 so không hướng dẫn (0,032)**, ×4,2 so chỉ vai trò; lại có **độ trễ
thấp nhất 7,1s** (quy tắc ép ngắn gọn) và **từ chối đúng lúc 0,20**. (Hai biến thể đầu F1 token cao hơn
vì "viết tuôn" nhưng không trích dẫn được.)

**Nói khi trình bày (~1 phút):** *"Bộ sinh dùng lệnh hướng dẫn gồm vai trò và 14 quy tắc bắt buộc: chỉ
dùng ngữ cảnh, mọi mức phạt phải kèm trích dẫn tới điểm khoản điều, copy nguyên văn số tiền, phân biệt
đúng điều theo loại xe, và ghép phạt tiền với trừ điểm qua tham chiếu chéo ba bước. Em còn tiêm gợi ý
động theo ý định và loại phương tiện. Kết quả Citation-F1 0,279, gấp 8,7 lần khi không có hướng dẫn."*

**Phản biện:** *Q: Prompt dài có hại tốc độ?* → ngược lại, quy tắc ép trả lời ngắn gọn nên P2 **nhanh
nhất** (7,1s). *Q: Vì sao F1 token của P2 thấp hơn?* → vì P2 không "viết tuôn"; trong pháp lý
Citation-F1 mới là thước đo thật.

---

## 17. Đầu ra có cấu trúc (Structured Output)  ·  *Mục 2.5.2 · `nodes.py` analyzer + generator*

**Cơ chế:** dùng `llm.with_structured_output(PydanticSchema)` — LLM trả về **JSON đã validate** theo lược
đồ (vd `AnalyzerOutput` với 5 trường; `GeneratedAnswer` với `answer` + `sources[]`). Loại bỏ hoàn toàn
bước parse chuỗi thủ công dễ lỗi. (Nội bộ: provider chuyển schema thành function-calling/JSON-mode.)

**Nói khi trình bày (~20s):** *"Cả nút phân tích và nút sinh đều dùng đầu ra có cấu trúc theo lược đồ
Pydantic, nên LLM trả JSON đã được kiểm tra hợp lệ, không phải bóc tách chuỗi thủ công."*

---

## 18. ⭐ Lọc trích dẫn (Citation Sanitation)  ·  *Mục 2.5.2 · `generator.py:_extract_cited_sources`*

**Vấn đề:** Dù prompt ràng buộc, LLM vẫn có thể **bịa một trích dẫn** (vd "Điều 99") không có trong ngữ
cảnh. Đây là **lớp phòng vệ cuối** trước khi tới người dùng.

**Cơ chế (3 bước, hậu xử lý — code thật):**
1. **Parse** mọi trích dẫn `[… Điều X, Khoản Y, Điểm Z — doc]` bằng regex → tuple `(doc_token, dieu,
   khoan, diem)` (Khoản/Điểm tùy chọn).
2. **Dựng tập văn bản hợp lệ** từ chính các đoạn đã truy xuất (`doc_id`, `ten_van_ban`) — để **không
   chấp nhận** tham chiếu tới văn bản LLM tự nghĩ ra.
3. **Lọc:** chỉ giữ đoạn nào có `(dieu, khoan, diem)` **khớp** một trích dẫn đã parse **VÀ** token văn
   bản khớp `doc_id/tên` của đoạn (khớp slug `168/2024`, hoặc tên dài ≥8 ký tự). Nếu **không parse được
   trích dẫn nào** → trả `[]` (không "đổ" cả top-k ra).

**Nói khi trình bày (~30s):** *"Lớp cuối cùng là lọc trích dẫn: em parse mọi trích dẫn trong câu trả
lời rồi đối chiếu với metadata của đúng các đoạn đã đưa vào ngữ cảnh. Nếu mô hình bịa một điều không có
trong ngữ cảnh, trích dẫn đó bị loại trước khi tới người dùng — đây là phòng vệ cuối chống ảo giác."*

**Phản biện:** *Q: Có loại nhầm trích dẫn đúng không?* → khớp theo (Điều, Khoản, Điểm) + slug văn bản
nên rất chặt; trường hợp không có trích dẫn nào (câu từ chối) thì trả rỗng là đúng. *Q: Đây có phải
"3 lớp phòng vệ"?* → đúng: (1) prompt ràng buộc — (2) lọc trích dẫn — (3) HITL cho web; cộng confidence
judge → kiểm soát ảo giác **theo chiều sâu** (đóng góp #3).

---
---

# PHẦN D — TÁC TỬ (AGENT) & HITL

## 19. Máy trạng thái LangGraph 6 nút  ·  *Mục 1.4.7 / 2.6 · `source/agent/graph.py`, `nodes.py`, `state.py`*

**Khái niệm cốt lõi (Bảng 2 báo cáo):** Nút (xử lý + cập nhật state) · Cạnh (cố định / **có điều kiện**) ·
Trạng thái dùng chung (`AgentState`) · **Checkpointer** (lưu snapshot mỗi bước) · `interrupt_before`
(dừng chờ phê duyệt).

**6 nút + định tuyến:** điểm vào `analyzer` → cạnh **có điều kiện** `route_by_category` rẽ theo `category`:
| category | nút đích |
|---|---|
| legal_rag | `legal_rag` (truy xuất lai + sinh) → END |
| chit_chat | `chit_chat` (mẫu tĩnh, không gọi LLM) → END |
| web_legal_search | `web_search` → `web_finalize` (HITL) → END |
| out_of_scope | `out_of_scope` (mẫu từ chối) → END |

**`AgentState` (TypedDict)** mang xuyên suốt: `query, raw_query, expanded_queries, chat_history,
category, intent, vehicle_type, multi_frame, chunks, answer, sources, refused, web_results,
warning_prefix, requires_approval, error`.

**Checkpointer (sự thật dễ sai):** **`SqliteSaver`** (dev) / **`AsyncSqliteSaver`** (FastAPI) /
**`AsyncPostgresSaver`** (production) — **KHÔNG phải MemorySaver**. Lưu snapshot vào DB tại mỗi bước nên
**hội thoại đa lượt** không mất trạng thái và **resume được sau khi tạm dừng**. Mỗi phiên một
**`thread_id`** riêng → cách ly đa người dùng. (Nhánh legal_rag → END trực tiếp; có `legal_fallback_router`
chuyển sang web khi từ chối/rỗng.)

**SSE streaming:** phát 3 loại sự kiện về giao diện: `token` (từng token), `cite` (danh sách trích dẫn
cuối), `pending` (đang chờ HITL).

**Nói khi trình bày (~1 phút):** *"Toàn bộ luồng trực tuyến là đồ thị LangGraph sáu nút, vào tại
analyzer, rồi cạnh có điều kiện định tuyến về bốn nhánh. Trạng thái dùng chung mang ngữ cảnh xuyên
suốt, và checkpointer lưu snapshot vào SQLite khi dev, Postgres khi production — nên hỗ trợ hội thoại
nhiều lượt và cho phép tạm dừng mà không mất trạng thái, mỗi phiên có thread_id riêng để cách ly người
dùng."*

**Phản biện:** *Q: Sao không viết if/else thường mà cần LangGraph?* → LangGraph cho **checkpointer +
interrupt** (HITL, hội thoại đa lượt, resume) sẵn có; tự viết sẽ phải tái hiện cơ chế lưu/resume state.
*Q: MemorySaver được không?* → không, MemorySaver in-memory mất state khi restart; production phải
Postgres để bền vững.

---

## 20. Con người trong vòng lặp (HITL)  ·  *Mục 1.4.9 / 2.6.3 · `nodes.py` web_search/web_finalize + `graph.py`*

**Vấn đề:** Khi kho nội bộ chưa phủ, hệ thống tra cứu web (Tavily). Thông tin internet **rủi ro** → cần
người kiểm duyệt trước khi trả về.

**Cơ chế:** analyzer xếp câu vào `web_legal_search` → nút `web_search` thu ≤3 đoạn trích Tavily → đồ thị
**`interrupt_before=["web_finalize"]`** (tạm dừng) → endpoint `/pending` trả danh sách phiên chờ → quản
trị viên ở `/admin` xem snippet + draft → **Approve** (resume `web_finalize`, gắn nhãn cảnh báo "tra
cứu internet") hoặc **Reject** (xóa phiên). Timeout client 10 phút; state vẫn lưu trong checkpointer để
xử lý muộn. (Dựa trên nguyên lý phê duyệt của WebGPT.)

**Nói khi trình bày (~40s):** *"Với câu cần tra cứu web, đồ thị dừng lại tại interrupt_before và chờ
phê duyệt. Quản trị viên xem các đoạn trích rồi duyệt hoặc từ chối — đảm bảo thông tin từ internet được
kiểm soát trước khi tới người dùng. Nhờ checkpointer, trạng thái không mất khi tạm dừng."*

**Phản biện:** *Q: HITL làm chậm trải nghiệm?* → chỉ áp cho nhánh web (hiếm, ~1% câu); nhánh corpus
chính trả ngay. *Q: Vì sao cần với pháp lý?* → thông tin sai trong tư vấn pháp lý gây hậu quả thật.

---
---

# PHẦN E — ĐÁNH GIÁ (METRICS)

## 21. Độ đo truy xuất: Recall@k, MRR, nDCG@10  ·  *Mục 2.7 / 3.x*

- **Recall@k** = (số khoản gold lọt vào top-k) / (tổng khoản gold). Đo "lấy đủ chưa". Khớp **cấp Khoản**
  (trùng cả doc_id+điều+khoản) và **cấp tài liệu** (chỉ doc_id).
- **MRR** (Mean Reciprocal Rank) = trung bình của $1/\text{rank}$ của kết quả đúng đầu tiên. Đo "đúng có
  sớm không".
- **nDCG@10** (Normalized Discounted Cumulative Gain): thưởng kết quả đúng ở thứ hạng cao, chiết khấu
  theo $\log_2(\text{rank}+1)$, chuẩn hóa về [0,1]. $\text{DCG@k}=\sum_{i=1}^{k}\frac{rel_i}{\log_2(i+1)}$,
  $\text{nDCG}=\text{DCG}/\text{IDCG}$.

**Số chốt:** hybrid R@10=0,324 → +mở rộng 0,581 → +reranker 0,481; MRR 0,329; nDCG@10 0,460.

---

## 22. Độ đo trích dẫn: Citation P / R / F1  ·  *Mục 2.7 / 3.6 — metric CỐT LÕI*

So tập trích dẫn LLM trả ra với **gold_citations** (cấp Khoản):
- **Precision** = trích đúng / tổng trích · **Recall** = trích đúng / tổng gold ·
  **F1** = $2PR/(P+R)$.

**Vì sao là metric cốt lõi (luận điểm xuyên suốt):** trong pháp lý người dùng phải **xác minh nguồn**.
**F1 token gây hiểu nhầm** (LLM "viết tuôn"/bịa nguồn vẫn cao F1) — minh chứng ở Mục 3.3, 3.6, 3.8.
Số chốt: P2 Citation-F1 **0,279** (×8,7 so zero-shot 0,032).

---

## 23. RAGAS — đánh giá bằng giám khảo LLM  ·  *Mục 1.4.11 / 3.9 · giám khảo gpt-4o-mini*

3 độ đo (chấm tự động bằng gpt-4o-mini làm trọng tài độc lập, n=35):
- **Độ chính xác ngữ cảnh (context precision) = 0,885** — trong 10 đoạn đưa vào, đa số liên quan
  (ngữ cảnh sạch).
- **Độ trung thực (faithfulness) = 0,550** — tỉ lệ khẳng định trong câu trả lời được ngữ cảnh hỗ trợ.
- **Độ phủ ngữ cảnh (context recall) = 0,452 = ĐIỂM NGHẼN** — truy xuất mới bắt ~45% thông tin cần thiết
  → lý do ưu tiên reranker + mở rộng truy vấn + Graph RAG.
- *answer_relevancy*: lỗi thư viện RAGAS 0.2 → không kết luận.

**Nói khi trình bày (~1 phút):** *"Em đánh giá độc lập bằng RAGAS với gpt-4o-mini làm giám khảo. Độ
chính xác ngữ cảnh 0,885 rất cao, xác nhận ngữ cảnh sạch. Độ trung thực 0,550. Nhưng độ phủ ngữ cảnh
chỉ 0,452 — đây mới là điểm nghẽn thật: truy xuất mới bắt khoảng 45% thông tin cần thiết, nên hướng
phát triển ưu tiên reranker GPU và Graph RAG."*

---
---

# CHEAT-SHEET — TÓM TẮT 1 DÒNG MỖI KỸ THUẬT (học thuộc nhanh)

| KT | 1 câu cốt lõi | Số chốt |
|---|---|---|
| Phân đoạn phân cấp | Biên đoạn = biên trích dẫn (Điều→Khoản→Điểm, làm giàu preamble) | 77,5%→11,7%; ×2,4 Cit-R |
| E5-base | Nhúng đa ngôn ngữ InfoNCE; prefix passage/query bắt buộc | R@10 0,424; sai prefix −30% |
| Qdrant HNSW | ANN + payload filter (lọc status tại DB) | 4.423×768d, ~26MB |
| BM25 | Từ khóa chính xác, k₁=1,5 b=0,75 | bù cho dense |
| Mở rộng truy vấn | 1–5 biến thể theo khung pháp lý | R@10 0,324→0,581 (+26) |
| RRF | Gộp theo thứ hạng, k=60, không cần normalize | 1/(60+rank) |
| Sibling enrichment | Kéo Khoản lân cận ±2 + đoạn "trừ điểm/bổ sung" | đặc trị NĐ 168 |
| Cross-reference | Regex + tra metadata kéo đúng đoạn được dẫn | chống loãng tham chiếu |
| Diversity cap | ≤3 đoạn/Khoản, sibling miễn trừ | ngữ cảnh trải nhiều khoản |
| Reranker | Cross-encoder đọc chung cặp (query,đoạn) | +48,5% R@10, ×156 latency |
| Lệnh hướng dẫn P2 | Vai trò + 14 quy tắc + gợi ý động (intent/vehicle) | Cit-F1 0,279 (×8,7), 7,1s |
| Lọc trích dẫn | Parse citation → chỉ giữ đoạn khớp metadata | chống bịa nguồn |
| LangGraph + HITL | Máy trạng thái + checkpointer + interrupt_before | Phân loại 0,975 |
| Metric cốt lõi | Citation-Recall, KHÔNG phải F1 token | F1 là "bẫy" |
| RAGAS | precision 0,885 / faith 0,550 / **recall 0,452 (nghẽn)** | giám khảo gpt-4o-mini |

> **Câu "chốt hạ" xuyên suốt (lặp lại khi trả lời phản biện):** *"Trong domain pháp lý, độ chính xác
> trích dẫn quan trọng hơn F1 token, vì người dùng phải xác minh được nguồn của từng thông tin."*
