# [kich_ban_ky_thuat_chuyen_sau.md](vscode-webview://0lqlr96qghopce52m75fa69501dcvco3rvlajh0v069rji0pc6sm/traffic_rag/doc/slides_v7/kich_ban_ky_thuat_chuyen_sau.md)Kịch bản thuyết trình — Bảo vệ ĐATN (30 phút)

> **Đề tài:** Xây dựng hệ thống hỏi đáp luật giao thông đường bộ Việt Nam ứng dụng LLM kết hợp RAG
> **SV:** Mạc Phú Phong — 106210059 — Lớp 21DTCLC1 · **GVHD:** TS. Trần Thị Minh Hạnh
> ĐH Đà Nẵng — Trường ĐH Bách Khoa — Khoa Điện tử - Viễn thông.
>
> Dùng kèm `Slide_BaoCao_MacPhuPhong_30phut.pptx` (30 slide). Lời thoại này đã có sẵn trong phần
> **ghi chú (speaker notes)** của từng slide. Mục tiêu: **29:50**, chừa ~1 phút buffer. Đã đồng bộ
> 100% với `BaoCao_MacPhuPhong_3.pdf` (số mục 3.x, số Hình 1–29, 3 đóng góp, Bảng 25/27).

## Mẹo canh giờ

- **Phần 1 (slide 1–7): 6:30** — nền tảng, đi nhanh, đừng sa đà lý thuyết.
- **Phần 2 (slide 8–15): 9:40** — trọng tâm kỹ thuật. Slide 11 (phân đoạn phân cấp) là đóng góp #1, nói kỹ.
- **Phần 3 (slide 16–25): 9:00** — trọng tâm kết quả. Mỗi thực nghiệm ~1 phút, bám 1–2 con số/biểu đồ.
- **Phần 4 (slide 26–30): 4:30** — chốt sản phẩm + kết luận, giữ năng lượng.
- Nếu cháy giờ: rút gọn slide 6, 9, 18, 21 (mỗi slide ~30s). Nếu dư giờ: demo trực tiếp ở slide 28.

---

## Bảng phân bổ thời gian

|   # | Slide                           | Mục/Hình báo cáo          | Thời lượng | Cộng dồn |
| --: | ------------------------------- | ------------------------- | ---------: | -------: |
|   1 | Bìa                             | —                         |       0:30 |     0:30 |
|   2 | Nội dung trình bày              | —                         |       0:20 |     0:50 |
|   3 | Đặt vấn đề & động lực           | 1.2                       |       1:10 |     2:00 |
|   4 | Mục tiêu · Phạm vi · 3 đóng góp | 1.3                       |       1:00 |     3:00 |
|   5 | RAG là gì                       | 1.4.5 · Hình 3            |       1:10 |     4:10 |
|   6 | Nền tảng: E5 / BM25 / RRF       | 1.4.3–1.4.6               |       1:20 |     5:30 |
|   7 | Hướng tác tử · LangGraph · HITL | 1.4.7–1.4.9               |       1:00 |     6:30 |
|   8 | Kiến trúc tổng thể (2 pipeline) | 2.2 · Hình 4–5            |       1:30 |     8:00 |
|   9 | Tech stack                      | Bảng 3                    |       0:30 |     8:30 |
|  10 | Bước 1 — Làm sạch PDF           | 2.3.1 · Bảng 4–5          |       1:00 |     9:30 |
|  11 | Bước 2 — Phân đoạn phân cấp     | 2.3.2 · Bảng 6–8          |       1:40 |    11:10 |
|  12 | Bước 3 — Nhúng & index Qdrant   | 2.3.3                     |       1:00 |    12:10 |
|  13 | Truy xuất lai 5 bước            | 2.4                       |       1:40 |    13:50 |
|  14 | Sinh văn bản — Lệnh hướng dẫn   | 2.5 · Bảng 10             |       1:00 |    14:50 |
|  15 | Agent LangGraph 6 nút + HITL    | 2.6 · Hình 7–8            |       1:20 |    16:10 |
|  16 | Thiết kế thực nghiệm            | 2.7 · Bảng 13             |       1:00 |    17:10 |
|  17 | Phân đoạn — then chốt           | Mục 3.3 · Hình 10         |       1:10 |    18:20 |
|  18 | Mô hình nhúng                   | Mục 3.4 · Hình 12         |       0:50 |    19:10 |
|  19 | Lệnh hướng dẫn (P0/P1/P2)       | Mục 3.6 · Hình 16         |       1:00 |    20:10 |
|  20 | Mở rộng truy vấn                | Mục 3.7.2 · Hình 19       |       1:00 |    21:10 |
|  21 | Xếp hạng lại & CSDL vector      | Mục 3.7.3 & 3.5 · Hình 21 |       0:50 |    22:00 |
|  22 | So sánh kiến trúc & 'bẫy' F1    | Mục 3.8 · Hình 22         |       1:10 |    23:10 |
|  23 | RAGAS (gpt-4o-mini)             | Mục 3.9 · Hình 23         |       1:00 |    24:10 |
|  24 | Chi phí & hiệu năng             | Mục 3.10 · Hình 24        |       0:50 |    25:00 |
|  25 | Tổng hợp quyết định             | Bảng 25                   |       0:20 |    25:20 |
|  26 | Triển khai production + CI/CD   | 4.2 · Hình 27             |       1:20 |    26:40 |
|  27 | Giao diện sản phẩm              | 4.3 · Hình 28–29          |       0:50 |    27:30 |
|  28 | Minh hoạ thực tế                | 4.4                       |       1:00 |    28:30 |
|  29 | Kết luận                        | 4.5 · Bảng 27             |       1:00 |    29:30 |
|  30 | Cảm ơn & Q&A                    | —                         |       0:20 |    29:50 |

---

## Kịch bản chi tiết

### Slide 1 — Bìa _(0:30)_

Kính chào quý thầy cô trong hội đồng. Em là Mạc Phú Phong, xin trình bày đồ án tốt nghiệp: **Xây
dựng hệ thống hỏi đáp luật giao thông đường bộ Việt Nam ứng dụng LLM kết hợp RAG**, dưới sự hướng
dẫn của cô Trần Thị Minh Hạnh. Báo cáo gồm 4 phần: bối cảnh & cơ sở lý thuyết, thiết kế hệ thống,
thực nghiệm đánh giá, và triển khai – kết luận.

### Slide 2 — Nội dung _(0:20)_

Phần một là bối cảnh và lý do chọn kiến trúc RAG. Phần hai là thiết kế hệ thống end-to-end. Phần ba
là mười thực nghiệm đánh giá. Phần cuối là triển khai thực tế và kết luận. Trọng tâm nằm ở phần hai
và ba.

### Slide 3 — Đặt vấn đề _(1:10)_

Pháp luật giao thông Việt Nam vừa trải qua đợt cải cách lớn nhất nhiều năm: Luật 35 và 36/2024, đặc
biệt **Nghị định 168/2024** tăng mạnh mức phạt và lần đầu áp dụng **trừ điểm giấy phép lái xe**.
Nhu cầu tra cứu chính xác tới điểm/khoản/điều tăng vọt. Nhưng cổng pháp luật chính thức buộc người
dùng tự biết tên văn bản và số điều; còn LLM thương mại chưa cập nhật corpus 2024–2026 nên hay bịa
nguồn, nhầm mức phạt — gọi là **ảo giác pháp lý**, rất nguy hiểm. RAG giải quyết bằng cách truy xuất
văn bản gốc đưa vào ngữ cảnh trước khi sinh, nhờ đó kiểm soát được nguồn trích dẫn.

### Slide 4 — Mục tiêu · Phạm vi · 3 đóng góp _(1:00)_

Mục tiêu: trợ lý hỏi đáp tiếng Việt trả lời đúng theo văn bản gốc với trích dẫn tới cấp điểm, khoản,
điều. Phạm vi gồm **24 văn bản** còn hiệu lực — 2 luật, 5 nghị định, 17 thông tư, trong đó có bổ
sung **Nghị định 81/2026 về đường sắt** để bao quát các quy định giao cắt đường bộ – đường sắt;
ngoài phạm vi là đường thuỷ, hàng không và tư vấn cá nhân hoá. Đề tài có **ba đóng góp chính**: thứ
nhất là thuật toán phân đoạn phân cấp theo cấu trúc Điều–Khoản–Điểm với năm cải tiến; thứ hai là quy
trình truy xuất lai đa truy vấn đặc thù cho văn bản pháp luật kèm bộ đánh giá 40 câu trên corpus
giao thông 2024–2026 chưa từng được xử lý; thứ ba là kiến trúc kiểm soát ảo giác pháp lý theo chiều
sâu gồm ba lớp — lệnh hướng dẫn ràng buộc trích dẫn, lọc trích dẫn sau sinh, và con người trong
vòng lặp.

### Slide 5 — RAG là gì _(1:10)_ · Hình 3

RAG hoạt động qua các bước: mã hoá truy vấn thành vector, tìm các đoạn văn gần nhất trong kho tài
liệu được kiểm soát, ghép vào ngữ cảnh rồi để LLM sinh câu trả lời kèm trích dẫn. Ưu điểm: cập nhật
văn bản mà không phải huấn luyện lại, và kiểm soát được nguồn nên hạn chế ảo giác. Với pháp luật
Việt Nam có ba thách thức đặc thù: cấu trúc phân cấp Điều–Khoản–Điểm, truy vấn ngắn mơ hồ, và câu
hỏi đa ý định.

### Slide 6 — Nền tảng kỹ thuật _(1:20)_

Truy xuất lai kết hợp hai nhánh bổ sung. Nhánh **dày đặc** dùng embedding E5-base 768 chiều, bắt
tương đồng ngữ nghĩa — "vượt đèn đỏ" và "không chấp hành hiệu lệnh đèn" tuy khác từ nhưng cùng
nghĩa. Nhánh **thưa** dùng BM25 dựa trên tần suất từ, rất mạnh khi truy vấn chứa mã chính xác như
"Điều 7" hay "Nghị định 168". Hai danh sách được hợp nhất bằng **RRF k=60** — gộp không cần chuẩn
hoá điểm, ổn định cho kho dưới mười nghìn tài liệu.

### Slide 7 — Hướng tác tử · LangGraph · HITL _(1:00)_

Đề tài dùng **LangGraph** — mô hình hoá ứng dụng LLM thành máy trạng thái với các nút, cạnh có điều
kiện, trạng thái dùng chung và bộ lưu điểm kiểm tra. Điểm quan trọng: checkpointer lưu bền vào
SQLite khi phát triển và Postgres khi production, nên hỗ trợ hội thoại nhiều lượt và cho phép tạm
dừng đồ thị chờ phê duyệt mà không mất trạng thái. **Con người trong vòng lặp** dùng để kiểm duyệt
thông tin tra cứu web. Khoảng trống nghiên cứu: chưa ai xử lý corpus giao thông 2024–2026 với cấu
trúc phân cấp và cơ chế trừ điểm mới — đây là chỗ đề tài lấp đầy.

### Slide 8 — Kiến trúc tổng thể _(1:30)_ · Hình 4–5

Hệ thống tách thành hai pipeline. **Ngoại tuyến** chỉ chạy khi thêm văn bản: 24 PDF được làm sạch,
phân đoạn phân cấp, nhúng E5-base rồi nạp vào Qdrant — collection 4.423 đoạn, kèm cache BM25. **Trực
tuyến** xử lý mỗi câu hỏi: nút phân tích phân tích ý định và mở rộng truy vấn, bộ định tuyến điều
hướng về một trong bốn nhánh — hỏi đáp pháp luật, hội thoại thường, tra cứu web có phê duyệt, ngoài
phạm vi. Nhánh hỏi đáp pháp luật truy vấn Qdrant rồi sinh câu trả lời, phát về giao diện theo từng
token qua SSE.

### Slide 9 — Tech stack _(0:30)_ · Bảng 3

Công nghệ: LangGraph điều phối agent; Gemini 3.1 Flash Lite sinh câu trả lời; embedding e5-base;
vector DB Qdrant; BM25 với tách từ pyvi; backend FastAPI streaming SSE; frontend Next.js 14; đánh
giá độc lập bằng gpt-4o-mini qua RAGAS; tra cứu web Tavily; LangSmith phân tích chi phí.

### Slide 10 — Bước 1: Làm sạch PDF _(1:00)_ · Bảng 4–5

Văn bản pháp luật là PDF layout phức tạp: bảng mức phạt, header/footer lặp, câu bị ngắt giữa chừng.
Em viết **ba chương trình làm sạch** riêng cho luật, nghị định, thông tư, dùng pdfplumber, lọc
header/footer, nối câu bị ngắt, bảo toàn bảng. Quan trọng là trường **status**: chỉ văn bản còn hiệu
lực mới được index; văn bản bị bãi bỏ như Luật GTĐB 2008 thì bỏ; văn bản bị bãi bỏ một phần như NĐ
100/2019 thì loại mảng đường bộ. Sau khi loại 2 văn bản hết hiệu lực, còn 24 văn bản được index.

### Slide 11 — Bước 2: Phân đoạn phân cấp _(1:40)_ — **ĐÓNG GÓP CỐT LÕI** · Bảng 6–8

Đây là đóng góp cốt lõi. Bộ phân đoạn tách văn bản theo đúng cấu trúc pháp lý ba cấp: Điều ngắn dưới
1.000 token giữ nguyên một đoạn; dài thì tách theo Khoản; Khoản dài hoặc nghị định nhiều Điểm thì
tách xuống từng **Điểm**. Năm cải tiến: giữ lại Điểm ngắn, hạ ngưỡng cho nghị định, ép tách khi nghị
định có ≥2 điểm, sửa mẫu nhận diện bắt cả "đ)" và điểm sau dấu chấm phẩy, và đặc biệt **làm giàu** —
mỗi đoạn Điểm tự mang theo phần mở đầu Khoản như "Phạt tiền từ X đến Y đồng đối với…". Nhờ vậy tỉ lệ
điểm của NĐ 168 không có đoạn riêng giảm từ **77,5% xuống 11,7%**. Cái hay: **biên đoạn trùng biên
trích dẫn** — lấy đúng đoạn là tự có đúng khoản, điểm để trích dẫn. Tổng 4.423 đoạn, 60% là cấp Điểm.

### Slide 12 — Bước 3: Nhúng & index _(1:00)_

E5-base mã hoá toàn bộ 4.423 đoạn. Yêu cầu bắt buộc của họ E5 là tiền tố: "passage:" khi nhúng đoạn
và "query:" khi nhúng truy vấn — dùng sai làm giảm ~30% recall (vấn đề kỹ thuật đầu tiên gặp).
Collection Qdrant: 768 chiều, cosine, HNSW. Ba chỉ mục payload trên định danh văn bản, status, chủ
đề để lọc nhanh. Mỗi đoạn mang siêu dữ liệu tới cấp điểm khoản điều. Cache BM25 tách từ sẵn bằng pyvi
nên khởi động lại dưới nửa giây. Toàn bộ index chỉ ~26 MB.

### Slide 13 — Truy xuất lai 5 bước _(1:40)_

Mỗi câu hỏi chạy năm bước. (1) Nút phân tích **mở rộng** truy vấn thành 1–5 biến thể và giải tham
chiếu hội thoại. (2) chạy song song nhánh **dày đặc** E5 — đã lọc status active ngay tại Qdrant — và
nhánh **thưa** BM25, mỗi nhánh top-2k. (3) **RRF** hợp nhất ra top-30 không trùng. (4) phần đặc thù
pháp luật: **làm giàu lân cận** kéo các khoản N±2 cùng Điều, và **giải tham chiếu chéo** dùng mẫu
chính quy tìm "Điều X Khoản Y" được nhắc tới rồi kéo luôn vào — đặc trị NĐ 168, nơi mức phạt tiền và
mức trừ điểm ở hai khoản khác nhau. Cuối cùng áp giới hạn đa dạng nguồn rồi cắt mười đoạn. Bước năm
là xếp hạng lại tuỳ chọn, mặc định tắt.

### Slide 14 — Sinh văn bản: Lệnh hướng dẫn _(1:00)_ · Bảng 10

Bộ sinh dùng **lệnh hướng dẫn đầy đủ quy tắc** (biến thể P2) gồm vai trò và 14 quy tắc bắt buộc. Vài
quy tắc quan trọng: chỉ dùng ngữ cảnh; mọi mức phạt phải kèm trích dẫn tới điểm khoản điều; thiếu
thông tin thì từ chối thay vì bịa; phân biệt đúng điều theo loại phương tiện — Điều 6 ô tô, Điều 7
mô tô; ghép mức phạt tiền với mức trừ điểm qua tham chiếu chéo. Gemini trả JSON có cấu trúc theo
Pydantic. Cuối cùng là **lọc trích dẫn**: so từng trích dẫn LLM trả về với siêu dữ liệu đúng 10 đoạn
trong lệnh; nếu LLM bịa một điều không có trong ngữ cảnh thì loại bỏ — lớp phòng vệ cuối chống ảo
giác trích dẫn.

### Slide 15 — Agent LangGraph 6 nút _(1:20)_ · Hình 7–8

Luồng trực tuyến điều phối bởi đồ thị LangGraph sáu nút, vào tại nút phân tích. Nút phân tích phân
tích ý định, loại xe, mở rộng truy vấn; bộ định tuyến điều hướng bốn nhánh. Nút hỏi đáp pháp luật
chạy truy xuất lai và sinh câu trả lời rồi kết thúc. Hội thoại thường và ngoài phạm vi dùng mẫu tĩnh.
Nhánh **tra cứu web** gọi Tavily lấy ≤3 đoạn trích rồi đồ thị **tạm dừng** tại nút tổng hợp kết quả
web nhờ interrupt_before — cơ chế con người trong vòng lặp. Quản trị viên vào trang quản trị, xem
đoạn trích, rồi phê duyệt để tiếp tục (gắn nhãn cảnh báo tra cứu internet) hoặc từ chối. Timeout 10
phút. Nhờ checkpointer, trạng thái không mất; mỗi phiên có định danh riêng đảm bảo cách ly nhiều
người dùng.

### Slide 16 — Thiết kế thực nghiệm _(1:00)_ · Bảng 13

Bộ đánh giá **40 câu / 8 danh mục** thực tiễn, mỗi danh mục năm câu — Phạt đơn nghĩa, Đa ý định,
Tham chiếu chéo, Thủ tục, Ngoài phạm vi, Ngắn đa nghĩa, Đa hành vi, Ràng buộc phương tiện. Mỗi câu
có đáp án chuẩn và trích dẫn chuẩn cấp khoản. Năm câu ngoài phạm vi được lọc khỏi đo truy xuất, giữ
35 câu pháp lý. Bộ metric đa chiều: Recall/MRR/nDCG; **citation P/R/F1** là metric cốt lõi;
token-F1/ROUGE; độ chính xác phân loại; RAGAS với gpt-4o-mini; chi phí, độ trễ, tỉ lệ lỗi. Mười thực
nghiệm chạy từ dưới lên, mỗi lần đổi một biến.

### Slide 17 — Phân đoạn (Mục 3.3) _(1:10)_ — **THEN CHỐT** · Hình 10

Thực nghiệm phân đoạn so sánh phân cấp với cắt cố định 512 token. Kết quả quan trọng nhất:
**Citation-Recall cấp khoản 0,410 so với 0,171 — gấp 2,4 lần**. Lý do: biên đoạn trùng biên trích
dẫn nên lấy đúng đoạn là có đúng khoản. Đáng chú ý, F1 token gần hoà (0,327 vs 0,325), vì cố định
512 tạo đoạn dài chứa nhiều n-gram trùng đáp án nhưng không trích dẫn đúng được — đây là lý do trong
domain pháp lý, **Citation-Recall mới là thước đo thực chất**, không phải F1.

### Slide 18 — Mô hình nhúng (Mục 3.4) _(0:50)_ · Hình 12

Thực nghiệm mô hình nhúng so sánh bốn mô hình. **e5-base thắng mọi chỉ số chất lượng, Recall@10 =
0,424**. e5-small rẻ và nhanh nhất, kém ~5 điểm, là phương án dự phòng. mpnet thua vì max_seq chỉ
128, cắt mất khoản dài. sbert yếu dù cũng 768 chiều vì không huấn luyện đối chiếu InfoNCE như E5. Em
chọn e5-base vì chất lượng cao nhất và giữ nguyên 768 chiều khớp collection đã có.

### Slide 19 — Lệnh hướng dẫn (Mục 3.6) _(1:00)_ · Hình 16

Thực nghiệm so ba biến thể lệnh hướng dẫn. Biến thể không hướng dẫn và chỉ khai báo vai trò có F1
token cao hơn vì viết tuôn nhiều chữ trùng đáp án, nhưng gần như không trích dẫn được. Biến thể
**đầy đủ 14 quy tắc** (P2 trong biểu đồ) đạt Citation-F1 0,279 — gấp 8,7 lần không hướng dẫn và 4,2
lần chỉ vai trò. Thêm hai hiệu ứng tốt: độ trễ thấp nhất 7,1 giây vì quy tắc ép trả lời ngắn gọn, và
tỷ lệ từ chối 0,20 nghĩa là biết từ chối đúng lúc thay vì bịa.

### Slide 20 — Mở rộng truy vấn (Mục 3.7.2) _(1:00)_ · Hình 19

Đây là kỹ thuật tác động lớn nhất: **Recall@10 từ 0,324 lên 0,581, cộng 26 điểm phần trăm** — mức
tăng lớn nhất toàn bộ thực nghiệm. nDCG tăng 14 điểm, độ trễ chỉ tăng 7%. Kết quả này đảo chiều so
kho cũ: khi đoạn còn dài, mở rộng truy vấn tạo nhiễu; với kho hiện tại đoạn cấp Điểm nhỏ và chính
xác, mở rộng truy vấn trở nên rất hữu ích. Vì vậy hệ thống bật tính năng này.

### Slide 21 — Xếp hạng lại & CSDL vector (Mục 3.7.3 & 3.5) _(0:50)_ · Hình 21

Thực nghiệm xếp hạng lại: bộ mã hoá chéo kéo **Recall@10 lên 0,481, tăng 48,5%** — lôi được khoản
đúng ở hạng 11–20 lên top 10. Nhưng cái giá là **độ trễ tăng 156 lần** trên CPU, ~110 giây, không
chấp nhận được real-time, nên tắt mặc định, chỉ bật khi có GPU. Thực nghiệm cơ sở dữ liệu vector so
Qdrant/Chroma/FAISS: recall xấp xỉ nhau; chênh độ trễ vài mili giây không đáng so với LLM. Em giữ
**Qdrant** vì có lọc payload tại tầng DB, HTTP API và cập nhật online.

### Slide 22 — So sánh kiến trúc & 'bẫy' F1 (Mục 3.8) _(1:10)_ · Hình 22

Thực nghiệm so ba kiến trúc trên đủ 40 câu. **RAG hướng tác tử đạt độ chính xác phân loại 0,975** —
đúng 39/40 câu, định tuyến 100% câu ngoài phạm vi — so với RAG cơ bản 0,875. Cần giải thích kỹ: kiến
trúc chỉ dùng Gemini lại có F1 cao nhất 0,370, nhưng đây là **sai lệch của metric**. Thứ nhất, với
câu ngoài phạm vi, Gemini từ chối tự nhiên, n-gram trùng nhiều với đáp án vốn cũng là câu từ chối.
Thứ hai, Gemini bịa nguồn pháp lý dài chi tiết, kéo F1 lên nhờ trùng từ bề mặt nhưng trích dẫn sai
hoàn toàn — đúng là ảo giác pháp lý. Vì vậy độ chính xác phân loại và Citation-Recall mới phản ánh
chất lượng thật. Độ trễ tác tử 52 giây bị thổi phồng do giãn cách 2 giây; thực tế ~20–25 giây.

### Slide 23 — RAGAS (Mục 3.9) _(1:00)_ · Hình 23

Thực nghiệm dùng RAGAS với gpt-4o-mini làm giám khảo độc lập. **Độ chính xác ngữ cảnh 0,885 rất
cao**, xác nhận các đoạn đưa vào ngữ cảnh đa số liên quan, ít nhiễu. **Độ trung thực 0,550**, hơn
nửa khẳng định được ngữ cảnh hỗ trợ. Nhưng **độ phủ ngữ cảnh chỉ 0,452 — đây mới là điểm nghẽn
thật**: truy xuất mới bắt ~45% thông tin cần thiết. Chính vì vậy xếp hạng lại và mở rộng truy vấn
được ưu tiên. Riêng độ liên quan câu trả lời bị lỗi thư viện RAGAS 0.2 nên không kết luận.

### Slide 24 — Chi phí & hiệu năng (Mục 3.10) _(0:50)_ · Hình 24

Thực nghiệm phân tích từ 1.964 lượt chạy trên hệ thống truy vết. Mỗi câu tốn **~0,0034 đô**, nút hỏi
đáp pháp luật chiếm 77%; mười nghìn truy vấn/tháng chỉ ~46 đô tiền LLM. Độ trễ trung bình 33 giây có
giãn cách, thực tế 20–25 giây. Điểm lưu ý: tỉ lệ lỗi Gemini **14,6%** do bản preview free tier bị
giới hạn tần suất; chuyển sang bản stable trả phí sẽ giảm xuống dưới 2%.

### Slide 25 — Tổng hợp quyết định (Bảng 25) _(0:20)_

Bảng này tổng hợp mười thực nghiệm: mỗi thành phần thiết kế đều có bằng chứng định lượng kèm số mục
trong báo cáo. Bốn dòng tô đậm là quyết định tác động lớn nhất: phân đoạn phân cấp, mở rộng truy
vấn, lệnh hướng dẫn đầy đủ quy tắc, và kiến trúc hướng tác tử.

### Slide 26 — Triển khai production + CI/CD _(1:20)_ · Hình 27

Hệ thống đã triển khai thực tế trên đám mây miễn phí, **hạ tầng 0 đồng/tháng**. Frontend Next.js
trên Vercel, backend FastAPI + LangGraph trên Hugging Face Spaces, kho vector Qdrant Cloud, xác thực
và checkpoint dùng chung Supabase Postgres; Gemini và Tavily là dịch vụ ngoài. Cùng một codebase và
Docker image phục vụ cả local lẫn cloud, chỉ khác biến môi trường. Về chất lượng, **CI/CD** trên
GitHub Actions có job smoke và job regression — bắt buộc Recall@10 trung bình ≥ 0,40, không đoạn đã
bãi bỏ lọt vào, độ trễ dưới 2 giây; fail thì chặn hợp nhất mã, ngăn cập nhật corpus làm tụt chất
lượng.

### Slide 27 — Giao diện sản phẩm _(0:50)_ · Hình 28–29

Giao diện chat trên Next.js 14: câu trả lời hiện dần từng token qua SSE, có **bảng trích dẫn** hiển
thị chính xác điều khoản điểm và mã văn bản để người dùng tự kiểm chứng, hỗ trợ hội thoại nhiều
lượt, gắn nhãn cảnh báo cho câu trả lời từ internet. Bên cạnh là **trang quản trị HITL**: quản trị
viên thấy danh sách phiên chờ, xem câu hỏi và đoạn trích Tavily rồi phê duyệt hoặc từ chối.
_(Hai ảnh chụp màn hình thực tế — Hình 28 và 29 — sẽ chèn khi đóng quyển.)_

### Slide 28 — Minh hoạ thực tế _(1:00)_

Hai ví dụ thực từ bộ đánh giá. Câu "vượt đèn đỏ đi xe máy phạt bao nhiêu" (phạt đơn nghĩa): hệ thống
trả đúng 800 nghìn–1 triệu, trích dẫn đúng Điều 7 Khoản 4 điểm p; nhờ làm giàu lân cận còn kéo thêm
Khoản 11 về trừ 2 điểm — đầy đủ hơn cả đáp án chuẩn tối thiểu. Câu đa ý định "vượt đèn đỏ và không
mang bằng lái": hệ thống tách đúng hai hành vi, trích dẫn đúng hai điều khoản, in đậm số tiền và
cộng tổng — đúng quy tắc 6 của lệnh hướng dẫn. _(Nếu hội đồng cho phép, em có thể demo trực tiếp.)_

### Slide 29 — Kết luận _(1:00)_ · Bảng 27

Tổng kết: **R@10 0,581** (×2,9 so phân đoạn cố định), **Citation-F1 0,279** (×8,7 so không hướng
dẫn), **độ chính xác phân loại 0,975**, độ chính xác ngữ cảnh 0,885, chi phí 0,0034 đô/câu, đã chạy
thực tế miễn phí với cổng CI. Hạn chế: bộ đánh giá nhỏ, Gemini free tier lỗi 14,6%, reranker cần
GPU, độ phủ ngữ cảnh 0,452 vẫn là điểm nghẽn, độ liên quan câu trả lời lỗi thư viện. Hướng phát
triển ba tầm: **ngắn hạn** mở rộng eval 80–100 câu, chuyển Gemini sang stable trả phí, tối ưu chi
phí; **trung hạn** bật reranker GPU, xây **Graph RAG** trên đồ thị tri thức pháp luật để nâng độ phủ
ngữ cảnh, đánh giá lại bằng giám khảo mạnh hơn; **dài hạn** tiến tới kiến trúc đa tác tử với tác tử
truy xuất đa bước.

### Slide 30 — Cảm ơn _(0:20)_

Phần trình bày của em đến đây là hết. Em cảm ơn quý thầy cô đã lắng nghe và rất mong nhận được câu
hỏi, góp ý từ hội đồng.

---

## Phụ lục — Câu hỏi hội đồng có thể hỏi (chuẩn bị trước)

- **Vì sao Citation-Recall quan trọng hơn F1?** Vì F1 token chỉ đo trùng n-gram bề mặt; LLM có thể
  "viết tuôn" hoặc bịa nguồn dài để F1 cao nhưng trích dẫn sai (xem Mục 3.3, 3.6, 3.8).
- **Vì sao tắt xếp hạng lại dù +48,5% recall?** Độ trễ ×156 trên CPU (~110s) không chấp nhận
  real-time; bật lại khi có GPU (dự kiến ~2–3s).
- **Độ phủ ngữ cảnh 0,452 thấp — xử lý thế nào?** Đây là điểm nghẽn đã nhận diện; hướng: reranker
  GPU, Graph RAG, mở rộng corpus (Bảng 27).
- **Vì sao Gemini lỗi 14,6%?** Bản preview free tier giới hạn tần suất; production chuyển stable trả
  phí → dưới 2%.
- **Số liệu có đáng tin với 35 câu?** Đây là hạn chế đã nêu; ưu tiên ngắn hạn mở rộng lên 80–100 câu.
- **Khác gì RAG pháp luật tiếng Việt trước đây?** Lấp khoảng trống corpus giao thông 2024–2026, cấu
  trúc Điều→Khoản→Điểm, trừ điểm GPLX, phân đoạn đặc thù + giải tham chiếu chéo + agent HITL.
- **Vì sao đưa Nghị định 81/2026 đường sắt vào dù đề tài là đường bộ?** Để bao quát các quy định
  giao cắt đường bộ – đường sắt mà người tham gia giao thông đường bộ phải tuân thủ.
