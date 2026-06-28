# KẾT LUẬN

Đồ án đã hoàn thành mục tiêu xây dựng một hệ thống trợ lý hỏi đáp pháp luật giao thông đường bộ Việt Nam dựa trên kiến trúc sinh văn bản tăng cường truy xuất hướng tác từ. Hệ thống giải quyết trực tiếp bài toán đặt ra ở chương 1: trả lời câu hỏi pháp luật bằng tiếng Việt với độ chính xác cao, căn cứ vào văn bản gốc và trích dẫn được tới cấp điểm, khoản, điều của từng văn bản trong kho dữ liệu giai đoạn 2024–2026.

Về mặt phương pháp, chương 2 đã trình bày một quy trình hoàn chỉnh gồm bộ phân đoạn phân cấp theo cấu trúc Điều–Khoản–Điểm, quy trình truy xuất lai năm bước kết hợp truy xuất dày đặc và thưa, mở rộng truy vấn, làm giàu ngữ cảnh lân cận, giải tham chiếu chéo, lệnh hướng dẫn với mười bốn quy tắc cùng lớp lọc trích dẫn, và máy trạng thái LangGraph sáu nút tích hợp HITL. Mỗi thành phần đều được đặt trên nền tảng lý thuyết vững chắc và gắn với một quyết định thiết kế cụ thể.

Về mặt thực nghiệm, chương 3 đã cung cấp bằng chứng định lượng cho từng quyết định thông qua mười thực nghiệm đối chứng. Ba phát hiện nổi bật là: phân đoạn phân cấp là yếu tố then chốt nhất cho độ chính xác trích dẫn; mở rộng truy vấn và lệnh hướng dẫn đầy đủ quy tắc là hai kỹ thuật mang lại hiệu quả cao nhất so với chi phí; và độ phủ ngữ cảnh vẫn là nút thắt chính cần cải thiện. Chương 4 đã đưa các kết quả này vào sản phẩm triển khai thực tế trên đám mây, với giao diện hỏi đáp, bảng điều khiển quản trị và minh họa qua sáu loại câu hỏi đại diện.

Bên cạnh các kết quả đạt được, hệ thống vẫn còn một số hạn chế đã được phân tích, đặc biệt là quy mô bộ đánh giá còn nhỏ, tỷ lệ lỗi của mô hình sinh ở bản xem trước và độ phủ ngữ cảnh chưa cao. Các hạn chế này định hướng cho lộ trình phát triển tiếp theo, bao gồm mở rộng bộ đánh giá, chuyển sang mô hình ổn định trả phí, kích hoạt bộ xếp hạng lại trên GPU và hướng tới tác từ truy xuất đa bước. Nhìn chung, đề tài đã chứng minh tính khả thi của việc kết hợp kiến trúc hướng tác từ với thiết kế phân đoạn và lệnh hướng dẫn đặc thù nhằm đạt độ chính xác trích dẫn cao trong lĩnh vực pháp luật tiếng Việt, đặt nền tảng cho việc phát triển các trợ lý pháp lý đáng tin cậy phục vụ người dùng phổ thông.

---

## Kết quả đạt được

Đề tài đã xây dựng thành công hệ thống **Traffic-RAG v7.0** — trợ lý hỏi đáp pháp luật giao thông Việt Nam dựa trên kiến trúc Agentic RAG, với các kết quả định lượng được xác minh qua mười thực nghiệm đối chứng RQ1–RQ10 trên bộ đánh giá 40 câu hỏi thực tiễn.

**Về chất lượng truy xuất:** Quy trình truy xuất lai kết hợp mở rộng truy vấn đạt Recall@10 = 0,581 — cải thiện 2,9 lần so với cơ sở phân đoạn cố định 512 token (0,200). Bộ xếp hạng lại bge-reranker-v2-m3 có thể nâng Recall@10 lên 0,481 (tăng 48,5% so với không dùng) khi tài nguyên GPU cho phép.

**Về độ chính xác trích dẫn:** Lệnh hướng dẫn P2 kết hợp với phân đoạn phân cấp và lọc trích dẫn đạt Citation-F1 = 0,279 — gấp 8,7 lần phương án không hướng dẫn (0,032) và gấp 4,2 lần phương án chỉ khai báo vai trò (0,067). Đây là cải thiện có ý nghĩa nhất so với RAG thông thường vì bảo đảm người dùng có thể xác minh nguồn gốc từng thông tin.

**Về kiến trúc hệ thống:** Luồng công việc hướng tác từ trên nền LangGraph đạt độ chính xác phân loại 0,975 (đúng 39 trên 40 câu), định tuyến đúng toàn bộ câu ngoài phạm vi. Mô hình giám khảo RAGAS xác nhận độ chính xác ngữ cảnh đạt 0,885, cho thấy ngữ cảnh đưa vào mô hình đa số liên quan và ít nhiễu.

**Về vận hành:** Chi phí ước tính khoảng 0,00344 USD mỗi câu, tương đương khoảng 46 USD mỗi tháng cho 10.000 truy vấn. Hệ thống tự động kiểm tra hồi quy điều kiện Recall@10 không thấp hơn 0,40 qua quy trình tích hợp liên tục, và có thể triển khai trong khoảng mười phút trên bất kỳ máy nào có Docker.

---

## Hạn chế

Hệ thống còn sáu hạn chế chính:

1. **Bộ đánh giá còn nhỏ:** 35 câu pháp lý với 4–5 câu mỗi danh mục tạo ra khoảng tin cậy rộng. Độ chính xác phân loại tổng thể ổn định, nhưng các độ đo theo từng danh mục có thể bị ảnh hưởng lớn bởi một, hai câu bất thường.

2. **Tỷ lệ lỗi của mô hình sinh cao:** Bản xem trước trên gói miễn phí bị giới hạn tần suất nghiêm ngặt, làm đội độ trễ (do phải giãn cách lời gọi) và gây thất bại khoảng 15% truy vấn. Hệ thống triển khai thực tế cần dùng bản ổn định có trả phí.

3. **Bộ xếp hạng lại chưa khả dụng trong vận hành thực tế:** Mức cải thiện 48,5% Recall@10 là đáng kể nhưng độ trễ tăng 156 lần trên CPU là không thể chấp nhận. Cần GPU để khai thác cải thiện này.

4. **Độ phủ ngữ cảnh 0,452 là điểm nghẽn:** Mô hình giám khảo RAGAS xác nhận truy xuất chỉ bắt được khoảng 45% thông tin cần thiết từ câu trả lời chuẩn. Quy trình truy xuất cần cải thiện thêm (truy vấn nhiều bước, xếp hạng lại trên GPU, mở rộng kho văn bản).

5. **Một độ đo của RAGAS không đo được:** Lỗi của thư viện RAGAS 0.2 khiến độ liên quan câu trả lời lỗi toàn bộ, làm thiếu một độ đo quan trọng đánh giá mức độ câu trả lời bám sát câu hỏi.

6. **Tỷ lệ từ chối 20%:** Một phần câu từ chối thực ra có thông tin trong kho nhưng truy xuất bỏ sót. Giảm các trường hợp từ chối sai cần cải thiện độ phủ truy xuất, đặc biệt cho danh mục tham chiếu chéo và đa hành vi.

---

## Hướng phát triển

Hệ thống hiện tại được tổ chức theo kiến trúc một tác từ đơn, trong đó một đồ thị trạng thái điều phối tuần tự các bước phân tích, truy xuất và sinh câu trả lời. Đây là điểm xuất phát hợp lý cho một hệ thống tư vấn pháp luật ưu tiên độ tin cậy, đồng thời mở ra lộ trình phát triển trên ba tầm: ngắn hạn tập trung vào vận hành và chi phí, trung hạn nâng cấp năng lực truy xuất, và dài hạn tiến tới kiến trúc tác từ nâng cao.

*Bảng KL.1. Hướng phát triển theo mức ưu tiên*

| Tầm | Hướng phát triển | Tác động dự kiến |
|---|---|---|
| **Ngắn hạn** | Mở rộng eval set lên 80–100 câu (10 câu/danh mục) | Thu hẹp khoảng tin cậy, phát hiện danh mục yếu |
| **Ngắn hạn** | Chuyển mô hình sinh sang bản ổn định có trả phí | Giảm tỷ lệ lỗi từ 14,6% xuống dưới 2% |
| **Ngắn hạn** | Tối ưu chi phí: lưu đệm lệnh hướng dẫn, phân tầng mô hình, gộp lô mã hoá, mở rộng truy vấn thích ứng | Giảm token tính phí và chi phí mỗi truy vấn |
| **Ngắn hạn** | Củng cố triển khai: giám sát và cảnh báo, tự mở rộng theo tải, chuyển đổi dự phòng mô hình, quản lý phiên bản kho văn bản | Tăng độ ổn định và khả năng phục hồi khi vận hành thực tế |
| **Trung hạn** | Kích hoạt bộ xếp hạng lại trên GPU | Recall@10 dự kiến 0,48, độ trễ khoảng 2–3 giây mỗi câu |
| **Trung hạn** | Xây dựng Graph RAG trên đồ thị tri thức pháp luật | Xử lý chuỗi tham chiếu nhiều bước, mô hình hoá quan hệ hiệu lực một cách tường minh |
| **Trung hạn** | Đánh giá lại bằng mô hình giám khảo mạnh hơn | Độ trung thực và độ phủ ngữ cảnh chính xác hơn |
| **Dài hạn** | Chuyển từ một tác từ sang kiến trúc đa tác từ chuyên biệt | Phân rã nhiệm vụ phức tạp, kiểm tra chéo, tăng độ tin cậy |
| **Dài hạn** | Tác từ truy xuất nhiều bước | Xử lý câu hỏi liên quan từ ba điều luật trở lên |
| **Dài hạn** | Trích xuất hành vi theo ngữ nghĩa | Phân định lỗi trong các vụ tai nạn phức tạp |

Tầm ngắn hạn — củng cố vận hành và tối ưu chi phí. Các cải tiến trong tầm này có thể thực hiện ngay mà không thay đổi kiến trúc cốt lõi. Về chi phí, lưu đệm phần lệnh hướng dẫn cố định, phân tầng mô hình (mô hình nhỏ cho phân tích và định tuyến, mô hình mạnh cho sinh câu trả lời) và điều tiết số truy vấn mở rộng theo độ phức tạp câu hỏi sẽ giảm đáng kể chi phí mỗi truy vấn. Về vận hành, bổ sung giám sát theo thời gian thực, cơ chế chuyển đổi mô hình dự phòng khi lỗi và quản lý phiên bản kho văn bản sẽ nâng độ ổn định phù hợp cho triển khai thực tế.

Tầm trung hạn — nâng cấp năng lực truy xuất với Graph RAG. Bên cạnh việc kích hoạt bộ xếp hạng lại khi có GPU, hướng đáng chú ý là tích hợp Graph RAG. Văn bản pháp luật giao thông có mật độ dẫn chiếu chéo cao mà cơ chế hiện tại chỉ bắt được qua biểu thức chính quy; Graph RAG biểu diễn mỗi điều khoản là một nút và các quan hệ dẫn chiếu, sửa đổi, bãi bỏ là các cạnh, cho phép truy xuất theo đường đi nhiều bước trên đồ thị, qua đó nâng độ phủ ngữ cảnh — điểm nghẽn chính được chỉ ra ở chương 3.

Tầm dài hạn — kiến trúc đa tác từ. Kiến trúc một tác từ đơn có thể phát triển thành hệ đa tác từ chuyên biệt: một tác từ phân tích, một tác từ truy xuất, một tác từ sinh câu trả lời và một tác từ thẩm định trích dẫn độc lập. Việc tách vai trò tạo cơ chế kiểm tra chéo, tăng độ tin cậy và cho phép xử lý các tình huống pháp lý đòi hỏi suy luận đa bước như phân định lỗi trong vụ tai nạn phức tạp. Hướng này được xếp dài hạn vì làm tăng độ trễ và chi phí, chỉ phù hợp sau khi đã giải quyết các nút thắt tài nguyên ở hai tầm trước.
