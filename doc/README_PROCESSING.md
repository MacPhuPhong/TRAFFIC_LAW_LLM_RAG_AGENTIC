# QUY TRÌNH CLEAN DỮ LIỆU LUẬT GIAO THÔNG CHO RAG

(Cập nhật Tháng 4/2026 - Thời điểm chuyển giao Luật 2008 -> 2024)

## Bước 1: Chuyển đổi định dạng

- Chuyển toàn bộ PDF sang định dạng **Markdown (.md)**.
- **Tại sao:** Markdown giữ được cấu trúc phân cấp (Header) và bảng biểu (Table) tốt nhất cho việc chunking theo ngữ nghĩa.

## Bước 2: Làm sạch dữ liệu (Cleaning)

- Loại bỏ các ký tự rác từ việc quét PDF (OCR lỗi).
- Loại bỏ thông tin "Thư viện pháp luật", "Mã tra cứu", "about:blank" ở chân trang.
- Hợp nhất các từ bị ngắt đôi do xuống dòng (Dòng tiếp theo bắt đầu bằng chữ thường).

## Bước 3: Đánh dấu quan hệ thay thế (Metadata & Status)

- Gán nhãn hiệu lực: Khi đưa vào Vector Database, mỗi đoạn văn bản cần có thuộc tính `status: active` hoặc `status: repealed`.
- **Ví dụ:** Tại các văn bản cũ (NĐ 100, TT 12/2017), các quy định về mức phạt đường bộ phải đánh dấu là hết hiệu lực.
- Xóa bỏ/Giảm trọng số các nội dung đã bị bãi bỏ cụ thể.

## Bước 4: Chunking Strategy (Cắt đoạn)

- **Kích thước chunk:** Ưu tiên cắt theo từng **Điều (Article)**.
- **Overlap:** Giữ overlap khoảng 10-15% để không mất ngữ cảnh giữa các Điều liền kề.
- **Phân cấp nội dung:** Định dạng `# Chương`, `## Điều`, `### Khoản`.
- **Metadata đi kèm mỗi chunk:**
  - `doc_id`: (Ví dụ: ND_168_2024)
  - `effective_date`: (Ví dụ: 01/01/2025)
  - `category`: (Ví dụ: Xử phạt, Đăng ký, Quy tắc)
  - `status`: active / repealed

## Bước 5: Kiểm định Bảng biểu

- Kiểm tra các bảng về mức lệ phí và tiêu chuẩn sức khỏe.
- Đảm bảo các cột không bị lệch để AI không trích xuất nhầm con số giữa các khu vực/hạng xe. Bảng phải chuyển qua Markdown tables chuẩn.
