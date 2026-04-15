import pdfplumber
import os

files_to_check = [
    "Data/raw/luat/luat_ttatgt_2024.pdf",
    "Data/raw/nghidinh/nd100_2019.pdf",
    "Data/raw/thongtu_gplx/tt4_2022_gplx_suadoi.pdf",
    "Data/raw/thongtu_transport/tt18_2024_transport_suadoi.pdf"
]

base_dir = "/media/pphong/D:/Do_An_Tot_Nghiep/GitHub1/traffic_rag"

print("--- PHÂN TÍCH NỘI DUNG PDF ---")
for rel_path in files_to_check:
    full_path = os.path.join(base_dir, rel_path)
    if os.path.exists(full_path):
        print(f"\nFile: {rel_path}")
        with pdfplumber.open(full_path) as pdf:
            # Lấy 2 trang đầu để xem tiêu đề và cách dẫn đề
            text = ""
            for i in range(min(2, len(pdf.pages))):
                text += pdf.pages[i].extract_text() + "\n"
            print(text[:1000]) # Hiển thị 1000 ký tự đầu
    else:
        print(f"File not found: {full_path}")
