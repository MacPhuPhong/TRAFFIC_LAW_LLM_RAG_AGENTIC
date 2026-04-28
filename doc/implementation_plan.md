# Lệnh Chạy Hệ Thống Traffic RAG (Local)

Tài liệu này tổng hợp toàn bộ các lệnh cần thiết để khởi chạy hệ thống Traffic Law Agentic RAG trên máy tính cá nhân.

> **Quan trọng — Working directory:** Tất cả lệnh trong tài liệu này (trừ khi ghi chú khác) đều chạy từ thư mục `traffic_rag/`. Trước khi bắt đầu:
>
> ```bash
> cd /media/pphong/D:/Do_An_Tot_Nghiep/GitHub1/traffic_rag
> ```

## 0. Môi trường Python (venv)

Virtual environment dùng chung cho project được đặt tại `~/venv/LLM_Agentic/`. Kích hoạt trước khi chạy bất kỳ lệnh Python nào:

```bash
source /home/pphong/venv/LLM_Agentic/bin/activate
```

> Nếu cần cài thêm dependencies, dùng `pip install -r ../requirements.txt` (file `requirements.txt` nằm ở root `GitHub1/`).

## 1. Hạ tầng (Infrastructure)

Hệ thống sử dụng **Qdrant** làm Vector Database. Chạy Qdrant thông qua Docker:

```bash
docker compose up -d qdrant
```

- **Qdrant Dashboard:** [http://localhost:6334/dashboard](http://localhost:6334/dashboard)
  *(host port 6334 → container port 6333, theo cấu hình trong `docker-compose.yml`)*

## 2. Chuẩn bị Dữ liệu (Data Pipeline) — *Tùy chọn*

Nếu bạn đã có file [Data/all_chunks.jsonl](../Data/all_chunks.jsonl) và muốn index lại vào Qdrant:

```bash
# (Tùy chọn) 2.1. Làm sạch dữ liệu PDF nếu xử lý lại từ đầu
python source/ingestion/clean_luat_pdfs.py
python source/ingestion/clean_nghidinh_pdfs.py
python source/ingestion/clean_thongtu_pdfs.py

# 2.2. Đẩy dữ liệu vào Qdrant (Indexing)
python -m source.indexing.indexer --recreate
```

## 3. Chạy Backend (FastAPI)

Có hai cách chạy backend — chọn **một** trong hai:

### Cách A — Chạy local bằng uvicorn (khuyên dùng khi đang phát triển)

```bash
# Đảm bảo đã kích hoạt venv (xem mục 0)
source /home/pphong/venv/LLM_Agentic/bin/activate

# Khởi chạy Backend ở port 8000
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Cách B — Chạy bằng Docker (cùng compose với Qdrant)

```bash
# Chạy cả qdrant + backend
docker compose up -d
```

- Backend chạy trong container `traffic_rag_backend`, expose ra **host port 8003**.
- **API Documentation:** [http://localhost:8003/docs](http://localhost:8003/docs)

> Lưu ý: Frontend ở mục 4 mặc định gọi sang backend tại `localhost:8000`. Nếu bạn dùng Cách B, cần chỉnh biến môi trường tương ứng trong `app/nextjs-app/` (ví dụ `NEXT_PUBLIC_API_URL`) sang `http://localhost:8003`.

## 4. Chạy Frontend (Next.js)

```bash
cd app/nextjs-app
npm install        # chỉ cần chạy lần đầu
npm run dev
```

- **Giao diện người dùng:** [http://localhost:3000](http://localhost:3000)

## Lưu ý về Cấu hình (.env)

File `.env` đặt ở **thư mục gốc `GitHub1/`** (không phải trong `traffic_rag/`) — `docker-compose.yml` đã trỏ tới `../.env`. Đảm bảo file [.env](../../.env) có đầy đủ:

- `API_KEY`: Google Gemini API Key.
- `TAVILY_API_KEY`: Dùng cho Web Search Agent.
- `LANGCHAIN_*`: (Tùy chọn) Dùng để debug với LangSmith.

## Kiểm tra trạng thái

| Thành phần | Lệnh kiểm tra |
|---|---|
| Container Docker | `docker ps` |
| Log Qdrant | `docker logs traffic_qdrant` |
| Backend (Cách A — local) | `ps aux \| grep uvicorn` |
| Backend (Cách B — Docker) | `docker logs -f traffic_rag_backend` |
| Frontend | Mở [http://localhost:3000](http://localhost:3000) |
