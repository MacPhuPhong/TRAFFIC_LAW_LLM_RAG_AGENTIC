# Hướng dẫn Triển khai Hệ thống Traffic Law RAG (Production Hybrid Cloud)

Tài liệu này hướng dẫn cách thiết lập và vận hành hệ thống Traffic Law Agentic RAG trên môi trường thực tế (Cloud Tier miễn phí).

## 1. Kiến trúc Hệ thống thực tế

Hệ thống được triển khai theo mô hình Hybrid Cloud để tối ưu hóa hiệu năng và chi phí ($0/tháng):

- **Frontend**: Next.js 14 triển khai trên **Vercel** (Hobby Tier).
- **Backend**: FastAPI + LangGraph triển khai trên **Hugging Face Spaces** (Docker SDK).
- **Vector Database**: **Qdrant Cloud** (Cluster us-east-1).
- **Data Persistence**: **Supabase Postgres** (Lưu trữ LangGraph checkpoints & NextAuth).
- **LLM**: **Google Gemini API** (Gemini 1.5 Flash / 3.1 Flash Lite).
- **Web Search**: **Tavily API**.

---

## 2. Các thành phần và URL Production

- **Giao diện người dùng**: [https://traffic-rag.vercel.app](https://traffic-rag.vercel.app)
- **Giao diện Admin (HITL)**: [https://traffic-rag.vercel.app/admin](https://traffic-rag.vercel.app/admin)
- **Backend API (Swagger)**: [https://pphong203-traffic-rag-backend.hf.space/docs](https://pphong203-traffic-rag-backend.hf.space/docs)

---

## 3. Cấu hình Biến môi trường (Environment Variables)

### 3.1 Backend (Hugging Face Secrets)

Cài đặt trong mục **Settings > Variables and secrets** của Space:

| Biến                   | Mô tả                                                                    |
| ---------------------- | ------------------------------------------------------------------------ |
| `API_KEY`              | Google Gemini API key.                                                   |
| `TAVILY_API_KEY`       | Tavily search API key.                                                   |
| `QDRANT_URL`           | URL cluster Qdrant Cloud (ví dụ: `https://...aws.cloud.qdrant.io:6333`). |
| `QDRANT_API_KEY`       | API key của Qdrant Cloud.                                                |
| `CHECKPOINT_DB_URL`    | Postgres URI từ Supabase (**Session Pooler**, port 5432).                |
| `CORS_ALLOWED_ORIGINS` | URL của Frontend (ví dụ: `https://traffic-rag.vercel.app`).              |
| `GENERATOR_MODEL`      | Mặc định: `gemini-3.1-flash-lite-preview`.                               |

### 3.2 Frontend (Vercel Env Vars)

Cài đặt trong **Project Settings > Environment Variables**:

| Biến              | Mô tả                                                                   |
| ----------------- | ----------------------------------------------------------------------- |
| `BACKEND_URL`     | URL của HF Space Backend (không có dấu / ở cuối).                       |
| `DATABASE_URL`    | Postgres URI từ Supabase (**Transaction Pooler**, port 6543).           |
| `NEXTAUTH_URL`    | URL của Frontend Vercel.                                                |
| `NEXTAUTH_SECRET` | Chuỗi ngẫu nhiên (tạo bằng `openssl rand -hex 32`).                     |
| `ADMIN_EMAILS`    | Danh sách email admin truy cập HITL (phân cách bằng dấu phẩy hoặc `*`). |

---

## 4. Quy trình Index dữ liệu lên Cloud

Khi có dữ liệu mới hoặc re-index:

1. Đảm bảo file `traffic_rag/Data/all_chunks.jsonl` đã được cập nhật.
2. Thiết lập env local:
   ```bash
   export QDRANT_URL="https://your-cluster.qdrant.io:6333"
   export QDRANT_API_KEY="your-key"
   ```
3. Chạy lệnh indexing:
   ```bash
   cd traffic_rag
   python -m source.indexing.indexer --recreate
   ```

---

## 5. Triển khai và Cập nhật (Deployment)

### 5.1 Cập nhật Backend

Backend được đồng bộ từ repo [hf-space-repo](file:///media/pphong/D:/Do_An_Tot_Nghiep/GitHub1/hf-space-repo). Khi có thay đổi:

1. Copy code mới vào `hf-space-repo/traffic_rag/`.
2. Commit và push lên Hugging Face:
   ```bash
   git add .
   git commit -m "Update backend"
   git push
   ```

### 5.2 Cập nhật Frontend

Vercel tự động deploy khi có commit mới vào nhánh `main` của GitHub repo.

---

## 6. Kiểm tra Trạng thái (Health Checks)

Sử dụng các lệnh sau để kiểm tra nhanh:

```bash
# Health check backend
curl -s https://pphong203-traffic-rag-backend.hf.space/health

# Test chat qua API backend
curl -X POST https://pphong203-traffic-rag-backend.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Phạt quá tốc độ ô tô bao nhiêu?"}'
```

---

## 7. Lưu ý bảo trì

1. **Supabase Auto-pause**: Gói miễn phí của Supabase sẽ tạm dừng nếu không có hoạt động trong 7 ngày. Cần truy cập dashboard hoặc ping định kỳ để duy trì.
2. **Quản lý Session**: Database URI cho Backend nên dùng **Session Pooler (5432)**, Frontend nên dùng **Transaction Pooler (6543)** để tối ưu kết nối.
3. **Rotating Keys**: Luôn cập nhật lại secrets nếu nghi ngờ bị lộ trong log build.

---

_Cập nhật ngày: 2026-05-20_
