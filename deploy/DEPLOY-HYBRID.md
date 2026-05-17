# Deploy Hybrid — Vercel + HF Spaces + Qdrant Cloud + Supabase

Triển khai Traffic RAG hoàn toàn miễn phí, URL cố định, không phụ thuộc PC nhà.

## Kiến trúc

```
   ┌────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
   │  Vercel        │─────▶│ HF Spaces (FastAPI)  │─────▶│ Qdrant Cloud     │
   │  Next.js FE    │      │ traffic-rag-backend  │      │ (1GB free)       │
   │  *.vercel.app  │      │ *.hf.space           │      └──────────────────┘
   └───────┬────────┘      └──────────┬───────────┘
           │                          │
           ▼                          ▼
   ┌──────────────────────────────────────────────────┐
   │ Supabase Postgres (NextAuth + LangGraph state)   │
   │ db.xxx.supabase.co                               │
   └──────────────────────────────────────────────────┘
```

## Cost summary

| Service | Tier | Cost |
|---|---|---|
| Vercel Hobby | 100GB bandwidth/mo, free SSL, custom domain | $0 |
| HF Spaces (CPU) | 16GB RAM, 2 vCPU, no sleep | $0 |
| Qdrant Cloud Free | 1GB cluster, single AZ | $0 |
| Supabase Free | 500MB Postgres, 50k MAU, 5GB bandwidth | $0 |
| Gemini API Free | ~15 req/min, 200/ngày | $0 |
| Tavily Free | 1000 search/tháng | $0 |
| **Tổng** | | **$0/tháng** |

---

## Bước 1 — Tạo tài khoản các service

Đăng ký free (không cần thẻ ngoài Supabase optional):

| Service | URL | Cần thẻ? |
|---|---|---|
| GitHub | https://github.com (nếu chưa có) | Không |
| Vercel | https://vercel.com/signup (login bằng GitHub) | Không |
| Hugging Face | https://huggingface.co/join | Không |
| Qdrant Cloud | https://cloud.qdrant.io | Không |
| Supabase | https://supabase.com/dashboard/sign-up | Không |

---

## Bước 2 — Supabase Postgres

1. Vào Supabase → **New project**:
   - Name: `traffic-rag`
   - Database password: (lưu lại — sẽ dùng)
   - Region: **Southeast Asia (Singapore)** (gần VN nhất)
2. Đợi ~2 phút project provision.
3. **Settings → Database → Connection string**:
   - Chọn tab **URI**
   - Copy chuỗi `postgresql://postgres:[PASSWORD]@db.xxx.supabase.co:5432/postgres`
   - **Thay `[PASSWORD]` bằng password thật**
4. Tạm gọi chuỗi này là `SUPABASE_URL`. Lưu vào notes.

> Lưu ý: Free tier auto-pause sau **7 ngày không request**. Cần access định kỳ để giữ DB sống.

---

## Bước 3 — Qdrant Cloud

1. Vào https://cloud.qdrant.io → **Clusters → Create**:
   - Provider: AWS
   - Region: **ap-southeast-1 (Singapore)** hoặc **ap-northeast-1 (Tokyo)**
   - Tier: **Free**
   - Name: `traffic-rag`
2. Đợi ~2 phút.
3. Click cluster → **API Keys → Create API Key** → copy.
4. **Cluster URL** (Overview tab) — copy URL dạng `https://xxx.aws.cloud.qdrant.io`.

Tạm gọi `QDRANT_URL` và `QDRANT_API_KEY`.

### Upload 2818 chunks lên Qdrant Cloud

Trên **máy local** (đang có project + venv):

```bash
source ~/venv/LLM_Agentic/bin/activate
cd /path/to/GitHub1/traffic_rag

export QDRANT_URL="https://xxx.aws.cloud.qdrant.io"
export QDRANT_API_KEY="eyJh..."

bash deploy/scripts/upload_qdrant_to_cloud.sh
```

Mất ~5-8 phút (embed 2818 chunks + upsert qua mạng). Khi xong, dashboard Qdrant Cloud sẽ thấy collection `Traffic_Law_Hybrid` với 2818 points.

---

## Bước 4 — Push code lên GitHub

Cần repo public hoặc private (HF Spaces có thể clone từ GitHub):

```bash
cd /path/to/GitHub1
git init
git remote add origin https://github.com/<username>/traffic-rag.git

# QUAN TRỌNG: không commit .env (đã có .gitignore?)
echo ".env" >> .gitignore
echo "*.env.local" >> .gitignore
echo "qdrant_storage/" >> .gitignore
echo "checkpoints/*.db" >> .gitignore

git add .
git commit -m "Initial deploy-ready commit"
git push -u origin main
```

---

## Bước 5 — Deploy Backend lên Hugging Face Spaces

### 5.1 Tạo Space

1. Vào https://huggingface.co/new-space:
   - Name: `traffic-rag-backend`
   - License: MIT
   - SDK: **Docker**
   - Hardware: **CPU basic - 16GB RAM** (free)
   - Visibility: Public (free tier yêu cầu public)
2. Click **Create Space**.

### 5.2 Copy Dockerfile vào Space

Space tự tạo 1 git repo. Trên local:

```bash
# Clone Space repo
git clone https://huggingface.co/spaces/<username>/traffic-rag-backend hf-space
cd hf-space

# Copy Dockerfile + README từ traffic_rag/deploy/huggingface vào root
cp /path/to/GitHub1/traffic_rag/deploy/huggingface/Dockerfile .
cp /path/to/GitHub1/traffic_rag/deploy/huggingface/README.md .

# Copy requirements.txt + traffic_rag/ vào root (build context)
cp /path/to/GitHub1/requirements.txt .
cp -r /path/to/GitHub1/traffic_rag .

# Loại bỏ thứ không cần
rm -rf traffic_rag/Data/raw traffic_rag/Data/cleaned \
  traffic_rag/Data/chroma_bench traffic_rag/Data/semantic_chunks \
  traffic_rag/research traffic_rag/logs traffic_rag/.git \
  traffic_rag/qdrant_storage traffic_rag/checkpoints \
  traffic_rag/app  # frontend đã deploy Vercel riêng

# Commit + push
git add .
git lfs install   # nếu dùng LFS cho file lớn (all_chunks.jsonl 4.5MB không cần)
git commit -m "Initial backend deploy"
git push
```

> **Lưu ý git push HF**: yêu cầu HF access token (Settings → Access Tokens → Write). Khi push lần đầu nó hỏi user/password — password là token.

### 5.3 Set Secrets

Vào Space → **Settings → Variables and secrets → New secret**:

| Key | Value |
|---|---|
| `API_KEY` | Gemini key |
| `TAVILY_API_KEY` | Tavily key |
| `QDRANT_URL` | https://xxx.aws.cloud.qdrant.io |
| `QDRANT_API_KEY` | eyJh... |
| `CHECKPOINT_DB_URL` | postgresql://postgres:PASS@db.xxx.supabase.co:5432/postgres |
| `CORS_ALLOWED_ORIGINS` | https://traffic-rag.vercel.app (cập nhật sau Bước 6) |
| `GENERATOR_MODEL` | gemini-3.1-flash-lite-preview |

### 5.4 Build + Test

Push code → HF tự build (8-15 phút lần đầu). Theo dõi ở tab **Logs**.

Khi log thấy `Application startup complete` → backend ready.

URL: `https://<username>-traffic-rag-backend.hf.space`

Test:
```bash
curl https://<username>-traffic-rag-backend.hf.space/health
# {"status":"ok"}
```

---

## Bước 6 — Deploy Frontend lên Vercel

### 6.1 Push DB schema lên Supabase

```bash
cd /path/to/GitHub1/traffic_rag/app/nextjs-app
export DATABASE_URL="postgresql://postgres:PASS@db.xxx.supabase.co:5432/postgres"
npx prisma db push
```

Tạo các table User/Session/Conversation/Message trong Supabase.

### 6.2 Import vào Vercel

1. Vào https://vercel.com/new → **Import Git Repository** → chọn repo GitHub.
2. **Root Directory**: `traffic_rag/app/nextjs-app` (Vercel sẽ build từ folder này).
3. Framework Preset: **Next.js** (auto detect).
4. **Environment Variables** (Settings → Environment Variables):

| Key | Value |
|---|---|
| `BACKEND_URL` | https://username-traffic-rag-backend.hf.space |
| `DATABASE_URL` | postgresql://postgres:PASS@db.xxx.supabase.co:5432/postgres |
| `NEXTAUTH_URL` | https://traffic-rag.vercel.app (Vercel sẽ cấp) |
| `NEXTAUTH_SECRET` | `openssl rand -hex 32` |
| `ADMIN_EMAILS` | `*` hoặc email cụ thể |
| `NEXT_PUBLIC_CF_ANALYTICS_TOKEN` | (optional) token CF Web Analytics |

5. Click **Deploy**. Build ~3 phút.

### 6.3 Cập nhật ngược CORS

Sau khi Vercel deploy xong, có URL `https://<your-app>.vercel.app`. Quay lại HF Spaces:
- Settings → Secrets → sửa `CORS_ALLOWED_ORIGINS = https://<your-app>.vercel.app`
- Space sẽ tự restart.

### 6.4 Custom domain (optional)

Vercel → Project → Settings → Domains → Add. Trỏ A record `@` về Vercel.
Update `NEXTAUTH_URL` thành domain mới + redeploy.

---

## Bước 7 — Kiểm tra end-to-end

1. Mở `https://<your-app>.vercel.app`
2. Hỏi: *"Vượt đèn đỏ ô tô bị phạt bao nhiêu?"*
3. Câu trả lời stream về với citations.
4. Admin: `https://<your-app>.vercel.app/admin` → login `admin@example.com` / `admin123`.

---

## Troubleshooting

| Lỗi | Xử lý |
|---|---|
| FE gọi BE 502/CORS | Check `CORS_ALLOWED_ORIGINS` ở HF Spaces khớp Vercel domain |
| FE 500 ở /api/chat | Check `BACKEND_URL` ở Vercel env vars |
| HF Space build fail | Log thường rõ — thường do `requirements.txt` thiếu hoặc Dockerfile path sai |
| Qdrant query trả rỗng | Re-run upload script (Bước 3) — collection có thể chưa index xong |
| Prisma `db push` fail | Check `DATABASE_URL`, password đúng, Supabase project Active không Paused |
| Gemini 429 quota | Đổi key trong HF Secrets, restart Space |
| HF Space sleep | Free tier KHÔNG sleep nếu là Docker SDK. Spaces có "sleep after 48h inactive" cho Gradio/Streamlit nhưng Docker thường stay alive — verify bằng dashboard |

---

## Lệnh sửa code rồi redeploy

Sau khi code thay đổi:

```bash
# Backend (HF Spaces)
cd hf-space
cp -r /path/to/GitHub1/traffic_rag ./
git add . && git commit -m "Update backend" && git push
# HF auto rebuild

# Frontend (Vercel)
cd /path/to/GitHub1
git add . && git commit -m "Update frontend" && git push
# Vercel auto rebuild
```

---

## Limitations free tier

- **Qdrant Free**: 1GB, 1 cluster, 1 AZ — đủ cho 2818 chunks. Nếu scale chục nghìn chunks → upgrade $25/mo.
- **Supabase Free**: 500MB Postgres, **auto-pause sau 7 ngày unused**. Nếu demo ít, set cron `curl https://your-app.vercel.app/api/health` mỗi 3 ngày để giữ alive.
- **HF Spaces CPU Free**: 16GB RAM, 2 vCPU, không GPU. Inference embedding model CPU ~200ms/query — chấp nhận được.
- **Vercel Hobby**: Serverless function timeout 10s. Vì FE chỉ proxy SSE từ BE (không tự inference) → không vấn đề.
- **Gemini Free**: 200 req/ngày — đủ demo. Cần nhiều hơn → đổi key xoay vòng hoặc upgrade.

---

## Migrate ngược về single VPS

Nếu muốn rời các free service về 1 VPS (Contabo $6.5/mo):
1. SQLite checkpoint (bỏ `CHECKPOINT_DB_URL`)
2. Local Qdrant Docker (bỏ `QDRANT_URL`)
3. Postgres local hoặc đổi Prisma về sqlite
4. Theo [DEPLOY.md](DEPLOY.md) hiện có
