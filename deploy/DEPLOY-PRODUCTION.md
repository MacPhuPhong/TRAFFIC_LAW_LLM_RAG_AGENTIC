# Deploy Production Log — Traffic RAG (2026-05-17)

Tài liệu này ghi lại **toàn bộ các bước đã thực hiện** để deploy hệ thống Traffic RAG lên production cloud miễn phí. Khác với [DEPLOY-HYBRID.md](DEPLOY-HYBRID.md) (hướng dẫn dự kiến), đây là **playback chính xác** mọi thao tác, mọi sự cố và cách khắc phục.

---

## 0. Tóm tắt kiến trúc đã deploy

```
        ┌──────────────────────────┐
        │  https://traffic-rag.    │
        │       vercel.app         │   Next.js 14 · Vercel Hobby (US East)
        │  Browser ⇄ Edge SSR/SSE  │
        └──────────────┬───────────┘
                       │ HTTPS · /api/chat (server-side proxy)
                       ▼
        ┌──────────────────────────────────┐
        │ https://pphong203-traffic-rag-    │
        │       backend.hf.space            │   FastAPI + LangGraph · HF Spaces (Docker, US East)
        │ uvicorn :7860 · 16GB RAM · 2 vCPU │
        └────────┬─────────────────┬───────┘
                 │ qdrant-client    │ psycopg (asyncio)
                 ▼                  ▼
   ┌───────────────────────┐  ┌────────────────────────────┐
   │ Qdrant Cloud Free 1GB │  │ Supabase Postgres Free 500MB│
   │ us-east-1 · 2818 pts  │  │ ap-northeast-1 (Tokyo)      │
   └───────────────────────┘  │ • NextAuth tables (Prisma)  │
                              │ • LangGraph checkpoint_*    │
                              └────────────────────────────┘

   External (HTTPS):
   • Gemini 2.5 Flash Lite (generativelanguage.googleapis.com)
   • Tavily web search (api.tavily.com)
```

**Tổng chi phí**: $0/tháng (tất cả nằm trong free tier).

**URLs production**:
- Chat: https://traffic-rag.vercel.app
- Admin: https://traffic-rag.vercel.app/admin (login `admin@example.com` / `admin123`)
- Backend Swagger: https://pphong203-traffic-rag-backend.hf.space/docs

---

## 1. Tổng quan thứ tự thực hiện

| Phase | Nội dung | Thời gian |
|---|---|---|
| **A** | Code patches (5 file Python + Prisma schema) | 10 phút |
| **B** | Push GitHub (fix `.gitignore` bug) | 5 phút |
| **C** | Qdrant Cloud: tạo cluster + upload 2818 chunks | 20 phút |
| **D** | Supabase: tạo project + lấy connection string | 5 phút |
| **E** | Hugging Face Spaces: tạo Space + push code + secrets | 20 phút (×2 retry) |
| **F** | Vercel: import + env vars + deploy | 10 phút |
| **G** | Update cross-references (CORS, NEXTAUTH_URL) | 5 phút |

**Tổng**: ~75 phút (bao gồm thời gian fix bug).

---

## 2. Phase A — Code patches

7 file đã sửa để code hoạt động cả local lẫn cloud thông qua env vars:

### 2.1 `source/rag_core/retriever.py`

Thêm 2 param mới cho Qdrant Cloud:
```python
def __init__(
    self,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    qdrant_url: str | None = None,        # NEW — Qdrant Cloud URL
    qdrant_api_key: str | None = None,    # NEW — Qdrant Cloud API key
    ...
):
    if qdrant_url:
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key)
```

### 2.2 `source/indexing/indexer.py`

Đọc cấu hình Qdrant từ env (thay vì hard-code):
```python
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "Traffic_Law_Hybrid")
```

Thêm `timeout=120s` cho client (xử lý latency cao khi upload từ VN sang Qdrant US/EU).

### 2.3 `api/main.py`

**Wire env vars vào retriever** (với `.strip()` chống trailing newline khi paste vào HF Spaces UI):
```python
_qdrant_url = (os.getenv("QDRANT_URL") or "").strip() or None
_qdrant_api_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
retriever = TrafficHybridRetriever(
    qdrant_host=os.getenv("QDRANT_HOST", "localhost").strip(),
    qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
    qdrant_url=_qdrant_url,
    qdrant_api_key=_qdrant_api_key,
    ...
)
```

**Checkpoint backend dual mode** — Postgres (Supabase) hoặc SQLite:
```python
ckpt_url = os.getenv("CHECKPOINT_DB_URL", "")
if ckpt_url.startswith(("postgres://", "postgresql://")):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    pg_url = ckpt_url.replace("postgres://", "postgresql://", 1)
    cm = AsyncPostgresSaver.from_conn_string(pg_url)
    checkpointer = await cm.__aenter__()
    app.state._ckpt_cm = cm
    await checkpointer.setup()
else:
    # ... SQLite fallback
```

**CORS middleware** cho cross-origin (Vercel → HF Spaces):
```python
_cors = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
if _cors:
    from fastapi.middleware.cors import CORSMiddleware
    origins = ["*"] if _cors == "*" else [o.strip() for o in _cors.split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
```

### 2.4 `requirements.txt`

Thêm 2 deps cho Postgres checkpoint:
```
langgraph-checkpoint-postgres
psycopg[binary,pool]
```

### 2.5 `app/nextjs-app/prisma/schema.prisma`

Đổi provider:
```diff
- provider = "sqlite"
+ provider = "postgresql"
```

### 2.6 `app/nextjs-app/src/lib/authOptions.ts` (file mới)

Tách `authOptions` khỏi route handler vì Next.js 14 không cho route file export gì khác ngoài HTTP method handlers. File mới chứa `authOptions`, các file cũ import từ đây.

### 2.7 `app/nextjs-app/src/app/api/auth/[...nextauth]/route.ts`

Rút gọn còn:
```typescript
import NextAuth from 'next-auth';
import { authOptions } from '@/lib/authOptions';
const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
```

---

## 3. Phase B — Fix `.gitignore` + push GitHub

**Bug phát hiện**: `.gitignore` có pattern `lib/` (cho Python build) đang match cả `app/nextjs-app/src/lib/` → toàn bộ 5 file Next.js library code (`useChat.ts`, `store.ts`, `admin.ts`, `types.ts`, `prisma.ts`) **chưa bao giờ được commit lên GitHub**. Nếu deploy Vercel sẽ fail.

**Fix**: anchor pattern bằng `/` để chỉ match root:
```diff
- lib/
+ /lib/
```

Thêm exception cho `Data/all_chunks.jsonl` (4.5MB, runtime cần cho BM25):
```diff
- Data/
- !Data/README.md
+ Data/*
+ !Data/README.md
+ !Data/all_chunks.jsonl
```

Push:
```bash
git add -A
git commit -m "Add deploy/hybrid + fix .gitignore"
git push origin main
```

---

## 4. Phase C — Qdrant Cloud

### 4.1 Tạo cluster

1. Đăng ký https://cloud.qdrant.io (free, không cần thẻ)
2. Create Cluster → **Free Tier** · AWS · **us-east-1** (N. Virginia — cùng datacenter với HF Spaces, tối ưu query latency runtime)
3. Tên cluster: `traffic-rag`
4. Provision ~2 phút

> **Lựa chọn region**: ban đầu tạo ở EU-West-2 (London) thì upload từ VN bị timeout. Đổi sang us-east-1 vì HF Spaces free tier chạy ở AWS us-east-1 → query backend↔Qdrant latency ~5ms (cùng DC), thay vì 200ms (EU↔US).

### 4.2 API key

Access Management → API Keys → Create → role `m` (read-write) → copy key `eyJh...`.

URL cluster: `https://b25af5db-...-us-east-1-1.aws.cloud.qdrant.io:6333`

### 4.3 Upload 2818 chunks

```bash
cd /path/to/GitHub1/traffic_rag
source ~/venv/LLM_Agentic/bin/activate

export QDRANT_URL="https://b25af5db-...-us-east-1-1.aws.cloud.qdrant.io:6333"
export QDRANT_API_KEY="eyJh..."
export QDRANT_TIMEOUT=180

python -m source.indexing.indexer --recreate
```

**Thời gian thực tế**: 13 phút embedding (CPU local) + 5 phút upsert + validation = **18 phút** từ VN sang us-east-1.

**Validation queries sau khi xong**: 3/3 PASS (top-1 đúng cho NĐ 168/2024, TT 47/2024, TT 79/2024).

---

## 5. Phase D — Supabase Postgres

### 5.1 Tạo project

1. Đăng ký https://supabase.com (free, không cần thẻ)
2. New Project:
   - Name: `traffic-rag`
   - Database password: đặt password mạnh (lưu vào password manager)
   - Region: **Tokyo (ap-northeast-1)** — gần VN nhất trong free tier
3. Provision ~2 phút

### 5.2 Connection strings

Supabase cung cấp 3 loại:

| Loại | Host | Port | Dùng cho |
|---|---|---|---|
| **Direct** | `db.<ref>.supabase.co` | 5432 | ❌ IPv6 only, HF Spaces không support |
| **Session Pooler** | `aws-1-ap-northeast-1.pooler.supabase.com` | 5432 | ✅ HF Spaces (LangGraph cần prepared statements) |
| **Transaction Pooler** | `aws-1-ap-northeast-1.pooler.supabase.com` | 6543 | ✅ Vercel/Prisma serverless |

**Sai lầm ban đầu**: dùng Direct connection cho HF → fail với `Network is unreachable` (HF Spaces không có IPv6 outbound). Đổi sang Session Pooler thì OK.

---

## 6. Phase E — Hugging Face Spaces

### 6.1 Tạo Space

1. https://huggingface.co/new-space
2. Name: `traffic-rag-backend` · Owner: `Pphong203` · SDK: **Docker** · Hardware: **CPU basic** (free, 16GB RAM) · Public
3. Access Token: Settings → Tokens → New → role **Write** (cần để `git push`)

### 6.2 Push code

```bash
cd /path/to/GitHub1
git clone https://Pphong203:<HF_TOKEN>@huggingface.co/spaces/Pphong203/traffic-rag-backend hf-space-repo
cd hf-space-repo

# Copy deploy config
cp ../traffic_rag/deploy/huggingface/Dockerfile .
cp ../traffic_rag/deploy/huggingface/README.md .

# Copy source (loại bỏ thứ không cần build)
cp ../requirements.txt .
cp -r ../traffic_rag .
rm -rf traffic_rag/{Data/raw,Data/cleaned,Data/chroma_bench,Data/semantic_chunks,research,logs,.git,qdrant_storage,checkpoints,app,doc,tests}

git add . && git commit -m "Initial backend deploy"
git push
```

**Sự cố push lần đầu**: YAML frontmatter trong README có `short_description` dài >60 ký tự → HF reject. Sửa thành chuỗi ngắn, push lại OK.

### 6.3 Set Secrets

Settings → Variables and secrets → New secret (7 secrets):

| Key | Source |
|---|---|
| `API_KEY` | Gemini API key từ `.env` local |
| `TAVILY_API_KEY` | Tavily key từ `.env` local |
| `QDRANT_URL` | từ Phase C |
| `QDRANT_API_KEY` | từ Phase C |
| `CHECKPOINT_DB_URL` | **Session Pooler URI** (port 5432), thay `[YOUR-PASSWORD]` bằng password thật |
| `CORS_ALLOWED_ORIGINS` | `*` (tạm; update sau Phase F) |
| `GENERATOR_MODEL` | `gemini-3.1-flash-lite-preview` |

### 6.4 Build + Startup

- Build Docker image: **~10 phút** (pull `python:3.10-slim` + cài deps + pre-download embedding model 1.1GB)
- Container startup: **~60-90s** (load model + connect Qdrant + Supabase + LangGraph setup)

### 6.5 Các bug phát sinh + cách fix

#### Bug 1 — `Illegal header value` từ Qdrant client

**Triệu chứng**: `httpcore.LocalProtocolError: Illegal header value b'eyJh...vss408k\n'`

**Nguyên nhân**: paste `QDRANT_API_KEY` vào HF UI có trailing `\n`. HTTP header không cho phép `\n`.

**Fix**: thêm `.strip()` trong [api/main.py](../api/main.py) khi đọc env, **và** re-paste secret cẩn thận hơn.

#### Bug 2 — `Network is unreachable` cho Supabase IPv6

**Triệu chứng**: `psycopg.OperationalError: connection to server at "2406:da14:1d62:...", port 5432 failed: Network is unreachable`

**Nguyên nhân**: dùng Direct Connection URI (`db.<ref>.supabase.co`) — IPv6-only. HF Spaces không có IPv6 outbound.

**Fix**: chuyển sang Session Pooler URI (`aws-1-ap-northeast-1.pooler.supabase.com:5432`) — IPv4.

#### Bug 3 — `failed to resolve host 'aws-0-XX-XXXXX-X...'`

**Triệu chứng**: paste nguyên placeholder của tôi (`aws-0-XX-XXXXX-X`) chứ không phải URI thật.

**Fix**: copy đúng URI từ Supabase Connect modal (region thật `ap-northeast-1`).

#### Bug 4 — `UnboundLocalError: local variable 'conn' referenced before assignment`

**Triệu chứng**: backend khởi động OK qua Postgres, nhưng line 178 `app.state._ckpt_conn = conn` fail vì `conn` chỉ định nghĩa trong nhánh SQLite.

**Fix**: gán `app.state._ckpt_conn = conn` ngay trong nhánh SQLite (không phải sau cùng), và shutdown handler dùng `getattr(app.state, "_ckpt_conn", None)` thay vì truy cập `conn` trực tiếp.

---

## 7. Phase F — Vercel

### 7.1 Push DB schema lên Supabase

```bash
cd /path/to/GitHub1/traffic_rag/app/nextjs-app

# Dùng Session Pooler URI (port 5432) — Transaction Pooler không hỗ trợ Prisma migrate
DATABASE_URL="postgresql://postgres.<ref>:<password>@aws-1-ap-northeast-1.pooler.supabase.com:5432/postgres" \
  ./node_modules/.bin/prisma db push
```

**Cảnh báo từ Prisma**: nó muốn drop 4 bảng `checkpoint_*` đã tạo bởi LangGraph (4 + 10 + 14 + 5 rows). Confirm "yes" → drop. **Sau đó restart HF Space** để LangGraph re-run `setup()` recreate tables.

> **Lưu ý**: lần deploy tới, nên chạy `prisma db push` **trước** khi HF Space tạo `checkpoint_*` tables. Hoặc cấu hình Prisma chỉ migrate trên schema `public.<NextAuth tables>` để không đụng các bảng LangGraph.

### 7.2 Tạo NEXTAUTH_SECRET

```bash
openssl rand -hex 32
# → ef83dda4bd5db4b89e683491328fb91f0abe61f08b2db820178cab8a4021e102
```

### 7.3 Import vào Vercel

1. https://vercel.com → Add New Project → Import `MacPhuPhong/TRAFFIC_LAW_LLM_RAG_AGENTIC`
2. **Root Directory**: `app/nextjs-app` ⚠️ quan trọng
3. Framework: Next.js (auto)
4. Build Command: từ `vercel.json` — `prisma generate && next build`

### 7.4 Environment Variables

| Key | Value |
|---|---|
| `BACKEND_URL` | `https://pphong203-traffic-rag-backend.hf.space` |
| `DATABASE_URL` | **Transaction Pooler** URI (port **6543**) |
| `NEXTAUTH_URL` | `https://traffic-rag.vercel.app` |
| `NEXTAUTH_SECRET` | từ §7.2 |
| `ADMIN_EMAILS` | `*` |

### 7.5 Bug: `"authOptions" is not a valid Route export field`

**Triệu chứng** sau khi deploy lần đầu:
```
Type error: Route "src/app/api/auth/[...nextauth]/route.ts" does not match the required types of a Next.js Route.
  "authOptions" is not a valid Route export field.
```

**Nguyên nhân**: Next.js 14 strict mode, route handlers chỉ được export HTTP method functions (`GET`, `POST`, ...) — không cho export biến arbitrary như `authOptions`.

**Fix**: tách `authOptions` ra file mới [`src/lib/authOptions.ts`](../app/nextjs-app/src/lib/authOptions.ts), route handler chỉ còn import và export GET/POST.

Push fix → Vercel auto-rebuild ~3 phút → deploy thành công.

---

## 8. Phase G — Cross-references update

### 8.1 Update `NEXTAUTH_URL` ở Vercel

Lần đầu để trống. Sau khi biết Vercel cấp URL `https://traffic-rag.vercel.app`, edit env var → Save → **Redeploy** (env vars không live-reload).

### 8.2 Update `CORS_ALLOWED_ORIGINS` ở HF Space

Đổi từ `*` sang chính xác URL Vercel:
```
CORS_ALLOWED_ORIGINS=https://traffic-rag.vercel.app
```

HF Space tự restart sau ~10-20s (không cần rebuild Docker image).

---

## 9. Phase F+ — End-to-end test

Bốn lệnh xác nhận toàn pipeline OK:

```bash
# 1. HF Space health
curl -s https://pphong203-traffic-rag-backend.hf.space/health
# → {"status":"ok"}

# 2. HF Space chat (test backend đầy đủ pipeline)
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"query":"vượt đèn đỏ ô tô phạt bao nhiêu"}' \
  https://pphong203-traffic-rag-backend.hf.space/chat | head -c 300
# → {"thread_id":"...","status":"completed","answer":"### Phạt đối với xe ô tô vượt đèn đỏ\n\n- Phạt tiền từ **18.000.000 đồng** đến **20.000.000 đồng** ..."}

# 3. Vercel frontend
curl -s -o /dev/null -w "%{http_code}\n" https://traffic-rag.vercel.app/
# → 200

# 4. Vercel /api/chat (SSE proxy → HF Space)
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"vượt đèn đỏ"}]}' \
  https://traffic-rag.vercel.app/api/chat | head -c 200
# → data: {"type":"token","value":"### Phạt"}\n\ndata: ...
```

---

## 10. Maintenance & Rotation

Vì 1 số secrets đã xuất hiện trong các log/chat (trong quá trình debug), **bắt buộc rotate trước khi public repo** hoặc sau khi xong demo.

### 10.1 Supabase DB password

1. https://supabase.com/dashboard → project → ⚙ Settings → Database → **Reset database password**
2. Tránh chứa `@ : / # ? + % &` (đỡ phải URL-encode)
3. Update 2 nơi:
   - HF Space `CHECKPOINT_DB_URL`
   - Vercel `DATABASE_URL`

### 10.2 Qdrant Cloud API key

1. https://cloud.qdrant.io → Access Management → API Keys → **Create** key mới → **Delete** key cũ
2. Update HF Space `QDRANT_API_KEY` → Space tự restart

### 10.3 HF Access Token

1. https://huggingface.co/settings/tokens → Invalidate token cũ → Create token mới (role Write)
2. Chỉ dùng cho `git push` code lên Space; không cần update runtime

### 10.4 Gemini API key

1. https://aistudio.google.com/app/apikey → Delete key cũ → Create key mới
2. Update HF Space `API_KEY` + `.env` local (nếu vẫn chạy local)

### 10.5 Update Frontend nếu URL Vercel đổi

Nếu vào **Vercel → Domains → Add Custom Domain** hoặc đổi project name:
- Update HF Space `CORS_ALLOWED_ORIGINS`
- Update Vercel env var `NEXTAUTH_URL`
- Redeploy Vercel

---

## 11. Cost & Limits

| Service | Tier | Limit thực tế | Đủ cho |
|---|---|---|---|
| Vercel Hobby | Free | 100GB bandwidth/tháng, SSL auto | 50k+ chat session/tháng |
| HF Spaces CPU basic | Free | 16GB RAM, 2 vCPU, không sleep (Docker SDK) | 24/7 uptime |
| Qdrant Cloud | Free | 1GB cluster, 1 AZ, 1 cluster/account | 2818 chunks × 768d = ~8MB (dư hàng trăm lần) |
| Supabase | Free | 500MB Postgres, **auto-pause sau 7 ngày unused** | Demo + dev |
| Gemini API | Free | ~15 req/min, 200 req/ngày | Demo, không production |
| Tavily | Free | 1000 search/tháng | Web fallback hiếm gọi |

**Đáng lưu ý**:
- Supabase free tier **pause sau 7 ngày không request** → cần cron ping định kỳ:
  ```bash
  # crontab -e
  0 */3 * * * curl -s https://traffic-rag.vercel.app/ > /dev/null
  ```
- Gemini quota daily reset 00:00 UTC. Nếu hết: rotate key hoặc đợi.
- HF Spaces Docker SDK **không sleep** (khác với Gradio/Streamlit Spaces). Container chạy 24/7 trừ khi manually stop.

---

## 12. Upgrade path

Khi nào nên trả tiền:

| Triệu chứng | Upgrade |
|---|---|
| Supabase pause khi user vắng | Supabase Pro $25/mo (no pause + 8GB) |
| Gemini quota hết liên tục | Trả tiền Gemini (no daily limit) |
| Qdrant query chậm | Qdrant Cloud paid (HA + scale) |
| Vercel function timeout 10s | Vercel Pro $20/mo (60s timeout) |
| Cần custom domain HTTPS | Vercel + domain $1-15/năm |

Hoặc migrate ngược về **1 VPS** (Contabo $6.5/mo) — xem [DEPLOY.md](DEPLOY.md).

---

## 13. Quick redeploy checklist

Sau khi sửa code, push lên repo tương ứng:

```bash
# Backend HF Space
cd /path/to/hf-space-repo
cp /path/to/GitHub1/traffic_rag/api/main.py ./traffic_rag/api/main.py
# (hoặc các file khác đã sửa)
git add . && git commit -m "Update backend" && git push
# → HF auto rebuild ~10 phút

# Frontend Vercel
cd /path/to/GitHub1/traffic_rag
git add . && git commit -m "Update frontend" && git push origin main
# → Vercel auto rebuild ~3 phút (chỉ rebuild thay đổi)
```

---

## 14. Tham khảo

- [DEPLOY-HYBRID.md](DEPLOY-HYBRID.md) — hướng dẫn lý thuyết (đã viết trước khi deploy)
- [DEPLOY.md](DEPLOY.md) — deploy alternative: 1 VPS Oracle Cloud ARM
- [huggingface/Dockerfile](huggingface/Dockerfile) — Docker image cho HF Space
- [scripts/upload_qdrant_to_cloud.sh](scripts/upload_qdrant_to_cloud.sh) — script upload data
- Báo cáo §9: [bao_cao_he_thong.md](../doc/bao_cao_he_thong.md)
- Sơ đồ §9: [so_do_he_thong.md](../doc/so_do_he_thong.md)
