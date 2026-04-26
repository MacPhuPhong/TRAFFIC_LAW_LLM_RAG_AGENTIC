# Trợ lý Luật Giao thông — Next.js 14

UI hi-fi cho chatbot pháp luật giao thông Việt Nam, kết hợp:

- **A** · Cấu trúc chat 3 cột quen thuộc
- **B** · Empty state với suggested prompts
- **C** · Citation linking (click `[3]` trong câu trả lời để mở nguồn)

Stack: **Next.js 14 App Router · TypeScript · Tailwind · Zustand · Prisma · NextAuth**.

## Cài đặt

```bash
cd nextjs-app
npm install
cp .env.example .env.local
# (Không bắt buộc) Khởi tạo DB SQLite
npx prisma generate
npx prisma db push
npm run dev
```

Mở http://localhost:3000.

## Kết nối Python backend

Trong `.env.local`:

```
BACKEND_URL=http://localhost:8000
```

Frontend gọi `POST /api/chat` (Next.js API route) → forward sang `${BACKEND_URL}/chat` của Python.

### Hợp đồng API mong đợi từ backend Python

Backend cần expose `POST /chat` nhận:

```json
{ "messages": [{ "role": "user", "content": "..." }] }
```

Trả về **một trong hai**:

**(1) SSE streaming** (khuyến nghị — ChatGPT-style token-by-token):

```
Content-Type: text/event-stream

data: {"type":"token","value":"Dựa "}
data: {"type":"token","value":"trên "}
data: {"type":"citations","value":[{"n":1,"title":"...","org":"...","date":"...","type":"Thông tư"}]}
data: [DONE]
```

**(2) JSON một lần** — Next.js sẽ tự fake streaming:

```json
{
  "answer": "Markdown trả lời với [3][4] inline...",
  "citations": [{ "n": 3, "title": "...", "org": "...", "date": "..." }]
}
```

### Ví dụ FastAPI backend (Python)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json, asyncio

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])

@app.post("/chat")
async def chat(body: dict):
    async def stream():
        # Replace with your RAG pipeline (LangChain/LlamaIndex/etc)
        answer = "Dựa trên Nghị định 168/2024 [1]..."
        for ch in answer:
            yield f"data: {json.dumps({'type':'token','value':ch})}\n\n"
            await asyncio.sleep(0.01)
        citations = [{"n": 1, "title": "Nghị định 168/2024", "org": "Chính phủ", "date": "26/12/2024", "type": "Nghị định"}]
        yield f"data: {json.dumps({'type':'citations','value':citations})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
```

## Tính năng

- ✅ Streaming token-by-token (SSE)
- ✅ Markdown rendering (`react-markdown` + GFM)
- ✅ Citation linking — click `[3]` mở card nguồn
- ✅ localStorage persistence (Zustand)
- ✅ Conversation history grouped theo ngày + search
- ✅ Pin / xoá hội thoại
- ✅ Sao chép · Xuất PDF (window.print) · Chia sẻ
- ✅ Theme tokens (navy / teal / dark) — đổi `data-theme` trên `<html>`
- ✅ Auth scaffold (NextAuth + Google)
- ✅ DB schema (Prisma) — User / Conversation / Message với citations

## Bật DB persistence (thay localStorage)

1. `npm i @auth/prisma-adapter`
2. Mở `src/app/api/auth/[...nextauth]/route.ts`, bỏ comment `adapter: PrismaAdapter(prisma)`
3. Tạo các API route `/api/conversations`, `/api/conversations/[id]/messages` lưu vào Prisma
4. Đổi `useChatStore` thành SWR/React Query gọi các route đó

Mặc định project dùng localStorage để bạn dev nhanh không cần DB.

## Cấu trúc thư mục

```
src/
  app/
    api/
      auth/[...nextauth]/route.ts   # NextAuth handler
      chat/route.ts                  # SSE proxy → Python backend
    globals.css                      # Tailwind + design tokens
    layout.tsx                       # Root layout + fonts
    page.tsx                         # Home → ChatPage
    providers.tsx                    # SessionProvider wrapper
  components/
    ChatPage.tsx                     # Layout chính
    Sidebar.tsx                      # Lịch sử hội thoại + search
    Composer.tsx                     # Textarea + send button
    AssistantMessage.tsx             # Markdown + citations
    EmptyState.tsx                   # Hero + suggestions
    Icon.tsx                         # SVG icon set
  lib/
    types.ts                         # Message / Conversation / Citation
    store.ts                         # Zustand + persist
    useChat.ts                       # SSE consumer
    prisma.ts                        # Prisma client singleton
prisma/schema.prisma                 # DB schema
```

## Đổi theme

```html
<html data-theme="teal">  <!-- hoặc "dark" -->
```

Tokens định nghĩa trong `globals.css`. Thêm `[data-theme='warm']` nếu muốn palette mới.

## Production

```bash
npm run build
npm start
```

Deploy được lên Vercel / Railway / VPS. Đổi `DATABASE_URL` sang Postgres và `provider = "postgresql"` trong `schema.prisma`.
