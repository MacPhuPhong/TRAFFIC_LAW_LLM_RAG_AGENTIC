// src/app/api/chat/route.ts — bridge Next.js client (messages[]) with the
// FastAPI Agentic-RAG backend (query + thread_id). Streams via SSE.
import type { NextRequest } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

type Msg = { role: 'user' | 'assistant'; content: string };

interface BackendSource {
  dieu?: string | number;
  khoan?: string | number;
  diem?: string;
  ten_van_ban?: string;
  doc_id?: string;
  url?: string;
  title?: string;
  van_ban_type?: string;
  ngay_ban_hanh?: string;
  excerpt?: string;
}

interface BackendChatResponse {
  thread_id: string;
  status: 'completed' | 'pending_web_review' | 'rejected';
  answer?: string | null;
  draft_answer?: string | null;
  sources?: BackendSource[];
  category?: string | null;
  error?: string | null;
}

function lastUserContent(messages: Msg[] | undefined): string {
  if (!Array.isArray(messages)) return '';
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') return messages[i].content || '';
  }
  return '';
}

function sourceToCitation(s: BackendSource, n: number) {
  const parts: string[] = [];
  if (s.dieu) parts.push(`Điều ${s.dieu}`);
  if (s.khoan) parts.push(`Khoản ${s.khoan}`);
  if (s.diem) parts.push(`Điểm ${s.diem}`);
  const loc = parts.join(' · ');
  const docTitle = s.ten_van_ban || s.title || s.doc_id || 'Nguồn';
  return {
    n,
    title: loc ? `${docTitle} — ${loc}` : docTitle,
    org: s.van_ban_type || (s.url ? 'Web' : 'Văn bản pháp luật'),
    date: s.ngay_ban_hanh || '',
    type: s.van_ban_type || (s.url ? 'Web' : 'Pháp luật'),
    excerpt: s.excerpt,
    url: s.url,
  };
}

function buildSSE(answer: string, sources: BackendSource[]): Response {
  const stream = new ReadableStream({
    start(controller) {
      const enc = new TextEncoder();
      const text = answer || '';
      const citations = (sources || []).map((s, i) => sourceToCitation(s, i + 1));

      // Token-by-token simulated streaming for a uniform UX.
      let i = 0;
      const tick = () => {
        if (i >= text.length) {
          if (citations.length) {
            controller.enqueue(
              enc.encode(
                `data: ${JSON.stringify({ type: 'citations', value: citations })}\n\n`,
              ),
            );
          }
          controller.enqueue(enc.encode('data: [DONE]\n\n'));
          controller.close();
          return;
        }
        const chunk = text.slice(i, i + 8);
        controller.enqueue(
          enc.encode(`data: ${JSON.stringify({ type: 'token', value: chunk })}\n\n`),
        );
        i += 8;
        setTimeout(tick, 12);
      };
      tick();
    },
  });
  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}

async function callBackend(
  query: string,
  thread_id: string | undefined,
): Promise<BackendChatResponse> {
  const res = await fetch(`${BACKEND_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, thread_id }),
  });
  if (!res.ok) throw new Error(`Backend /chat returned ${res.status}`);
  return (await res.json()) as BackendChatResponse;
}

async function autoApprove(thread_id: string): Promise<BackendChatResponse> {
  // V1: auto-approve HITL web answers. A future iteration can surface the
  // draft + Approve/Reject buttons in the chat UI.
  const res = await fetch(`${BACKEND_URL}/resume/${thread_id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approved: true }),
  });
  if (!res.ok) throw new Error(`Backend /resume returned ${res.status}`);
  return (await res.json()) as BackendChatResponse;
}

export async function POST(req: NextRequest) {
  let body: { messages?: Msg[]; thread_id?: string };
  try {
    body = await req.json();
  } catch {
    return new Response('Invalid JSON', { status: 400 });
  }

  const query = lastUserContent(body.messages);
  if (!query) return new Response('No user message', { status: 400 });

  try {
    let resp = await callBackend(query, body.thread_id);

    if (resp.status === 'pending_web_review') {
      const tid = resp.thread_id || body.thread_id;
      if (tid) resp = await autoApprove(tid);
    }

    const answer =
      resp.answer ||
      resp.draft_answer ||
      resp.error ||
      'Không có nội dung trả về từ trợ lý.';
    return buildSSE(answer, resp.sources ?? []);
  } catch (err: any) {
    const msg = `⚠️ Không thể kết nối backend (${BACKEND_URL}): ${err?.message ?? err}`;
    return buildSSE(msg, []);
  }
}
