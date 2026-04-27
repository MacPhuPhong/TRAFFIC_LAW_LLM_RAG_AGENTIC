// src/app/api/chat/status/route.ts — poll endpoint for clients to learn
// whether a thread paused at admin review has been finalised.
import type { NextRequest } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

function sourceToCitation(s: any, n: number) {
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

export async function GET(req: NextRequest) {
  const tid = req.nextUrl.searchParams.get('thread_id');
  if (!tid) return Response.json({ error: 'thread_id required' }, { status: 400 });

  try {
    const res = await fetch(`${BACKEND_URL}/pending/${tid}`, { cache: 'no-store' });
    if (!res.ok) {
      return Response.json({ status: 'unknown' }, { status: 200 });
    }
    const data: any = await res.json();
    const values = data?.values ?? {};
    const next = data?.next_nodes ?? [];

    if (Array.isArray(next) && next.includes('web_finalize')) {
      return Response.json({ status: 'pending', thread_id: tid });
    }

    const answer = values.answer || values.draft_answer || '';
    const sources = Array.isArray(values.sources) ? values.sources : [];
    return Response.json({
      status: values.refused ? 'rejected' : 'completed',
      thread_id: tid,
      answer,
      citations: sources.map((s: any, i: number) => sourceToCitation(s, i + 1)),
    });
  } catch (err: any) {
    return Response.json(
      { status: 'error', message: err?.message ?? String(err) },
      { status: 200 },
    );
  }
}
