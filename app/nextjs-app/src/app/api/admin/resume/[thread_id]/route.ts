// src/app/api/admin/resume/[thread_id]/route.ts — admin-gated wrapper for
// FastAPI POST /resume/{thread_id}.
import { isAdminRequest } from '@/lib/admin';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

interface ResumeBody {
  approved: boolean;
  override?: { draft_answer?: string } & Record<string, unknown>;
}

export async function POST(
  req: Request,
  { params }: { params: { thread_id: string } },
) {
  const auth = await isAdminRequest();
  if (!auth.ok) {
    return Response.json({ error: auth.reason ?? 'Forbidden' }, { status: 403 });
  }

  let body: ResumeBody;
  try {
    body = (await req.json()) as ResumeBody;
  } catch {
    return Response.json({ error: 'Invalid JSON' }, { status: 400 });
  }
  if (typeof body?.approved !== 'boolean') {
    return Response.json({ error: '`approved` (boolean) required' }, { status: 400 });
  }

  try {
    const res = await fetch(`${BACKEND_URL}/resume/${params.thread_id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return Response.json(data, { status: res.status });
    return Response.json(data);
  } catch (err: any) {
    return Response.json(
      { error: `Cannot reach backend: ${err?.message ?? err}` },
      { status: 502 },
    );
  }
}
