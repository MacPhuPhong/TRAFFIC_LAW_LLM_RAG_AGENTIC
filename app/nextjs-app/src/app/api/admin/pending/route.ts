// src/app/api/admin/pending/route.ts — proxy to FastAPI GET /pending,
// gated by NextAuth + ADMIN_EMAILS allowlist.
import { isAdminRequest } from '@/lib/admin';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

export async function GET() {
  const auth = await isAdminRequest();
  if (!auth.ok) {
    return Response.json({ error: auth.reason ?? 'Forbidden' }, { status: 403 });
  }

  try {
    const res = await fetch(`${BACKEND_URL}/pending`, { cache: 'no-store' });
    if (!res.ok) {
      return Response.json({ error: `Backend returned ${res.status}` }, { status: 502 });
    }
    return Response.json(await res.json());
  } catch (err: any) {
    return Response.json(
      { error: `Cannot reach backend: ${err?.message ?? err}` },
      { status: 502 },
    );
  }
}
