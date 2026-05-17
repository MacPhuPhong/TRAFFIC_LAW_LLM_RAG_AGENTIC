// src/lib/admin.ts — shared helpers to gate the /admin surface.
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/authOptions';

export interface AdminCheck {
  ok: boolean;
  email: string | null;
  reason?: string;
}

function parseAllowlist(raw: string | undefined): { wildcard: boolean; emails: Set<string> } {
  const trimmed = (raw ?? '').trim();
  if (!trimmed) return { wildcard: false, emails: new Set() };
  if (trimmed === '*') return { wildcard: true, emails: new Set() };
  const emails = new Set(
    trimmed.split(',').map((e) => e.trim().toLowerCase()).filter(Boolean),
  );
  return { wildcard: false, emails };
}

export async function isAdminRequest(): Promise<AdminCheck> {
  const session = await getServerSession(authOptions);
  const email = (session?.user?.email ?? null)?.toLowerCase() ?? null;
  const { wildcard, emails } = parseAllowlist(process.env.ADMIN_EMAILS);

  // Dev wildcard: skip auth entirely so the admin console is reachable
  // without Google OAuth credentials configured.
  if (wildcard) {
    return { ok: true, email: email ?? 'dev@local' };
  }
  if (emails.size === 0) {
    return {
      ok: false,
      email,
      reason: 'ADMIN_EMAILS chưa được cấu hình. Đặt ADMIN_EMAILS=email@... (hoặc * để tạm mở) trong .env.local.',
    };
  }
  if (!email) {
    return { ok: false, email, reason: 'Bạn cần đăng nhập trước.' };
  }
  if (!emails.has(email)) {
    return { ok: false, email, reason: 'Tài khoản này không có quyền admin.' };
  }
  return { ok: true, email };
}
