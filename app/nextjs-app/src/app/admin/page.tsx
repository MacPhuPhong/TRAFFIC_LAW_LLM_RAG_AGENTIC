// src/app/admin/page.tsx — server component. Gates access via NextAuth +
// ADMIN_EMAILS, then renders the client-side review console.
import Link from 'next/link';
import { isAdminRequest } from '@/lib/admin';
import AdminConsole from './AdminConsole';

export const dynamic = 'force-dynamic';

export default async function AdminPage() {
  const auth = await isAdminRequest();

  if (!auth.ok) {
    return (
      <main className="min-h-screen bg-slate-50 flex items-center justify-center px-4">
        <div className="max-w-md w-full bg-white border border-slate-200 rounded-xl shadow-sm p-8">
          <h1 className="text-2xl font-bold text-slate-900 mb-2">
            Khu vực dành cho Admin
          </h1>
          <p className="text-sm text-slate-600 mb-6">{auth.reason}</p>
          <div className="flex gap-3">
            <Link
              href="/api/auth/signin"
              className="flex-1 text-center rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-4 py-2"
            >
              Đăng nhập
            </Link>
            <Link
              href="/"
              className="flex-1 text-center rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700 text-sm font-semibold px-4 py-2"
            >
              Về trang chính
            </Link>
          </div>
          {auth.email && (
            <p className="mt-4 text-xs text-slate-500">
              Đang đăng nhập với: <code>{auth.email}</code>
            </p>
          )}
        </div>
      </main>
    );
  }

  return <AdminConsole adminEmail={auth.email ?? ''} />;
}
