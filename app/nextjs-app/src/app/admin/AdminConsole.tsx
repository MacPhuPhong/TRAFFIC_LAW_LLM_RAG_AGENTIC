'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';

interface PendingItem {
  thread_id: string;
  query: string;
  draft_answer: string | null;
  sources: any[];
  category: string | null;
  model_info: string | null;
  expanded_query: string | null;
  created_at: number;
}

interface PendingResp {
  count: number;
  items: PendingItem[];
}

function relTime(ts: number): string {
  const diff = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (diff < 60) return `${diff}s trước`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m trước`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h trước`;
  return `${Math.floor(diff / 86400)}d trước`;
}

export default function AdminConsole({ adminEmail }: { adminEmail: string }) {
  const [items, setItems] = useState<PendingItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const res = await fetch('/api/admin/pending', { cache: 'no-store' });
      const data: PendingResp | { error: string } = await res.json();
      if (!res.ok || 'error' in data) {
        throw new Error(('error' in data && data.error) || `HTTP ${res.status}`);
      }
      setItems(data.items);
      setDrafts((prev) => {
        const next = { ...prev };
        for (const it of data.items) {
          if (next[it.thread_id] === undefined) next[it.thread_id] = it.draft_answer ?? '';
        }
        return next;
      });
      if (selectedId && !data.items.find((x) => x.thread_id === selectedId)) {
        setSelectedId(data.items[0]?.thread_id ?? null);
      } else if (!selectedId) {
        setSelectedId(data.items[0]?.thread_id ?? null);
      }
    } catch (e: any) {
      setErr(e?.message ?? 'Không tải được danh sách.');
    } finally {
      setLoading(false);
    }
  }, [selectedId]);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 7000);
    return () => clearInterval(id);
  }, [refresh]);

  const selected = useMemo(
    () => items.find((x) => x.thread_id === selectedId) ?? null,
    [items, selectedId],
  );

  const decide = useCallback(
    async (tid: string, approved: boolean) => {
      setBusy(tid);
      setErr(null);
      try {
        const editedDraft = drafts[tid];
        const original = items.find((x) => x.thread_id === tid)?.draft_answer ?? '';
        const override =
          approved && editedDraft && editedDraft.trim() !== original.trim()
            ? { draft_answer: editedDraft.trim() }
            : undefined;
        const res = await fetch(`/api/admin/resume/${tid}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ approved, override }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
        setFlash(
          approved ? 'Đã phê duyệt và gửi cho người dùng.' : 'Đã từ chối câu trả lời.',
        );
        await refresh();
      } catch (e: any) {
        setErr(e?.message ?? 'Lỗi khi xử lý.');
      } finally {
        setBusy(null);
      }
    },
    [drafts, items, refresh],
  );

  return (
    <main className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-slate-900">
              ⚖️ Bảng điều khiển Admin · Duyệt câu trả lời từ Internet
            </h1>
            <p className="text-xs text-slate-500 mt-0.5">
              Đăng nhập: <code>{adminEmail}</code> · Tổng số chờ duyệt:{' '}
              <span className="font-semibold text-slate-700">{items.length}</span>
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={refresh}
              disabled={loading}
              className="text-sm rounded-lg border border-slate-200 hover:bg-slate-100 px-3 py-1.5 text-slate-700"
            >
              {loading ? 'Đang tải…' : 'Làm mới'}
            </button>
            <Link
              href="/"
              className="text-sm rounded-lg border border-slate-200 hover:bg-slate-100 px-3 py-1.5 text-slate-700"
            >
              ← Chat
            </Link>
          </div>
        </div>
      </header>

      {flash && (
        <div className="max-w-7xl mx-auto px-6 mt-3">
          <div className="rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-sm px-4 py-2 flex items-center justify-between">
            <span>{flash}</span>
            <button onClick={() => setFlash(null)} className="text-emerald-600">
              ✕
            </button>
          </div>
        </div>
      )}
      {err && (
        <div className="max-w-7xl mx-auto px-6 mt-3">
          <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-sm px-4 py-2">
            {err}
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-6 py-6 grid grid-cols-12 gap-6">
        <aside className="col-span-12 md:col-span-4 lg:col-span-3">
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-4 py-2 border-b border-slate-200 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Câu hỏi đang chờ
            </div>
            {items.length === 0 && !loading && (
              <div className="px-4 py-8 text-center text-sm text-slate-500">
                Không có yêu cầu nào đang chờ duyệt.
              </div>
            )}
            <ul className="divide-y divide-slate-100">
              {items.map((it) => (
                <li key={it.thread_id}>
                  <button
                    onClick={() => setSelectedId(it.thread_id)}
                    className={`w-full text-left px-4 py-3 hover:bg-slate-50 transition ${
                      selectedId === it.thread_id ? 'bg-blue-50' : ''
                    }`}
                  >
                    <div className="text-sm font-medium text-slate-900 line-clamp-2">
                      {it.query || <em className="text-slate-400">(không có nội dung)</em>}
                    </div>
                    <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
                      <span className="rounded-full bg-slate-100 px-2 py-0.5 font-medium">
                        {it.category ?? '—'}
                      </span>
                      <span>{relTime(it.created_at)}</span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </aside>

        <section className="col-span-12 md:col-span-8 lg:col-span-9">
          {!selected && (
            <div className="bg-white border border-slate-200 rounded-xl p-12 text-center text-slate-500 text-sm">
              Chọn một câu hỏi ở danh sách bên trái để bắt đầu duyệt.
            </div>
          )}
          {selected && (
            <article className="bg-white border border-slate-200 rounded-xl shadow-sm">
              <header className="px-6 py-4 border-b border-slate-200">
                <div className="text-xs uppercase tracking-wide text-slate-500 font-semibold">
                  Câu hỏi của người dùng
                </div>
                <h2 className="mt-1 text-lg font-semibold text-slate-900">
                  {selected.query}
                </h2>
                <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-500">
                  <span>
                    Thread: <code>{selected.thread_id}</code>
                  </span>
                  <span>· Category: {selected.category ?? '—'}</span>
                  <span>· Model: {selected.model_info ?? '—'}</span>
                  <span>· Tạo: {relTime(selected.created_at)}</span>
                </div>
              </header>

              <div className="px-6 py-5">
                <label className="block text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
                  Bản nháp (có thể chỉnh sửa trước khi phê duyệt)
                </label>
                <textarea
                  value={drafts[selected.thread_id] ?? ''}
                  onChange={(e) =>
                    setDrafts((prev) => ({ ...prev, [selected.thread_id]: e.target.value }))
                  }
                  rows={12}
                  className="w-full rounded-lg border border-slate-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 px-3 py-2 text-sm font-mono leading-relaxed"
                />
              </div>

              {selected.sources?.length > 0 && (
                <div className="px-6 py-5 border-t border-slate-200 bg-slate-50">
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-3">
                    Nguồn (
                    {selected.sources.length})
                  </div>
                  <ul className="space-y-2 text-sm">
                    {selected.sources.map((s: any, i: number) => (
                      <li key={i} className="flex gap-2">
                        <span className="text-slate-400 shrink-0">[{i + 1}]</span>
                        <div className="min-w-0">
                          {s.url ? (
                            <a
                              href={s.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:underline break-all"
                            >
                              {s.title || s.url}
                            </a>
                          ) : (
                            <span className="text-slate-700">
                              {s.ten_van_ban || s.doc_id || 'Văn bản'}
                            </span>
                          )}
                          {s.excerpt && (
                            <p className="mt-0.5 text-xs text-slate-500 line-clamp-3">
                              {s.excerpt}
                            </p>
                          )}
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <footer className="px-6 py-4 border-t border-slate-200 flex items-center justify-end gap-3 bg-white rounded-b-xl">
                <button
                  onClick={() => decide(selected.thread_id, false)}
                  disabled={busy === selected.thread_id}
                  className="rounded-lg border border-rose-200 hover:bg-rose-50 text-rose-700 text-sm font-semibold px-4 py-2 disabled:opacity-50"
                >
                  Từ chối
                </button>
                <button
                  onClick={() => decide(selected.thread_id, true)}
                  disabled={busy === selected.thread_id}
                  className="rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-5 py-2 disabled:opacity-50"
                >
                  {busy === selected.thread_id ? 'Đang xử lý…' : 'Phê duyệt & gửi'}
                </button>
              </footer>
            </article>
          )}
        </section>
      </div>
    </main>
  );
}
