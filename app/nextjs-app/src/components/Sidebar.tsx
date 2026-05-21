// src/components/Sidebar.tsx
'use client';

import { useMemo, useState } from 'react';
import { useChatStore, groupByDate } from '@/lib/store';
import { Icon } from './Icon';
import { Logo } from './Logo';
import { LOGO_VARIANT, BRAND_NAME, BRAND_SUBTITLE } from '@/lib/config';

export function Sidebar({
  onCollapse,
  onNavigate,
}: {
  onCollapse?: () => void;
  onNavigate?: () => void;
}) {
  const {
    conversations, activeId, setActive, newConversation, deleteConversation, togglePin,
  } = useChatStore();
  const [q, setQ] = useState('');
  const [menuId, setMenuId] = useState<string | null>(null);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return conversations;
    return conversations.filter((c) => c.title.toLowerCase().includes(s));
  }, [q, conversations]);

  const groups = groupByDate(filtered);

  return (
    <aside className="w-[85vw] max-w-[300px] md:w-[280px] h-full shrink-0 bg-surface border-r border-border flex flex-col">
      {/* Brand */}
      <div className="px-[18px] pt-[18px] pb-3 flex items-center gap-3">
        <Logo size={36} variant={LOGO_VARIANT} />
        <div className="min-w-0 flex-1">
          <div className="font-semibold text-[14.5px] tracking-tightish">{BRAND_NAME}</div>
          <div className="text-[11.5px] text-text-faint">{BRAND_SUBTITLE}</div>
        </div>
      </div>

      {/* New chat */}
      <div className="px-3.5 pb-2.5">
        <button onClick={() => { newConversation(); onNavigate?.(); }} className="btn-primary w-full">
          <Icon name="plus" size={15} strokeWidth={2.2} />
          Cuộc hội thoại mới
        </button>
      </div>

      {/* Search */}
      <div className="px-3.5 pb-3">
        <div className="flex items-center gap-2 px-2.5 py-2 bg-surface-muted rounded-lg">
          <Icon name="search" size={14} className="text-text-faint" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Tìm trong lịch sử..."
            className="flex-1 bg-transparent outline-none text-[12.5px] text-text"
          />
        </div>
      </div>

      {/* Groups */}
      <div className="flex-1 overflow-y-auto px-2 pb-4">
        {groups.length === 0 && (
          <div className="px-3 py-6 text-[12.5px] text-text-faint text-center">
            Chưa có hội thoại nào.<br />Bấm "Cuộc hội thoại mới" để bắt đầu.
          </div>
        )}
        {groups.map(([label, items]) => (
          <div key={label} className="mb-3.5">
            <div className="text-[10.5px] font-semibold text-text-faint uppercase tracking-[0.06em] px-2 py-1.5">
              {label}
            </div>
            {items.map((c) => {
              const active = activeId === c.id;
              const open = menuId === c.id;
              return (
                <div
                  key={c.id}
                  onClick={() => { setActive(c.id); onNavigate?.(); }}
                  className={`group relative flex items-center gap-2 px-2 py-1.5 rounded-md mb-px cursor-pointer transition ${
                    active ? 'bg-accent-soft text-primary font-medium' : 'text-text-muted hover:bg-surface-muted'
                  }`}
                >
                  <span className="text-[12.5px] truncate flex-1">{c.title}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); setMenuId(open ? null : c.id); }}
                    className="icon-btn !w-6 !h-6 opacity-100 md:opacity-0 md:group-hover:opacity-100"
                  >
                    <Icon name="more" size={14} />
                  </button>
                  {open && (
                    <div
                      className="absolute right-1 top-8 z-10 bg-surface border border-border rounded-lg shadow-soft py-1 min-w-[140px]"
                      onMouseLeave={() => setMenuId(null)}
                    >
                      <button
                        onClick={(e) => { e.stopPropagation(); togglePin(c.id); setMenuId(null); }}
                        className="w-full text-left px-3 py-1.5 text-[12px] text-text-muted hover:bg-surface-muted flex items-center gap-2"
                      >
                        <Icon name="pin" size={13} /> {c.pinned ? 'Bỏ ghim' : 'Ghim'}
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); deleteConversation(c.id); setMenuId(null); }}
                        className="w-full text-left px-3 py-1.5 text-[12px] text-danger hover:bg-surface-muted flex items-center gap-2"
                      >
                        <Icon name="trash" size={13} /> Xoá
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="px-3.5 py-3 border-t border-border-soft flex items-center gap-2.5">
        <div className="w-7 h-7 rounded-full bg-surface-muted grid place-items-center text-text-muted text-[11px] font-semibold">
          NV
        </div>
        <div className="flex-1 text-[12px] text-text font-medium">Người dùng</div>
        <button onClick={onCollapse} className="icon-btn" title="Thu gọn">
          <Icon name="panel" size={14} />
        </button>
      </div>
    </aside>
  );
}
