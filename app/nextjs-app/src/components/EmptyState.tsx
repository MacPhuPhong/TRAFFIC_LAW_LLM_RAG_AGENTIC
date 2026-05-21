// src/components/EmptyState.tsx
'use client';

import { Icon } from './Icon';
import { Logo } from './Logo';
import { LOGO_VARIANT } from '@/lib/config';

interface Props { onPick: (q: string) => void; }

const SUGGESTIONS = [
  { cat: 'Mức phạt', icon: 'fine', q: 'Mức phạt vượt đèn đỏ với xe máy theo Nghị định 168/2024' },
  { cat: 'Thủ tục', icon: 'doc', q: 'Thủ tục đăng ký xe máy lần đầu cần giấy tờ gì' },
  { cat: 'GPLX', icon: 'license', q: 'Quy định trừ điểm giấy phép lái xe năm 2025' },
  { cat: 'Hiệu lực', icon: 'calendar', q: 'Các thông tư giao thông còn hiệu lực hiện nay' },
];

export function EmptyState({ onPick }: Props) {
  const hour = new Date().getHours();
  const greeting = hour < 11 ? 'Chào buổi sáng' : hour < 18 ? 'Chào buổi chiều' : 'Chào buổi tối';

  return (
    <div className="max-w-[760px] mx-auto px-8 py-10">
      <div className="text-center mb-9">
        <Logo size={64} variant={LOGO_VARIANT} className="mx-auto mb-4" />
        <h1 className="m-0 text-3xl font-[650] tracking-tighter2 text-text">
          {greeting}, hỏi gì về luật giao thông?
        </h1>
        <p className="mt-2 mb-0 text-text-muted text-[15px]">
          Tra cứu mức phạt, quy định đăng ký, GPLX và các thông tư mới nhất theo Nghị định 168/2024.
        </p>
      </div>

      <div className="flex items-center gap-2 mb-3 text-[12px] text-text-faint font-semibold uppercase tracking-[0.06em]">
        <Icon name="sparkle" size={13} className="text-accent" /> Câu hỏi gợi ý
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
        {SUGGESTIONS.map((p) => (
          <button
            key={p.q}
            onClick={() => onPick(p.q)}
            className="text-left bg-surface border border-border rounded-xl p-3.5 flex gap-3 transition hover:border-primary/30 hover:shadow-soft"
          >
            <div className="w-8 h-8 rounded-lg bg-accent-soft text-primary grid place-items-center shrink-0">
              <Icon name={p.icon} size={16} />
            </div>
            <div className="min-w-0">
              <div className="text-[10.5px] font-semibold text-text-faint uppercase tracking-[0.06em] mb-0.5">
                {p.cat}
              </div>
              <div className="text-[13px] text-text leading-snug">{p.q}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
