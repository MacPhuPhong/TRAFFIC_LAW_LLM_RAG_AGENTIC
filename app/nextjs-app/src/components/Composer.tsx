// src/components/Composer.tsx
'use client';

import { useEffect, useRef } from 'react';
import { Icon } from './Icon';
import { useChatStore } from '@/lib/store';

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
  hint?: string;
  big?: boolean;
}

export function Composer({ onSend, disabled, placeholder, hint, big }: Props) {
  const { draft, setDraft } = useChatStore();
  const ref = useRef<HTMLTextAreaElement>(null);

  // Autosize
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, big ? 220 : 160) + 'px';
  }, [draft, big]);

  function submit() {
    const t = draft.trim();
    if (!t || disabled) return;
    onSend(t);
    setDraft('');
  }

  return (
    <div className="w-full">
      <div
        className={`flex items-end gap-2 bg-surface border border-border rounded-2xl shadow-composer transition focus-within:border-primary/40 ${
          big ? 'px-4 py-3' : 'pl-3.5 pr-2.5 py-2.5'
        }`}
      >
        <button className="icon-btn shrink-0" title="Đính kèm">
          <Icon name="paperclip" size={16} />
        </button>
        <textarea
          ref={ref}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder={placeholder ?? 'Nhập câu hỏi về luật giao thông...'}
          rows={1}
          className={`flex-1 resize-none bg-transparent outline-none text-text font-sans ${
            big ? 'text-[15px] py-1' : 'text-[14.5px] py-1.5'
          }`}
        />
        <button
          onClick={submit}
          disabled={disabled || !draft.trim()}
          className="w-9 h-9 rounded-lg grid place-items-center text-white shrink-0 transition disabled:opacity-40 disabled:cursor-not-allowed"
          style={{ background: 'var(--primary)' }}
          title="Gửi (Enter)"
        >
          <Icon name="send" size={15} strokeWidth={2.2} />
        </button>
      </div>
      {hint && <div className="text-[11px] text-text-faint text-center mt-2">{hint}</div>}
    </div>
  );
}
