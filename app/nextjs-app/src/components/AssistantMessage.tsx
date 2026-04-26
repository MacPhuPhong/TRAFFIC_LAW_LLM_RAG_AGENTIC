// src/components/AssistantMessage.tsx
'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Citation, Message } from '@/lib/types';
import { Icon } from './Icon';

interface Props { msg: Message; streaming?: boolean; }

// Replace [3] / [3][4] inline tokens with clickable badges
function withCitations(text: string, onClick: (n: number) => void, active: number | null) {
  const parts: (string | JSX.Element)[] = [];
  let lastIdx = 0;
  const re = /\[(\d+)\]/g;
  let m;
  let key = 0;
  while ((m = re.exec(text))) {
    if (m.index > lastIdx) parts.push(text.slice(lastIdx, m.index));
    const n = Number(m[1]);
    parts.push(
      <button
        key={`c${key++}`}
        onClick={() => onClick(n)}
        className={`ref-badge ${active === n ? 'active' : ''}`}
      >
        {n}
      </button>
    );
    lastIdx = m.index + m[0].length;
  }
  if (lastIdx < text.length) parts.push(text.slice(lastIdx));
  return parts;
}

export function AssistantMessage({ msg, streaming }: Props) {
  const [active, setActive] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const [open, setOpen] = useState(false);
  const citations = msg.citations ?? [];

  function onCite(n: number) {
    setActive(n);
    setOpen(true);
    // Scroll citation into view
    requestAnimationFrame(() => {
      document.getElementById(`citation-${n}`)?.scrollIntoView({
        behavior: 'smooth', block: 'nearest',
      });
    });
  }

  async function copy() {
    await navigator.clipboard.writeText(msg.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="flex gap-3.5 mb-3.5">
      <div
        className="w-8 h-8 rounded-[10px] grid place-items-center text-white shrink-0"
        style={{ background: 'linear-gradient(135deg, var(--primary), var(--primary-hover))' }}
      >
        <Icon name="gavel" size={16} strokeWidth={1.8} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-2 text-[12px] text-text-faint">
          <span className="font-semibold text-text">Trợ lý Luật Giao thông</span>
          {citations.length > 0 && (
            <>
              <span>•</span>
              <span>Dựa trên {citations.length} nguồn</span>
            </>
          )}
        </div>

        <div className="prose-answer text-[14.5px] text-text">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              p: ({ children }) => <p>{processChildren(children, onCite, active)}</p>,
              li: ({ children }) => <li>{processChildren(children, onCite, active)}</li>,
            }}
          >
            {msg.content || (streaming ? '...' : '')}
          </ReactMarkdown>
        </div>

        {/* Citations collapsible */}
        {citations.length > 0 && (
          <div className="mt-3 border border-border rounded-xl bg-surface overflow-hidden">
            <button
              onClick={() => setOpen(!open)}
              className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-[13px] font-medium text-text"
            >
              <Icon name="book" size={15} className="text-primary" />
              Nguồn trích dẫn
              <span className="chip">{citations.length}</span>
              <span className={`ml-auto text-text-faint transition ${open ? 'rotate-180' : ''}`}>
                <Icon name="chevron-down" size={14} />
              </span>
            </button>
            {open && (
              <div className="border-t border-border-soft p-2.5 grid grid-cols-1 md:grid-cols-2 gap-2">
                {citations.map((c) => (
                  <CitationCard key={c.n} c={c} active={active === c.n} onClick={() => setActive(c.n)} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Action toolbar */}
        {!streaming && msg.content && (
          <div className="flex items-center gap-1 mt-3">
            <button onClick={copy} className="btn-ghost">
              <Icon name={copied ? 'check' : 'copy'} size={14} />
              {copied ? 'Đã chép' : 'Sao chép'}
            </button>
            <button className="btn-ghost" onClick={() => window.print()}>
              <Icon name="pdf" size={14} /> Xuất PDF
            </button>
            <button className="btn-ghost">
              <Icon name="share" size={14} /> Chia sẻ
            </button>
            <div className="flex-1" />
            <button className="icon-btn" title="Hữu ích">
              <Icon name="thumb-up" size={14} />
            </button>
            <button className="icon-btn" title="Không hữu ích">
              <Icon name="thumb-down" size={14} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function processChildren(children: any, onCite: (n: number) => void, active: number | null): any {
  if (typeof children === 'string') return withCitations(children, onCite, active);
  if (Array.isArray(children)) return children.map((c, i) => (
    <span key={i}>{processChildren(c, onCite, active)}</span>
  ));
  return children;
}

function CitationCard({ c, active, onClick }: { c: Citation; active: boolean; onClick: () => void }) {
  return (
    <div
      id={`citation-${c.n}`}
      onClick={onClick}
      className={`grid grid-cols-[28px_1fr] gap-2.5 p-2.5 rounded-lg cursor-pointer transition ${
        active ? 'bg-accent-soft border border-border' : 'hover:bg-surface-muted border border-transparent'
      }`}
    >
      <div
        className={`w-7 h-7 rounded-md grid place-items-center text-[11px] font-bold tabular-nums ${
          active ? 'bg-primary text-white' : 'bg-surface-muted text-text-muted'
        }`}
      >
        {c.n}
      </div>
      <div className="min-w-0">
        <div className="text-[12.5px] font-semibold text-text truncate">{c.title}</div>
        <div className="text-[11px] text-text-faint mt-0.5 flex items-center gap-1.5">
          <span>{c.org}</span>
          <span>•</span>
          <span>{c.date}</span>
        </div>
      </div>
    </div>
  );
}
