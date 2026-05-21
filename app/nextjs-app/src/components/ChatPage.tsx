// src/components/ChatPage.tsx — main page orchestration
'use client';

import { useEffect, useRef, useState } from 'react';
import { Sidebar } from './Sidebar';
import { Composer } from './Composer';
import { AssistantMessage } from './AssistantMessage';
import { EmptyState } from './EmptyState';
import { Icon } from './Icon';
import { useChatStore } from '@/lib/store';
import { useChat } from '@/lib/useChat';

export function ChatPage() {
  const { conversations, activeId, newConversation, setDraft } = useChatStore();
  const { send, streaming } = useChat();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const conv = conversations.find((c) => c.id === activeId) ?? null;

  // Open sidebar by default on desktop; stay closed on mobile.
  useEffect(() => {
    if (typeof window !== 'undefined' && window.matchMedia('(min-width: 768px)').matches) {
      setSidebarOpen(true);
    }
  }, []);

  // Autoscroll on new content
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [conv?.messages, streaming]);

  function closeSidebarOnMobile() {
    if (typeof window !== 'undefined' && window.matchMedia('(max-width: 767px)').matches) {
      setSidebarOpen(false);
    }
  }

  function handleSend(text: string) {
    let cid = activeId;
    if (!cid) cid = newConversation();
    send({ conversationId: cid, content: text });
  }

  function pickSuggestion(q: string) {
    setDraft(q);
    // Focus the textarea after state settles
    setTimeout(() => {
      const ta = document.querySelector<HTMLTextAreaElement>('textarea');
      ta?.focus();
    }, 50);
  }

  return (
    <div className="h-screen flex bg-bg text-text overflow-hidden">
      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black/40 z-40"
          onClick={() => setSidebarOpen(false)}
          aria-hidden
        />
      )}

      {/* Sidebar: drawer on mobile, in-flow on desktop */}
      {sidebarOpen && (
        <div className="fixed md:relative inset-y-0 left-0 z-50 md:z-auto">
          <Sidebar
            onCollapse={() => setSidebarOpen(false)}
            onNavigate={closeSidebarOnMobile}
          />
        </div>
      )}

      <main className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="flex items-center justify-between px-3 md:px-7 py-3.5 border-b border-border bg-surface">
          <div className="flex items-center gap-2.5 min-w-0">
            {!sidebarOpen && (
              <button onClick={() => setSidebarOpen(true)} className="icon-btn shrink-0">
                <Icon name="menu" size={16} />
              </button>
            )}
            <div className="flex items-center gap-2 text-text-muted text-[13px] min-w-0">
              <span className="hidden md:inline">Tra cứu mức phạt, quy định, GPLX</span>
              <span className="chip chip-accent truncate">Nghị định 168/2024</span>
            </div>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <button className="icon-btn"><Icon name="share" size={15} /></button>
            <button className="icon-btn"><Icon name="more" size={16} /></button>
          </div>
        </header>

        {/* Body */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {!conv || conv.messages.length === 0 ? (
            <div className="min-h-full flex items-center">
              <div className="w-full">
                <EmptyState onPick={pickSuggestion} />
              </div>
            </div>
          ) : (
            <div className="max-w-[820px] mx-auto px-3 md:px-7 py-6 pb-32">
              {conv.messages.map((m, i) =>
                m.role === 'user' ? (
                  <div key={m.id} className="flex justify-end mb-7">
                    <div
                      className="max-w-[85%] md:max-w-[78%] px-4 py-3 rounded-2xl rounded-br-sm text-white text-[14.5px] leading-relaxed break-words"
                      style={{ background: 'var(--primary)' }}
                    >
                      {m.content}
                    </div>
                  </div>
                ) : (
                  <AssistantMessage
                    key={m.id}
                    msg={m}
                    streaming={streaming && i === conv.messages.length - 1}
                  />
                )
              )}
            </div>
          )}
        </div>

        {/* Composer dock */}
        <div
          className="px-3 md:px-7 pb-5 pt-3"
          style={{ background: 'linear-gradient(180deg, transparent, var(--bg) 30%)', marginTop: -40, position: 'relative' }}
        >
          <div className="max-w-[820px] mx-auto">
            <Composer
              onSend={handleSend}
              disabled={streaming}
              big={!conv || conv.messages.length === 0}
              hint="Câu trả lời chỉ mang tính tham khảo. Vui lòng đối chiếu với văn bản pháp luật chính thức."
            />
          </div>
        </div>
      </main>
    </div>
  );
}
