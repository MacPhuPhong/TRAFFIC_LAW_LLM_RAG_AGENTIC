// src/lib/useChat.ts — handles streaming + citation parsing
'use client';

import { useCallback, useState } from 'react';
import { useChatStore } from './store';
import type { Citation } from './types';

interface SendOptions { conversationId: string; content: string; }

export function useChat() {
  const { appendMessage, updateLastAssistant } = useChatStore();
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = useCallback(async ({ conversationId, content }: SendOptions) => {
    setError(null);
    // Read fresh state — closure'd `conversations` would be stale on the very
    // first send (when newConversation() was just called in the same tick).
    const conv = useChatStore.getState().conversations.find((c) => c.id === conversationId);
    if (!conv) return;

    // Append user msg + empty assistant placeholder
    const userMsg = {
      id: crypto.randomUUID(), role: 'user' as const, content, createdAt: Date.now(),
    };
    const assistantMsg = {
      id: crypto.randomUUID(), role: 'assistant' as const, content: '',
      citations: [] as Citation[], createdAt: Date.now(),
    };
    appendMessage(conversationId, userMsg);
    appendMessage(conversationId, assistantMsg);

    setStreaming(true);
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          thread_id: conversationId,
          messages: [...conv.messages, userMsg].map((m) => ({
            role: m.role, content: m.content,
          })),
        }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let acc = '';
      let citations: Citation[] = [];
      let pendingThreadId: string | null = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === '[DONE]') continue;
          try {
            const ev = JSON.parse(payload);
            if (ev.type === 'token') {
              acc += ev.value;
              updateLastAssistant(conversationId, { content: acc });
            } else if (ev.type === 'citations') {
              citations = ev.value;
              updateLastAssistant(conversationId, { citations });
            } else if (ev.type === 'pending') {
              pendingThreadId = ev.value as string;
            }
          } catch { /* ignore bad line */ }
        }
      }
      updateLastAssistant(conversationId, { content: acc, citations });

      if (pendingThreadId) {
        // Poll status until admin decides (or 10 minutes max).
        const startedAt = Date.now();
        const TIMEOUT_MS = 10 * 60 * 1000;
        const POLL_MS = 5000;
        const tid = pendingThreadId;
        const poll = async () => {
          while (Date.now() - startedAt < TIMEOUT_MS) {
            await new Promise((r) => setTimeout(r, POLL_MS));
            try {
              const sres = await fetch(`/api/chat/status?thread_id=${encodeURIComponent(tid)}`, {
                cache: 'no-store',
              });
              const sjson = await sres.json();
              if (sjson.status === 'completed' || sjson.status === 'rejected') {
                updateLastAssistant(conversationId, {
                  content: sjson.answer || acc,
                  citations: sjson.citations ?? citations,
                });
                return;
              }
            } catch {
              /* ignore transient poll errors */
            }
          }
          updateLastAssistant(conversationId, {
            content:
              acc +
              '\n\n⏰ _Quá thời gian chờ duyệt. Hãy thử hỏi lại hoặc liên hệ admin._',
          });
        };
        // fire and forget — UI shows the holding message in the meantime
        void poll();
      }
    } catch (e: any) {
      setError(e?.message ?? 'Đã có lỗi xảy ra');
      updateLastAssistant(conversationId, {
        content: '⚠️ Không thể kết nối đến trợ lý. Vui lòng thử lại.',
      });
    } finally {
      setStreaming(false);
    }
  }, [appendMessage, updateLastAssistant]);

  return { send, streaming, error };
}
