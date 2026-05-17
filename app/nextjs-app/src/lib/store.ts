// src/lib/store.ts — Zustand store with localStorage persistence
'use client';

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Conversation, Message } from './types';

interface State {
  conversations: Conversation[];
  activeId: string | null;
  draft: string;
  setDraft: (s: string) => void;
  setActive: (id: string | null) => void;
  newConversation: () => string;
  appendMessage: (cid: string, m: Message) => void;
  updateLastAssistant: (cid: string, patch: Partial<Message>) => void;
  renameConversation: (cid: string, title: string) => void;
  deleteConversation: (cid: string) => void;
  togglePin: (cid: string) => void;
}

export const useChatStore = create<State>()(
  persist(
    (set) => ({
      conversations: [],
      activeId: null,
      draft: '',
      setDraft: (s) => set({ draft: s }),
      setActive: (id) => set({ activeId: id }),
      newConversation: () => {
        const id = crypto.randomUUID();
        const now = Date.now();
        const conv: Conversation = {
          id, title: 'Cuộc hội thoại mới', messages: [], createdAt: now, updatedAt: now,
        };
        set((s) => ({ conversations: [conv, ...s.conversations], activeId: id }));
        return id;
      },
      appendMessage: (cid, m) => set((s) => ({
        conversations: s.conversations.map((c) => c.id === cid
          ? { ...c, messages: [...c.messages, m], updatedAt: Date.now(),
              title: c.messages.length === 0 && m.role === 'user'
                ? m.content.slice(0, 60) : c.title }
          : c),
      })),
      updateLastAssistant: (cid, patch) => set((s) => ({
        conversations: s.conversations.map((c) => {
          if (c.id !== cid) return c;
          const msgs = [...c.messages];
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'assistant') {
              msgs[i] = { ...msgs[i], ...patch };
              break;
            }
          }
          return { ...c, messages: msgs, updatedAt: Date.now() };
        }),
      })),
      renameConversation: (cid, title) => set((s) => ({
        conversations: s.conversations.map((c) => c.id === cid ? { ...c, title } : c),
      })),
      deleteConversation: (cid) => set((s) => ({
        conversations: s.conversations.filter((c) => c.id !== cid),
        activeId: s.activeId === cid ? null : s.activeId,
      })),
      togglePin: (cid) => set((s) => ({
        conversations: s.conversations.map((c) => c.id === cid ? { ...c, pinned: !c.pinned } : c),
      })),
    }),
    { name: 'tlgt-chat-store', version: 1 }
  )
);

// Group conversations by relative time
export function groupByDate(convs: Conversation[]) {
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today); yesterday.setDate(yesterday.getDate() - 1);
  const sevenAgo = new Date(today); sevenAgo.setDate(sevenAgo.getDate() - 7);
  const thirtyAgo = new Date(today); thirtyAgo.setDate(thirtyAgo.getDate() - 30);

  const groups: Record<string, Conversation[]> = {
    'Đã ghim': [], 'Hôm nay': [], 'Hôm qua': [], '7 ngày qua': [], 'Tháng trước': [], 'Cũ hơn': [],
  };
  for (const c of convs) {
    if (c.pinned) { groups['Đã ghim'].push(c); continue; }
    const d = new Date(c.updatedAt);
    if (d >= today) groups['Hôm nay'].push(c);
    else if (d >= yesterday) groups['Hôm qua'].push(c);
    else if (d >= sevenAgo) groups['7 ngày qua'].push(c);
    else if (d >= thirtyAgo) groups['Tháng trước'].push(c);
    else groups['Cũ hơn'].push(c);
  }
  return Object.entries(groups).filter(([, v]) => v.length > 0);
}
