// src/lib/types.ts
export type Role = 'user' | 'assistant';

export interface Citation {
  n: number;
  title: string;
  org: string;
  date: string;
  type: string;
  excerpt?: string;
  url?: string;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  citations?: Citation[];
  createdAt: number;
}

export interface Conversation {
  id: string;
  title: string;
  pinned?: boolean;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}
