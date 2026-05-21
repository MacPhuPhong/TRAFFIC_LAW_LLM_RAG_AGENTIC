// src/components/LoadingStates.tsx
// Loading / streaming indicators: TypingDots, ThinkingPill, ShimmerLine, TypingCursor.
'use client';

import { Icon } from './Icon';

export function TypingDots({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-text-muted text-[13px] font-medium">
      {label && <span>{label}</span>}
      <span className="inline-flex items-center text-primary">
        <span className="tlgt-dot" />
        <span className="tlgt-dot" />
        <span className="tlgt-dot" />
      </span>
    </span>
  );
}

export function TypingCursor() {
  return <span className="tlgt-cursor" aria-hidden />;
}

interface ThinkingPillProps {
  icon: string;
  label: string;
  status: 'done' | 'doing' | 'todo';
}

export function ThinkingPill({ icon, label, status }: ThinkingPillProps) {
  const isDone = status === 'done';
  const isDoing = status === 'doing';
  return (
    <div
      className={`flex items-center gap-2.5 px-3 py-1.5 rounded-full text-[12.5px] border ${
        isDoing
          ? 'bg-accent-soft border-border text-primary font-medium'
          : isDone
          ? 'bg-surface border-border-soft text-text-muted'
          : 'border-border-soft text-text-faint'
      }`}
    >
      <span className="w-4 h-4 grid place-items-center">
        {isDone ? (
          <span className="text-success">
            <Icon name="check" size={12} strokeWidth={2.5} />
          </span>
        ) : isDoing ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="tlgt-spin">
            <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.2" opacity=".25" />
            <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
          </svg>
        ) : (
          <span className="text-text-faint">
            <Icon name={icon} size={12} />
          </span>
        )}
      </span>
      {label}
    </div>
  );
}

export function ShimmerLine({
  width = '100%',
  height = 12,
  delay = 0,
}: {
  width?: string | number;
  height?: number;
  delay?: number;
}) {
  return (
    <div
      style={{
        width,
        height,
        borderRadius: 6,
        background:
          'linear-gradient(90deg, var(--surface-muted) 0%, var(--border-soft) 50%, var(--surface-muted) 100%)',
        backgroundSize: '200% 100%',
        animation: `tlgt-shimmer 1.6s linear ${delay}s infinite`,
      }}
    />
  );
}

// Convenience: ordered RAG-style thinking pipeline.
// Pass the index of the active step (0-based); earlier ones are 'done',
// later ones 'todo'. Pass `steps={null}` to hide.
interface PipelineProps {
  steps: { icon: string; label: string }[];
  active: number; // -1 = all done
}

export function ThinkingPipeline({ steps, active }: PipelineProps) {
  return (
    <div className="flex flex-col gap-1.5">
      {steps.map((s, i) => {
        const status = active === -1 || i < active ? 'done' : i === active ? 'doing' : 'todo';
        return <ThinkingPill key={i} icon={s.icon} label={s.label} status={status} />;
      })}
    </div>
  );
}
