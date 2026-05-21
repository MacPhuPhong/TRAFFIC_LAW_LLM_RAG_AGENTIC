// src/components/Logo.tsx
// 4 logo variants for Trợ lý Luật Giao thông.
// Renders inside a colored badge of any size.
'use client';

export type LogoVariant = 'shieldSignal' | 'balanceRoad' | 'stopJustice' | 'wheelBook';

interface Props { size?: number; variant?: LogoVariant; className?: string; }

export function Logo({ size = 36, variant = 'shieldSignal', className }: Props) {
  const inner = size * 0.68;
  return (
    <div
      className={className}
      style={{
        width: size,
        height: size,
        borderRadius: size * 0.28,
        background: 'linear-gradient(135deg, var(--primary), var(--primary-hover))',
        display: 'grid',
        placeItems: 'center',
        color: '#fff',
        boxShadow: '0 1px 0 rgba(255,255,255,.3) inset, 0 6px 16px rgba(15,45,92,.20)',
        flexShrink: 0,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background:
            'radial-gradient(120% 60% at 30% 0%, rgba(255,255,255,.18), transparent 60%)',
          pointerEvents: 'none',
        }}
      />
      <Mark size={inner} variant={variant} />
    </div>
  );
}

function Mark({ size, variant }: { size: number; variant: LogoVariant }) {
  switch (variant) {
    case 'balanceRoad':
      return (
        <svg viewBox="0 0 32 32" width={size} height={size} fill="none"
          stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M16 8v17M11 27h10" />
          <path d="M6 11h20" strokeWidth="2.2" />
          <path d="M11 11h.01M16 11h.01M21 11h.01" strokeWidth="1.6" opacity=".55" />
          <path d="M3.5 11 6 16h6L9.5 11M22.5 11 26 16h-6l2.5-5" />
          <circle cx="16" cy="7.5" r="1.3" fill="#fff" stroke="none" />
        </svg>
      );
    case 'stopJustice':
      return (
        <svg viewBox="0 0 32 32" width={size} height={size} fill="none">
          <path d="M11.5 4h9L27 10.5v9L20.5 26h-9L5 19.5v-9z"
            stroke="#fff" strokeWidth="1.8" strokeLinejoin="round" />
          <g stroke="#fff" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
            <path d="m13.5 11 5.5 5.5M11.5 13l5.5 5.5" />
            <path d="m18 14.5-5 5-1.7-1.7 5-5" />
            <path d="M10 21.5h5" />
          </g>
        </svg>
      );
    case 'wheelBook':
      return (
        <svg viewBox="0 0 32 32" width={size} height={size} fill="none"
          stroke="#fff" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="16" cy="16" r="11" />
          <circle cx="16" cy="16" r="2.4" fill="#fff" stroke="none" />
          <path d="M16 5v3.5M16 23.5V27M5 16h3.5M23.5 16H27" />
          <path d="M10 13.5c2-.6 4-.4 6 .8 2-1.2 4-1.4 6-.8v5c-2-.6-4-.4-6 .8-2-1.2-4-1.4-6-.8z"
            fill="rgba(255,255,255,.15)" />
          <path d="M16 14.3v5.3" />
        </svg>
      );
    case 'shieldSignal':
    default:
      return (
        <svg viewBox="0 0 32 32" width={size} height={size} fill="none">
          <path
            d="M16 3 6 6.4v9.5c0 6.4 4.3 11.2 10 12.5 5.7-1.3 10-6.1 10-12.5V6.4z"
            stroke="#fff" strokeWidth="1.8" strokeLinejoin="round"
          />
          <circle cx="16" cy="11" r="1.7" fill="#fff"
            style={{ animation: 'tlgt-signal 2.4s infinite', animationDelay: '0s' }} />
          <circle cx="16" cy="16" r="1.7" fill="#fff"
            style={{ animation: 'tlgt-signal 2.4s infinite', animationDelay: '.8s' }} />
          <circle cx="16" cy="21" r="1.7" fill="#fff"
            style={{ animation: 'tlgt-signal 2.4s infinite', animationDelay: '1.6s' }} />
        </svg>
      );
  }
}
