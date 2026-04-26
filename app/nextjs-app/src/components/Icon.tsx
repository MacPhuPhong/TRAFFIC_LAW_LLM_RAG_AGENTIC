// src/components/Icon.tsx
'use client';

interface Props { name: string; size?: number; className?: string; strokeWidth?: number; }

export function Icon({ name, size = 16, className, strokeWidth = 1.6 }: Props) {
  const p = {
    width: size, height: size, viewBox: '0 0 24 24', fill: 'none',
    stroke: 'currentColor', strokeWidth, strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const, className,
  };
  switch (name) {
    case 'plus': return <svg {...p}><path d="M12 5v14M5 12h14"/></svg>;
    case 'search': return <svg {...p}><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>;
    case 'send': return <svg {...p}><path d="M12 19V5M5 12l7-7 7 7"/></svg>;
    case 'panel': return <svg {...p}><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></svg>;
    case 'sparkle': return <svg {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8"/></svg>;
    case 'copy': return <svg {...p}><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>;
    case 'share': return <svg {...p}><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8.6 13.5 6.8 4M15.4 6.5l-6.8 4"/></svg>;
    case 'pdf': return <svg {...p}><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>;
    case 'thumb-up': return <svg {...p}><path d="M7 10v10H4V10zM7 10l4-7a2 2 0 0 1 2 2v4h5a2 2 0 0 1 2 2.4l-1.4 7A2 2 0 0 1 16.6 20H7"/></svg>;
    case 'thumb-down': return <svg {...p} style={{transform:'scale(1,-1)'}}><path d="M7 10v10H4V10zM7 10l4-7a2 2 0 0 1 2 2v4h5a2 2 0 0 1 2 2.4l-1.4 7A2 2 0 0 1 16.6 20H7"/></svg>;
    case 'refresh': return <svg {...p}><path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/></svg>;
    case 'chevron-down': return <svg {...p}><path d="m6 9 6 6 6-6"/></svg>;
    case 'gavel': return <svg {...p}><path d="m14 4 6 6M11 7l6 6M9 9l6 6"/><path d="m17 13-9 9-3-3 9-9M3 21h7"/></svg>;
    case 'doc': return <svg {...p}><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M8 13h8M8 17h6"/></svg>;
    case 'license': return <svg {...p}><rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="12" r="2.5"/><path d="M14 10h4M14 14h3"/></svg>;
    case 'fine': return <svg {...p}><circle cx="12" cy="12" r="9"/><path d="M9 9.5C9 8 10.3 7 12 7s3 1 3 2.5c0 2.5-3 2-3 4.5M12 17.5h.01"/></svg>;
    case 'calendar': return <svg {...p}><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>;
    case 'paperclip': return <svg {...p}><path d="m21 11-9 9a5 5 0 0 1-7-7l9-9a3.5 3.5 0 0 1 5 5l-8.5 8.5a2 2 0 0 1-3-3l7.5-7.5"/></svg>;
    case 'book': return <svg {...p}><path d="M4 5a2 2 0 0 1 2-2h12v18H6a2 2 0 0 1-2-2zM4 19a2 2 0 0 1 2-2h12"/></svg>;
    case 'external': return <svg {...p}><path d="M14 4h6v6M20 4 10 14M19 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h6"/></svg>;
    case 'pin': return <svg {...p}><path d="M9 4h6l-1 6 4 4H6l4-4z"/><path d="M12 14v7"/></svg>;
    case 'more': return <svg {...p}><circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/></svg>;
    case 'check': return <svg {...p}><path d="m5 13 4 4L19 7"/></svg>;
    case 'trash': return <svg {...p}><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M6 6l1 14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-14"/></svg>;
    case 'menu': return <svg {...p}><path d="M3 6h18M3 12h18M3 18h18"/></svg>;
    default: return null;
  }
}
