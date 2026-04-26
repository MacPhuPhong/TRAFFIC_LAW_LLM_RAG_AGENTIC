import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-sans)', 'system-ui', 'sans-serif'],
        serif: ['var(--font-serif)', 'Georgia', 'serif'],
      },
      colors: {
        bg: 'var(--bg)',
        surface: 'var(--surface)',
        'surface-muted': 'var(--surface-muted)',
        border: 'var(--border)',
        'border-soft': 'var(--border-soft)',
        text: 'var(--text)',
        'text-muted': 'var(--text-muted)',
        'text-faint': 'var(--text-faint)',
        primary: 'var(--primary)',
        'primary-hover': 'var(--primary-hover)',
        accent: 'var(--accent)',
        'accent-soft': 'var(--accent-soft)',
        chip: 'var(--chip)',
        success: 'var(--success)',
        warn: 'var(--warn)',
        danger: 'var(--danger)',
      },
      letterSpacing: { tightish: '-0.01em', tighter2: '-0.02em' },
      boxShadow: {
        soft: '0 1px 2px rgba(15,45,92,.08), 0 4px 16px rgba(15,45,92,.06)',
        composer: '0 4px 24px rgba(15,45,92,.10), 0 1px 2px rgba(15,45,92,.06)',
      },
    },
  },
  plugins: [],
};
export default config;
