@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    --font-serif: 'Source Serif 4', 'Iowan Old Style', Georgia, serif;

    /* Navy palette (default) */
    --bg: #F7F8FA;
    --surface: #FFFFFF;
    --surface-muted: #F1F3F7;
    --border: #E5E8EE;
    --border-soft: #EEF1F5;
    --text: #0E1B2C;
    --text-muted: #5B6675;
    --text-faint: #8A94A3;
    --primary: #0F2D5C;
    --primary-hover: #16407F;
    --accent: #2563EB;
    --accent-soft: #E7EEFB;
    --chip: #EEF2F8;
    --success: #0F7A4A;
    --warn: #B45309;
    --danger: #B42318;
  }

  [data-theme='teal'] {
    --bg: #F6F9F9;
    --surface-muted: #F1F5F5;
    --border: #E2EAEA;
    --border-soft: #ECF1F1;
    --text: #0B2024;
    --text-muted: #536267;
    --text-faint: #8A9698;
    --primary: #0E6B6B;
    --primary-hover: #0B5757;
    --accent: #0E9F9F;
    --accent-soft: #E0F2F2;
    --chip: #ECF3F3;
  }

  [data-theme='dark'] {
    --bg: #0B1220;
    --surface: #111A2E;
    --surface-muted: #16203A;
    --border: #25304A;
    --border-soft: #1C2640;
    --text: #E6ECF5;
    --text-muted: #98A4BA;
    --text-faint: #6A7793;
    --primary: #4A78D6;
    --primary-hover: #6B92E0;
    --accent: #6B92E0;
    --accent-soft: #1E2C4A;
    --chip: #1A2440;
  }

  html, body { height: 100%; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    -webkit-font-smoothing: antialiased;
  }
  *::selection { background: rgba(37, 99, 235, 0.18); }
}

@layer components {
  .chip {
    @apply inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium border;
    background: var(--chip);
    border-color: var(--border);
    color: var(--text-muted);
  }
  .chip-success {
    background: #E7F5EE;
    border-color: #C9E8D6;
    color: var(--success);
  }
  .chip-accent {
    background: var(--accent-soft);
    border-color: var(--border);
    color: var(--primary);
  }
  .ref-badge {
    @apply inline-flex items-center justify-center min-w-[18px] h-[18px] px-1.5 ml-1
           text-[10px] font-semibold rounded-md cursor-pointer tabular-nums transition;
    background: var(--accent-soft);
    border: 1px solid var(--border);
    color: var(--primary);
  }
  .ref-badge:hover { background: var(--primary); color: white; }
  .ref-badge.active { background: var(--primary); color: white; }

  .btn-primary {
    @apply inline-flex items-center justify-center gap-2 px-3 py-2 rounded-lg
           text-[13px] font-semibold text-white transition;
    background: var(--primary);
  }
  .btn-primary:hover { background: var(--primary-hover); }

  .btn-ghost {
    @apply inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[12px]
           font-medium transition;
    color: var(--text-muted);
    border: 1px solid var(--border-soft);
  }
  .btn-ghost:hover { background: var(--surface-muted); color: var(--text); }

  .icon-btn {
    @apply grid place-items-center w-7 h-7 rounded-md cursor-pointer transition;
    color: var(--text-muted);
  }
  .icon-btn:hover { background: var(--surface-muted); color: var(--text); }

  /* Markdown content styling */
  .prose-answer h1, .prose-answer h2, .prose-answer h3 {
    color: var(--text);
    font-weight: 650;
    letter-spacing: -0.01em;
    margin: 1.4em 0 0.6em;
  }
  .prose-answer h1 { font-size: 22px; }
  .prose-answer h2 { font-size: 18px; }
  .prose-answer h3 { font-size: 16px; }
  .prose-answer p { margin: 0 0 0.9em; line-height: 1.65; }
  .prose-answer ul, .prose-answer ol { margin: 0 0 0.9em; padding-left: 1.4em; }
  .prose-answer li { margin: 0.3em 0; line-height: 1.6; }
  .prose-answer strong { color: var(--text); font-weight: 600; }
  .prose-answer code {
    background: var(--surface-muted);
    padding: 0.1em 0.4em;
    border-radius: 4px;
    font-size: 0.92em;
  }
  .prose-answer blockquote {
    border-left: 3px solid var(--accent);
    background: var(--surface-muted);
    padding: 0.8em 1em;
    margin: 1em 0;
    color: var(--text-muted);
  }
}

/* ─── Animations (typing, shimmer, pop, etc.) ──────────────────────── */
@keyframes tlgt-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
@keyframes tlgt-bounce {
  0%, 80%, 100% { transform: translateY(0); opacity: .45; }
  40% { transform: translateY(-3px); opacity: 1; }
}
@keyframes tlgt-shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
@keyframes tlgt-pulse { 0%, 100% { opacity: .55; } 50% { opacity: 1; } }
@keyframes tlgt-fadein {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: none; }
}
@keyframes tlgt-pop {
  0% { transform: scale(.6); opacity: 0; }
  60% { transform: scale(1.15); }
  100% { transform: scale(1); opacity: 1; }
}
@keyframes tlgt-spin { to { transform: rotate(360deg); } }
@keyframes tlgt-signal {
  0%, 33% { opacity: .25; }
  16% { opacity: 1; }
}

.tlgt-cursor::after {
  content: '';
  display: inline-block;
  width: 2px;
  height: 1.05em;
  vertical-align: -2px;
  margin-left: 2px;
  background: currentColor;
  animation: tlgt-blink 1s steps(2) infinite;
}
.tlgt-dot {
  display: inline-block;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: currentColor;
  margin: 0 2px;
  animation: tlgt-bounce 1.2s infinite;
}
.tlgt-dot:nth-child(2) { animation-delay: .15s; }
.tlgt-dot:nth-child(3) { animation-delay: .3s; }
.tlgt-fade-in { animation: tlgt-fadein .35s ease-out both; }
.tlgt-pop { animation: tlgt-pop .35s cubic-bezier(.34, 1.56, .64, 1) both; }
.tlgt-pulse { animation: tlgt-pulse 1.8s ease-in-out infinite; }
.tlgt-spin { animation: tlgt-spin 1.2s linear infinite; }
