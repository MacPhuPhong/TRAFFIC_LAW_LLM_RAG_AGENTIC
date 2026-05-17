import type { Metadata, Viewport } from 'next';
import Script from 'next/script';
import { Inter, Source_Serif_4 } from 'next/font/google';
import './globals.css';
import { Providers } from './providers';

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  themeColor: '#ffffff',
};

const inter = Inter({
  subsets: ['latin', 'vietnamese'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-sans',
});
const serif = Source_Serif_4({
  subsets: ['latin', 'vietnamese'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-serif',
});

export const metadata: Metadata = {
  title: 'Trợ lý Luật Giao thông',
  description: 'Tra cứu mức phạt, quy định đăng ký xe, GPLX theo Nghị định 168/2024',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const cfToken = process.env.NEXT_PUBLIC_CF_ANALYTICS_TOKEN;
  return (
    <html lang="vi" className={`${inter.variable} ${serif.variable}`}>
      <body>
        <Providers>{children}</Providers>
        {cfToken && (
          <Script
            id="cf-analytics"
            strategy="afterInteractive"
            src="https://static.cloudflareinsights.com/beacon.min.js"
            data-cf-beacon={JSON.stringify({ token: cfToken })}
          />
        )}
      </body>
    </html>
  );
}
