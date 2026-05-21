// src/lib/config.ts — brand & feature constants
import type { LogoVariant } from '@/components/Logo';

// Đổi 1 dòng này để swap logo toàn app
export const LOGO_VARIANT: LogoVariant = 'shieldSignal';
// Tuỳ chọn: 'shieldSignal' | 'balanceRoad' | 'stopJustice' | 'wheelBook'

export const BRAND_NAME = 'Trợ lý Luật Giao thông';
export const BRAND_SUBTITLE = 'Tra cứu thông minh';

// Pipeline steps shown while AI is processing
export const THINKING_STEPS = [
  { icon: 'search', label: 'Tìm văn bản liên quan' },
  { icon: 'book', label: 'Đối chiếu điều luật' },
  { icon: 'sparkle', label: 'Tổng hợp câu trả lời' },
];
