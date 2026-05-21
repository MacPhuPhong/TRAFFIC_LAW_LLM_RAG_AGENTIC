// src/app/api/auth/[...nextauth]/route.ts
// Next.js 14 disallows non-method exports from route handlers, so authOptions
// lives in @/lib/authOptions (also has CredentialsProvider for /admin demo).
import NextAuth from 'next-auth';
import { authOptions } from '@/lib/authOptions';

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
