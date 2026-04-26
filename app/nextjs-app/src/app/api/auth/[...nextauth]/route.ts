// src/app/api/auth/[...nextauth]/route.ts
import NextAuth, { type NextAuthOptions } from 'next-auth';
import GoogleProvider from 'next-auth/providers/google';

// To enable DB-backed sessions:
//   npm i @auth/prisma-adapter
//   import { PrismaAdapter } from '@auth/prisma-adapter';
//   import { prisma } from '@/lib/prisma';
//   then add `adapter: PrismaAdapter(prisma),` below.

export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID ?? '',
      clientSecret: process.env.GOOGLE_CLIENT_SECRET ?? '',
    }),
  ],
  session: { strategy: 'jwt' },
};

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
