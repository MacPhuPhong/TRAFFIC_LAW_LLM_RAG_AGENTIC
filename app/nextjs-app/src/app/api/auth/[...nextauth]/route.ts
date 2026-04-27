// src/app/api/auth/[...nextauth]/route.ts
import NextAuth, { type NextAuthOptions } from 'next-auth';
import GoogleProvider from 'next-auth/providers/google';
import CredentialsProvider from 'next-auth/providers/credentials';

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
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: "Email", type: "text", placeholder: "admin@example.com" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        if (credentials?.email === 'admin@example.com' && credentials?.password === 'admin123') {
          return { id: '1', name: 'Admin', email: 'admin@example.com' };
        }
        return null;
      }
    }),
  ],
  session: { strategy: 'jwt' },
};

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
