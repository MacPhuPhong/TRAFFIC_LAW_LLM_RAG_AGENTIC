/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // If you want to forward /api/chat to the Python backend instead of using
  // the local proxy route, uncomment and set BACKEND_URL.
  // async rewrites() {
  //   return [
  //     { source: '/api/backend/:path*', destination: `${process.env.BACKEND_URL}/:path*` },
  //   ];
  // },
};
export default nextConfig;
