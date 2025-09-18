// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: [
    '@fullcalendar/react',
    '@fullcalendar/core',
    '@fullcalendar/common',
    '@fullcalendar/daygrid',
    '@fullcalendar/timegrid',
  ],
};

module.exports = nextConfig;
