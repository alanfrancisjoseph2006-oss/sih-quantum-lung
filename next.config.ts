import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev overlay badge sits on top of the sidebar collapse control.
  devIndicators: false,
  eslint: { ignoreDuringBuilds: false },
};

export default nextConfig;
