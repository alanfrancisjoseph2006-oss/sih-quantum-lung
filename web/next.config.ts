import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /**
   * The dev overlay badge renders bottom-left, directly over the sidebar's
   * collapse control. Hidden so the shell behaves the same in development as it
   * does in production. Compile and runtime errors are still surfaced.
   */
  devIndicators: false,
};

export default nextConfig;
