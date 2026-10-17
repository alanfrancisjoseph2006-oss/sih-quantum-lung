import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "QuantumLung — AI-Assisted Lung Screening & Radiology Decision Support",
  description:
    "Hybrid quantum-classical AI for chest CT analysis. AI-assisted research and clinical " +
    "decision-support tool. Not a replacement for professional medical diagnosis.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        {/* Skip link — keyboard users should not tab through the whole sidebar. */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-[100]
                     focus:rounded-[var(--radius-md)] focus:bg-[var(--color-primary)] focus:px-3
                     focus:py-2 focus:text-[13px] focus:font-medium focus:text-white"
        >
          Skip to main content
        </a>
        {children}
      </body>
    </html>
  );
}
