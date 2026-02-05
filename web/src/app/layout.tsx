import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AI Job Finder - 나에게 맞는 AI 직무 찾기",
  description: "AI 부트캠프 수강생을 위한 맞춤형 직무 추천 서비스. 간단한 테스트로 당신의 성향에 맞는 AI 직무를 찾아보세요.",
};

import AdBanner from "@/components/AdBanner";
import Script from "next/script";

// ... (imports)

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <Script
          async
          src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-6922660715404171"
          crossOrigin="anonymous"
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen flex flex-col`}
      >
        <main className="flex-1 w-full max-w-[1920px] mx-auto flex gap-4 p-4">

          {/* Left Ad (Desktop Only) */}
          <aside className="hidden xl:block w-[200px] shrink-0 sticky top-4 h-fit">
            <AdBanner dataAdSlot="1234567890" className="h-[600px] w-full" />
          </aside>

          {/* Main Content */}
          <section className="flex-1 w-full min-w-0">
            {children}
          </section>

          {/* Right Ad (Desktop Only) */}
          <aside className="hidden xl:block w-[200px] shrink-0 sticky top-4 h-fit">
            <AdBanner dataAdSlot="0987654321" className="h-[600px] w-full" />
          </aside>

        </main>

        {/* Mobile footer Ad (Visible only on small screens) */}
        <div className="xl:hidden w-full p-4">
          <AdBanner dataAdSlot="1122334455" className="h-[100px] w-full" />
        </div>
      </body>
    </html>
  );
}
