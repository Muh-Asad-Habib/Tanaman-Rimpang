import type { Metadata } from "next";
import localFont from "next/font/local";
import { SiteHeader } from "@/components/site-header";
import { SiteFooter } from "@/components/site-footer";
import "./globals.css";

const bodyFont = localFont({ src: "../../node_modules/@fontsource-variable/manrope/files/manrope-latin-wght-normal.woff2", variable: "--font-body", display: "swap" });
const displayFont = localFont({ src: [
  { path: "../../node_modules/@fontsource-variable/lora/files/lora-latin-wght-normal.woff2", style: "normal" },
  { path: "../../node_modules/@fontsource-variable/lora/files/lora-latin-wght-italic.woff2", style: "italic" },
], variable: "--font-display", display: "swap" });

export const metadata: Metadata = {
  title: { default: "Rimpang — Kenali dari akarnya", template: "%s | Rimpang" },
  description: "Jelajahi sepuluh jenis rimpang Nusantara. Kenali ciri, temukan perbedaannya, dan siapkan pengenalan lokal melalui kamera.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="id"
      className={`${bodyFont.variable} ${displayFont.variable}`}
    >
      <body><a href="#konten" className="skip-link">Langsung ke konten</a><SiteHeader />{children}<SiteFooter /></body>
    </html>
  );
}
