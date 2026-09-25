import type { Metadata, Viewport } from "next";
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
  title: { default: "Rimpang — Kenali lewat kamera", template: "%s | Rimpang" },
  description: "Ruang pengenalan 10 jenis rimpang melalui kamera dan foto lokal. Gambar tetap di perangkat, dengan katalog dan panduan untuk belajar.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#f5f6f1",
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
