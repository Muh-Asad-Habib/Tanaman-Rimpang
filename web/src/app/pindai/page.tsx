import type { Metadata } from "next";
import { Scanner } from "@/components/scanner";
import { Icon } from "@/components/icons";
export const metadata: Metadata = { title: "Ruang pindai" };
export default function Scan() {
  return <main id="konten" className="container page-main"><div className="scanner-heading"><div><p className="eyebrow">KENALI DARI AKARNYA</p><h1>Ruang <em>pindai.</em></h1><p>Kamera Anda, langkah pertama untuk mengenal lebih dekat.</p></div><span className="privacy-chip"><Icon name="shield" size={16} /> Diproses di perangkat Anda</span></div><Scanner /></main>;
}
