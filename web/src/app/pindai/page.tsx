import type { Metadata } from "next";
import { Scanner } from "@/components/scanner";
import { Icon } from "@/components/icons";
export const metadata: Metadata = { title: "Ruang pindai" };
export default function Scan() {
  return <main id="konten" tabIndex={-1} className="container scanner-main"><div className="scanner-heading"><h1>Pindai rimpang</h1><span className="privacy-chip"><Icon name="shield" size={16} />Gambar tidak diunggah</span></div><Scanner /></main>;
}
