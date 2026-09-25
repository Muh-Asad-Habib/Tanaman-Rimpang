import type { Metadata } from "next";
import { CatalogGrid } from "@/components/catalog-grid";

export const metadata: Metadata = { title: "Jelajahi rimpang" };
export default function Catalog() {
  return <main id="konten" tabIndex={-1} className="container page-main">
    <div className="page-heading"><p className="eyebrow">KATALOG RIMPANG</p><h1>Sepuluh jenis,<br /><em>kenali bedanya.</em></h1><p>Telusuri bentuk, aroma, dan ciri rimpang. Cari nama atau pilih kelompok untuk membandingkan.</p></div>
    <CatalogGrid /><div className="catalog-disclaimer">Katalog ini membantu belajar mengenali, bukan memastikan keamanan konsumsi. Identitas dari nama lokal dan foto perlu dikonfirmasi bila masih meragukan.</div>
  </main>;
}
