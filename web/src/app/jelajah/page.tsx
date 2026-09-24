import type { Metadata } from "next";
import { CatalogGrid } from "@/components/catalog-grid";

export const metadata: Metadata = { title: "Jelajahi rimpang" };
export default function Catalog() {
  return <main id="konten" className="container page-main">
    <div className="page-heading"><p className="eyebrow">CATATAN BOTANI / KOLEKSI</p><h1>Sepuluh rimpang.<br /><em>Sepuluh karakter.</em></h1><p>Jelajahi bentuk, aroma, dan cerita rimpang yang dekat dengan keseharian kita.</p></div>
    <CatalogGrid /><div className="catalog-disclaimer">Katalog ini membantu belajar mengenali, bukan memastikan keamanan konsumsi. Identitas dari nama lokal dan foto perlu dikonfirmasi bila masih meragukan.</div>
  </main>;
}
