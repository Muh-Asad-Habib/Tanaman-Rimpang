import Link from "next/link";
export default function NotFound() {
  return <main id="konten" className="container empty-state page-main"><p className="eyebrow">404 / BELUM DITEMUKAN</p><h1>Sepertinya kita<br /><em>keluar dari kebun.</em></h1><p>Halaman yang Anda cari tidak tersedia.</p><Link href="/jelajah" className="button button-dark">Kembali ke koleksi</Link></main>;
}
