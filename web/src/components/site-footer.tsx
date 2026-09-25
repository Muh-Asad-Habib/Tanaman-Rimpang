import Link from "next/link";

export function SiteFooter() {
  return <footer className="site-footer"><div className="container footer-inner">
    <span>Rimpang &copy; 2026</span>
    <p>Untuk belajar, bukan acuan konsumsi atau diagnosis.</p>
    <nav aria-label="Navigasi footer"><Link href="/jelajah">Katalog</Link><Link href="/panduan">Panduan & privasi</Link></nav>
  </div></footer>;
}
