import Link from "next/link";
import { Icon } from "./icons";

export function SiteFooter() {
  return <footer className="site-footer"><div className="container">
    <div className="footer-top"><div><Link href="/" className="brand"><Icon name="leaf" size={25} />rimpang.</Link>
      <p>Mengenal yang dekat.<br />Menghargai yang tumbuh.</p></div>
      <div className="footer-links"><Link href="/jelajah">Jelajahi rimpang</Link><Link href="/pindai">Ruang pindai</Link><Link href="/panduan">Panduan penggunaan</Link></div>
      <p className="footer-note">Catatan kecil tentang kekayaan<br />rimpang di sekitar kita.<br /><span>Dibuat untuk belajar, bukan diagnosis.</span></p>
    </div><div className="footer-bottom"><span>Rimpang &copy; 2026</span><span>Berakar pada pengetahuan, tumbuh bersama teknologi.</span><span>Indonesia <span aria-hidden="true">↗</span></span></div>
  </div></footer>;
}
