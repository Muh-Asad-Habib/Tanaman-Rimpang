import Image from "next/image";
import Link from "next/link";
import { Icon } from "@/components/icons";
import { PlantCard } from "@/components/plant-card";
import { plants } from "@/lib/catalog";

export default function Home() {
  return <main id="konten" tabIndex={-1}>
    <section className="container hero">
      <div className="hero-copy"><p className="eyebrow"><span className="small-line" /> DARI TANAH, UNTUK KITA</p>
        <h1>Serupa bentuknya.<br /><em>Berbeda ceritanya.</em></h1>
        <p className="hero-description">Jahe atau lengkuas? Kunyit atau temulawak?<br className="desktop-break" /> Mari mengenal rimpang lebih dekat, mulai dari akarnya.</p>
        <div className="hero-actions"><Link href="/pindai" className="button button-dark">Mulai mengenali <Icon name="diagonal" size={19} /></Link>
          <Link href="/jelajah" className="text-link">Lihat koleksi <Icon name="arrow" size={18} /></Link></div>
        <div className="hero-footnote"><Icon name="leaf" size={16} /><span>10 jenis rimpang. Satu ruang untuk mengenal.</span></div>
      </div>
      <div className="hero-art">
        <div className="botanical-ring ring-one" /><div className="botanical-ring ring-two" />
        <div className="hero-top-note"><span>CATATAN BOTANI</span><span>01 — 10</span></div>
        <div className="hero-specimen hero-ginger"><Image src="/images/rimpang/jahe.webp" alt="Rimpang jahe dengan ruas dan cabangnya" fill preload sizes="(max-width: 800px) 65vw, 420px" /></div>
        <div className="hero-specimen hero-turmeric"><Image src="/images/rimpang/kunyit.webp" alt="Rimpang kunyit" fill preload sizes="(max-width: 800px) 50vw, 310px" /></div>
        <div className="specimen-caption caption-ginger"><span className="caption-point" /><div>Jahe<small>Zingiber officinale</small></div></div>
        <div className="specimen-caption caption-turmeric"><span className="caption-point" /><div>Kunyit<small>Curcuma longa</small></div></div>
        <span className="vertical-note">KEKAYAAN YANG TUMBUH DI SEKITAR KITA</span>
        <div className="hero-art-bottom"><span>Kenali dari akarnya.</span><Icon name="leaf" size={27} /></div>
      </div>
    </section>
    <div className="values-strip"><div className="container values-inner">
      <span><Icon name="book" size={18} /> Pengetahuan yang dekat</span><span><Icon name="scan" size={18} /> Dirancang untuk kamera Anda</span><span><Icon name="shield" size={18} /> Foto tetap di perangkat</span>
    </div></div>
    <section className="container section collection-section">
      <div className="section-heading"><div><p className="eyebrow">KOLEKSI RIMPANG</p><h2>Kenalan dengan<br /><em>yang sering kita temui.</em></h2></div>
        <div className="section-intro"><p>Bukan sekadar bumbu dapur. Setiap rimpang punya bentuk, aroma, dan karakter tersendiri.</p><Link href="/jelajah" className="text-link">Jelajahi 10 jenis <Icon name="arrow" size={18} /></Link></div></div>
      <div className="featured-grid">{[plants[0], plants[3], plants[2], plants[6]].map((plant, index) => <PlantCard key={plant.id} plant={plant} index={index} />)}</div>
    </section>
    <section className="how-section"><div className="container how-layout">
      <div className="how-visual"><Image src="/images/rimpang/temu-kunci.webp" alt="Bentuk memanjang temu kunci" fill sizes="(max-width: 800px) 80vw, 450px" /><span className="how-visual-note">Bentuk kecil.<br /><em>Banyak hal untuk dikenali.</em></span><span className="specimen-number">BOESENBERGIA ROTUNDA</span></div>
      <div className="how-content"><p className="eyebrow">SEDERHANA, DARI AWAL</p><h2>Dekatkan kamera.<br /><em>Temukan bedanya.</em></h2>
        <div className="step"><span>01</span><div><h3>Siapkan rimpangnya</h3><p>Letakkan di permukaan yang bersih, dengan cahaya yang cukup.</p></div></div>
        <div className="step"><span>02</span><div><h3>Arahkan kamera Anda</h3><p>Gunakan kamera atau pilih foto. Gambar tetap berada di perangkat.</p></div></div>
        <div className="step"><span>03</span><div><h3>Kenali, lalu pelajari</h3><p>Jelajahi ciri setiap jenis. Contoh hasil tersedia melalui demo tampilan.</p></div></div>
        <Link href="/pindai" className="text-link">Buka ruang pindai <Icon name="arrow" size={18} /></Link>
        <p className="honesty-note"><span className="status-dot" /> Model identifikasi sedang disiapkan. Demo bukan prediksi nyata.</p>
      </div>
    </div></section>
    <section className="container field-note section"><span className="eyebrow">SEBUAH CATATAN</span><h2>Yang tumbuh dekat dengan kita,<br />layak untuk <em>dikenal lebih dalam.</em></h2><p>Rimpang adalah ruang belajar tentang kekayaan tanaman di sekitar kita.<br />Teknologi membantu melihat. Pengetahuan membantu memahami.</p><Link href="/panduan" className="button button-outline">Baca panduan <Icon name="arrow" size={18} /></Link></section>
  </main>;
}
