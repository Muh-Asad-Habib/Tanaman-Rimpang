import Image from "next/image";
import Link from "next/link";
import { Icon } from "@/components/icons";
import { ModelAvailability } from "@/components/model-availability";

export default function Home() {
  return <main id="konten" tabIndex={-1} className="container home-main">
    <section className="home-hero" aria-labelledby="home-title">
      <div className="home-copy">
        <p className="eyebrow">KAMERA LOKAL · 10 JENIS RIMPANG</p>
        <h1 id="home-title">Kenali rimpang,<br />lewat <em>kamera.</em></h1>
        <p className="home-description">Arahkan kamera ke rimpang atau pilih foto dari perangkat. Mulai mengenal bentuk dan perbedaannya.</p>
        <Link href="/pindai" className="button button-primary home-cta"><Icon name="camera" size={21} />Mulai pindai<Icon name="arrow" size={21} /></Link>
        <p className="home-privacy"><Icon name="shield" size={17} />Kamera hanya aktif setelah Anda mengizinkan. Gambar tidak diunggah.</p>
      </div>
      <div className="home-specimen" aria-hidden="true">
        <Image src="/images/rimpang/jahe.webp" alt="" width={180} height={150} sizes="180px" loading="eager" />
        <span>Jahe · satu dari sepuluh jenis</span>
      </div>
    </section>
    <section className="home-support" aria-label="Persiapan dan status pemindaian">
      <div className="home-preparation">
        <h2><Icon name="sun" size={19} />Cahaya cukup, objek terpisah.</h2>
        <p>Letakkan hingga 5 rimpang di permukaan polos. Pastikan seluruh objek terlihat jelas.</p>
        <Link href="/panduan" className="text-link">Panduan singkat<Icon name="arrow" size={16} /></Link>
      </div>
      <ModelAvailability />
    </section>
  </main>;
}
