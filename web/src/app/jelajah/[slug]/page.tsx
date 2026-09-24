import Image from "next/image";
import Link from "next/link";
import { notFound } from "next/navigation";
import { plants } from "@/lib/catalog";
import { Icon } from "@/components/icons";
import { PlantCard } from "@/components/plant-card";

export function generateStaticParams() { return plants.map((plant) => ({ slug: plant.slug })); }
export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params; return { title: plants.find((plant) => plant.slug === slug)?.name ?? "Rimpang tidak ditemukan" };
}
export default async function Detail({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const plant = plants.find((item) => item.slug === slug);
  if (!plant) notFound();
  return <main id="konten" tabIndex={-1} className="container page-main"><Link href="/jelajah" className="back-link"><Icon name="arrow" style={{ transform: "rotate(180deg)" }} size={17} /> Kembali ke koleksi</Link>
    <div className="detail-grid"><div className="detail-art" style={{ background: plant.color }}><span className="specimen-number">SPESIMEN {String(plant.id + 1).padStart(2, "0")}</span><Image src={plant.image} alt={`Contoh rimpang ${plant.name}`} fill loading="eager" sizes="(max-width: 800px) 95vw, (max-width: 1352px) 50vw, 590px" /><span className="detail-caption">Setiap rimpang punya karakter.</span></div>
      <div className="detail-copy"><p className="eyebrow">{plant.family}</p><h1>{plant.name}</h1><p className="scientific">{plant.scientific}</p><p className="detail-subtitle">{plant.subtitle}</p>
        {[["Bentuk & warna", plant.shape], ["Aroma", plant.aroma], ["Di keseharian", plant.use]].map(([title, text]) => <div className="detail-fact" key={title}><h2>{title}</h2><p>{text}</p></div>)}
        <Link href="/pindai" className="button button-dark">Buka ruang pindai <Icon name="scan" size={19} /></Link>
      </div></div>
    <div className="botanical-note"><Icon name="info" size={23} /><div><h2>Perhatikan sebelum membedakan</h2><p>{plant.distinguish}</p>{plant.review && <p className="review-note">Nama lokal dan label spesimen masih memerlukan peninjauan ahli sebelum dipakai sebagai data model.</p>}<a href={plant.source} target="_blank" rel="noreferrer">Rujukan nama botani: Kew, Plants of the World Online <span aria-hidden="true">↗</span></a><p className="fine-print">Rujukan taksonomi bukan validasi foto atau rekomendasi medis. Tidak ada dosis maupun anjuran pengobatan pada katalog ini.</p></div></div>
    <section className="section"><div className="section-heading"><h2>Kenali juga <em>yang lainnya.</em></h2><Link href="/jelajah" className="text-link">Seluruh koleksi <Icon name="arrow" size={17} /></Link></div>
      <div className="featured-grid">{plants.filter((p) => p.id !== plant.id).slice(0, 4).map((p, i) => <PlantCard key={p.id} plant={p} index={i} />)}</div></section>
  </main>;
}
