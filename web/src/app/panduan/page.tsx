import Link from "next/link";
import type { Metadata } from "next";
import { Icon } from "@/components/icons";
import { ModelAvailability } from "@/components/model-availability";
export const metadata: Metadata = { title: "Panduan penggunaan" };
const questions = [
  ["Apakah foto saya dikirim atau disimpan?", "Tidak. Halaman pindai tidak mengirim foto, video, atau crop ke server. Pratinjau berada di memori browser dan dilepas saat diganti atau halaman ditutup. Browser tetap memerlukan jaringan untuk membuka situs serta mengunduh aset/model."],
  ["Berapa objek yang dapat dikenali?", "Sistem disiapkan untuk 10 jenis rimpang hasil panen dengan maksimal 5 objek aktif dalam satu tampilan. Ini berbeda dengan mengenali semua tanaman atau daun. Performa dan ketepatan model nanti perlu diukur pada perangkat nyata."],
  ["Mengapa kamera tidak terbuka?", "Berikan izin kamera melalui pengaturan browser. Pastikan kamera tidak sedang dipakai aplikasi lain dan situs dibuka melalui HTTPS. Di komputer pengembang, localhost diperbolehkan; alamat HTTP jaringan lokal biasanya tidak mendukung izin kamera."],
  ["Apakah hasil bisa menjadi acuan konsumsi atau pengobatan?", "Tidak. Pengetahuan visual tidak menjamin identitas maupun keamanan konsumsi. Nama lokal dapat berbeda antar daerah. Bila ragu, minta konfirmasi ahli tanaman; jangan menggunakan aplikasi sebagai diagnosis atau petunjuk dosis."],
  ["Mengapa hasil bisa berbeda pada setiap perangkat?", "Pencahayaan, fokus, kamera dan kemampuan komputasi dapat berbeda. Browser yang mendukung WebGPU dapat menggunakan akselerasi, sedangkan mode CPU/WASM mungkin lebih lambat. Target realtime tidak dijanjikan sebelum model diuji pada perangkat tersebut."],
];
export default function Guide() {
  return <main id="konten" tabIndex={-1} className="container page-main"><div className="page-heading"><p className="eyebrow">PANDUAN PENGGUNAAN</p><h1>Mulai dengan<br /><em>gambar yang jelas.</em></h1><p>Siapkan rimpang, izinkan kamera saat Anda siap, lalu lihat status dan hasil di ruang pindai.</p></div>
    <div className="guide-tips">{([{ icon: "sun", title: "Cahaya yang cukup", text: "Pilih cahaya alami yang merata. Hindari sorotan langsung dan bayangan yang menutup objek." }, { icon: "leaf", title: "Objek terlihat utuh", text: "Pisahkan rimpang di permukaan polos. Jangan menumpuk atau menutupi objek dengan tangan." }, { icon: "scan", title: "Dekat, tetapi tetap fokus", text: "Jaga kamera stabil dan pastikan seluruh rimpang berada dalam bingkai." }] as const).map((item, i) => <div key={item.title} className="guide-tip"><span className="tip-number">0{i + 1}</span><Icon name={item.icon} size={32} /><h2>{item.title}</h2><p>{item.text}</p></div>)}</div>
    <section className="faq-section"><div><p className="eyebrow">PERTANYAAN UMUM</p><h2>Tentang kamera,<br />model & privasi.</h2></div><div className="faq-list">
      <details open><summary>Apakah identifikasi sudah bisa digunakan?<span aria-hidden="true">+</span></summary><ModelAvailability /><p>Status ini mengikuti manifest model yang terpasang. Identifikasi baru berjalan setelah model berhasil dimuat di ruang pindai. Demo selalu merupakan ilustrasi, bukan prediksi nyata.</p></details>
      {questions.map(([question, answer]) => <details key={question}><summary>{question}<span aria-hidden="true">+</span></summary><p>{answer}</p></details>)}</div></section>
    <div className="guide-cta"><h2>Siap mencoba?</h2><Link href="/pindai" className="button button-primary">Mulai pindai <Icon name="arrow" size={18} /></Link></div>
  </main>;
}
