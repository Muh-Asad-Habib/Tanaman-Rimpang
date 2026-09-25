# Desain: kamera lebih dahulu

## Arah visual

- Hijau hutan `#173b2b` untuk hero singkat; off-white `#f5f6f1` untuk halaman,
  putih `#ffffff` untuk workspace, dan hijau muda `#c7e7a0` untuk aksi utama.
- Teks `#20372b`, sekunder `#5c6a61`, garis `#dce2d8`, permukaan sage `#edf2e9`.
  Pemberitahuan model dibedakan dengan teks dan warna, bukan warna saja.
- Manrope adalah font dominan untuk judul, isi, navigasi, dan kontrol.
  Lora hanya aksen singkat pada judul dan keterangan botani.
  Keduanya tetap di-host lokal melalui `next/font/local`.
- Tombol pill dengan tinggi 44–58px; panel beradius 14–24px. Tanpa library UI,
  animasi dekoratif, foto hutan besar, logo mitra, atau wordmark raksasa.
- Inspirasi nuansa: [Earth Day oleh Lumios Digital](https://dribbble.com/shots/27282245-Earth-Day-Brand-Recognition-Sustainable-Web-Design).
  Tidak menyalin teks, foto, logo, maupun komposisi referensi. Pratinjau referensi
  hanya artefak perencanaan privat, bukan aset aplikasi. Beranda hanya memakai
  satu aksen jahe kecil dari aset existing, disembunyikan di layar ringkas.

## Hierarki halaman

- `/` tetap beranda, bukan kamera yang langsung menyala. Satu hero dengan judul
  pengenalan melalui kamera, deskripsi pendek, dan CTA **Mulai pindai**.
  CTA harus terlihat pada viewport awal desktop dan ponsel.
- Satu bagian pendukung berisi persiapan singkat dan status model. Tidak ada
  galeri unggulan atau rangkaian promosi di beranda.
- Header ringkas: identitas, katalog, panduan, dan akses pindai. Footer satu
  baris fleksibel berisi informasi belajar dan navigasi, bukan panel promosi.
- `/pindai` adalah workspace terang, tanpa hero pemasaran. Pada desktop,
  preview mendapat sekitar tiga perempat lebar workspace; status dan hasil
  berada di sidebar. Foto, pilihan perangkat, demo, dan tips bersifat sekunder.
- Di ponsel/tablet, preview dan kontrol mendahului status/hasil. Aktivasi atau
  penghentian kamera tetap aksi paling jelas, termasuk ketika melihat foto.
  Demo dapat ditutup tanpa meminta izin kamera.
- Katalog, pencarian/filter, sepuluh halaman detail, dan panduan dipertahankan.
  Informasi identitas, sumber botani, ketidakpastian, dan keselamatan tidak dihapus.

## Status dan privasi

- Beranda dan FAQ mengambil `/models/manifest.json` dengan `cache: "no-store"`
  melalui `fetchManifest`, helper yang juga dipakai engine. Manifest divalidasi
  dengan kontrak kelas existing; tidak ada status "belum dilatih" hardcoded.
- `ready` di beranda/FAQ hanya berarti bundle dinyatakan tersedia oleh manifest.
  UI tidak mengklaim model sudah dimuat atau perangkat kompatibel.
  Status aktif di ruang pindai baru berasal dari engine setelah Worker siap.
- Loading, manifest gagal, unavailable, dan retry tetap eksplisit. Kamera/foto
  hanya pratinjau sampai model berhasil dimuat. Bundle aktual tidak diganti
  untuk memenuhi desain.
- Demo diberi label ilustrasi pada preview dan hasil; tidak menjadi fallback
  prediksi. Identitas objek, ketidakpastian, tautan detail, dan batas lima objek
  tetap terlihat dalam hasil nyata.
- Tidak mengirim foto, video, atau crop pengguna ke server. Jaringan hanya
  untuk halaman, aset, manifest, dan model; kamera perlu klik eksplisit.
- Lifecycle start/stop, pergantian kamera, pembatalan operasi, unmount,
  visibility cleanup, URL foto, validasi MIME/10 MB/32 MP, serta frame/session
  gate dipertahankan. Tidak mengubah urutan label atau kontrak inferensi.

## Responsif dan aksesibilitas

- Container fluid maksimum 1320px, mendukung mulai 320px dan safe area.
  Breakpoint 360/580/800/900/1100px. Workspace satu kolom pada 900px ke bawah.
- `fitPreview` menentukan permukaan sesuai rasio sumber dan ukuran stage aktual.
  Video/foto serta bbox memakai permukaan yang sama, tanpa `object-fit: cover`.
  ResizeObserver dan event resize video menjaga geometri setelah rotasi.
- Pada landscape dengan tinggi maksimal 600px, stage dipadatkan ke 150–230px,
  header tidak sticky, dan kontrol tetap di luar frame. Tidak menutupi gambar
  dengan kontrol atau memaksa pemotongan frame.
- Kontrol utama minimum 44px; input pencarian/pilihan perangkat 16px. Kartu dan
  tautan detail bisa ditemukan tanpa hover. Menu mobile mendukung Escape,
  navigasi, interaksi luar, dan perubahan breakpoint.
- Focus ring, skip link ke main, native details, reduced motion, serta zoom
  browser/pinch zoom tetap tersedia. Ukuran dan urutan DOM mengutamakan kamera,
  bukan CSS visual reorder yang mengacaukan navigasi keyboard.
- QA mencakup lebar 320, 390, 768, 1024, 1440, landscape, dan reflow/zoom.
  Pada 390×844 dan 1440×900, preview dan tombol kamera utama harus terlihat
  bersama. Fake camera/browser emulation bukan validasi perangkat fisik.
