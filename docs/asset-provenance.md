# Asal aset

Sepuluh foto rimpang berasal dari `SKRIPSI_ALYA\static\plants`, disalin atas
permintaan pemilik proyek. Original tidak diubah. Salinan di
`web\public\images\rimpang` disiapkan oleh `web\scripts\prepare-images.py`.

- `*-photo.webp`: foto dengan backdrop asli, dioptimalkan ukuran/kompresinya.
- `*.webp`: cutout dekoratif dengan pemisahan backdrop studio biru; bukan
  ground-truth anotasi maupun input training.
- Nama berkas mengikuti kelas sumber. Validasi botani seluruh spesimen tetap
  memerlukan ahli. Foto katalog bukan bukti hasil classifier.
- Kepemilikan/lisensi asli belum diverifikasi independen; tinjau sebelum penggunaan
  publik/komersial. Tidak mengunduh stok foto pihak ketiga.

Font Lora dan Manrope didistribusikan lewat paket `@fontsource-variable`
yang dipin dalam lockfile, dengan lisensi SIL Open Font License pada paket.
Font disajikan secara lokal melalui `next/font/local`.
Ikon garis dan identitas visual SVG dibuat khusus di source aplikasi.
