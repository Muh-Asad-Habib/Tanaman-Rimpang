# Rimpang

Web editorial untuk mengenal sepuluh jenis rimpang hasil panen. Next.js App Router,
TypeScript dan Tailwind, dengan kamera/foto lokal serta integrasi ONNX Runtime Web.

## Menjalankan web

Node.js >=22.18 direkomendasikan. Dari direktori proyek:

```powershell
Set-Location web
npm ci
npm run dev
```

Buka http://localhost:3000. Untuk produksi: `npm run build`, lalu `npm start`.
Script `predev`/`prebuild` menyiapkan aset ONNX Runtime dari paket dengan versi
yang sama. Jalankan `npm run lint`, `npm run typecheck`, dan `npm test` untuk
pemeriksaan kode. Font dan gambar disajikan lokal, tidak melalui hotlink.

## Status model

**Belum ada model terlatih.** Kamera dan foto berfungsi sebagai pratinjau, bukan
identifikasi. Mode demo memiliki anotasi ilustrasi yang jelas, tidak menggunakan
foto/kamera pengguna, dan tidak menjadi bukti akurasi.

Integrasi disiapkan untuk YOLO11n deteksi generik -> crop -> EfficientNetV2-B0
+ CBAM, sepuluh kelas, maksimal lima objek. Target frekuensi dan akurasi harus
diukur setelah training. Confidence bukan jaminan kebenaran maupun keamanan konsumsi.

`web/public/models/manifest.json` sengaja menyatakan `status: unavailable`.
Jangan menggantinya menjadi ready tanpa bundle trained dengan kelas, preprocessing,
checksum dan output yang sesuai `shared/model-manifest.schema.json`.

## Struktur

- `web`: halaman, kamera, katalog dan runtime inference browser.
- `shared`: urutan kelas dan kontrak bundle/data untuk TypeScript serta Python.
- `training`: toolkit training server; tidak dijalankan saat web dimulai.
- `data`: lokasi sumber/anotasi/manifest dan turunan dataset, bukan data fiktif.
- `artifacts`: keluaran training/ekspor, diabaikan git.
- `docs`: dokumentasi arsitektur, desain dan persiapan server.

## Kamera dan privasi

Kamera memerlukan HTTPS atau localhost. HTTP melalui IP LAN bukan secure context
untuk kamera HP. Akses dari HP ke hosting perlu HTTPS yang valid; jangan mematikan
pengamanan browser. Tidak ada API upload/predict, perekaman atau telemetry gambar.
Browser masih membutuhkan jaringan untuk membuka situs/mengunduh aset.

Preview/hasil diproses dalam memori dan dibersihkan saat sesi berganti. WebGPU
dipakai jika model dan perangkat mendukung; WASM single-thread adalah fallback
yang dinyatakan, bukan janji realtime di setiap HP.

## Aset dan sumber pengetahuan

Foto katalog adalah salinan foto proyek SKRIPSI_ALYA yang dioptimalkan untuk web.
Cutout dekoratif bukan data training. Lihat `docs/asset-provenance.md`. Tautan
Kew Plants of the World Online pada detail adalah rujukan taksonomi, bukan
validasi foto/dataset. Beberapa nama lokal memerlukan peninjauan ahli.

Situs ini tidak menyediakan diagnosis, dosis atau jaminan keamanan konsumsi.
Tinjau hak pemakaian dataset, foto, Ultralytics dan pretrained weights sebelum
mendistribusikan produk. Repo ini tidak memberi lisensi baru atas aset sumber.
