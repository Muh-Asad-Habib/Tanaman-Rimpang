# Tanaman Rimpang

Web pengenalan 10 jenis rimpang hasil panen dengan tampilan responsif untuk
desktop, tablet, dan HP. Tersedia katalog, pencarian, detail tanaman,
pratinjau kamera/foto lokal, dan demo hasil multiobjek.

**Status:** sistem web dan toolkit training sudah tersedia. Model belum dilatih;
demo hanya ilustrasi, bukan hasil identifikasi.

## Arsitektur dan algoritma

| Bagian | Teknologi / fungsi |
| --- | --- |
| Web | Next.js App Router, TypeScript, Tailwind CSS |
| Inferensi | ONNX Runtime Web di browser, WebGPU atau WASM |
| Training server | Python, PyTorch, Ultralytics, timm |

```text
Kamera / foto -> YOLO11n -> crop objek -> EfficientNetV2-B0 + CBAM -> hasil
```

- **YOLO11n:** menemukan posisi rimpang.
- **EfficientNetV2-B0:** mengklasifikasikan 10 jenis rimpang.
- **CBAM:** attention pada fitur sebelum klasifikasi.

Sistem disiapkan untuk maksimal 5 objek sekaligus. Foto/video tidak dikirim
ke server; ketepatan dan kecepatan inferensi diukur setelah model dilatih.

## Menjalankan web

Gunakan Node.js 22.18 atau lebih baru. Dari root proyek:

```powershell
Set-Location web
npm ci
npm run dev
```

Buka **http://localhost:3000**. Produksi: `npm run build`, lalu `npm start`.
Kamera membutuhkan **HTTPS atau localhost**.

## Struktur proyek

```text
web/        Aplikasi Next.js dan inferensi browser
shared/     Daftar kelas dan kontrak data/model
training/   Persiapan dataset, training, dan ekspor ONNX
data/       Sumber, anotasi, dan dataset
artifacts/  Hasil training dan bundle model
docs/       Panduan teknis
```

Training dijalankan terpisah di server. Alurnya: siapkan dataset dan anotasi ->
split per kelompok -> training -> evaluasi -> ekspor ONNX -> pasang model ke web.

Panduan: [dataset](docs/dataset-guide.md) · [training server](docs/server-training.md) ·
[integrasi model](docs/model-integration.md) · [arsitektur](docs/architecture.md).

## Lisensi

Kode proyek menggunakan [MIT License](LICENSE).
Dependensi dan aset pihak ketiga tetap mengikuti lisensi masing-masing.
