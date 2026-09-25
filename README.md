# Tanaman Rimpang

Aplikasi web untuk mengenali jenis rimpang lewat kamera. Arahkan kamera HP atau
laptop ke rimpang, lalu aplikasi akan memberi kotak pada tiap rimpang yang
terlihat beserta nama jenisnya. Pengenalan berjalan di browser, jadi gambar
tidak dikirim ke server mana pun.

Ada 10 jenis yang bisa dikenali: jahe, jahe merah, kencur, kunyit, kunyit
putih, lempuyang, lengkuas, temu hitam, temu kunci, dan temulawak.

## Fitur

- Pindai langsung dari kamera, sampai 5 rimpang sekaligus
- Bisa juga memakai foto dari galeri
- Katalog berisi keterangan setiap jenis rimpang
- Tampilan menyesuaikan HP, tablet, dan desktop

## Cara kerja

```text
Kamera / foto -> YOLO11n -> potong tiap rimpang -> EfficientNetV2-B0 + CBAM -> hasil
```

Pengenalan dilakukan dalam dua tahap:

1. **YOLO11n** mencari letak setiap rimpang pada gambar.
2. Tiap rimpang yang ditemukan dipotong, lalu **EfficientNetV2-B0** dengan
   modul attention **CBAM** menentukan jenisnya.

Hasilnya berupa kotak, nama jenis, dan skor keyakinan. Kalau skornya terlalu
rendah, rimpang ditandai "belum dikenali". Kedua model disimpan dalam format
ONNX dan dijalankan dengan ONNX Runtime Web. Aplikasi memakai WebGPU bila
tersedia, dan otomatis beralih ke CPU (WASM) bila tidak.

## Teknologi

| Bagian | Teknologi |
| --- | --- |
| Web | Next.js, TypeScript, Tailwind CSS |
| Model | YOLO11n, EfficientNetV2-B0 + CBAM (PyTorch) |
| Inferensi | ONNX Runtime Web |

## Menjalankan

Pastikan Node.js versi 22.18 atau lebih baru sudah terpasang.

```powershell
cd web
npm ci
npm run dev
```

Buka http://localhost:3000 lalu masuk ke menu **Pindai**. Untuk versi
produksi, jalankan `npm run build` kemudian `npm start`.

Browser hanya mengizinkan kamera di localhost atau HTTPS.

## Struktur folder

```text
web/       aplikasi web beserta model ONNX
shared/    daftar kelas dan format data
training/  pengolahan dataset dan pelatihan model
```

## Lisensi

Kode di repositori ini memakai [MIT License](LICENSE). Dataset dan pustaka
pihak ketiga mengikuti lisensinya masing-masing.