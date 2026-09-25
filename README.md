# Tanaman Rimpang

Web pengenal **10 jenis rimpang** langsung dari kamera. Arahkan kamera HP atau
laptop ke rimpang, dan web akan menandai setiap rimpang serta menyebutkan
jenisnya. Semua proses berjalan di perangkat, sehingga foto dan video tidak
dikirim ke server.

Jenis yang dikenali: jahe, jahe merah, kencur, kunyit, kunyit putih, lempuyang,
lengkuas, temu hitam, temu kunci, dan temulawak.

📄 **Hasil akurasi:** [docs/hasil-akurasi.pdf](docs/hasil-akurasi.pdf)

## Fitur

- Pemindaian kamera realtime, maksimal 5 rimpang dalam satu layar
- Pilih foto dari galeri sebagai alternatif kamera
- Katalog dan detail setiap jenis rimpang
- Tampilan responsif untuk HP, tablet, dan desktop

## Cara kerja

```text
Kamera / foto → YOLO11n (cari rimpang) → potong tiap objek
             → EfficientNetV2-B0 + CBAM (tentukan jenis) → hasil di layar
```

1. **Deteksi.** YOLO11n menemukan posisi setiap rimpang pada gambar.
2. **Potong.** Setiap rimpang yang ditemukan dipotong menjadi gambar kecil.
3. **Klasifikasi.** EfficientNetV2-B0 dengan attention CBAM menentukan jenis
   rimpang pada setiap potongan.
4. **Tampilkan.** Kotak, nama jenis, dan skor keyakinan muncul di layar. Jika
   skornya rendah, objek ditandai "belum dikenali".

Kedua model berformat ONNX dan dijalankan di browser dengan ONNX Runtime Web
(WebGPU, dengan cadangan otomatis ke CPU/WASM).

## Teknologi

| Bagian | Teknologi |
| --- | --- |
| Web | Next.js, TypeScript, Tailwind CSS |
| Inferensi | ONNX Runtime Web (WebGPU / WASM) |
| Model | YOLO11n + EfficientNetV2-B0 + CBAM (PyTorch, diekspor ke ONNX) |

## Menjalankan

Butuh **Node.js 22.18** atau lebih baru.

```powershell
cd web
npm ci
npm run dev
```

Buka **http://localhost:3000**, lalu pilih menu **Pindai**. Untuk mode
produksi, jalankan `npm run build` lalu `npm start`.

Kamera hanya bisa dipakai lewat **localhost atau HTTPS**.

## Struktur

```text
web/       Aplikasi web dan model ONNX (web/public/models)
shared/    Daftar kelas dan kontrak data
training/  Kode pengolahan data dan pelatihan model
docs/      Dokumentasi teknis dan laporan akurasi
```

## Catatan

Model saat ini berstatus **eksperimental**. Label data dibuat otomatis tanpa
pemeriksaan manual, jadi hasil pengenalan masih bisa keliru. Sebagian dataset
berlisensi non-komersial (CC BY-NC-SA 4.0).

## Lisensi

Kode proyek menggunakan [MIT License](LICENSE). Dataset dan dependensi pihak
ketiga mengikuti lisensi masing-masing.
