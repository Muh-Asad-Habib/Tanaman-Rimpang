# Memasang hasil training ke Next.js

Bundle berisi `detector.onnx`, `classifier.onnx`, `manifest.json` dan laporan
kalibrasi. Manifest mengikuti `shared/model-manifest.schema.json`, dengan class
slugs identik `shared/labels.json`. Script export tidak memasang model otomatis.

Sesudah evaluasi bundle, jalankan dari root proyek:

```text
python -m training.scripts.install_web_bundle --bundle artifacts/web-v1
```

Installer memeriksa kontrak kelas, nama berkas dan checksum, menyalin artefak
dengan nama berbasis hash, baru mengganti manifest aktif secara atomic. Model
lama tidak dihapus agar halaman yang sedang memuatnya tidak rusak. Bersihkan
artefak tak terpakai secara terencana setelah redeployment, bukan saat startup.
Installer bukan validator kebenaran semantik trained weights; gunakan hanya
bundle exporter yang sudah dievaluasi. File ONNX yang hash-nya benar tetap
dapat memiliki operator/output yang tidak didukung browser.

Rebuild/redeploy Next.js jika hosting menggunakan artifact immutable. Pada
halaman pindai tekan "Periksa kembali" atau muat ulang. Jika unduhan, checksum,
provider atau tensor gagal, pesan error muncul; aplikasi tidak beralih ke
prediksi demo atau upload server.

Versi1: detector RGB0..1 letterbox pad114, batch1 float32, output1x5xN tanpa NMS.
Classifier menerima crop padded yang diregangkan ke persegi, RGB0..1 kemudian
mean/std dari manifest, output logits1x10. Threshold, preprocessing, class order
dan resolusi merupakan bagian kontrak, bukan angka bebas di UI.

Perbedaan interpolasi PIL/canvas perlu dibandingkan menggunakan fixture gambar,
bukan hanya tensor nol. Default WASM single-thread menghindari ketergantungan
COOP/COEP. WebGPU dicoba jika tersedia. Tidak ada jaminan kecepatan mobile
sebelum pengukuran nyata, termasuk ketika lima ROI diklasifikasikan.
