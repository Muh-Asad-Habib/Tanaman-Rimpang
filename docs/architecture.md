# Arsitektur sistem

## Batas domain

Next.js menyajikan konten/server-rendered HTML dan aset. Browser menangani kamera,
decode foto, demo, serta (setelah model tersedia) inferensi. Tidak ada layanan
inference Python, API upload, database atau autentikasi yang diperlukan untuk web.

`shared\labels.json` mengunci index 0-9. Katalog dan hasil model menggunakan index
yang sama. Dataset dan training harus mengikuti urutan tersebut, bukan urutan
alphabet bawaan ImageFolder tanpa mapping.

## Alur model

1. `InferenceEngine.initialize` mengambil manifest tanpa cache permanen.
2. `unavailable` menjadi state UI nyata; tidak membuat Worker/model palsu.
3. `ready` divalidasi terhadap versi, kelas, dimensi, normalisasi dan nama artefak.
4. Worker mengambil kedua ONNX satu origin dan memverifikasi SHA-256.
5. WebGPU dicoba; kegagalan provider ditampilkan ketika beralih ke WASM.
6. Detector menerima letterbox RGB/NCHW float32 [0,1], pad114.
7. Output `1x5xN` -> filter -> NMS -> koordinat normalized frame asli.
8. Maksimal lima track diklasifikasikan dari frame yang sama, dengan margin crop,
   resize persegi dan mean/std yang ditentukan manifest.
9. Probabilitas disaring dan distabilkan terpisah per track. Label ambigu tidak
   dipaksakan menjadi kelas; transisi/muncul kembali membutuhkan konfirmasi baru.

Main thread mempunyai satu frame aktif; frame antara dilewati. Worker menyimpan
paling banyak satu frame terbaru bila ada pergantian sesi ketika GPU masih sibuk.
Session ID membatalkan hasil lama. Bitmap dan tensor dilepas setelah digunakan.
`dispose` menutup Worker dan menginvalidasi semua pekerjaan aktif.

## Keterbatasan terukur

Graph/IO hanya FP32 dengan batch1, detector tanpa embedded NMS, classifier1x10.
FP16/INT8/arsitektur lain membutuhkan versi kontrak atau adaptasi yang diuji.
Smoothing, threshold dan tracking awal bukan hasil kalibrasi penelitian.

PIL/PyTorch dan canvas browser dapat berbeda dalam interpolasi/rounding. Sebelum
release model, bandingkan tensor identik dan gambar lengkap terpisah; lakukan
evaluasi end-to-end, bukan hanya classifier pada crop ground-truth.
Tidak ada klaim 98,06% atau 2Hz/objek berdasarkan demo.
