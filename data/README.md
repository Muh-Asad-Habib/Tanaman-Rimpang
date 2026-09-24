# Data belum dipulihkan

Folder ini sengaja tidak memuat dataset maupun file gambar kosong. `.gitkeep`
hanya mempertahankan direktori, bukan sampel data.

1. Pulihkan sumber asli ke `raw` di server.
2. Tinjau lisensi/provenance dan identitas spesies.
3. Anotasi setiap instance dengan jenis dan bbox (x,y,width,height normalized,
   bukan YOLO center coordinates). Foto negatif memakai `objects: []`.
4. Catat groupId yang sama untuk spesimen/sesi/video yang berkaitan.
5. Audit dan siapkan data sesuai `docs/dataset-guide.md`.

Jangan meletakkan data pribadi/kredensial di manifest atau menggunakan foto
katalog web sebagai pengganti dataset penelitian.
