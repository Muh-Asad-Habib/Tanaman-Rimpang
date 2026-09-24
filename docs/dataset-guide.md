# Data dan anotasi

Belum ada dataset yang dipulihkan dalam repo ini. Mulai dari sumber pada
`data/sources.json`, dengan izin pemilik dan pemeriksaan lisensi.

## Master manifest

Gunakan `shared/dataset-manifest.schema.json`. Bbox adalah **x kiri, y atas,
lebar, tinggi**, semuanya normalized0..1 terhadap gambar dengan orientasi
yang sudah benar. Ini berbeda dari koordinat center YOLO. Gambar dengan EXIF
rotasi harus dinormalisasi SEBELUM anotasi; validator menolaknya agar bbox
tidak bergeser diam-diam.

Contoh format di bawah hanya dokumentasi, bukan manifest data asli:

```json
{
  "schemaVersion": 1,
  "labelsVersion": 1,
  "images": [{
    "path": "sesi-001/foto-001.jpg",
    "sourceId": "koleksi-sendiri",
    "groupId": "spesimen-jahe-001",
    "license": "periksa-hak-penggunaan",
    "objects": [{"classId": 0, "bbox": [0.2, 0.1, 0.4, 0.6]}]
  }]
}
```

`groupId` bersifat GLOBAL, bukan otomatis berbeda karena sourceId berbeda.
Semua foto/frame dari spesimen, video atau sesi yang sama harus satu kelompok.
Setiap objek mendapat bbox dan classId dari shared labels. Foto negatif punya
`objects: []`; jangan mengarang bbox penuh gambar. Ambiguitas jenis perlu ahli.

Audit:

```powershell
python -m training.scripts.audit_dataset --manifest data/manifests/master.json --data-root data/raw
```

Persiapan:

```powershell
python -m training.scripts.prepare_dataset --manifest data/manifests/master.json --data-root data/raw --output data/prepared/v1 --seed 42 --crop-margin 0.1
```

Perintah sama berjalan dari root proyek di Linux. Tidak menimpa direktori
keluaran existing atau mengubah original. Jalur CLI boleh menyesuaikan server.

Semua train/valid/test harus mengandung 10 kelas. Diperlukan sekurangnya tiga
kelompok independen per kelas. Split random per kelompok mencoba distribusi
70/15/15 dengan coverage; untuk data yang sulit diseimbangkan, sediakan field
`split` pada SEMUA baris setelah peninjauan. Validator menolak kelompok yang
melintasi split. Crop selalu mengikuti induk gambar.

Keluaran:

- `detector`: gambar dan YOLO label generic ID0, termasuk negatif kosong.
- `classifier`: crop GT per kelas/split dengan margin, resolusi asli crop.
- `manifest.json`: provenance, checksum original dan split.
- `report.json`: jumlah gambar/objek/kelompok, seed dan crop margin.

Path dalam manifest keluaran tetap relatif terhadap `data-root` original, bukan
foto JPEG prepared. Jika memindahkan hasil prepared ke server lain, perbarui
`path` absolut pada `detector/data.yaml`; jangan mengubah provenance original.

Exact byte duplicates ditolak. Ini belum merupakan deteksi near-duplicate:
lakukan audit perceptual/manual, khususnya foto satu sesi. Jangan menggabungkan
salinan preprocessed dan split menjadi sampel tambahan.

Data studio satu objek belum cukup untuk live multiobjek. Tambahkan 1-5 objek,
campuran jenis, berbagai sumber/lokasi/kamera, oklusi dan unknown/non-rimpang.
Jangan membuat valid/test dari augmentasi train. Sisakan holdout lapangan
independen untuk evaluasi akhir.
