# Menjalankan training nanti di server

**Training tidak dijalankan pada tahap pembangunan web.** Skrip tersedia tetapi
akurasi, kapasitas GPU dan ekspor trained model belum dibuktikan dengan dataset nyata.

1. Salin source proyek, `shared`, `training`, dan metadata sumber ke server.
2. Buat virtualenv Python3.11/3.12 yang didukung dependency. Pasang pasangan
   torch/torchvision CUDA dari kanal resmi yang sesuai driver server.
3. `python -m pip install -r training/requirements.txt`
4. Catat `python -m pip freeze` dalam folder eksperimen untuk reproduksi.
5. Pulihkan/anotasi/audit data mengikuti `dataset-guide.md`.

Jalankan dari root proyek; contoh jalur menggunakan separator yang juga
diterima Python di Windows:

```text
python -m training.scripts.prepare_dataset --manifest data/manifests/master.json --data-root data/raw --output data/prepared/v1
python -m training.scripts.train_detector --data data/prepared/v1/detector/data.yaml --output artifacts/detector-v1 --device 0
python -m training.scripts.train_classifier --data data/prepared/v1/classifier --output artifacts/classifier-v1 --device cuda
```

YOLO memakai pretrained `yolo11n.pt` dan classifier memakai
`tf_efficientnetv2_b0.in1k`. Download bobot awal baru dapat terjadi ketika perintah
TRAIN dijalankan, bukan saat web dimulai/ketika `--help` dibuka.
SE bawaan EfficientNetV2 tidak dihapus; satu CBAM tambahan sebelum pooling.
Jalankan ablation dengan `cbam:false` dalam salinan config. Config tanpa CBAM
bukan pengganti diam-diam arsitektur proposal.

Classifier memakai crop, jitter train, class weights train saja, seleksi best
berdasarkan macro-F1 valid dan early stopping. Mean/std0.5 adalah konfigurasi
bobot TF EfficientNetV2 yang dipilih; gunakan konfigurasi yang konsisten saat
training/export/browser. Script saat ini melatih GT crops; eksperimen lanjutan
perlu predicted crops/jitter yang merepresentasikan kesalahan detector.

Resume classifier menggunakan `--resume artifacts/classifier-v1/last.pt` dengan
config dan direktori output yang sama, memulihkan optimizer, epoch dan RNG.
Detector resume memakai checkpoint `weights/last.pt` dan konfigurasi checkpoint.
Jangan mengubah split tengah resume. Checkpoint pickle hanya boleh berasal dari
training sendiri atau sumber yang benar-benar dipercaya.

Default workers/batch bukan jaminan cocok di semua server. Ubah config sebelum
eksperimen; jangan mengambil semua CPU/GPU mesin bersama tanpa alokasi.

## Evaluasi dan kalibrasi sebelum ekspor

Uji detector recall/mAP, classifier per-class/macro-F1, end-to-end bbox+jenis,
unknown false accept dan coverage; pisahkan predicted crop dari GT crop.
Pilih ambang pada validasi kelompok, bukan test. Ukur sedikitnya beberapa seed
untuk perbandingan dan laporkan ketidakpastian. Jangan mengutip angka notebook
lama untuk model ini.

Exporter memerlukan laporan JSON nyata yang mengikat SHA256 kedua checkpoint:

```text
status: validated
split: valid
protocol: group-disjoint
classSlugs: array slug dalam urutan shared/labels.json
detectorCheckpointSha256: SHA256 file detector terpilih
classifierCheckpointSha256: SHA256 file classifier terpilih
detectorInputSize: 416
classifierInputSize: 224
cropMargin: 0.1
detectorScoreThreshold: hasil kalibrasi
iouThreshold: hasil kalibrasi
classifierScoreThreshold: hasil kalibrasi
minMargin: hasil kalibrasi
```

Sengaja tidak disediakan laporan dengan angka palsu siap pakai.

```text
python -m training.scripts.export_models --detector artifacts/detector-v1/weights/best.pt --classifier artifacts/classifier-v1/best.pt --calibration artifacts/calibration.json --output artifacts/web-v1 --version rimpang-v1
```

Default export FP32, batch1, ONNX opset17. Torch/ORT CPU numeric smoke bukan
evaluasi akurasi maupun pengganti paritas gambar/browser. Sesudah ekspor lakukan
golden-image checks pada browser dan perangkat HP/laptop sebenarnya.
