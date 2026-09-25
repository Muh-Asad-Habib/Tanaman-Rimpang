# Training di server terisolasi

Training terpisah dari pembangunan dan startup web. Model tidak dinyatakan siap
sebelum data ditinjau, training selesai, dan evaluasi/kalibrasi memenuhi kontrak.
Redesign kamera dapat dikerjakan sambil menunggu tahap data/training.

## Workspace dan pemindahan source

Server proyek: `muhasadhabib@10.33.33.11`. Gunakan root Linux
`/home/muhasadhabib/tanaman-rimpang`, terpisah dari proyek lain:

```text
code/<source-id>/        snapshot source dengan checksum
envs/train-v1/           Python 3.11 untuk data/training
envs/annotate-v1/        environment Label Studio yang terpisah
cache/                  cache dependency dan pretrained weights
data/<dataset-id>/       incoming, raw, images, annotations, manifests, prepared
services/label-studio/   state dan kredensial privat, tidak ikut hasil training
runs/<run-id>/           jobs/log, configs, detector, classifier, evaluation, web
transfers/              arsip source dan hasil beserta checksum
```

Dari root proyek Windows, upload hanya source yang diperlukan:

```powershell
.\training\scripts\server_training.ps1 -Action Upload -SourceId rimpang-source-v1
.\training\scripts\server_training.ps1 -Action Bootstrap -SourceId rimpang-source-v1
```

`Upload` mengecualikan web/node_modules, cache, IDE, dataset, `.git`, dan secrets.
Snapshot existing tidak ditimpa. Arsip dipindahkan sebagai berkas, bukan lewat
pipeline biner PowerShell. Host key harus sudah dipercaya; jangan mematikan
pemeriksaan SSH untuk melewati kegagalan koneksi.

`Bootstrap` memasang dependency data dan Label Studio pada environment terpisah.
Tambahkan `-WithCuda` bila siap memasang stack training: pasangan
torch 2.8.0/torchvision 0.23.0 dari kanal resmi cu128, kemudian requirements proyek.
Pasangan ini memerlukan driver CUDA yang sesuai. Jangan memakai environment
aplikasi lain atau melakukan instalasi dengan sudo. Simpan `pip freeze` tiap run.
Instalasi dependency tidak menjalankan training atau mengunduh dataset otomatis.
Jika koneksi kanal PyTorch bermasalah, pilih secara eksplisit
`-TorchIndex https://pypi.org/simple`: distribusi Linux resmi PyTorch 2.8.0 pada
PyPI juga memakai CUDA 12.8. Bootstrap tetap memeriksa runtime CUDA dan satu GPU;
tidak ada fallback diam-diam ke wheel CPU atau versi model lain.

## Job persisten dan sumber daya bersama

`server_training.ps1 -Action Start` menjalankan modul `training.scripts.*` di
tmux melalui supervisor. Pengguna telah mengizinkan job tetap berjalan setelah
SSH atau sesi pengembangan ditutup. Contoh pemeriksaan CLI ringan:

```powershell
.\training\scripts\server_training.ps1 -Action Start -SourceId rimpang-source-v1 `
  -RunId preflight-v1 -JobId cli-help `
  -PythonModule training.scripts.prepare_dataset -Arguments @("--help")
.\training\scripts\server_training.ps1 -Action Status -SourceId rimpang-source-v1 `
  -RunId preflight-v1 -JobId cli-help
.\training\scripts\server_training.ps1 -Action Logs -SourceId rimpang-source-v1 `
  -RunId preflight-v1 -JobId cli-help
```

Periksa `status.json`, bukan hanya keberadaan sesi tmux. Sukses memiliki exit
code 0; kegagalan/putusnya supervisor tidak disamarkan. `-ResumeJob` hanya
mengizinkan ulang supervisor failed/interrupted; argumen resume checkpoint
model tetap harus diberikan dengan benar. Job completed memakai JobId baru.
Status `orphaned` berarti child job masih hidup tanpa supervisor. Jangan
menjalankan salinan kedua; periksa PID spesifik yang tercatat terlebih dahulu.

Supervisor membatasi affinity maksimal 8 CPU dan thread numerik 4. Config model
tetap memakai 4 data-loader workers. Jalankan detector dan classifier bergiliran
pada GPU fisik 0 yang telah dialokasikan; gunakan `-Gpu 0` atau UUID GPU 0 yang
telah diperiksa. Default tanpa `-Gpu` menyembunyikan seluruh GPU, sesuai job data
dan anotasi. Jangan memakai GPU 1 atau menghentikan proses pengguna lain.
Periksa ulang VRAM/disk sebelum training; nilai kosong saat inspeksi bukan
reservasi. Slurm yang belum siap bukan alasan mengubah konfigurasi global.

## Pemulihan dataset Drive

Dataset aktif: `data/drive-v2` (5.507 gambar + `ATRIBUSI.txt` + `rename_mapping.txt`).

```powershell
.\training\scripts\server_training.ps1 -Action Start -SourceId rimpang-source-v2 `
  -RunId dataset-drive-v2 -JobId download -PythonModule training.scripts.download_dataset `
  -Arguments @("--dataset-root", "/home/muhasadhabib/tanaman-rimpang/data/drive-v2")
.\training\scripts\server_training.ps1 -Action Start -SourceId rimpang-source-v2 `
  -RunId dataset-drive-v2 -JobId audit -PythonModule training.scripts.audit_raw_dataset `
  -Arguments @("--dataset-root", "/home/muhasadhabib/tanaman-rimpang/data/drive-v2", "--near-duplicates")
```

gdown hanya dipakai untuk listing folder. Unduhan per-berkas via `uc?id=` ditolak
Google ("many accesses") setelah puluhan request anonim, sehingga berkas diambil
dari `drive.usercontent.google.com` dengan jeda 0,5 detik dan backoff (±2,5 jam).
Audit single-core menormalkan EXIF ke PNG lossless (±2 jam). Keduanya dapat
dilanjutkan dengan perintah yang sama bila terputus.

## Anotasi privat melalui SSH

Setelah gambar sumber dipulihkan dan turunan EXIF-normalized siap, jalankan
`training.scripts.annotation_service` dari environment anotasi. Contoh perintah
Linux di server, dari snapshot source:

```bash
BASE=/home/muhasadhabib/tanaman-rimpang
DATA="$BASE/data/drive-v2"
"$BASE/envs/annotate-v1/bin/python" -m training.scripts.annotation_service \
  --service "$BASE/services/label-studio" --port 8087 \
  start --base "$BASE" --images "$DATA/images"
```

Gunakan `Start` wrapper dengan `-Environment annotate` dan RunId layanan yang
terpisah jika ingin supervisor/tmux persisten. Jangan menaruh job layanan yang
selalu hidup dalam RunId training yang nantinya akan dipaketkan.

Layanan bind hanya `127.0.0.1:8087`; pemilik SSH mendapat akun privat yang dibuat
secara acak, disimpan pada `services/label-studio/credentials.json` mode 600.
Ambil kredensial secara privat menggunakan akun SSH sendiri; jangan mengirimkannya
ke repo, screenshot, laporan, atau log. Analytics dimatikan dan local-file root
hanya menunjuk ke `DATA/images`, bukan home server atau seluruh filesystem.

Buka tunnel dari komputer lokal:

```powershell
.\training\scripts\server_training.ps1 -Action Tunnel -Port 8087
```

Lalu buka `http://localhost:8087`. Tunnel adalah koneksi lokal foreground;
tutup dengan Ctrl+C ketika tidak diperlukan. Menutup tunnel tidak menghentikan
layanan anotasi atau training di server.

Subcommand `annotation_service import` menerima `--tasks`, `--label-config`,
dan `--project-state` hasil persiapan anotasi. Ia membuat project dan memastikan
inventaris task cocok; pengulangan melanjutkan task yang belum diimpor.
Subcommand `export` mengambil seluruh task, termasuk yang belum selesai.
Ekspor Label Studio **belum** berarti anotasi valid: importer review tetap harus
menolak draft/skipped/konflik, bbox invalid, group belum jelas, dan negatif yang
tidak ditandai eksplisit.

## Persiapan data dan training

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

## Mengembalikan hasil ke komputer lokal

Paketkan hanya run yang job-nya sudah berhenti. Source dataset dan kredensial
Label Studio tidak termasuk artefak training.

```powershell
.\training\scripts\server_training.ps1 -Action Collect `
  -SourceId rimpang-source-v1 -RunId rimpang-v1
```

Hasil dipindahkan ke `artifacts\rimpang-v1` melalui staging, diperiksa checksum
arsip serta seluruh berkas, kemudian diterbitkan sebagai direktori baru.
Direktori hasil yang sudah ada tidak ditimpa. Arsip remote identik dapat dipakai
ulang setelah transfer gagal, sedangkan checkpoint/dataset server tidak dihapus.
Gunakan `-Destination` untuk direktori lokal baru bila mengambil snapshot lain.

`-AllowIncomplete` khusus mengambil diagnosis/log/hasil parsial dan mencatat
`status: incomplete` serta daftar artefak yang belum ada. Opsi tersebut bukan
persetujuan memasang model dan tidak mengubah manifest web.
Setelah bundle layak, ikuti `model-integration.md`; jangan menyalin checkpoint
PyTorch langsung ke direktori model browser.
