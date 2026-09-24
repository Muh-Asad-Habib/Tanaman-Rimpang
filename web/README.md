# Rimpang Web

Next.js App Router, TypeScript, font lokal Lora/Manrope, dan ONNX Runtime Web.
Panduan proyek lengkap tersedia di [README utama](../README.md).

```powershell
npm ci
npm run dev
```

Buka http://localhost:3000. Produksi: `npm run build`, lalu `npm start`.
Jangan menyalin folder `web` saja ke deployment: direktori `shared` di sebelahnya
dibutuhkan untuk urutan label dan kontrak.

Halaman: `/`, `/jelajah`, `/jelajah/[slug]`, `/panduan`, `/pindai`.
Kamera membutuhkan HTTPS atau localhost. Foto tetap di perangkat. Demo memiliki
label ilustrasi dan tidak berjalan pada foto/kamera pengguna.

Manifest awal sengaja `unavailable`; belum ada model terlatih. Ikuti
[integrasi model](../docs/model-integration.md) untuk memasang bundle hasil
training server. `predev`/`prebuild` menyalin aset ORT lokal sesuai versi paket.

Perintah pengembangan: `npm run lint`, `npm run typecheck`, dan `npm test`.
