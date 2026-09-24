# Desain: catatan botani yang hangat

- Ivory `#f8f6f0`, teks hutan `#243d30`, muted `#636b5d`, aksen kunyit `#b68636`.
- Lora untuk judul/editorial dan Manrope untuk isi/interaksi; fallback tersedia.
- Ruang kosong, aturan tipografi dan foto spesimen menggantikan dekorasi generik.
- Konten bahasa Indonesia, dengan nama ilmiah sebagai informasi sekunder.
- Container fluid dengan max-width 1240px; layout kamera menjadi satu kolom pada
  lebar 800px ke bawah. Mendukung layar mulai 320px dan safe area perangkat.
- Breakpoint 360/580/800/900/1100/1600px, dengan ukuran fluid di antaranya.
- Teks isi utama 14-16px, teks sekunder umumnya 12-13px. Label ilustrasi tetap
  lebih kecil. Input pencarian dan pemilihan kamera 16px agar tidak memicu zoom
  otomatis saat fokus di ponsel.
- Kontrol utama memiliki tinggi minimal 44px. Kartu tetap menampilkan petunjuk
  tautan pada layar sentuh tanpa membutuhkan hover.
- Ukuran unduhan gambar mengikuti layout kartu. Foto utama detail dan empat
  kartu pertama katalog dimuat eager; kartu lainnya tetap lazy.
- Header sticky; navigasi ringkas pada lebar 900px ke bawah. Menu menutup saat
  navigasi, interaksi di luar, atau pergantian breakpoint. Escape menutup menu
  dan mengembalikan fokus ke tombol pembuka.
- Pratinjau kamera/foto mengikuti ukuran bingkai aktual dengan contain fit.
  Pada orientasi landscape dengan tinggi maksimal 600px, area aktif dipadatkan
  ke 170-270px. Placeholder tetap cukup tinggi agar petunjuk terbaca.
- Perubahan ukuran video memperbarui rasio dan menghapus koordinat hasil lama.
  Kotak objek memakai permukaan yang sama dengan foto/video, bukan letterbox.
- Focus ring, skip link ke main yang dapat menerima fokus, semantik heading,
  native details, reduced-motion; pinch zoom dan zoom browser tidak dibatasi.
- Demo dan unavailable adalah bagian desain produk, bukan error yang disembunyikan.

Halaman utama tidak memuat statistik palsu/review rekaan. Foto/katalog merupakan
konten editorial; anotasi demo tidak pernah menjadi hasil inferensi.
