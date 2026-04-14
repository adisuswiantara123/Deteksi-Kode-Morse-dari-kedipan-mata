# 👁️ Deteksi Kode Morse via Kedipan Mata (Eye Blink Morse Detector)

Halo! Selamat datang di project **Deteksi Kode Morse**. Project ini adalah aplikasi berbasis Python yang memungkinkan kamu untuk mengetik pesan atau berkomunikasi menggunakan kode Morse hanya dengan **berkedip** di depan kamera. Keren, kan?

Aplikasi ini menggunakan teknologi *Computer Vision* untuk mendeteksi wajah dan mata secara real-time, lalu menerjemahkan durasi kedipanmu menjadi titik (.) atau garis (-) dalam kode Morse.

## ✨ Fitur Utama
- **Deteksi Real-time**: Menggunakan webcam untuk mendeteksi kedipan secara instan.
- **Adaptive Threshold**: Aplikasi secara pintar menyesuaikan sensitivitas kedipan berdasarkan kebiasaan berkedipmu.
- **Live Decoding**: Langsung menerjemahkan kode Morse menjadi karakter dan kata di layar.
- **Visual Feedback**: Menampilkan kontur mata dan status kedipan langsung di jendela video.

## 🛠️ Persiapan
Sebelum menjalankan aplikasi ini, pastikan kamu sudah menyiapkan beberapa hal berikut:

1. **Python**: Versi 3.x terinstal di komputermu.
2. **Library Python**:
   - `opencv-python` (untuk akses kamera dan pemrosesan gambar)
   - `dlib` (untuk deteksi titik wajah)
   - `numpy` & `scipy` (untuk perhitungan matematis)

   Kamu bisa menginstalnya dengan perintah:
   ```bash
   pip install opencv-python dlib numpy scipy
   ```

3. **Face Landmark Model**: Pastikan file `shape_predictor_68_face_landmarks.dat` ada di folder yang sama dengan script ini.

## 🚀 Cara Menjalankan
Cukup jalankan script utamanya lewat terminal atau command prompt:

```bash
python morse_kedip_mata.py
```

## 📖 Cara Menggunakan (Aturan Main)
Setelah aplikasi terbuka, kamu bisa mulai berkedip dengan aturan berikut:

- **Titik (.)**: Berkedip dengan cepat (kurang dari 0.3 detik).
- **Garis (-)**: Berkedip sedikit lebih lama/lambat (lebih dari 0.3 detik).
- **Spasi Karakter**: Diam (mata terbuka) selama kurang lebih 1 detik untuk menyelesaikan satu karakter.
- **Spasi Kata**: Diam selama kurang lebih 3 detik untuk memulai kata baru.

> [!TIP]
> Posisikan wajah tepat di depan kamera dan pastikan pencahayaannya cukup supaya deteksi matanya makin akurat!

## ⌨️ Shortcut
- Tekan tombol **'q'** pada keyboard untuk menutup aplikasi.

---

Dibuat dengan ❤️ untuk eksplorasi *Computer Vision*. Kalau ada ide atau nemu bug, langsung aja di-oprek ya! Selamat mencoba! 😉
