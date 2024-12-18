# BMI Classification Using Voting Classifier Algorithm
Model Prediksi BMI

## Deskripsi Proyek
Proyek ini dirancang untuk memprediksi Body Mass Index (BMI) individu berdasarkan fitur-fitur tertentu seperti tinggi badan, berat badan, dan data terkait lainnya. Model ini menggunakan teknik machine learning untuk menganalisis dan memprediksi kategori BMI (misalnya, kurus, normal, kelebihan berat badan, obesitas). Proyek ini dapat digunakan dalam aplikasi kesehatan, sistem pelacakan kebugaran, atau alat edukasi.

## Instalasi

Untuk mengatur proyek ini, ikuti langkah-langkah berikut:

1. Clone repositori ini ke komputer Anda:
   ```bash
   git clone https://github.com/BlackBox-Ops/ML-Bmi-Predict.git
   ```

2. Install paket Python yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```
Pastikan Anda memiliki Python 3.7 atau versi yang lebih baru sebelum melanjutkan.

3. Masuk ke direktori proyek:
   ```bash
   cd Scripts/
   ```
4. Dan jalankan kode program di direktori Scripts:
   ```bash
   python api_model.py
   ```

5. jika pada kode program api_model.py berhasil dijalankan maka akan tampil alamat address sebagai berikut
   ```bash
   http://127.0.0.1:8000 
   ```

## Penggunaan

Model ini dilatih menggunakan dataset yang mengandung data terkait BMI. Setelah proyek diatur, Anda dapat menggunakan model ini untuk:
- Memprediksi kategori BMI individu berdasarkan fitur input mereka.
- Menganalisis tren BMI pada berbagai demografi (jika memungkinkan).

### Menjalankan Proyek
1. Siapkan dataset Anda: Pastikan data input Anda mengikuti format yang diharapkan (misalnya, mencakup `tinggi`, `berat`, dan kolom relevan lainnya).
2. Latih model (jika diperlukan) atau muat model yang sudah dilatih.
3. Gunakan model untuk membuat prediksi pada data baru.

Lihat file `api_model.py` atau skrip terkait untuk instruksi penggunaan spesifik.

## Informasi Dataset
Dataset yang digunakan untuk proyek ini mengandung kolom-kolom berikut:
- **Tinggi**: Tinggi badan individu (dalam sentimeter atau inci).
- **Berat**: Berat badan individu (dalam kilogram atau pon).
- **BMI**: Body Mass Index yang dihitung atau kategori yang sesuai (digunakan sebagai variabel target).

## Contoh Dataset

Berikut adalah cuplikan data dari file CSV yang digunakan dalam proyek:

| Tinggi (cm) | Berat (kg) | BMI Kategori     |
|-------------|------------|------------------|
| 170         | 65         | Normal           |
| 160         | 70         | Overweight       |
| 180         | 85         | Obesitas         |

*Dataset lengkap dapat ditemukan di folder `./Data/bmi.csv`.*

---

Pastikan dataset Anda sudah dibersihkan dan diproses sebelum digunakan dengan model.

## Fitur Utama Model
- Memperkirakan kategori BMI dengan akurasi tinggi.
- Pipeline yang mudah digunakan untuk pelatihan dan prediksi.
- Format input yang fleksibel untuk berbagai struktur dataset.

Silakan sesuaikan dan kembangkan proyek ini sesuai dengan kebutuhan spesifik Anda.
