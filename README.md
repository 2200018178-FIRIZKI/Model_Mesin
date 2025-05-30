Proyek Machine Learning Analisis Keuangan
Proyek ini adalah implementasi dari serangkaian model Machine Learning untuk menganalisis data transaksi keuangan pribadi. Tujuannya adalah untuk memberikan wawasan otomatis, mendeteksi anomali, dan memberikan rekomendasi cerdas untuk manajemen keuangan yang lebih baik.

🚀 Modul Utama Proyek
Proyek ini dibagi menjadi tiga modul utama dengan tujuan yang spesifik:

Klasifikasi Kategori Transaksi (Kontributor: Khadafi)

Tujuan: Mengklasifikasikan transaksi secara otomatis ke dalam kategori yang relevan (misalnya: "Makanan", "Transportasi", "Hiburan") berdasarkan deskripsi (Note) dan jumlah (Amount).
Jenis ML: Supervised Learning (Klasifikasi Multi-Kelas).
Model: Naive Bayes, Random Forest, atau SVM.
Deteksi Anomali Transaksi (Kontributor: Adhim)

Tujuan: Mengidentifikasi transaksi yang tidak biasa atau mencurigakan yang mungkin merupakan penipuan atau kesalahan input.
Jenis ML: Unsupervised Learning.
Model: Isolation Forest atau One-Class SVM.
Rekomendasi & Tips Keuangan (Kontributor: Arya)

Tujuan: Memberikan saran dan tips penghematan yang dipersonalisasi berdasarkan pola pengeluaran pengguna.
Pendekatan: Sistem berbasis aturan (Rule-Based) dengan dukungan clustering (K-Means) opsional.
🛠️ Teknologi yang Digunakan
Bahasa Pemrograman: Python 3.8+
Library Utama:
Pandas: Untuk manipulasi dan analisis data.
Scikit-learn: Untuk membangun dan mengevaluasi model Machine Learning.
NLTK: Untuk pra-pemrosesan data teks.
Deployment API: FastAPI
Database: PostgreSQL
📂 Struktur Folder
financial_ml_project/
│
├── 📁 1_klasifikasi_transaksi_khadafi/
│   ├── 📄 app.py
│   ├── 📁 data/
│   ├── 📁 notebooks/
│   ├── 📁 src/
│   ├── 📁 models/
│   └── 📄 requirements.txt
│
├── 📁 2_deteksi_anomali_adhim/
│   ├── 📄 app.py
│   └── ... (struktur serupa)
│
├── 📁 3_rekomendasi_keuangan_arya/
│   ├── 📄 app.py
│   └── ... (struktur serupa)
│
├── 📁 database/
│   └── 📄 schema.sql
│
├── 📄 .gitignore
└── 📄 README.md
⚙️ Instalasi & Setup
Ikuti langkah-langkah berikut untuk menjalankan proyek ini secara lokal.

Clone Repositori

Bash

git clone https://github.com/[nama-user-anda]/financial_ml_project.git
cd financial_ml_project
Buat dan Aktifkan Virtual Environment

Bash

# Membuat venv
python -m venv venv

# Mengaktifkan di Windows
venv\Scripts\activate

# Mengaktifkan di macOS/Linux
source venv/bin/activate
Instal Dependensi untuk Setiap Modul
Anda perlu menginstal requirements.txt untuk setiap modul yang ingin Anda jalankan. Mulailah dengan modul pertama.

Bash

# Masuk ke direktori modul
cd 1_klasifikasi_transaksi_khadafi

# Install library yang dibutuhkan
pip install -r requirements.txt
Ulangi langkah ini untuk direktori 2_deteksi_anomali_adhim dan 3_rekomendasi_keuangan_arya jika diperlukan.

Setup Database (Opsional)
Pastikan server PostgreSQL Anda berjalan. Jalankan skrip schema.sql untuk membuat tabel yang diperlukan.

Bash

psql -U [nama_user] -h [host] -d [nama_db] -f database/schema.sql
🏃 Cara Menjalankan
Setiap modul memiliki server API-nya sendiri.

Contoh: Menjalankan Modul Klasifikasi Transaksi
Latih Model (jika belum ada)
Dari dalam direktori 1_klasifikasi_transaksi_khadafi/, jalankan skrip training:

Bash

python src/train.py
Jalankan Server API
Masih di dalam direktori 1_klasifikasi_transaksi_khadafi/, jalankan server Uvicorn:

Bash

uvicorn app:app --reload
Server akan berjalan di http://127.0.0.1:8000.

Gunakan Endpoint
Buka browser Anda dan kunjungi http://127.0.0.1:8000/docs untuk melihat dokumentasi API interaktif.

Contoh Request menggunakan cURL:

Bash

curl -X 'POST' \
  'http://127.0.0.1:8000/predict/category' \
  -H 'Content-Type: application/json' \
  -d '{
    "note": "Makan siang nasi padang",
    "amount": 25000
  }'
Contoh Response:

JSON

{
  "note": "Makan siang nasi padang",
  "amount": 25000,
  "predicted_category": "Makanan"
}
👥 Kontributor
Khadafi - Klasifikasi Kategori Transaksi
Adhim - Deteksi Anomali Transaksi
Arya - Rekomendasi & Tips Keuangan#   M o d e l _ M e s i n  
 