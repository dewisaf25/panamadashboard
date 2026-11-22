# ⚡ Panama Load Forecast Dashboard

Tema **biru** dengan aksesoris bertema listrik (gelombang kurva beban, petir, menara transmisi, kartu KPI berikon). Semua gaya ada di `utils.py` (CSS) dan `.streamlit/config.toml` (tema), tanpa file aset eksternal selain `assets/logo.svg`.

Dashboard Streamlit yang menampilkan **model terbaik (LightGBM)** untuk meramalkan beban listrik nasional Panama per jam, **24 jam ke depan**.
Model ini dipilih dari 11 model yang dibandingkan di notebook (baseline, klasik, ML, DL, transfer learning).

## Halaman
| Halaman | Isi |
|---|---|
| 🏠 Ringkasan | KPI akurasi (MAE, MAPE, RMSE, R²), split 70:10:20, temuan utama |
| 📊 Eksplorasi Data | tren, pola jam × hari, suhu, libur & sekolah, korelasi |
| 🎯 Performa Model | aktual vs prediksi, galat per jam / jenis hari / bulan, 10 hari terburuk |
| 🔍 Variabel Berpengaruh | suhu, libur, sekolah, jam, hari: 4 metode importance + SHAP |
| 🔮 Forecast Explorer | ramalan vs aktual pada tanggal pilihan, unduh CSV |
| 📘 Metodologi & Catatan | cara kerja, asumsi, keterbatasan |

## Struktur
```
app.py            # router + sidebar
utils.py          # loader data, tema, helper plot
forecasting.py    # pipeline fitur + pelatihan LightGBM (dijalankan sekali saat dashboard dibuka)
views/            # satu file per halaman
data/             # data per jam + hasil resmi notebook
assets/           # gambar SHAP
requirements.txt  # termasuk lightgbm
packages.txt      # libgomp1 (dibutuhkan LightGBM di Streamlit Cloud)
```

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
Saat halaman *Performa Model* / *Forecast Explorer* pertama dibuka, LightGBM dilatih ± 30–60 detik (di-cache setelahnya).

## Deploy dari GitHub (Streamlit Community Cloud)
1. Buat repository di GitHub, lalu **Add file → Upload files** dan unggah **seluruh isi folder ini** (termasuk `data/`, `assets/`, `views/`,
   `.streamlit/`, dan `packages.txt`). `app.py` harus berada di root repo.
2. Buka <https://share.streamlit.io>, login dengan GitHub, klik **Create app**.
3. Isi Repository, Branch `main`, Main file path `app.py`. Di *Advanced settings* pilih Python 3.11 atau 3.12.
4. Klik **Deploy**. Setiap perubahan di GitHub memperbarui dashboard otomatis.

## Opsional: pakai prediksi persis dari notebook
Jika ingin angka halaman Performa/Forecast identik dengan notebook, jalankan sel **“Export untuk dashboard”** di notebook terbaru, lalu taruh
`dashboard_predictions.csv.gz` di folder `data/`. Dashboard otomatis memakainya (tanpa melatih ulang).

## Catatan
Cuaca jam target memakai data aktual sebagai pengganti ramalan cuaca, jadi akurasi riil akan sedikit lebih rendah. Lihat halaman Metodologi.
