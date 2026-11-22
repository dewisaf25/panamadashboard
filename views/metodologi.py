import streamlit as st

from utils import ASSETS, divider, note, page_header, section

page_header("📘 Metodologi & Catatan", "Bagaimana model terbaik dipilih, cara kerjanya, asumsi, dan keterbatasannya.")

section("Mengapa LightGBM", "🏆")
st.markdown("""
Di notebook, **11 model** dibandingkan: baseline (naive), time series klasik (Dynamic Regression + ARMA), Machine Learning (Ridge, LightGBM),
Deep Learning (GRU), dan transfer learning (pre-train + fine-tune). **LightGBM paling akurat** pada data Test total maupun pra-COVID
(berdasarkan MAE, RMSE, MAPE, dan R²). Dashboard ini hanya menampilkan model tersebut.

Alasan teknisnya: LightGBM menangkap hubungan **non-linear** suhu–beban (AC menyala di atas ambang suhu), interaksi **jam × hari × libur**,
tahan terhadap fitur yang saling berkorelasi, cepat dilatih, dan menyediakan ukuran pentingnya variabel.
""")

divider()
section("Skema peramalan", "🧭", "tanpa kebocoran data")
st.markdown("""
- **Titik asal (origin)** adalah pukul **23:00** hari sebelumnya. Dari situ diramalkan 24 jam hari berikutnya (00:00 – 23:00).
- Satu model global untuk horizon 1–24 jam (*direct multi-horizon*), sehingga tidak ada akumulasi galat rekursif.
- Fitur historis hanya memakai `nat_demand` sampai origin: lag 24, 48, 72, 168, 336 jam, nilai di origin, serta rata-rata 24 dan 168 jam terakhir.
- Kalender, libur, dan sekolah diketahui di muka. **Cuaca jam target memakai data aktual sebagai pengganti ramalan cuaca**,
  sehingga akurasi di dunia nyata (dengan ramalan NWP) akan sedikit lebih buruk.
- **Split kronologis 70 : 10 : 20** (tanpa diacak): Train ≤ 3 Nov 2018, Validasi 4 Nov 2018 – 22 Mei 2019 (early stopping),
  Test 23 Mei 2019 – 26 Jun 2020 (memuat masa normal dan guncangan COVID-19 sejak ± Maret 2020).
- **Normalisasi level:** target dan fitur lag dibagi rata-rata 168 jam terakhir di origin, agar model pohon tidak terganggu tren kenaikan beban.
- **Anomali:** 8 jam bernilai < 600 MWh (mis. pemadaman 20 Jan 2019) diinterpolasi untuk training dan fitur; evaluasi memakai data mentah.
""")

divider()
section("Analisis variabel", "🔍")
st.markdown("""
Pengaruh variabel diukur **per kelompok** (suhu, kelembapan, angin, hujan/awan, libur, sekolah, jam, hari, bulan, lag historis) dengan empat cara:
gain bawaan LightGBM, permutasi pada model penuh, *ablation* (model dilatih ulang tanpa kelompok), dan model **tanpa lag historis**
yang mengukur pengaruh intrinsik variabel non-historis. SHAP melengkapi sebagai pandangan per fitur.
""")

divider()
section("Catatan jujur", "📝")
note("""<ul style="margin:0;padding-left:18px">
<li><b>Periode COVID:</b> LightGBM tetap terbaik pada MAE, tetapi RMSE-nya (77,4) sedikit di atas rata-rata 7 hari (72,5), jadi keunggulannya
menyempit saat pola konsumsi berubah mendadak. Periode COVID di Test hanya ± 3,5 bulan.</li>
<li><b>Cuaca:</b> memakai data aktual, bukan ramalan, sehingga hasil ini adalah batas atas akurasi.</li>
<li><b>Satu pembagian waktu</b> (bukan validasi silang berulang), sehingga ketidakpastian estimasi tidak dihitung.</li>
<li><b>Pelatihan di aplikasi:</b> bila file <code>dashboard_predictions.csv.gz</code> tidak ada, dashboard melatih ulang LightGBM dengan pengaturan
yang sama dengan notebook. Angka dapat berbeda sangat tipis dari angka resmi notebook.</li>
</ul>""", "warn")

shap_p = ASSETS / "shap_summary.png"
if shap_p.exists():
    with st.expander("🧠 Ringkasan SHAP per fitur (dari notebook)"):
        st.image(str(shap_p), caption="SHAP summary LightGBM model penuh, data Test (3.000 sampel)")
