import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (B600, B800, B900, BEST, CYAN, NAIVE, SKY, SPLITS, COVID_START, card, divider, hero, kpi, load_hourly,
                   load_importance, load_results, note, plot, section, shade, style_fig, vline)

res = load_results()
te, pre, cov = res["Test total"], res["Pra-COVID"], res["Saat COVID"]
imp = load_importance()
h = load_hourly()
b, nv = te.loc[BEST], te.loc[NAIVE]
gain = 100 * (1 - b["MAE"] / nv["MAE"])

hero("⚡ Peramalan Beban Listrik Nasional Panama",
     "Model terbaik dari eksperimen: <b>LightGBM</b> yang meramalkan beban per jam untuk <b>24 jam ke depan</b> (day-ahead), "
     "dipilih dari 11 model (baseline, time series klasik, Machine Learning, Deep Learning, dan transfer learning).",
     tags=["Model: LightGBM", "Horizon: 24 jam", "Split kronologis 70 : 10 : 20", "Data: Jan 2015 – Jun 2020"])

section("Akurasi pada data Test", "🎯", "23 Mei 2019 – 26 Jun 2020")
c1, c2, c3, c4 = st.columns(4)
with c1: kpi("MAE", f"{b['MAE']:.1f} MWh", f"pra-COVID {pre.loc[BEST,'MAE']:.1f} · COVID {cov.loc[BEST,'MAE']:.1f}", B600, "🎯")
with c2: kpi("MAPE", f"{b['MAPE%']:.2f}%", f"pra-COVID {pre.loc[BEST,'MAPE%']:.2f}% · COVID {cov.loc[BEST,'MAPE%']:.2f}%", B800, "📉")
with c3: kpi("RMSE", f"{b['RMSE']:.1f} MWh", f"pra-COVID {pre.loc[BEST,'RMSE']:.1f} · COVID {cov.loc[BEST,'RMSE']:.1f}", CYAN, "📏")
with c4: kpi("R²", f"{b['R2']:.3f}", f"pra-COVID {pre.loc[BEST,'R2']:.3f} · COVID {cov.loc[BEST,'R2']:.3f}", SKY, "📈")
st.caption(f"Sebagai pembanding, patokan sederhana 'beban minggu lalu pada jam yang sama' memiliki MAE {nv['MAE']:.1f} MWh, "
           f"sehingga LightGBM **{gain:.0f}% lebih akurat**.")

divider()
section("Rentang data & pembagian data", "🗓️", "Train 70% · Validasi 10% · Test 20%")
daily = h.set_index("datetime")["nat_clean"].resample("D").mean()
fig = go.Figure(go.Scatter(x=daily.index, y=daily.values, mode="lines", line=dict(color=B900, width=1.1),
                           name="Rata-rata harian", hovertemplate="%{x|%d %b %Y}<br>%{y:.0f} MWh<extra></extra>"))
for name, (a, z, col) in SPLITS.items():
    shade(fig, a, pd.Timestamp(z) + pd.Timedelta(days=1), col, name)
vline(fig, COVID_START, B900, "Awal COVID-19")
fig.update_yaxes(title="MWh")
plot(style_fig(fig, 340, legend_bottom=False))

divider()
section("Temuan utama", "💡")
exo = imp["ΔMAE permutasi (eksogen-saja)"].dropna().sort_values(ascending=False)
lag_gain = imp.loc["Historis (lag)", "Gain % (full)"]
worse = 100 * (cov.loc[BEST, "MAE"] / pre.loc[BEST, "MAE"] - 1)
a, c, d = st.columns(3)
with a:
    card("🎯 Akurat pada kondisi normal",
         f"Sebelum COVID, rata-rata galat hanya <b>{pre.loc[BEST,'MAPE%']:.2f}%</b> ({pre.loc[BEST,'MAE']:.1f} MWh) dengan R² "
         f"{pre.loc[BEST,'R2']:.2f}. Model menangkap pola jam, hari, libur, dan efek non-linear suhu.", B600)
with c:
    card("🦠 COVID-19 menurunkan akurasi",
         f"Sejak Maret 2020 pola konsumsi berubah mendadak. MAE naik <b>{worse:.0f}%</b> "
         f"({pre.loc[BEST,'MAE']:.1f} → {cov.loc[BEST,'MAE']:.1f} MWh). Ini wajar karena model belajar dari pola era normal.", SKY)
with d:
    card("🔍 Pendorong utama ramalan",
         f"Beban historis (kemarin dan minggu lalu) menyumbang ±<b>{lag_gain:.0f}%</b> gain model. Di antara variabel non-historis, "
         f"yang terpenting: <b>{exo.index[0]}</b>, <b>{exo.index[1]}</b>, lalu <b>{exo.index[2]}</b>.", B800)

st.markdown("&nbsp;")
note("💡 <b>Navigasi:</b> <b>Eksplorasi Data</b> untuk pola beban · <b>Performa Model</b> untuk kualitas ramalan (per jam, jenis hari, bulan) · "
     "<b>Variabel Berpengaruh</b> untuk suhu/libur/sekolah/jam/hari · <b>Forecast Explorer</b> untuk ramalan vs aktual pada tanggal pilihan.")
