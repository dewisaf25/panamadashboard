import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (B600, B900, BEST, DOW_ID, NAIVE, TEST_END, TEST_START, load_hourly, load_predictions, metrics, note,
                   page_header, plot, style_fig, table)

page_header("🔮 Forecast Explorer", "Lihat ramalan day-ahead LightGBM vs beban aktual pada tanggal pilihan (periode Test).")
pred, source = load_predictions()
h = load_hourly()
if source.startswith("baseline"):
    note(f"⚠️ Model LightGBM tidak dapat disiapkan ({source[9:]}). Pastikan <code>lightgbm</code> ada di requirements.txt "
         "atau unggah <code>data/dashboard_predictions.csv.gz</code> dari notebook.", "warn")
    st.stop()

hol_days = sorted(h.loc[(h["holiday"] == 1) & (h["dow"] < 5) & (h["date"] >= TEST_START) & (h["date"] <= TEST_END), "date"].unique())
presets = {"Pilih sendiri": None,
           "Hari kerja biasa (13 Nov 2019)": pd.Timestamp("2019-11-13"),
           "Akhir pekan (16 Nov 2019)": pd.Timestamp("2019-11-16"),
           "Awal pembatasan COVID (± 25 Mar 2020)": pd.Timestamp("2020-03-25")}
for d in hol_days[:6]:
    presets[f"Libur nasional ({pd.Timestamp(d).strftime('%d %b %Y')})"] = pd.Timestamp(d)

c1, c2, c3 = st.columns([1.6, 1.2, 1.2])
with c1:
    sel = st.selectbox("Lompat cepat ke", list(presets.keys()))
with c2:
    start = presets[sel] or pd.Timestamp("2019-11-13")
    day = st.date_input("Tanggal", value=start.date(), min_value=TEST_START.date(), max_value=TEST_END.date())
with c3:
    span = st.radio("Rentang", ["1 hari (24 jam)", "7 hari"], horizontal=True)
show_naive = st.checkbox("Tampilkan patokan Naive-168 (beban minggu lalu jam yang sama)", value=True)

t0 = pd.Timestamp(day)
t1 = t0 + pd.Timedelta(days=1 if span.startswith("1") else 7)
w = pred[(pred["datetime"] >= t0) & (pred["datetime"] < t1)]
if w.empty:
    st.warning("Tidak ada data pada rentang itu.")
    st.stop()

fig = go.Figure()
fig.add_trace(go.Scatter(x=w["datetime"], y=w["actual"], mode="lines", name="Aktual", line=dict(color=B900, width=3.4),
                         hovertemplate="%{x|%d %b %H:%M}<br>Aktual: %{y:.0f} MWh<extra></extra>"))
fig.add_trace(go.Scatter(x=w["datetime"], y=w[BEST], mode="lines", name="Prediksi LightGBM", line=dict(color=B600, width=2.6),
                         hovertemplate="%{x|%d %b %H:%M}<br>LightGBM: %{y:.0f} MWh<extra></extra>"))
if show_naive:
    fig.add_trace(go.Scatter(x=w["datetime"], y=w[NAIVE], mode="lines", name="Naive-168", line=dict(color="#94A3B8", width=1.8, dash="dot"),
                             hovertemplate="%{x|%d %b %H:%M}<br>Naive-168: %{y:.0f} MWh<extra></extra>"))
fig.add_trace(go.Scatter(x=pd.concat([w["datetime"], w["datetime"][::-1]]),
                         y=pd.concat([w[BEST], w["actual"][::-1]]), fill="toself", fillcolor="rgba(37,99,235,.12)",
                         line=dict(width=0), name="Selisih", hoverinfo="skip"))
fig.update_yaxes(title="MWh")
plot(style_fig(fig, 440, f"Ramalan vs aktual: {t0.strftime('%d %b %Y')} ({DOW_ID[t0.dayofweek]})"))

rows = []
for name, col in (("LightGBM", BEST), ("Naive-168", NAIVE)):
    ok = w[col].notna()
    rows.append({"Model": name, **metrics(w.loc[ok, "actual"], w.loc[ok, col])})
mt = pd.DataFrame(rows).set_index("Model")
a, b, c, d = st.columns(4)
a.metric("MAE (LightGBM)", f"{mt.loc['LightGBM','MAE']:.1f} MWh", f"{mt.loc['LightGBM','MAE']-mt.loc['Naive-168','MAE']:+.1f} vs Naive-168", delta_color="inverse")
b.metric("MAPE", f"{mt.loc['LightGBM','MAPE%']:.2f}%")
c.metric("RMSE", f"{mt.loc['LightGBM','RMSE']:.1f} MWh")
pk = w.loc[w["actual"].idxmax()]
d.metric("Puncak aktual", f"{pk['actual']:.0f} MWh", f"prediksi {pk[BEST]:.0f} MWh pada jam {pk['datetime'].hour:02d}:00", delta_color="off")

with st.expander("📋 Tabel jam demi jam"):
    tab = w[["datetime", "actual", BEST, NAIVE]].copy()
    tab["Selisih"] = tab[BEST] - tab["actual"]
    tab.columns = ["Waktu", "Aktual", "Prediksi LightGBM", "Naive-168", "Selisih (pred − aktual)"]
    table(tab.set_index("Waktu").round(1))

st.download_button("⬇️ Unduh data rentang ini (CSV)", data=w[["datetime", "actual", BEST, NAIVE]].to_csv(index=False).encode("utf-8"),
                   file_name=f"prediksi_{t0.date()}.csv", mime="text/csv")
st.caption("Skema day-ahead: ramalan dibuat pukul 23:00 hari sebelumnya untuk 24 jam berikutnya. "
           "Cuaca jam target memakai data aktual sebagai pengganti ramalan cuaca.")
