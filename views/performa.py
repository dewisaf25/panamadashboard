import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (B200, B600, B800, B900, BEST, COV_COLOR, COVID_START, CYAN, DOW_ID, NAIVE, PRE_COLOR, SKY, divider, kpi,
                   load_hourly, load_predictions, metrics, note, page_header, plot, section, style_fig, table, vline)

page_header("🎯 Performa Model", "Seberapa akurat LightGBM pada data Test: per periode, jam, jenis hari, bulan, dan hari terburuk.")
pred, source = load_predictions()
h = load_hourly()
if source.startswith("baseline"):
    note(f"⚠️ Model LightGBM tidak dapat disiapkan ({source[9:]}). Halaman ini butuh paket <code>lightgbm</code> "
         "atau file <code>data/dashboard_predictions.csv.gz</code> dari notebook.", "warn")
    st.stop()

p = pred.copy()
p["err"] = p[BEST] - p["actual"]
p["abs"] = p["err"].abs()
p["period"] = np.where(p["datetime"] < COVID_START, "Pra-COVID", "Saat COVID")
p["hour"] = p["datetime"].dt.hour
p["date"] = p["datetime"].dt.normalize()
hol = h.set_index("datetime")["holiday"]
p["holiday"] = p["datetime"].map(hol).fillna(0)
p["Jenis hari"] = np.where(p["holiday"] == 1, "Libur nasional", np.where(p["datetime"].dt.dayofweek >= 5, "Akhir pekan", "Hari kerja"))
COL = {"Pra-COVID": PRE_COLOR, "Saat COVID": COV_COLOR}

# ------------------------------------------------------------------ metrik per periode
rows = []
for name, m in (("Test total", np.ones(len(p), bool)), ("Pra-COVID", (p["period"] == "Pra-COVID").values),
                ("Saat COVID", (p["period"] == "Saat COVID").values)):
    r = {"Periode": name, "Jumlah jam": int(m.sum())}
    r.update({f"LightGBM {k}": v for k, v in metrics(p.loc[m, "actual"], p.loc[m, BEST]).items()})
    r["Naive-168 MAE"] = metrics(p.loc[m, "actual"], p.loc[m, NAIVE])["MAE"]
    rows.append(r)
mt = pd.DataFrame(rows).set_index("Periode")
cols = st.columns(3)
for col, (name, r), color, ic in zip(cols, mt.iterrows(), [B800, B600, SKY], ["📊", "✅", "🦠"]):
    with col:
        kpi(name, f"MAE {r['LightGBM MAE']:.1f} MWh", f"MAPE {r['LightGBM MAPE%']:.2f}% · RMSE {r['LightGBM RMSE']:.1f} · R² {r['LightGBM R2']:.3f}", color, ic)
st.caption("Angka dihitung dari prediksi pada halaman ini; dapat berbeda tipis dari angka resmi di notebook bila model dilatih ulang.")
with st.expander("📋 Tabel metrik + patokan Naive-168"):
    table(mt.round(3))

# ------------------------------------------------------------------ aktual vs prediksi
divider()
section("Aktual vs prediksi", "🎯", "tiap titik = satu jam pada data Test")
fig = go.Figure()
for per, g in p.groupby("period"):
    fig.add_trace(go.Scattergl(x=g["actual"], y=g[BEST], mode="markers", name=per,
                               marker=dict(size=3, color=COL[per], opacity=.35)))
lo, hi = float(p["actual"].min()), float(p["actual"].max())
fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=B900, dash="dash", width=1.4))
fig.update_layout(xaxis_title="Aktual (MWh)", yaxis_title="Prediksi (MWh)")
plot(style_fig(fig, 430, "Garis putus-putus = prediksi sempurna"))

# ------------------------------------------------------------------ per jam & jenis hari
a, b = st.columns(2)
with a:
    g = p.groupby(["hour", "period"])["abs"].mean().reset_index()
    fig = go.Figure()
    for per in ("Pra-COVID", "Saat COVID"):
        d = g[g["period"] == per]
        fig.add_trace(go.Scatter(x=d["hour"], y=d["abs"], mode="lines+markers", name=per, line=dict(color=COL[per], width=2.4)))
    fig.update_layout(xaxis_title="Jam target", yaxis_title="MAE (MWh)")
    plot(style_fig(fig, 360, "MAE per jam dalam sehari"))
with b:
    g = p.groupby(["Jenis hari", "period"])["abs"].mean().reset_index()
    order = ["Hari kerja", "Akhir pekan", "Libur nasional"]
    fig = go.Figure()
    for per in ("Pra-COVID", "Saat COVID"):
        d = g[g["period"] == per].set_index("Jenis hari").reindex(order)
        fig.add_trace(go.Bar(x=order, y=d["abs"], name=per, marker_color=COL[per], text=d["abs"].round(1), textposition="outside"))
    fig.update_layout(barmode="group", yaxis_title="MAE (MWh)", xaxis_title="")
    plot(style_fig(fig, 360, "MAE per jenis hari"))

# ------------------------------------------------------------------ per bulan & harian
mo = p.groupby(p["datetime"].dt.to_period("M").astype(str)).agg(MAE=("abs", "mean"), n=("abs", "size")).reset_index()
mo.columns = ["Bulan", "MAE", "n"]
fig = go.Figure(go.Bar(x=mo["Bulan"], y=mo["MAE"], marker_color=[COV_COLOR if m >= "2020-03" else PRE_COLOR for m in mo["Bulan"]],
                       text=mo["MAE"].round(1), textposition="outside", hovertemplate="%{x}<br>MAE %{y:.1f} MWh<extra></extra>"))
fig.update_layout(yaxis_title="MAE (MWh)", xaxis_title="")
plot(style_fig(fig, 340, "MAE per bulan (biru muda = periode COVID)", legend_bottom=False))

daily = p.groupby("date")["abs"].mean()
roll = daily.rolling(7, min_periods=3).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily.index, y=daily.values, mode="lines", name="MAE harian", line=dict(color=B200, width=1)))
fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", name="Rata-rata bergulir 7 hari", line=dict(color=B600, width=2.6)))
vline(fig, COVID_START, B900, "COVID-19")
fig.update_layout(yaxis_title="MAE (MWh)")
plot(style_fig(fig, 340, "MAE harian sepanjang periode Test"))

# ------------------------------------------------------------------ hari terburuk
divider()
section("10 hari dengan galat terbesar", "🚨")
dd = p.groupby("date").agg(MAE=("abs", "mean"), rata_aktual=("actual", "mean"), libur=("holiday", "max")).reset_index()
dd["MAPE"] = (100 * (p["abs"] / p["actual"]).groupby(p["date"]).mean()).values
dd["Hari"] = dd["date"].dt.dayofweek.map(lambda i: DOW_ID[i])
dd["Libur"] = np.where(dd["libur"] == 1, "Ya", "")
worst = dd.sort_values("MAE", ascending=False).head(10)
worst["Tanggal"] = worst["date"].dt.strftime("%d %b %Y")
table(worst[["Tanggal", "Hari", "Libur", "MAE", "MAPE", "rata_aktual"]].rename(
    columns={"MAE": "MAE (MWh)", "MAPE": "MAPE (%)", "rata_aktual": "Rata-rata aktual (MWh)"}).set_index("Tanggal").round(1))
note("Hari-hari dengan galat terbesar umumnya jatuh pada <b>masa awal pembatasan COVID</b> dan <b>hari libur</b>, yaitu saat pola beban "
     "menyimpang dari kebiasaan minggu-minggu sebelumnya yang menjadi dasar fitur historis model.")
