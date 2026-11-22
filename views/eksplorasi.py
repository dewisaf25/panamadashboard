import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (B200, B500, B600, B700, B800, B900, COV_COLOR, COVID_START, CYAN, DOW_ID, PRE_COLOR, SKY, SPLITS, kpi,
                   load_hourly, note, page_header, plot, shade, style_fig, vline)

page_header("📊 Eksplorasi Data", "Pola beban listrik Panama: tren, rutinitas harian/mingguan, efek suhu, libur, dan sekolah.")
h = load_hourly()
n_anom = int((h["nat_demand"] < 600).sum())

c1, c2, c3, c4 = st.columns(4)
with c1: kpi("Jumlah observasi", f"{len(h):,}".replace(",", "."), "per jam, tanpa jam hilang", B600, "🕒")
with c2: kpi("Rata-rata beban", f"{h['nat_clean'].mean():.0f} MWh", f"min {h['nat_clean'].min():.0f} · maks {h['nat_clean'].max():.0f}", B800, "⚡")
with c3: kpi("Hari libur nasional", f"{int(h.groupby('date')['holiday'].max().sum())} hari", "flag `holiday` = 1", CYAN, "🎌")
with c4: kpi("Anomali (< 600 MWh)", f"{n_anom} jam", "mis. pemadaman 20 Jan 2019, dibersihkan", SKY, "🔌")
st.markdown("&nbsp;")

t1, t2, t3, t4, t5 = st.tabs(["📈 Tren", "🕒 Pola jam & hari", "🌡️ Suhu", "🎌 Libur & sekolah", "🔗 Korelasi"])

# ------------------------------------------------------------------ Tren
with t1:
    freq = st.radio("Agregasi", ["Harian", "Mingguan", "Bulanan"], horizontal=True, index=1)
    rule = {"Harian": "D", "Mingguan": "W", "Bulanan": "MS"}[freq]
    s = h.set_index("datetime")["nat_clean"].resample(rule).mean()
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=B600, width=1.6),
                               fill="tozeroy", fillcolor="rgba(37,99,235,.10)", name=freq,
                               hovertemplate="%{x|%d %b %Y}<br>%{y:.0f} MWh<extra></extra>"))
    for name, (a, b, col) in SPLITS.items():
        shade(fig, a, pd.Timestamp(b) + pd.Timedelta(days=1), col, name, 0.07)
    vline(fig, COVID_START, B900, "COVID-19")
    fig.update_yaxes(title="MWh", rangemode="tozero")
    plot(style_fig(fig, 400, f"Rata-rata beban ({freq.lower()})", legend_bottom=False))

    yr = h.groupby("year")["nat_clean"].agg(["mean", "count"]).reset_index()
    yr["label"] = yr["year"].astype(str) + np.where(yr["count"] < 8000, " (parsial)", "")
    fig2 = px.bar(yr, x="label", y="mean", text=yr["mean"].round(0), color_discrete_sequence=[B500])
    fig2.update_traces(textposition="outside", hovertemplate="%{x}<br>%{y:.0f} MWh<extra></extra>")
    fig2.update_yaxes(title="MWh", range=[0, yr["mean"].max() * 1.15]); fig2.update_xaxes(title="")
    plot(style_fig(fig2, 330, "Rata-rata beban per tahun"))
    note("Beban naik bertahap 2015–2019, lalu turun tajam sejak <b>Maret 2020</b> (pembatasan COVID-19). "
         "Inilah yang membuat periode Test menjadi ujian ketahanan model.")

# ------------------------------------------------------------------ Pola
with t2:
    per = st.radio("Periode", ["Semua data", "Pra-COVID (< 15 Mar 2020)", "Saat COVID (≥ 15 Mar 2020)"], horizontal=True)
    d = h
    if per.startswith("Pra"): d = h[h["datetime"] < COVID_START]
    elif per.startswith("Saat"): d = h[h["datetime"] >= COVID_START]
    hm = d.pivot_table(index="dow", columns="hour", values="nat_clean", aggfunc="mean")
    hm.index = DOW_ID
    fig = px.imshow(hm, aspect="auto", color_continuous_scale="Blues", labels=dict(x="Jam", y="", color="MWh"))
    plot(style_fig(fig, 340, "Rata-rata beban: hari × jam", legend_bottom=False))

    prof = d.groupby(["weekend", "hour"])["nat_clean"].mean().reset_index()
    prof["Tipe hari"] = prof["weekend"].map({False: "Hari kerja", True: "Akhir pekan"})
    fig = px.line(prof, x="hour", y="nat_clean", color="Tipe hari", markers=True,
                  color_discrete_map={"Hari kerja": B700, "Akhir pekan": SKY})
    fig.update_layout(xaxis_title="Jam", yaxis_title="MWh")
    plot(style_fig(fig, 340, "Profil beban rata-rata per jam"))

    wk = h[~h["weekend"] & (h["holiday"] == 0)].copy()
    wk["Periode"] = np.where(wk["datetime"] < COVID_START, "Pra-COVID", "Saat COVID")
    cmp_ = wk.groupby(["Periode", "hour"])["nat_clean"].mean().reset_index()
    fig = px.line(cmp_, x="hour", y="nat_clean", color="Periode", markers=True,
                  color_discrete_map={"Pra-COVID": PRE_COLOR, "Saat COVID": COV_COLOR})
    fig.update_layout(xaxis_title="Jam", yaxis_title="MWh")
    plot(style_fig(fig, 340, "Hari kerja biasa: profil sebelum vs saat COVID"))
    note("Dua puncak harian (siang & malam) dan beban akhir pekan yang lebih rendah adalah pola rutin. "
         "Saat COVID, seluruh kurva bergeser turun, terutama pada jam kerja.")

# ------------------------------------------------------------------ Suhu
with t3:
    slot = st.radio("Jam", ["Semua jam", "Dini hari (0–5)", "Pagi (6–11)", "Siang (12–17)", "Malam (18–23)"], horizontal=True)
    rng = {"Semua jam": (0, 23), "Dini hari (0–5)": (0, 5), "Pagi (6–11)": (6, 11),
           "Siang (12–17)": (12, 17), "Malam (18–23)": (18, 23)}[slot]
    city = st.selectbox("Sumber suhu", ["Rata-rata 3 kota", "Tocumen", "Santiago", "David"])
    col = {"Rata-rata 3 kota": "T_mean", "Tocumen": "T2M_toc", "Santiago": "T2M_san", "David": "T2M_dav"}[city]
    d = h[h["hour"].between(*rng) & h["nat_clean"].notna() & (h["datetime"] < COVID_START)].copy()
    d["bin"] = (d[col] / 0.5).round() * 0.5
    g = d.groupby("bin")["nat_clean"].agg(["mean", "std", "count"]).reset_index()
    g = g[g["count"] >= 20]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["bin"], y=g["mean"] + g["std"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["bin"], y=g["mean"] - g["std"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(56,189,248,.22)", name="± 1 simpangan baku", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["bin"], y=g["mean"], mode="lines+markers", line=dict(color=B600, width=2.4),
                             name="Rata-rata beban", hovertemplate="%{x:.1f} °C<br>%{y:.0f} MWh<extra></extra>"))
    fig.update_layout(xaxis_title="Suhu (°C)", yaxis_title="MWh")
    plot(style_fig(fig, 400, f"Suhu vs beban: {slot.lower()} (data pra-COVID)"))
    corr = d[[col, "nat_clean"]].corr().iloc[0, 1]
    note(f"Korelasi suhu–beban pada pilihan ini: <b>{corr:.2f}</b>. Hubungannya <b>non-linear</b> (landai di suhu rendah, naik tajam "
         "saat panas karena AC), alasan LightGBM unggul dari model linear. Pengaruh suhu murni lebih kecil daripada yang terlihat di sini, "
         "karena suhu ikut naik-turun searah dengan jam (siang panas, siang juga jam sibuk).")

# ------------------------------------------------------------------ Libur & sekolah
with t4:
    daily = h.groupby("date").agg(dem=("nat_clean", "mean"), hol=("holiday", "max"), dow=("dow", "first"),
                                  sch=("school", "max"), hid=("Holiday_ID", "max"))
    ordinary = daily.loc[(daily["hol"] == 0) & (daily["dow"] < 5), "dem"]
    base = ordinary.reindex(daily.index).rolling(29, center=True, min_periods=5).mean()
    daily["rel"] = 100 * (daily["dem"] / base - 1)

    hol = daily[(daily["hol"] == 1) & (daily["dow"] < 5)].dropna(subset=["rel"])
    ca, cb = st.columns([1, 1])
    with ca:
        fig = px.box(hol, y="rel", points="all", color_discrete_sequence=[B600])
        fig.add_hline(y=0, line_dash="dash", line_color="#64748B")
        fig.update_layout(yaxis_title="% terhadap hari kerja biasa (± 2 minggu)", xaxis_title="")
        fig.update_xaxes(showticklabels=False)
        plot(style_fig(fig, 380, "Efek libur nasional pada hari kerja", legend_bottom=False))
    with cb:
        d = h[~h["weekend"] & (h["holiday"] == 0) & h["nat_clean"].notna()].copy()
        d["Status"] = np.where(d["school"] == 1, "Sekolah masuk", "Libur sekolah")
        prof = d.groupby(["Status", "hour"])["nat_clean"].mean().reset_index()
        fig = px.line(prof, x="hour", y="nat_clean", color="Status", markers=True,
                      color_discrete_map={"Sekolah masuk": B700, "Libur sekolah": SKY})
        fig.update_layout(xaxis_title="Jam", yaxis_title="MWh")
        plot(style_fig(fig, 380, "Hari kerja: sekolah masuk vs libur sekolah"))

    n_hol = len(hol)
    st.markdown(f"**Ringkasan:** pada **{n_hol} hari libur nasional yang jatuh di hari kerja**, beban rata-rata berselisih "
                f"**{hol['rel'].mean():+.1f}%** terhadap hari kerja biasa di sekitarnya (median {hol['rel'].median():+.1f}%; negatif = lebih rendah).")
    by_id = hol.groupby("hid")["rel"].agg(["mean", "count"]).reset_index().sort_values("mean")
    by_id["ID"] = "Holiday_ID " + by_id["hid"].astype(int).astype(str)
    fig = px.bar(by_id, x="mean", y="ID", orientation="h", text=by_id["mean"].round(1), color="mean",
                 color_continuous_scale="Blues_r", hover_data={"count": True, "mean": ":.1f"})
    fig.update_layout(xaxis_title="% terhadap hari kerja biasa", yaxis_title="", coloraxis_showscale=False)
    plot(style_fig(fig, max(320, 22 * len(by_id)), "Efek per jenis hari libur (Holiday_ID)", legend_bottom=False))
    note("Libur menurunkan beban (aktivitas industri/perkantoran berhenti), dengan besaran berbeda per jenis libur. "
         "Efek sekolah tampak kecil dan bercampur dengan musim (libur sekolah jatuh pada bulan tertentu), "
         "sehingga sulit dipisahkan tanpa model.")

# ------------------------------------------------------------------ Korelasi
with t5:
    cols = [c for c in h.columns if c.split("_")[0] in ("T2M", "QV2M", "TQL", "W2M")] + ["holiday", "Holiday_ID", "school"]
    cr = h[cols + ["nat_clean"]].corr()["nat_clean"].drop("nat_clean").sort_values()
    fig = go.Figure(go.Bar(x=cr.values, y=cr.index, orientation="h",
                           marker_color=np.where(cr.values >= 0, B600, "#94A3B8"),
                           text=[f"{v:.2f}" for v in cr.values], textposition="outside"))
    fig.update_layout(xaxis_title="Korelasi Pearson dengan nat_demand", yaxis_title="")
    plot(style_fig(fig, 480, "Korelasi tiap variabel dengan beban nasional", legend_bottom=False))
    note("Suhu paling berkorelasi (≈ 0,65), libur berkorelasi negatif. <b>Korelasi bukan ukuran pengaruh di dalam model</b>: "
         "jam dan hari (yang tidak muncul di sini karena bersifat siklik) justru paling dominan. Lihat halaman Variabel Berpengaruh.")
