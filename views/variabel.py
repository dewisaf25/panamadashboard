import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import ASSETS, B200, B600, B800, CYAN, SKY, divider, kpi, load_importance, note, page_header, plot, section, style_fig, table

page_header("🔍 Variabel Berpengaruh", "Seberapa besar pengaruh suhu, libur, sekolah, jam, dan hari terhadap akurasi ramalan LightGBM.")
imp = load_importance()
ASKED = ["Suhu", "Hari libur", "Sekolah", "Jam", "Hari (dow)"]          # variabel yang ditanyakan di tujuan riset

METHODS = {
    "Permutasi, model eksogen-saja": ("ΔMAE permutasi (eksogen-saja)",
        "Model **tanpa lag historis**; satu kelompok variabel diacak lalu diukur kenaikan MAE. Mengukur pengaruh *intrinsik* variabel."),
    "Ablation, model penuh": ("ΔMAE ablation (full)",
        "Model penuh **dilatih ulang tanpa** satu kelompok. Lebih adil karena model bisa menutup kehilangan lewat variabel lain."),
    "Permutasi, model penuh": ("ΔMAE permutasi (full)",
        "Kelompok diacak pada model penuh. Pengaruh variabel lain tampak mengecil karena lag historis sudah membawa informasinya."),
    "Gain LightGBM (%)": ("Gain % (full)",
        "Kontribusi pengurangan error saat pohon bercabang. Cepat, tetapi bias ke variabel dengan banyak nilai unik."),
}
c1, c2 = st.columns([1.2, 2])
with c1:
    method = st.radio("Metode pengukuran", list(METHODS.keys()))
with c2:
    st.markdown("&nbsp;")
    note(METHODS[method][1])
col = METHODS[method][0]
s = imp[col].dropna().sort_values()
unit = "%" if "Gain" in col else "MWh"
colors = [B600 if i in ASKED else B200 for i in s.index]
colors = [CYAN if i == "Historis (lag)" else c for i, c in zip(s.index, colors)]
fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colors,
                       text=[f"{v:.2f}" for v in s.values], textposition="outside",
                       hovertemplate="%{y}<br>%{x:.3f} " + unit + "<extra></extra>"))
fig.update_layout(xaxis_title=("Kontribusi gain (%)" if unit == "%" else "Kenaikan MAE (MWh)"), yaxis_title="")
plot(style_fig(fig, 420, f"{method}", legend_bottom=False))
st.caption("🔵 variabel yang ditanyakan (suhu, libur, sekolah, jam, hari) · 🔷 lag historis · ⚪ variabel lain")

# ------------------------------------------------------------------ konsensus
divider()
section("Peringkat konsensus", "🏅", "tanpa lag historis")
ex = imp.drop(index="Historis (lag)")
ranks = ex.rank(ascending=False, method="min").astype(int)
ranks["Rata-rata peringkat"] = ranks.mean(axis=1)
ranks = ranks.sort_values("Rata-rata peringkat")
fig = px.imshow(ranks.drop(columns="Rata-rata peringkat"), text_auto=True, aspect="auto",
                color_continuous_scale="Blues_r", labels=dict(color="Peringkat"))
fig.update_xaxes(side="top")
plot(style_fig(fig, 400, None, legend_bottom=False))
top = ranks.index[:3].tolist()
st.markdown(f"**Peringkat teratas lintas metode:** {top[0]}, {top[1]}, {top[2]}.")

# ------------------------------------------------------------------ jawaban 5 variabel
divider()
section("Jawaban untuk 5 variabel yang ditanyakan", "✅")
exo = imp["ΔMAE permutasi (eksogen-saja)"]
abl = imp["ΔMAE ablation (full)"]
rows = []
for v in ASKED:
    rows.append({"Variabel": v, "Pengaruh intrinsik (ΔMAE, eksogen-saja)": exo[v],
                 "Kontribusi unik di model penuh (ΔMAE ablation)": abl[v],
                 "Peringkat (eksogen-saja)": int(exo.drop("Historis (lag)").rank(ascending=False)[v])})
tab = pd.DataFrame(rows).sort_values("Peringkat (eksogen-saja)").set_index("Variabel")
table(tab.round(3))

a, b, c = st.columns(3)
with a:
    kpi("Jam & hari (rutinitas)", f"{exo['Jam']:.0f} + {exo['Hari (dow)']:.0f} MWh", "ΔMAE permutasi, eksogen-saja", B600, "🕒")
with b:
    kpi("Libur", f"{exo['Hari libur']:.1f} MWh", f"peringkat ke-{int(exo.drop('Historis (lag)').rank(ascending=False)['Hari libur'])} dari 9 kelompok", CYAN, "🎌")
with c:
    kpi("Suhu · Sekolah", f"{exo['Suhu']:.1f} · {exo['Sekolah']:.2f} MWh", "kecil setelah rutinitas diperhitungkan", SKY, "🌡️")

note(f"<b>Interpretasi:</b> (1) <b>Jam</b> adalah penentu terbesar bentuk kurva harian (siklus siang-malam), disusul <b>hari</b> "
     f"(kerja vs akhir pekan). (2) <b>Libur</b> berdampak menengah: jumlah hari terbatas, tetapi efek per hari besar. (3) <b>Suhu</b> sangat "
     f"berkorelasi dengan beban (≈ 0,65), namun kontribusi <i>uniknya</i> kecil karena suhu berpola searah dengan jam, dan level beban "
     f"harian sudah ditangkap lag historis. (4) <b>Sekolah</b> hampir tidak memberi informasi tambahan; kemungkinan polanya sudah tertangkap oleh hari, bulan, dan libur (saling tumpang tindih). "
     f"(5) Di model penuh, <b>lag historis</b> mendominasi (gain {imp.loc['Historis (lag)','Gain % (full)']:.0f}%): beban kemarin & minggu lalu "
     f"adalah prediktor terbaik.")

shap_p = ASSETS / "shap_summary.png"
if shap_p.exists():
    with st.expander("🧠 Ringkasan SHAP per fitur (dari notebook)", expanded=False):
        st.image(str(shap_p), caption="SHAP summary LightGBM model penuh, data Test (3.000 sampel)")
