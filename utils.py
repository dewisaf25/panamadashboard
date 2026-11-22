"""Utilitas bersama: loader data, tema, komponen UI."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DATA, ASSETS = ROOT / "data", ROOT / "assets"

# ---------------------------------------------------------------- konstanta
SPLITS = {  # sesuai notebook (70 : 10 : 20, kronologis)
    "Train (70%)":    ("2015-01-03", "2018-11-03", "#1E3A8A"),
    "Validasi (10%)": ("2018-11-04", "2019-05-22", "#38BDF8"),
    "Test (20%)":     ("2019-05-23", "2020-06-26", "#2563EB"),
}
TEST_START, TEST_END = pd.Timestamp("2019-05-23"), pd.Timestamp("2020-06-26")
COVID_START = pd.Timestamp("2020-03-15")
ANOMALY_THR = 600

DOW_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

# Palet biru
B900, B800, B700, B600, B500 = "#0B1F4B", "#1E3A8A", "#1D4ED8", "#2563EB", "#3B82F6"
B400, B300, B200, B100 = "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE"
SKY, CYAN, SLATE = "#38BDF8", "#0EA5E9", "#64748B"
PRE_COLOR, COV_COLOR = B700, SKY          # pra-COVID vs saat COVID
BLUE_SEQ = [B600, SKY, B800, B400, CYAN, B300]

BEST = "LightGBM (ML)"
NAIVE = "Naive-168 (minggu lalu)"
RESULT_FILES = {"Test total": "hasil_test.csv", "Pra-COVID": "hasil_test_pra_covid.csv",
                "Saat COVID": "hasil_test_covid.csv"}


# ---------------------------------------------------------------- tampilan
import urllib.parse as _u


def _svg(svg: str) -> str:
    return 'url("data:image/svg+xml;utf8,' + _u.quote(svg, safe="/:=,;") + '")'


_WAVE = _svg("<svg xmlns='http://www.w3.org/2000/svg' width='600' height='90' viewBox='0 0 600 90'><path d='M0 62 C40 62 55 30 100 30 S150 18 200 24 S260 58 300 54 S360 16 410 20 S470 56 520 58 S575 62 600 62' fill='none' stroke='white' stroke-opacity='0.5' stroke-width='2.5' stroke-linecap='round'/></svg>")
_BOLT = _svg("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M13 2 L4 14 h7 l-2 8 l11 -13 h-7 z' fill='white' fill-opacity='0.20' stroke='white' stroke-opacity='0.55' stroke-width='0.5' stroke-linejoin='round'/></svg>")
_BOLT_BLUE = _svg("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M13 2 L4 14 h7 l-2 8 l11 -13 h-7 z' fill='%232563EB' fill-opacity='0.10'/></svg>")
_BOLT_SIDE = _svg("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M13 2 L4 14 h7 l-2 8 l11 -13 h-7 z' fill='white' fill-opacity='0.06'/></svg>")
_TOWER = _svg("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 160' fill='none' stroke='white' stroke-opacity='0.55' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M32 158 L46 22 M68 158 L54 22 M50 6 L50 22'/><path d='M6 42 H94 M16 66 H84 M38 110 H62 M36 130 H64'/><path d='M46 22 L66 66 M54 22 L34 66 M34 66 L62 110 M66 66 L38 110 M38 110 L64 130 M62 110 L36 130 M36 130 L68 158 M64 130 L32 158'/><path d='M6 42 V52 M94 42 V52 M16 66 V76 M84 66 V76'/></svg>")

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
@keyframes waveMove { to { background-position: -600px 100%, -420px 100%; } }
@keyframes floaty { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-12px); } }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(56,189,248,.7); } 70% { box-shadow: 0 0 0 9px rgba(56,189,248,0); } 100% { box-shadow: 0 0 0 0 rgba(56,189,248,0); } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes shimmer { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
[data-testid="stAppViewContainer"] { background: radial-gradient(1100px 480px at 88% -8%, #DBEAFE 0%, rgba(219,234,254,0) 62%), radial-gradient(900px 420px at -8% 108%, #E0F2FE 0%, rgba(224,242,254,0) 58%), #F5F9FF; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: __BOLT_SIDE__ no-repeat 112% 96% / 230px, linear-gradient(185deg, #0B1F4B 0%, #1E3A8A 62%, #1D4ED8 130%); border-right: 1px solid rgba(147,197,253,.25); }
[data-testid="stSidebar"] * { color: #DBEAFE !important; }
[data-testid="stSidebarNav"] a { border-radius: 10px; margin: 2px 0; transition: background .2s; }
[data-testid="stSidebarNav"] a:hover { background: rgba(147,197,253,.16); }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: linear-gradient(90deg, rgba(56,189,248,.35), rgba(56,189,248,.08)); border-left: 3px solid #38BDF8; }
.hero { position: relative; overflow: hidden; background: linear-gradient(118deg, #0B1F4B 0%, #1D4ED8 56%, #38BDF8 135%); border-radius: 24px; padding: 34px 40px 84px; color: #fff; margin-bottom: 26px; box-shadow: 0 16px 44px rgba(29,78,216,.30); }
.hero::before { content: ""; position: absolute; inset: 0; background-image: radial-gradient(rgba(255,255,255,.16) 1.2px, transparent 1.5px); background-size: 22px 22px; opacity: .6; -webkit-mask-image: linear-gradient(90deg, transparent 5%, #000 100%); mask-image: linear-gradient(90deg, transparent 5%, #000 100%); }
.hero .wave { position: absolute; left: 0; right: 0; bottom: 0; height: 90px; background-image: __WAVE__, __WAVE__; background-repeat: repeat-x; background-size: 600px 90px, 420px 63px; background-position: 0 100%, 0 100%; animation: waveMove 16s linear infinite; opacity: .95; }
.hero .bolt { position: absolute; right: 40px; top: 16px; width: 190px; height: 190px; background: __BOLT__ no-repeat center / contain; animation: floaty 5s ease-in-out infinite; filter: drop-shadow(0 0 18px rgba(125,211,252,.65)); }
.hero .tower { position: absolute; right: 250px; bottom: 0; width: 104px; height: 160px; background: __TOWER__ no-repeat center bottom / contain; opacity: .85; }
.hero .orb { position: absolute; border-radius: 50%; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,.55), rgba(255,255,255,.04)); animation: floaty 7s ease-in-out infinite; }
.hero-in { position: relative; z-index: 2; }
.hero h1 { color: #fff; font-size: 2.25rem; font-weight: 800; margin: 10px 0 8px 0; letter-spacing: -0.6px; line-height: 1.15; }
.hero p { color: #DBEAFE; font-size: 1.02rem; margin: 0; max-width: 760px; line-height: 1.6; }
.hero .tag { display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25); backdrop-filter: blur(4px); padding: 4px 13px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin: 16px 8px 0 0; color: #fff; }
.live { display: inline-flex; align-items: center; gap: 8px; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); padding: 4px 12px; border-radius: 999px; font-size: .75rem; font-weight: 700; letter-spacing: .6px; text-transform: uppercase; color: #fff; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: #38BDF8; display: inline-block; animation: pulse 1.8s infinite; }
.phead { position: relative; margin-bottom: 20px; padding: 20px 26px; background: linear-gradient(120deg, #fff 0%, #EAF3FF 100%); border: 1px solid #DBEAFE; border-radius: 20px; box-shadow: 0 4px 16px rgba(30,64,175,.07); overflow: hidden; }
.phead::after { content: ""; position: absolute; right: 18px; top: 50%; width: 120px; height: 120px; transform: translateY(-50%) rotate(8deg); background: __BOLT_BLUE__ no-repeat center / contain; }
.phead::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 6px; background: linear-gradient(180deg, #2563EB, #38BDF8); }
.phead .crumb { font-size: .72rem; font-weight: 700; letter-spacing: .8px; text-transform: uppercase; color: #2563EB; }
.phead h2 { font-weight: 800; font-size: 1.65rem; color: #0B1F4B; margin: 2px 0 0 0; letter-spacing: -0.4px; }
.phead p { color: #475569; margin: 4px 0 0 0; font-size: .97rem; max-width: 760px; }
.sec { display: flex; align-items: center; gap: 12px; margin: 30px 0 14px 0; }
.sec .bar { width: 6px; height: 30px; border-radius: 6px; background: linear-gradient(180deg, #2563EB, #38BDF8); }
.sec .ico { width: 34px; height: 34px; border-radius: 11px; display: flex; align-items: center; justify-content: center; background: #DBEAFE; font-size: 1.05rem; }
.sec h3 { margin: 0; padding: 0; font-size: 1.28rem; font-weight: 800; color: #0B1F4B; letter-spacing: -0.3px; }
.sec .sub { margin-left: auto; color: #64748B; font-size: .86rem; }
.bolt-div { display: flex; align-items: center; gap: 14px; margin: 26px 0; color: #2563EB; }
.bolt-div::before, .bolt-div::after { content: ""; flex: 1; height: 2px; border-radius: 2px; background: linear-gradient(90deg, rgba(37,99,235,0), #93C5FD, rgba(37,99,235,0)); }
.kpi { position: relative; background: #fff; border: 1px solid #DBEAFE; border-radius: 18px; padding: 18px 18px 15px 18px; overflow: hidden; height: 100%; box-shadow: 0 4px 14px rgba(30,64,175,.08); transition: transform .2s ease, box-shadow .2s ease; animation: fadeUp .55s ease both; }
.kpi:hover { transform: translateY(-4px); box-shadow: 0 12px 28px rgba(30,64,175,.18); }
.kpi::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 5px; background: linear-gradient(90deg, var(--accent, #2563EB), #38BDF8); }
.kpi .ic { position: absolute; right: 14px; top: 16px; width: 40px; height: 40px; border-radius: 13px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; background: #EFF6FF; }
.kpi-l { color: #64748B; font-size: .76rem; font-weight: 700; text-transform: uppercase; letter-spacing: .7px; padding-right: 48px; }
.kpi-v { color: #0B1F4B; font-size: 1.7rem; font-weight: 800; line-height: 1.25; margin-top: 4px; letter-spacing: -0.5px; }
.kpi-s { color: #64748B; font-size: .82rem; margin-top: 3px; }
.card { position: relative; background: #fff; border: 1px solid #DBEAFE; border-radius: 18px; padding: 20px 22px; height: 100%; box-shadow: 0 4px 14px rgba(30,64,175,.07); color: #0B1F4B; transition: transform .2s ease, box-shadow .2s ease; animation: fadeUp .6s ease both; }
.card:hover { transform: translateY(-3px); box-shadow: 0 12px 26px rgba(30,64,175,.15); }
.card h4 { margin: 0 0 8px 0; font-size: 1.02rem; font-weight: 700; color: #0B1F4B; }
.card p, .card li { color: #334155; font-size: .93rem; line-height: 1.6; margin: 0; }
.chip { display: inline-block; padding: 2px 11px; border-radius: 999px; font-size: .75rem; font-weight: 600; color: #fff; }
.note { background: linear-gradient(100deg, #EFF6FF, #F8FBFF); border: 1px solid #BFDBFE; border-left: 5px solid #2563EB; border-radius: 14px; padding: 13px 18px; color: #1E3A8A; font-size: .92rem; line-height: 1.55; }
.warn { background: linear-gradient(100deg, #E0F2FE, #F0F9FF); border: 1px solid #7DD3FC; border-left: 5px solid #0EA5E9; border-radius: 14px; padding: 13px 18px; color: #075985; font-size: .92rem; line-height: 1.55; }
h3 { color: #0B1F4B; font-weight: 700; }
div[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #DBEAFE; border-radius: 18px; padding: 8px; box-shadow: 0 4px 14px rgba(30,64,175,.07); overflow: hidden; }
[data-testid="stDataFrame"] { border: 1px solid #DBEAFE; border-radius: 14px; overflow: hidden; box-shadow: 0 3px 10px rgba(30,64,175,.05); }
[data-testid="stExpander"] { background: #fff; border: 1px solid #DBEAFE; border-radius: 14px; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #DBEAFE; border-left: 5px solid #2563EB; border-radius: 14px; padding: 12px 16px; box-shadow: 0 3px 10px rgba(30,64,175,.06); }
[data-testid="stMetricLabel"] { color: #64748B; font-weight: 600; }
[data-testid="stMetricValue"] { color: #0B1F4B; font-weight: 800; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: #E3EEFF; padding: 6px; border-radius: 14px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 16px; font-weight: 600; color: #1E3A8A; background: transparent; }
.stTabs [aria-selected="true"] { background: #fff; color: #1D4ED8; box-shadow: 0 2px 8px rgba(30,64,175,.16); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stDownloadButton button { background: linear-gradient(120deg, #1D4ED8, #38BDF8); background-size: 200% 100%; color: #fff; border: 0; border-radius: 12px; font-weight: 700; padding: 8px 20px; box-shadow: 0 6px 16px rgba(29,78,216,.30); transition: transform .15s ease; }
.stDownloadButton button:hover { transform: translateY(-2px); animation: shimmer 1.6s linear infinite; color: #fff; }
.stDownloadButton button p { color: #fff; }
.brand-chips span { display: inline-block; background: rgba(147,197,253,.18); border: 1px solid rgba(147,197,253,.35); border-radius: 999px; padding: 2px 10px; font-size: .72rem; font-weight: 600; margin: 0 6px 6px 0; color: #DBEAFE; }
.side-status { display: flex; align-items: center; gap: 8px; font-size: .8rem; color: #BFDBFE; margin: 6px 0 10px; }
@media (max-width: 900px) { .hero .tower, .hero .bolt, .phead::after { display: none; } .hero { padding: 26px 22px 78px; } .hero h1 { font-size: 1.7rem; } }
@media (prefers-reduced-motion: reduce) { .hero .wave, .hero .bolt, .hero .orb, .dot, .kpi, .card { animation: none !important; } }
#MainMenu, footer { visibility: hidden; }
</style>
"""
_CSS = (_CSS.replace("__WAVE__", _WAVE).replace("__BOLT_SIDE__", _BOLT_SIDE).replace("__BOLT_BLUE__", _BOLT_BLUE)
        .replace("__BOLT__", _BOLT).replace("__TOWER__", _TOWER))


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title, subtitle, tags=()):
    chips = "".join(f'<span class="tag">{t}</span>' for t in tags)
    orbs = ('<div class="orb" style="width:48px;height:48px;right:262px;top:16px"></div>'
            '<div class="orb" style="width:22px;height:22px;right:205px;top:178px;animation-delay:1.2s"></div>'
            '<div class="orb" style="width:34px;height:34px;right:22px;top:150px;animation-delay:2s"></div>')
    st.markdown(f'<div class="hero">{orbs}<div class="tower"></div><div class="bolt"></div><div class="wave"></div>'
                f'<div class="hero-in"><span class="live"><span class="dot"></span>Dashboard aktif</span>'
                f'<h1>{title}</h1><p>{subtitle}</p>{chips}</div></div>', unsafe_allow_html=True)


def page_header(title, subtitle=""):
    st.markdown(f'<div class="phead"><div class="crumb">⚡ Panama Load Forecast</div><h2>{title}</h2><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def section(title, icon="⚡", sub=""):
    s = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="sec"><div class="bar"></div><div class="ico">{icon}</div><h3>{title}</h3>{s}</div>',
                unsafe_allow_html=True)


def divider(icon="⚡"):
    st.markdown(f'<div class="bolt-div">{icon}</div>', unsafe_allow_html=True)


def kpi(label, value, sub="", color="#2563EB", icon=""):
    ic = f'<div class="ic">{icon}</div>' if icon else ""
    st.markdown(f'<div class="kpi" style="--accent:{color}">{ic}<div class="kpi-l">{label}</div>'
                f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>', unsafe_allow_html=True)


def card(title, body, color="#2563EB"):
    st.markdown(f'<div class="card" style="border-top:5px solid {color}"><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def note(text, kind="note"):
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


def chip(text, color):
    return f'<span class="chip" style="background:{color}">{text}</span>'


def sidebar_footer():
    st.markdown('<div class="brand-chips"><span>LightGBM</span><span>Day-ahead 24 jam</span><span>Panama 2015–2020</span></div>'
                '<div class="side-status"><span class="dot"></span>Model terbaik aktif</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- plotly helpers
def style_fig(fig, height=420, title=None, legend_bottom=True):
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="white", plot_bgcolor="white", colorway=BLUE_SEQ,
        margin=dict(l=10, r=10, t=56 if title else 24, b=10),
        font=dict(family="Inter, sans-serif", size=13, color="#0B1F4B"),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16, color="#0B1F4B")))
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0))
    fig.update_xaxes(showgrid=True, gridcolor="#EAF2FF", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EAF2FF", zeroline=False)
    return fig


def plot(fig, **kw):
    """Kompatibel lintas versi Streamlit (width='stretch' vs use_container_width)."""
    try:
        st.plotly_chart(fig, width="stretch", theme=None, **kw)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, **kw)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def shade(fig, x0, x1, color, label=None, opacity=0.10):
    """Blok warna vertikal (hindari bug add_vrect+anotasi pada sumbu tanggal)."""
    fig.add_shape(type="rect", xref="x", yref="paper", x0=str(x0), x1=str(x1), y0=0, y1=1,
                  fillcolor=color, opacity=opacity, line_width=0, layer="below")
    if label:
        fig.add_annotation(x=str(x0), xref="x", y=1, yref="paper", text=label, showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color=color))


def vline(fig, x, color="#0B1F4B", label=None):
    fig.add_shape(type="line", xref="x", yref="paper", x0=str(x), x1=str(x), y0=0, y1=1,
                  line=dict(color=color, width=1.6, dash="dash"))
    if label:
        fig.add_annotation(x=str(x), xref="x", y=1, yref="paper", text=label, showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color=color))


# ---------------------------------------------------------------- data
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAPE%": float(np.mean(np.abs(e / y)) * 100),
            "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}


@st.cache_data(show_spinner=False)
def load_hourly():
    df = pd.read_csv(DATA / "hourly_data.csv.gz", parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.normalize()
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["T_mean"] = df[["T2M_toc", "T2M_san", "T2M_dav"]].mean(axis=1)
    df["weekend"] = df["dow"] >= 5
    df["nat_clean"] = df["nat_demand"].where(df["nat_demand"] >= ANOMALY_THR)   # anomali -> NaN
    return df


@st.cache_data(show_spinner=False)
def load_results():
    """Hasil resmi dari notebook (LightGBM + patokan Naive-168) untuk 3 periode."""
    return {k: pd.read_csv(DATA / f, index_col=0) for k, f in RESULT_FILES.items()}


@st.cache_data(show_spinner=False)
def load_importance():
    return pd.read_csv(DATA / "importance_variabel.csv", index_col=0)


def _naive_frame():
    h = load_hourly()
    y = h["nat_demand"].where(h["nat_demand"] >= ANOMALY_THR).interpolate(limit_direction="both")
    d = pd.DataFrame({"datetime": h["datetime"], "actual": h["nat_demand"], NAIVE: y.shift(168).values})
    return d[(d["datetime"] >= TEST_START) & (d["datetime"] < TEST_END + pd.Timedelta(days=1))].reset_index(drop=True)


@st.cache_data(show_spinner="Menyiapkan model LightGBM (hanya sekali, ± 30–60 detik)...")
def load_predictions():
    """Return (df, sumber). df: datetime, actual, LightGBM (ML), Naive-168.
    sumber: 'file' (hasil ekspor notebook) | 'trained' (dilatih di aplikasi) | 'baseline:<pesan>' (gagal)."""
    p = DATA / "dashboard_predictions.csv.gz"
    if p.exists():
        d = pd.read_csv(p, parse_dates=["datetime"])
        if BEST in d.columns:
            if NAIVE not in d.columns:
                d = d.merge(_naive_frame()[["datetime", NAIVE]], on="datetime", how="left")
            return d[["datetime", "actual", BEST, NAIVE]], "file"
    try:
        import forecasting
        return forecasting.train_and_predict(DATA / "hourly_data.csv.gz"), "trained"
    except Exception as e:  # noqa: BLE001
        return _naive_frame(), f"baseline:{type(e).__name__}: {str(e)[:150]}"
