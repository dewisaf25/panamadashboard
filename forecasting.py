"""Replikasi pipeline notebook untuk model terbaik (LightGBM day-ahead). Dilatih sekali saat dashboard dibuka."""
import numpy as np
import pandas as pd

TRAIN_END, VAL_START, VAL_END = "2018-11-03", "2018-11-04", "2019-05-22"
TEST_START, TEST_END = "2019-05-23", "2020-06-26"
SEED, OUTAGE_THR = 42, 600

LAGCOLS = ["lag_24", "lag_48", "lag_72", "lag_168", "lag_336", "same_hour_mean7", "same_hour_std7", "y_origin", "mean24_origin"]
GROUPS = {
    "Suhu":             ["T2M_toc", "T2M_san", "T2M_dav", "T_mean", "T_roll24", "T_max24"],
    "Kelembapan":       ["QV2M_toc", "QV2M_san", "QV2M_dav"],
    "Angin":            ["W2M_toc", "W2M_san", "W2M_dav"],
    "Hujan/Awan (TQL)": ["TQL_toc", "TQL_san", "TQL_dav"],
    "Hari libur":       ["holiday", "Holiday_ID", "hol_prev", "hol_next"],
    "Sekolah":          ["school"],
    "Jam":              ["hour", "hour_sin", "hour_cos"],
    "Hari (dow)":       ["dow", "is_weekend", "dow_sin", "dow_cos"],
    "Bulan/Musim":      ["month", "month_sin", "month_cos"],
    "Historis (lag)":   LAGCOLS,
}
FEATS = [c for g in GROUPS.values() for c in g]
LGB_PARAMS = dict(objective="regression", n_estimators=4000, learning_rate=0.03, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)


def _shift(a, k):
    out = np.full(len(a), np.nan)
    out[k:] = a[:-k]
    return out


def build_dataset(path):
    raw = pd.read_csv(path, parse_dates=["datetime"])
    df = raw.set_index("datetime").sort_index().asfreq("h")
    idx, n = df.index, len(df)
    pos, hour = np.arange(n), df.index.hour.values
    y_raw = df["nat_demand"]
    yv = y_raw.where(y_raw >= OUTAGE_THR).interpolate(limit_direction="both").values.astype(float)

    HF = pd.DataFrame(index=idx)
    for c in df.columns:
        if c.split("_")[0] in ("T2M", "QV2M", "TQL", "W2M"):
            HF[c] = df[c]
    HF["T_mean"] = df[[c for c in df.columns if c.startswith("T2M_")]].mean(axis=1)
    HF["T_roll24"] = HF["T_mean"].rolling(24, min_periods=1).mean()
    HF["T_max24"] = HF["T_mean"].rolling(24, min_periods=1).max()
    HF["hour"], HF["dow"], HF["month"] = hour, idx.dayofweek, idx.month
    HF["is_weekend"] = (HF["dow"] >= 5).astype(int)
    for name, per in (("hour", 24), ("dow", 7), ("month", 12)):
        HF[f"{name}_sin"] = np.sin(2 * np.pi * HF[name] / per)
        HF[f"{name}_cos"] = np.cos(2 * np.pi * HF[name] / per)
    date = pd.Series(idx.normalize(), index=idx)
    dh = df["holiday"].groupby(date.values).max()
    HF["holiday"], HF["Holiday_ID"] = df["holiday"], df["Holiday_ID"]
    HF["hol_prev"] = date.map(dh.shift(1).fillna(0)).values
    HF["hol_next"] = date.map(dh.shift(-1).fillna(0)).values
    HF["school"] = df["school"]

    D = HF.copy()
    for k in (24, 48, 72, 168, 336):
        D[f"lag_{k}"] = _shift(yv, k)
    S = np.vstack([_shift(yv, 24 * j) for j in range(1, 8)])
    D["same_hour_mean7"], D["same_hour_std7"] = S.mean(0), S.std(0)
    o = pos - (hour + 1)                                   # origin = 23:00 hari sebelumnya
    oc = np.clip(o, 0, None)
    r24 = pd.Series(yv).rolling(24).mean().values
    r168 = pd.Series(yv).rolling(168).mean().values
    D["y_origin"] = np.where(o >= 0, yv[oc], np.nan)
    D["mean24_origin"] = np.where(o >= 0, r24[oc], np.nan)
    D["lvl"] = np.where(o >= 0, r168[oc], np.nan)
    D["time"], D["date"] = idx, idx.normalize()
    D["y"], D["y_clean"] = y_raw.values, yv
    return D.dropna(subset=LAGCOLS + ["lvl"]).reset_index(drop=True)


def _xmat(D):
    X = D[FEATS].copy()
    for c in LAGCOLS:
        X[c] = X[c] / D["lvl"]
    return X


def train_and_predict(path):
    import lightgbm as lgb
    D = build_dataset(path)
    X, lvl = _xmat(D), D["lvl"].values
    tgt = (D["y_clean"] / D["lvl"]).values
    dm = lambda a, b: ((D["date"] >= pd.Timestamp(a)) & (D["date"] <= pd.Timestamp(b))).values
    m_tr, m_va, m_te = dm("2000-01-01", TRAIN_END), dm(VAL_START, VAL_END), dm(TEST_START, TEST_END)

    mdl = lgb.LGBMRegressor(**LGB_PARAMS)
    mdl.fit(X[m_tr], tgt[m_tr], eval_set=[(X[m_va], tgt[m_va])], eval_metric="l1",
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    out = D.loc[m_te, ["time", "y"]].rename(columns={"time": "datetime", "y": "actual"}).reset_index(drop=True)
    out["LightGBM (ML)"] = mdl.predict(X[m_te]) * lvl[m_te]
    out["Naive-168 (minggu lalu)"] = D.loc[m_te, "lag_168"].values
    return out
