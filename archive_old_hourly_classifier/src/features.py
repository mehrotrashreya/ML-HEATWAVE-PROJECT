import numpy as np, pandas as pd

HORIZON = 24
BASE = ["temp_c", "rh_pct", "wind_ms", "solar_wm2", "risk_index"]

def make_features(df, horizon=HORIZON, with_target=True):
    """df: hourly, time-indexed, continuous. Har feature sirf time t tak ka data use karta hai."""
    X = df[BASE].copy()
    for lag in [1, 3, 6, 24]:
        for c in ["temp_c", "rh_pct", "risk_index"]:
            X[f"{c}_lag{lag}"] = df[c].shift(lag)
    X["temp_roll6_mean"]  = df.temp_c.rolling(6).mean()
    X["temp_roll24_mean"] = df.temp_c.rolling(24).mean()
    X["temp_roll24_max"]  = df.temp_c.rolling(24).max()
    X["rh_roll24_mean"]   = df.rh_pct.rolling(24).mean()
    X["risk_roll24_max"]  = df.risk_index.rolling(24).max()
    X["temp_trend3"] = df.temp_c - df.temp_c.shift(3)
    X["rh_trend3"]   = df.rh_pct - df.rh_pct.shift(3)
    hr, mo, doy = df.index.hour, df.index.month, df.index.dayofyear
    X["hour_sin"]  = np.sin(2*np.pi*hr/24);   X["hour_cos"]  = np.cos(2*np.pi*hr/24)
    X["month_sin"] = np.sin(2*np.pi*mo/12);   X["month_cos"] = np.cos(2*np.pi*mo/12)
    X["doy_sin"]   = np.sin(2*np.pi*doy/365); X["doy_cos"]   = np.cos(2*np.pi*doy/365)
    if not with_target:
        return X.dropna()
    y = df["risk_class"].shift(-horizon)          # t+24 ka risk
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask].astype(int)
