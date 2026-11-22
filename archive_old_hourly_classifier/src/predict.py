"""Prediction pipeline: recent weather -> features -> XGBoost -> heat-stress risk 24 h ahead.

Flow (same features as training, see features.py):
  Open-Meteo recent hourly data -> risk_index -> make_features -> model.predict_proba
The risk class predicted is for (last observed hour + 24 h).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))
from features import make_features  # noqa: E402

CLASSES = ["Low", "Moderate", "High", "Very High"]
COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#c0392b"}
ADVICE = {
    0: "Normal conditions. Usual precautions are enough.",
    1: "Stay hydrated and limit long, strenuous work in direct sun.",
    2: "Avoid strenuous outdoor activity in the afternoon; take frequent shade and water breaks.",
    3: "Avoid outdoor exposure during peak hours; watch for dizziness, heavy sweating or confusion.",
}

# Coarse weather-grid sample points across Delhi (NOT neighbourhood-level measurements).
LOCATIONS = {
    "Central Delhi": (28.61, 77.21),
    "North Delhi (Rohini)": (28.74, 77.12),
    "North-West (Narela)": (28.85, 77.09),
    "North-East (Shahdara)": (28.68, 77.29),
    "East Delhi (Mayur Vihar)": (28.61, 77.30),
    "South Delhi (Saket)": (28.52, 77.22),
    "South-West (Dwarka)": (28.59, 77.05),
    "West Delhi (Janakpuri)": (28.62, 77.08),
    "Airport (IGI)": (28.56, 77.10),
}

API_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation"
RENAME = {"temperature_2m": "temp_c", "relative_humidity_2m": "rh_pct",
          "wind_speed_10m": "wind_ms", "shortwave_radiation": "solar_wm2"}
COLS = ["temp_c", "rh_pct", "wind_ms", "solar_wm2"]


def load_artifacts():
    with open(os.path.join(ROOT, "data", "processed", "risk_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(ROOT, "models", "model_meta.json")) as f:
        meta = json.load(f)
    from xgboost import XGBClassifier
    json_path = os.path.join(ROOT, "models", "model.json")
    if os.path.exists(json_path):          # preferred: version-safe format
        model = XGBClassifier()
        model.load_model(json_path)
    else:
        import joblib
        model = joblib.load(os.path.join(ROOT, "models", "model.joblib"))
    return cfg, meta, model


def heat_index_at(temp_c, rh_pct, wind_ms, solar_wm2, a):
    """Same apparent-temperature index used for labels in training."""
    e = (rh_pct / 100) * 6.105 * np.exp(17.27 * temp_c / (237.7 + temp_c))
    q = a * solar_wm2
    return temp_c + 0.348 * e - 0.70 * wind_ms + 0.70 * q / (wind_ms + 10) - 4.25


def add_risk(df, cfg):
    df = df.copy()
    df["risk_index"] = heat_index_at(df.temp_c, df.rh_pct, df.wind_ms, df.solar_wm2, cfg["a"])
    df["risk_class"] = pd.cut(df["risk_index"], [-np.inf, *cfg["thresholds"], np.inf],
                              labels=[0, 1, 2, 3]).astype(int)
    return df


def fetch_recent(lat, lon, past_days=4):
    """Recent hourly weather (up to the current hour, IST) from Open-Meteo."""
    params = dict(latitude=lat, longitude=lon, hourly=HOURLY, wind_speed_unit="ms",
                  timezone="Asia/Kolkata", past_days=past_days, forecast_days=1)
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"]).rename(columns=RENAME)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).floor("h")
    df = df[df.index <= now][COLS]                  # never use future hours as input
    df = df.interpolate(limit=3).dropna()
    df["rh_pct"] = df["rh_pct"].clip(0, 100)
    df["wind_ms"] = df["wind_ms"].clip(lower=0)
    df["solar_wm2"] = df["solar_wm2"].clip(lower=0)
    return df


def predict_from_df(df, cfg, meta, model, hist_hours=72):
    """df: hourly, time-indexed, columns COLS. Returns risk 24 h after the last row."""
    df = add_risk(df, cfg)
    X = make_features(df, with_target=False)
    if X.empty:
        raise ValueError("Not enough recent hourly data (need at least ~25 hours).")
    x = X.iloc[[-1]][meta["features"]]
    t = x.index[0]
    proba = model.predict_proba(x)[0]
    # 24-h timeline: forecast issued at hour s covers hour s+24, so last 24 rows -> next 24 hours
    Xl = X.iloc[-24:][meta["features"]]
    pl = model.predict_proba(Xl)
    tl = pd.DataFrame({"target_time": Xl.index + pd.Timedelta(hours=24),
                       "pred_class": pl.argmax(1), "p_high_plus": pl[:, 2:].sum(1)})
    return {
        "time": t,
        "target_time": t + pd.Timedelta(hours=24),
        "now": {c: float(df.loc[t, c]) for c in COLS + ["risk_index"]},
        "class_now": int(df.loc[t, "risk_class"]),
        "pred_class": int(np.argmax(proba)),
        "proba": [float(p) for p in proba],
        "timeline": tl,
        "history": df.iloc[-hist_hours:][COLS + ["risk_index"]],
    }


def live_prediction(lat, lon, cfg, meta, model):
    return predict_from_df(fetch_recent(lat, lon), cfg, meta, model)
