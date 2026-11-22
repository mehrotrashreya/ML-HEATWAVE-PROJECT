"""Auto-generated from 03_xgboost_training.ipynb - live prediction for the Delhi 3-day Tmax model."""
import numpy as np
import pandas as pd
import requests

from src.features import (make_calendar, add_historical_features, add_seasonal_features, add_trend_features,
                          FEATURE_COLUMNS as ALL_FEATURES, META_COLS, HORIZONS, VARS)
from src.heatwave_rules import classify_heatwave

FORECAST_API_URL = 'https://api.open-meteo.com/v1/forecast'
LAT, LON, TZ = 28.61, 77.21, 'Asia/Kolkata'
API_VARS = {'temperature_2m_max': 'tmax', 'relative_humidity_2m_mean': 'humidity', 'wind_speed_10m_max': 'wind', 'shortwave_radiation_sum': 'solar'}


def fetch_recent_daily(latitude=LAT, longitude=LON, past_days=7, timeout=30):
    # Recent daily weather from the operational Forecast API (same units as training). Fails loudly.
    params = {"latitude": latitude, "longitude": longitude, "daily": ",".join(API_VARS), "timezone": TZ,
              "temperature_unit": "celsius", "wind_speed_unit": "kmh", "past_days": past_days, "forecast_days": 1}
    try:
        resp = requests.get(FORECAST_API_URL, params=params, timeout=timeout)
    except requests.RequestException as err:
        raise RuntimeError(f"Network/API request failed: {err}") from err
    if resp.status_code != 200:
        raise RuntimeError(f"Open-Meteo returned HTTP {resp.status_code}: {resp.text[:300]}")
    daily = resp.json().get("daily")
    if not daily or "time" not in daily:
        raise RuntimeError("API response has no 'daily' block")
    out = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for api_name, name in API_VARS.items():
        if api_name not in daily:
            raise RuntimeError(f"API response is missing {api_name}")
        out[name] = pd.to_numeric(pd.Series(daily[api_name]), errors="coerce")
    return out


def last_three_completed_days(df, today=None):
    # Drop today (incomplete) and keep the 3 latest COMPLETE consecutive days with no missing values.
    if today is None:
        today = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()
    done = df[df["date"] < today].sort_values("date").tail(3).reset_index(drop=True)
    if len(done) < 3 or done[VARS].isna().any().any():
        raise ValueError("Need 3 completed days with all four variables present.")
    if (done["date"].diff().dropna() != pd.Timedelta(days=1)).any():
        raise ValueError("The 3 input days are not consecutive.")
    return done


def build_live_features(last_three):
    # Same code path as training: T = last completed day; one row per horizon; seasonal features of the TARGET date.
    cal = make_calendar(last_three)
    if len(cal) != 3 or cal.isna().any().any():
        raise ValueError("Need exactly 3 consecutive completed days with no missing values.")
    hist = add_trend_features(add_historical_features(cal)).iloc[[-1]]
    origin = hist.index[0]
    rows = []
    for h in HORIZONS:
        r = hist.copy().reset_index(drop=True)
        td = origin + pd.Timedelta(days=h)
        r["forecast_horizon"] = h
        r = pd.concat([r, add_seasonal_features([td])], axis=1)
        r.insert(0, "target_date", td)
        r.insert(0, "forecast_origin", origin)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)[META_COLS + ALL_FEATURES]


def predict_next_three_days(model, last_three, feature_columns=None):
    # One feature row per horizon -> the SAME regressor -> the separate heatwave rule.
    rows = build_live_features(last_three)
    cols = feature_columns or ALL_FEATURES
    pred = model.predict(rows[cols])
    out = rows[META_COLS + ["forecast_horizon"]].copy()
    out["predicted_tmax"] = pred
    codes = [classify_heatwave(float(t)) for t in pred]
    out["category_code"] = [c for c, _ in codes]
    out["category"] = [n for _, n in codes]
    return out

