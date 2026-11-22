"""Auto-generated from 02_model_training.ipynb - feature engineering for the Delhi Tmax regression."""
import numpy as np
import pandas as pd

VARS = ['tmax', 'humidity', 'wind', 'solar']
LAGS = [0, 1, 2]
HORIZONS = [1, 2, 3]
TARGET = 'target_tmax'
META_COLS = ['forecast_origin', 'target_date']
HIST_FEATURES = ['tmax_d0', 'tmax_d1', 'tmax_d2', 'humidity_d0', 'humidity_d1', 'humidity_d2', 'wind_d0', 'wind_d1', 'wind_d2', 'solar_d0', 'solar_d1', 'solar_d2']
SEASONAL_FEATURES = ['month_sin', 'month_cos', 'day_sin', 'day_cos']
TREND_FEATURES = ['tmax_mean_3d', 'tmax_trend_3d', 'humidity_mean_3d', 'solar_sum_3d']
HORIZON_FEATURE = ['forecast_horizon']
FEATURE_COLUMNS = ['tmax_d0', 'tmax_d1', 'tmax_d2', 'humidity_d0', 'humidity_d1', 'humidity_d2', 'wind_d0', 'wind_d1', 'wind_d2', 'solar_d0', 'solar_d1', 'solar_d2', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'tmax_mean_3d', 'tmax_trend_3d', 'humidity_mean_3d', 'solar_sum_3d', 'forecast_horizon']


def make_calendar(df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").set_index("date")
    full = pd.date_range(out.index.min(), out.index.max(), freq="D")
    return out[VARS].reindex(full).rename_axis("date")


def add_historical_features(daily):
    # shift(k) moves data FORWARD in time: row T receives the value observed on T-k (past only).
    feats = pd.DataFrame(index=daily.index)
    for var in VARS:
        for k in LAGS:
            feats[f"{var}_d{k}"] = daily[var].shift(k)
    feats.index.name = "forecast_origin"
    return feats


def add_seasonal_features(target_dates):
    # Cyclic encoding of the TARGET date (the day being predicted).
    #   month_sin/cos = sin/cos(2*pi*(month-1)/12)
    #   day_sin/cos   = sin/cos(2*pi*(day_of_year-1)/days_in_year), days_in_year = 365 or 366 (leap-aware),
    #                   so 1 Jan is always angle 0 and the year always closes the circle.
    td = pd.DatetimeIndex(target_dates)
    month_angle = 2 * np.pi * (td.month.values - 1) / 12
    day_angle = 2 * np.pi * (td.dayofyear.values - 1) / np.where(td.is_leap_year, 366, 365)
    return pd.DataFrame({"month_sin": np.sin(month_angle), "month_cos": np.cos(month_angle),
                         "day_sin": np.sin(day_angle), "day_cos": np.cos(day_angle)})


def add_trend_features(feats):
    # Only the 3-day INPUT window is used -> strictly backward-looking.
    out = feats.copy()
    out["tmax_mean_3d"] = out[["tmax_d0", "tmax_d1", "tmax_d2"]].mean(axis=1, skipna=False)
    out["tmax_trend_3d"] = out["tmax_d0"] - out["tmax_d2"]
    out["humidity_mean_3d"] = out[["humidity_d0", "humidity_d1", "humidity_d2"]].mean(axis=1, skipna=False)
    out["solar_sum_3d"] = out[["solar_d0", "solar_d1", "solar_d2"]].sum(axis=1, skipna=False)
    return out


def build_supervised(daily):
    # One origin -> up to 3 rows (horizon 1, 2, 3): same weather inputs, different horizon / target date / target.
    hist = add_trend_features(add_historical_features(daily)).iloc[max(LAGS):]   # first 2 days have no 3-day window
    parts = []
    for h in HORIZONS:
        part = hist.copy()
        part["forecast_origin"] = part.index
        part["target_date"] = part["forecast_origin"] + pd.Timedelta(days=h)
        part["forecast_horizon"] = h
        part[TARGET] = daily["tmax"].reindex(part["target_date"]).values       # actual Tmax on T+h (NaN if absent)
        seas = add_seasonal_features(part["target_date"]); seas.index = part.index
        parts.append(pd.concat([part, seas], axis=1).reset_index(drop=True))
    sup = pd.concat(parts, ignore_index=True)
    sup["has_missing_predictor"] = sup[FEATURE_COLUMNS].isna().any(axis=1).astype(int)
    sup = sup[sup[TARGET].notna()].copy()                                      # never fabricate targets
    sup = sup.sort_values(["target_date", "forecast_horizon"]).reset_index(drop=True)
    return sup[META_COLS + FEATURE_COLUMNS + [TARGET, "has_missing_predictor"]]

