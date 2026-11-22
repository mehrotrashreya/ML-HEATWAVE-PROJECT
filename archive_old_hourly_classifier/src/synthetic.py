import numpy as np, pandas as pd

COLS = ["temp_c", "rh_pct", "wind_ms", "solar_wm2"]

def heat_index_at(temp_c, rh_pct, wind_ms, solar_wm2, a=0.2):
    e = (rh_pct/100) * 6.105 * np.exp(17.27*temp_c/(237.7+temp_c))
    q = a * solar_wm2
    return temp_c + 0.348*e - 0.70*wind_ms + 0.70*q/(wind_ms+10) - 4.25

def add_risk(df, cfg):
    df = df.copy()
    df["risk_index"] = heat_index_at(df.temp_c, df.rh_pct, df.wind_ms, df.solar_wm2, cfg["a"])
    df["risk_class"] = pd.cut(df["risk_index"], [-np.inf, *cfg["thresholds"], np.inf],
                              labels=[0, 1, 2, 3]).astype(int)
    return df

def block_bootstrap(src, n_years, cfg, block_days=5, window=7, noise=True, seed=0):
    rng = np.random.default_rng(seed)
    years = sorted(set(src.index.year))
    by_year = {y: src[src.index.year == y][COLS].reset_index(drop=True) for y in years}
    blocks = []
    for _ in range(n_years):
        d = 0
        while d < 365:
            L = min(block_days, 365 - d)
            data = by_year[years[rng.integers(len(years))]]
            ndays = len(data) // 24
            s = int(np.clip(d + rng.integers(-window, window + 1), 0, ndays - L))
            blk = data.iloc[s*24:(s+L)*24].copy()
            if noise:                      # block ke andar constant shift, taaki continuity bani rahe
                blk["temp_c"]    += rng.normal(0, 0.7)
                blk["rh_pct"]    += rng.normal(0, 3.0)
                blk["wind_ms"]   *= np.exp(rng.normal(0, 0.10))
                blk["solar_wm2"] *= np.exp(rng.normal(0, 0.05))   # raat ka 0 zero hi rehta hai
            blocks.append(blk)
            d += L
    syn = pd.concat(blocks, ignore_index=True)
    # physical constraints
    syn["rh_pct"]    = syn["rh_pct"].clip(0, 100)
    syn["wind_ms"]   = syn["wind_ms"].clip(lower=0)
    syn["solar_wm2"] = syn["solar_wm2"].clip(lower=0)
    syn["temp_c"]    = syn["temp_c"].clip(src.temp_c.min() - 2, src.temp_c.max() + 2)
    # synthetic calendar: non-leap saal, taaki hour/month/day-of-year features sahi bane
    ys = [y for y in range(2001, 2300) if not (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))][:n_years]
    idx = np.concatenate([pd.date_range(f"{y}-01-01", periods=365*24, freq="h") for y in ys])
    syn.index = pd.DatetimeIndex(idx, name="time")
    return add_risk(syn, cfg)
