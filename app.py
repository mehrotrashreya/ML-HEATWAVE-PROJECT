"""Delhi Heat Watch: interactive 3-day temperature and heatwave dashboard.  Run:  streamlit run app.py"""
import json
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import xgboost as xgb
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.features import VARS                                                                  # noqa: E402
from src.live import API_VARS, fetch_recent_daily, last_three_completed_days, predict_next_three_days  # noqa: E402

st.set_page_config(page_title="Delhi Heat Watch", page_icon="🌡️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.stApp{background:radial-gradient(1100px 480px at 88% -8%,rgba(240,80,110,.20),transparent 60%),#0b1020}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:4.2rem;max-width:1280px}
.nav{display:flex;justify-content:space-between;align-items:center;padding:10px 4px 14px 4px;margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,.08)}
.brand{font-size:24px;font-weight:800;color:#fff}.brand b{color:#f0506e}.nav-r{font-size:13px;color:#9aa6c4}
.hero{border-radius:20px;padding:24px 30px;color:#fff;margin:2px 0 10px 0;box-shadow:0 8px 30px #0006;border:1px solid rgba(255,255,255,.08)}
.hero h1{margin:8px 0 0 0;font-size:38px;font-weight:800;line-height:1.15;color:#fff}.hero p{margin:8px 0 12px 0;font-size:16px;opacity:.95;max-width:900px}
.tag{display:inline-block;background:#f0506e;color:#fff;font-size:12px;font-weight:700;letter-spacing:.06em;padding:4px 12px;border-radius:99px}
.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.14);border-radius:99px;padding:5px 13px;font-size:13px;color:#e8ecf5}
.card{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.10);border-radius:16px;padding:16px 18px;box-shadow:0 4px 16px #0004;height:100%;color:#e8ecf5}
.card .big{font-size:34px;font-weight:800;line-height:1.1;color:#fff}.card .small{font-size:13px;color:#9aa6c4}
.small{font-size:13px;color:#9aa6c4}
.pill{display:inline-block;padding:3px 12px;border-radius:99px;font-weight:700;font-size:13px;vertical-align:middle}
.soft{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:12px 16px;color:#e8ecf5}
[data-testid="stMetric"]{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:8px 14px}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid rgba(255,255,255,.08)}
.stTabs [data-baseweb="tab"]{padding:8px 16px;font-weight:600;border-radius:10px 10px 0 0}
.stTabs [aria-selected="true"]{color:#f0506e}
.st-key-prev_day button,.st-key-next_day button{border-radius:50%;width:48px;height:48px;background:#000;color:#fff;border:1px solid #333;font-size:24px;padding:0}
iframe{border-radius:16px;border:1px solid rgba(255,255,255,.10)}
</style>""", unsafe_allow_html=True)

LEVELS = [("Comfortable", "#2ecc71"), ("Hot", "#f1c40f"), ("Very hot", "#e67e22"), ("Heatwave", "#c0392b"), ("Severe heatwave", "#7f1d1d")]
UNITS = {"tmax": "°C", "humidity": "%", "wind": "km/h", "solar": "MJ/m²"}
PLACES = {"Delhi (central)": (28.61, 77.21), "Gurugram": (28.46, 77.03), "Noida": (28.57, 77.39), "Faridabad": (28.41, 77.31),
          "Ghaziabad": (28.67, 77.45), "Sonipat": (28.99, 77.02), "Meerut": (28.98, 77.71), "Hisar": (29.15, 75.72),
          "Jaipur": (26.91, 75.79), "Agra": (27.18, 78.01)}
HOME = "Delhi (central)"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LATEST = pd.Timestamp.today().normalize() - pd.Timedelta(days=7)          # archive data lags by ~5 days
ADVICE = {
    "General public": ["Normal day. Usual precautions are enough.", "Warm to hot. Drink water regularly and use shade.",
                       "Very hot. Avoid the midday sun (12-4 pm), wear light cotton, drink often.",
                       "Heatwave. Stay indoors in the afternoon, check on neighbours, use cool places.",
                       "Severe heatwave. Avoid going out; seek medical help for dizziness, confusion or fainting."],
    "Outdoor workers": ["Normal schedule.", "Keep water handy and take shade breaks.", "Move heavy work to early morning or evening; rest every hour.",
                        "Stop heavy outdoor work at midday; use cooling breaks.", "Suspend outdoor work in the afternoon; keep medical help on stand-by."],
    "Elderly & children": ["Normal day.", "Keep fluids nearby; avoid long sun exposure.", "Stay in cool rooms in the afternoon; no midday outdoor play.",
                           "Stay indoors, drink fluids often, watch for vomiting or dizziness.", "Stay in a cool place all day; urgent medical help for confusion or hot dry skin."]}


def fg(c):
    return "#111" if c in ("#2ecc71", "#f1c40f") else "#fff"


def pill(lv):
    n, c = LEVELS[lv]
    return f'<span class="pill" style="background:{c};color:{fg(c)}">{n}</span>'


def show(fig, h=380, top=50):
    fig.update_layout(margin=dict(l=10, r=10, t=top, b=10), height=h, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def build_map(geo, k, names, center, zoom, home):
    mp = folium.Map(location=center, zoom_start=zoom, tiles=None)
    folium.TileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", attr="© OpenStreetMap © CARTO", name="Dark").add_to(mp)
    folium.TileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", attr="© OpenStreetMap contributors", name="OpenStreetMap").add_to(mp)
    folium.LayerControl(collapsed=True).add_to(mp)
    for pl in names:
        if pl not in geo:
            continue
        pp = geo[pl]
        la, lo = PLACES[pl]
        lv, t = int(pp["level"].iloc[k]), float(pp["pred"].iloc[k])
        col = LEVELS[lv][1]
        folium.CircleMarker([la, lo], radius=24 if pl == home else 19, color="#ffffff", weight=3 if pl == home else 1.5, fill=True,
                            fill_color=col, fill_opacity=.92, tooltip=f"{pl}: {t:.1f} °C · {LEVELS[lv][0]}").add_to(mp)
        folium.Marker([la, lo], icon=folium.DivIcon(icon_size=(44, 20), icon_anchor=(22, 10),
                      html=f"<div style='text-align:center;font:700 12px sans-serif;color:{fg(col)};pointer-events:none'>{t:.0f}°</div>")).add_to(mp)
    leg = "".join(f"<div><span style='color:{c}'>●</span> {n}</div>" for n, c in LEVELS)
    mp.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:14px;left:14px;z-index:9999;background:#0b1020cc;padding:8px 12px;border-radius:10px;"
        f"font:12px sans-serif;color:#e8ecf5;border:1px solid #ffffff22'>{leg}</div>"))
    return mp


def read(path, **kw):
    p = ROOT / path
    return pd.read_csv(p, **kw) if p.exists() else None


def mv(d, key):                       # metric lookup that tolerates a damaged degree sign in the key
    for k, v in d.items():
        if (key == "R2" and k.startswith("R") and not k.startswith("RM") and len(k) <= 3) or (key != "R2" and k.startswith(key)):
            return v
    return float("nan")


# ---------------------------------------------------------------- model, data, helpers
@st.cache_resource
def load_model():
    mp, jp = ROOT / "models" / "xgb_tmax_regressor.json", ROOT / "models" / "model_metadata.json"
    if not (mp.exists() and jp.exists()):
        return None, None
    m = xgb.XGBRegressor()
    m.load_model(mp)
    return m, json.loads(jp.read_text(encoding="utf-8"))


model, meta = load_model()
clean = read("data/processed/delhi_daily_clean.csv", parse_dates=["date"])
if model is None or clean is None:
    st.error("Model or dataset not found. Run notebooks 01 -> 02 -> 03 first, then start the app from the heatwave-delhi folder.")
    st.stop()
FEATS = tuple(meta["selected_features_in_order"])
CLEAN_END = clean["date"].max()


@st.cache_data
def normals():
    d = clean.set_index("date")["tmax"]
    doy = d.groupby(d.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    ext = pd.concat([doy.iloc[-15:], doy, doy.iloc[:15]]).rolling(15, center=True).mean().iloc[15:-15]
    ext.index = range(1, 367)
    return ext


NORM = normals()


def normal_for(d):
    return float(NORM.loc[min(pd.Timestamp(d).dayofyear, 366)])


def level(t, normal, alert):
    dep = t - normal
    if t >= 47 or (t >= 40 and dep > 6.4):
        return 4
    if t >= 45 or (t >= 40 and dep >= 4.5):
        return 3
    return 2 if t >= alert else (1 if t >= alert - 5 else 0)


def adjust(p):
    """Peak-aware correction fitted on the 2023-24 validation years (the model under-predicts the hottest days)."""
    p = np.asarray(p, float)
    return p + np.where(p >= 36, np.clip(0.461 * (p - 36) - 1.231, 0, None), 0)


@st.cache_data
def residual_pools():
    v = read("reports/model/validation_predictions.csv")
    pools = {h: np.array([0.0]) for h in (1, 2, 3)}
    if v is not None:
        v["adj"] = adjust(v["predicted"].values)
        v["res"] = v["actual"] - v["adj"]
        for h in (1, 2, 3):
            s = v[v["forecast_horizon"] == h]
            hot = s[s["adj"] >= 36]
            pools[h] = (hot if len(hot) >= 40 else s)["res"].values
    return pools


POOLS = residual_pools()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_range(lat, lon, start, end):
    prm = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, daily=",".join(API_VARS), timezone="Asia/Kolkata",
               temperature_unit="celsius", wind_speed_unit="kmh")
    r = requests.get(ARCHIVE_URL, params=prm, timeout=40)
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({"date": pd.to_datetime(d["time"]), **{n: d[a] for a, n in API_VARS.items()}})


@st.cache_data(ttl=3600, show_spinner=False)
def live_series(lat, lon):
    return fetch_recent_daily(lat, lon, past_days=14)


def series_for(place, T):
    lat, lon = PLACES[place]
    if T is None:
        df = live_series(lat, lon)
        t = last_three_completed_days(df)["date"].max()
        return df[df["date"] <= t].reset_index(drop=True), t
    start, end = T - pd.Timedelta(days=14), min(T + pd.Timedelta(days=3), LATEST)
    if place == HOME and end <= CLEAN_END:
        df = clean[(clean["date"] >= start) & (clean["date"] <= end)][["date"] + VARS]
    else:
        df = fetch_range(lat, lon, str(start.date()), str(end.date()))
    return df.reset_index(drop=True), T


def make_forecast(series, T, adj_on, alert):
    last3 = series[series["date"] <= T].tail(3).reset_index(drop=True)
    if len(last3) < 3 or last3[VARS].isna().any().any():
        return None, None
    p = predict_next_three_days(model, last3, list(FEATS))
    p["raw"] = p["predicted_tmax"]
    p["pred"] = adjust(p["raw"]) if adj_on else p["raw"]
    p["actual"] = p["target_date"].map(series.set_index("date")["tmax"])
    p["normal"] = p["target_date"].map(normal_for)
    p["level"] = [level(a, b, alert) for a, b in zip(p["pred"], p["normal"])]
    p["lo"] = [a + np.quantile(POOLS[int(h)], .1) for a, h in zip(p["pred"], p["forecast_horizon"])]
    p["hi"] = [a + np.quantile(POOLS[int(h)], .9) for a, h in zip(p["pred"], p["forecast_horizon"])]
    p["p_alert"] = [float((a + POOLS[int(h)] >= alert).mean() * 100) for a, h in zip(p["pred"], p["forecast_horizon"])]
    p["p_45"] = [float((a + POOLS[int(h)] >= 45).mean() * 100) for a, h in zip(p["pred"], p["forecast_horizon"])]
    return p, last3


@st.cache_data(show_spinner=False)
def backtest(_model, feats, ser, adj_on):
    ser = ser.dropna().reset_index(drop=True)
    act = ser.set_index("date")["tmax"]
    rows = []
    for i in range(2, len(ser)):
        l3 = ser.iloc[i - 2:i + 1].reset_index(drop=True)
        if (l3["date"].iloc[2] - l3["date"].iloc[0]).days != 2:
            continue
        for _, r in predict_next_three_days(_model, l3, list(feats)).iterrows():
            a = act.get(r["target_date"])
            if a is not None and pd.notna(a):
                rows.append((l3["date"].iloc[2], r["target_date"], int(r["forecast_horizon"]),
                             float(adjust(r["predicted_tmax"])) if adj_on else float(r["predicted_tmax"]), float(a)))
    return pd.DataFrame(rows, columns=["origin", "target_date", "horizon", "pred", "actual"])


def score(bt, cut):
    if bt.empty:
        return {}
    e = bt["pred"] - bt["actual"]
    a, p = bt["actual"] >= cut, bt["pred"] >= cut
    return {"MAE (°C)": round(e.abs().mean(), 2), "RMSE (°C)": round(float(np.sqrt((e ** 2).mean())), 2), "Bias (°C)": round(e.mean(), 2),
            f"Hot days ≥{cut:.0f}°C caught": f"{int((a & p).sum())}/{int(a.sum())}", "False alarms": int((~a & p).sum())}


# ---------------------------------------------------------------- header and controls
st.markdown("""<div class="nav"><span class="brand">🌡️ Delhi<b>Heat</b>Watch</span>
<span class="nav-r">3-day temperature &amp; heatwave outlook · XGBoost · ERA5 / Open-Meteo</span></div>""", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1.3, 1.3, 1.4])
mode = c1.radio("Mode", ["Live now", "Pick a date"], horizontal=True)
if mode == "Pick a date":
    dflt = pd.Timestamp("2026-06-10") if LATEST >= pd.Timestamp("2026-06-10") else CLEAN_END
    pick = c2.date_input("Date you want to look at", value=dflt.date(), min_value=pd.Timestamp("2015-01-03").date(), max_value=LATEST.date())
    T_req = pd.Timestamp(pick)
else:
    T_req = None
    c2.markdown("<div class='soft' style='margin-top:26px'>Uses the latest weather from Open-Meteo</div>", unsafe_allow_html=True)
place = c3.selectbox("Location", list(PLACES), help="The model was trained on central Delhi only. Other places are extrapolation.")
with st.expander("⚙️ Settings (alert level, peak adjustment, who is reading)"):
    s1, s2, s3 = st.columns(3)
    alert = s1.slider("'Very hot' alert from (°C)", 36, 44, 40)
    adj_on = s2.checkbox("Peak-aware adjustment", value=True, help="Corrects the model's tendency to under-predict the hottest days.")
    who = s3.radio("Advice for", list(ADVICE), horizontal=True)

try:
    series, T = series_for(place, T_req)
except Exception as err:
    st.error(f"Could not load weather data ({str(err)[:120]}). Check your internet, or choose a date up to 2025 for Delhi (works offline).")
    st.stop()
P, last3 = make_forecast(series, T, adj_on, alert)
if P is None:
    st.warning("Not enough weather data for that date. Please pick another date.")
    st.stop()
obs = last3.iloc[-1]
nrm_T = normal_for(T)
lvl_T = level(obs["tmax"], nrm_T, alert)
worst = int(P["level"].max())
peak = P.loc[P["pred"].idxmax()]
if "fday" not in st.session_state:
    st.session_state["fday"] = 0


def _step(n):
    st.session_state["fday"] = (st.session_state["fday"] + n) % 3


fday = st.session_state["fday"]
d1 = P.iloc[fday]
wd = ["falls well below", "is below", "is close to", "is above", "is well above"]
dep1 = d1["pred"] - d1["normal"]
phrase = wd[int(np.digitize(dep1, [-3, -1, 1, 3]))]
colw = LEVELS[int(d1["level"])][1]
ha, hb, hc = st.columns([0.06, 0.88, 0.06])
ha.markdown("<div style='height:100px'></div>", unsafe_allow_html=True)
ha.button("‹", key="prev_day", on_click=_step, args=(-1,))
hc.markdown("<div style='height:100px'></div>", unsafe_allow_html=True)
hc.button("›", key="next_day", on_click=_step, args=(1,))
dots = " ".join("●" if i == fday else "○" for i in range(3))
hb.markdown(f"""<div class="hero" style="background:linear-gradient(120deg,#0f172a 0%,#16213e 58%,{colw} 230%)">
<span class="tag">⭐ DAY +{int(d1['forecast_horizon'])} OUTLOOK</span> <span class="small" style="margin-left:8px">based on weather up to {T:%A, %d %B %Y}</span>
<h1>{d1['target_date']:%A, %d %b}: about {d1['pred']:.0f} °C {pill(int(d1['level']))}</h1>
<p>That {phrase} what is normal for this time of year ({d1['normal']:.0f} °C, {dep1:+.1f} °C). The hottest day in the next 3 days is
<b>{peak['target_date']:%a %d %b} at about {peak['pred']:.0f} °C</b>.</p>
<div class="chips"><span class="chip">📅 {d1['target_date']:%d %b %Y}</span><span class="chip">📍 {place}</span>
<span class="chip">🎯 {d1['p_alert']:.0f}% chance of ≥ {alert} °C</span><span class="chip">↔ likely {d1['lo']:.0f}-{d1['hi']:.0f} °C</span>
<span class="chip" style="margin-left:auto">{dots}</span></div></div>""", unsafe_allow_html=True)
if worst >= 3:
    st.error(f"⚠️ **Heatwave conditions expected.** {ADVICE[who][worst]}")
elif worst == 2:
    st.warning(f"**Very hot days ahead.** {ADVICE[who][2]}")

if worst >= 2 and not st.session_state.get("toasted"):
    st.toast(f"{LEVELS[worst][0]} expected: up to {peak['pred']:.0f} °C on {peak['target_date']:%a %d %b}", icon="🔥")
    st.session_state["toasted"] = True

rows, geo = [], {}
with st.spinner("Loading nearby places..."):
    for pl in PLACES:
        try:
            s_, t_ = series_for(pl, T_req)
            pp, l3 = make_forecast(s_, t_, adj_on, alert)
            if pp is None:
                continue
            geo[pl] = pp
            rows.append({"Place": pl, "Validated": "✔ trained here" if pl == HOME else "extrapolation", "Last Tmax (°C)": round(l3["tmax"].iloc[-1], 1),
                         "Day +1 (°C)": round(pp["pred"].iloc[0], 1), "Day +2 (°C)": round(pp["pred"].iloc[1], 1),
                         "Day +3 (°C)": round(pp["pred"].iloc[2], 1), "Worst level": LEVELS[int(pp["level"].max())][0]})
        except Exception:
            continue
NCR = ["Delhi (central)", "Gurugram", "Noida", "Faridabad", "Ghaziabad", "Sonipat"]

t_fc, t_cmp, t_acc, t_learn, t_mod = st.tabs(["🏠 Forecast", "📍 Compare places", "🎯 How accurate?", "📚 Learn", "🛠️ Model details"])

# ---------------------------------------------------------------- forecast tab
with t_fc:
    st.markdown(f"#### Weather on {T:%d %B %Y} (selected day)")
    m = st.columns(6)
    m[0].metric("Max temperature", f"{obs['tmax']:.1f} °C", f"{obs['tmax'] - nrm_T:+.1f} °C vs normal", delta_color="off")
    m[1].metric("Humidity", f"{obs['humidity']:.0f} %")
    m[2].metric("Wind (max)", f"{obs['wind']:.0f} km/h")
    m[3].metric("Sunshine energy", f"{obs['solar']:.1f} MJ/m²")
    m[4].metric("Normal for date", f"{nrm_T:.1f} °C")
    m[5].markdown(f"<div class='soft'>Heat level<br>{pill(lvl_T)}</div>", unsafe_allow_html=True)
    st.markdown("#### Next 3 days")
    cols = st.columns(3)
    for c, (_, r) in zip(cols, P.iterrows()):
        act = f"Actual: <b>{r['actual']:.1f} °C</b> (error {r['pred'] - r['actual']:+.1f})" if pd.notna(r["actual"]) else "Actual: not available yet"
        c.markdown(f"""<div class="card"><div class="small">{r['target_date']:%A %d %b} · day +{int(r['forecast_horizon'])}</div>
        <div class="big">{r['pred']:.1f} °C</div>{pill(int(r['level']))}
        <div class="small" style="margin-top:8px">Likely range {r['lo']:.0f}-{r['hi']:.0f} °C · normal {r['normal']:.0f} °C<br>
        Chance ≥ {alert} °C: <b>{r['p_alert']:.0f}%</b> · ≥ 45 °C: <b>{r['p_45']:.0f}%</b><br>{act}</div></div>""", unsafe_allow_html=True)
    st.markdown(f"<div class='soft' style='margin-top:12px'>👉 <b>What to do ({who.lower()}):</b> {ADVICE[who][worst]} "
                f"<span class='small'>(general guidance, not medical advice)</span></div>", unsafe_allow_html=True)
    hist = series[series["date"] <= T].tail(14)
    fd = list(P["target_date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(hist["date"]) + fd, y=[normal_for(d) for d in list(hist["date"]) + fd], name="Normal",
                             line=dict(color="#9ca3af", dash="dot")))
    fig.add_trace(go.Scatter(x=fd + fd[::-1], y=list(P["hi"]) + list(P["lo"])[::-1], fill="toself", fillcolor="rgba(228,87,86,.15)",
                             line=dict(width=0), name="Likely range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["tmax"], name="Observed", mode="lines+markers", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=[T] + fd, y=[obs["tmax"]] + list(P["pred"]), name="Forecast", mode="lines+markers",
                             line=dict(color="#e45756", dash="dash")))
    if P["actual"].notna().any():
        fig.add_trace(go.Scatter(x=fd, y=P["actual"], name="Actual", mode="markers", marker=dict(symbol="diamond", size=12, color="#111")))
    fig.add_trace(go.Scatter(x=[T], y=[obs["tmax"]], mode="markers+text", text=["Selected day"], textposition="top center",
                             marker=dict(symbol="star", size=16, color="#f59e0b"), showlegend=False))
    fig.add_hline(y=alert, line_dash="dot", line_color="#e67e22", annotation_text=f"Alert {alert} °C", annotation_position="top left")
    fig.add_hline(y=45, line_dash="dot", line_color="#c0392b", annotation_text="IMD heatwave (absolute) 45 °C", annotation_position="top left")
    fig.update_layout(yaxis_title="°C", legend=dict(orientation="h", y=-0.2))
    mday = st.radio("Show on map", ["Day +1", "Day +2", "Day +3"], horizontal=True, key="mapday")
    ch, mh = st.columns([3, 2])
    with ch:
        st.markdown("##### 📈 Temperature outlook")
        show(fig, 440, top=10)
    with mh:
        st.markdown("##### 🗺️ Heat map: Delhi NCR")
        st_folium(build_map(geo, int(mday[-1]) - 1, NCR, [28.68, 77.2], 9, place), height=440, use_container_width=True, returned_objects=[])
        st.caption("Coarse weather-grid points (~25 km), not neighbourhood data. Number = forecast Tmax; ring = selected location.")

# ---------------------------------------------------------------- compare tab
with t_cmp:
    st.markdown("#### Compare places around Delhi")
    st.info("The model was trained on **central Delhi only**. Other places use the same model without separate validation, "
            "so treat them as an indication, not a verified forecast. Use the accuracy check below to see how well it travels.")
    if not rows:
        st.warning("Could not load other places (needs internet).")
    else:
        hz = st.radio("Show map for", ["Day +1", "Day +2", "Day +3"], horizontal=True, key="mapday2")
        st_folium(build_map(geo, int(hz[-1]) - 1, list(PLACES), [28.2, 77.0], 7, place), height=470, use_container_width=True, returned_objects=[])
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    if st.button("Check how accurate the model is at each place (2026 data, takes ~1 minute)"):
        out = []
        with st.spinner("Downloading 2026 weather and testing the model at each place..."):
            for pl, (la, lo) in PLACES.items():
                try:
                    s26 = fetch_range(la, lo, "2026-01-01", str(LATEST.date()))
                    bt = backtest(model, FEATS, s26.tail(150), adj_on)
                    sc = score(bt[bt["horizon"] == 1], alert)
                    out.append({"Place": pl, "Day+1 MAE (°C)": sc.get("MAE (°C)"), "Bias (°C)": sc.get("Bias (°C)"), "Days tested": len(bt) // 3})
                except Exception:
                    continue
        if out:
            st.dataframe(pd.DataFrame(out), hide_index=True)
            st.caption("Lower MAE = better. If far-away places have higher error, the model does not transfer well, which is why training on several locations is the next improvement.")

# ---------------------------------------------------------------- accuracy tab
with t_acc:
    st.markdown("#### 1. Untouched test year (2025)")
    tp = read("reports/model/test_predictions.csv", parse_dates=["target_date"])
    if tp is not None:
        tp["adj"] = adjust(tp["predicted"].values)
        hh = st.radio("Forecast horizon", [1, 2, 3], horizontal=True, format_func=lambda x: f"Day +{x}")
        g = tp[tp["forecast_horizon"] == hh].sort_values("target_date")
        f = px.line(g, x="target_date", y=["actual", "predicted", "adj"], labels={"value": "°C", "target_date": "", "variable": ""},
                    title=f"2025: actual vs predicted (day +{hh}); 'adj' = peak-aware adjustment")
        show(f)
        cmp = []
        for nm, col in (("Raw model", "predicted"), ("With peak adjustment", "adj")):
            row = {"Version": nm, "MAE (°C)": round((tp["actual"] - tp[col]).abs().mean(), 2)}
            for cut in (40, 42):
                a, p = tp["actual"] >= cut, tp[col] >= cut
                row[f"Hot days ≥{cut}°C caught"] = f"{int((a & p).sum())}/{int(a.sum())}"
                row[f"False alarms ≥{cut}°C"] = int((~a & p).sum())
            cmp.append(row)
        st.dataframe(pd.DataFrame(cmp), hide_index=True)
        st.caption("The adjustment was fitted on 2023-24 and only evaluated on 2025. 2025 had no day at or above 45 °C, so heatwave detection at 45 °C cannot be judged on this year.")
    st.markdown("#### 2. 2026 so far (genuinely unseen data)")
    if st.button("Run the 2026 check for Delhi (needs internet)"):
        with st.spinner("Downloading 2026 weather and forecasting every day..."):
            try:
                s26 = fetch_range(*PLACES[HOME], "2026-01-01", str(LATEST.date()))
                bt26 = backtest(model, FEATS, s26, adj_on)
                st.session_state["bt26"] = bt26
            except Exception as err:
                st.error(f"Could not run the 2026 check: {str(err)[:100]}")
    bt26 = st.session_state.get("bt26")
    if bt26 is not None and len(bt26):
        sc = pd.DataFrame([{"Horizon": f"Day +{h}", **score(bt26[bt26["horizon"] == h], alert)} for h in (1, 2, 3)])
        st.dataframe(sc, hide_index=True)
        g = bt26[bt26["horizon"] == 1].sort_values("target_date")
        show(px.line(g, x="target_date", y=["actual", "pred"], labels={"value": "°C", "target_date": "", "variable": ""},
                     title="2026: actual vs day +1 forecast"))

# ---------------------------------------------------------------- learn tab
with t_learn:
    l1, l2 = st.columns(2)
    with l1:
        st.markdown("#### What do the colours mean?")
        for i, (n, c) in enumerate(LEVELS):
            desc = ["Normal weather for the season.", f"About {alert - 5} to {alert} °C. Be sensible outdoors.", f"{alert} °C or more. Avoid the midday sun.",
                    "Meets the India Meteorological Department's heatwave rule.", "Meets the severe heatwave rule. Dangerous."][i]
            st.markdown(f"<div class='soft' style='margin-bottom:6px'>{pill(i)} &nbsp;{desc}</div>", unsafe_allow_html=True)
    with l2:
        st.markdown("#### What is a heatwave?")
        st.markdown("""The India Meteorological Department (IMD) treats **40 °C as the starting point in the plains**. A heatwave is then declared
when the maximum is **4.5 to 6.4 °C above normal**, or **at least 45 °C**; a severe heatwave is **more than 6.4 °C above normal**, or **47 °C or more**.
IMD also needs the condition at 2 or more stations for 2 days. This dashboard applies the same idea to one point, using the 2015-2025
average for each date as "normal", so it is an approximation of the official declaration.""")
    st.markdown("#### Frequently asked")
    with st.expander("Why not just call every day above 40 °C a heatwave?"):
        st.write("In Delhi, 40 °C is normal in May and June, so a flat 40 °C rule would label most of the summer a heatwave. IMD therefore also checks how far above normal the day is. Use the alert slider in Settings to choose when *you* want a 'very hot' warning.")
    with st.expander("How does the forecast work?"):
        st.write("The model looks at the last 3 days (maximum temperature, humidity, wind, sunshine) plus the time of year, and predicts the maximum temperature for the next 3 days. It is trained on 2015-2022, tuned on 2023-24 and tested on 2025.")
    with st.expander("How accurate is it?"):
        st.write(f"On the 2025 test year the typical error is about {mv(meta['metrics']['test_all_horizons'], 'MAE'):.1f} °C (day +1 is the most accurate). It is weakest on the very hottest days, which is why a likely range and the chance of crossing each limit are shown.")
    with st.expander("Is this medical advice?"):
        st.write("No. It is an environmental heat estimate with general precautions. Follow official IMD and health-department alerts.")

# ---------------------------------------------------------------- model tab
with t_mod:
    ta = meta["metrics"]["test_all_horizons"]
    k = st.columns(4)
    k[0].metric("Test MAE (2025)", f"{mv(ta, 'MAE'):.2f} °C")
    k[1].metric("Test RMSE", f"{mv(ta, 'RMSE'):.2f} °C")
    k[2].metric("Test R²", f"{mv(ta, 'R2'):.3f}")
    k[3].metric("Trees / features", f"{meta['n_estimators']} / {len(FEATS)}")
    tm = read("reports/model/metrics_test.csv", index_col=0)
    if tm is not None:
        st.markdown("**Test metrics by horizon**")
        st.dataframe(tm.round(3))
    fi = read("reports/model/feature_importance.csv", index_col=0)
    if fi is not None:
        top = fi.sort_values("perm_MAE_increase", ascending=False).head(10).reset_index()
        show(px.bar(top, x="perm_MAE_increase", y=top.columns[0], orientation="h", title="Most important inputs (permutation importance)"), 340)
    d = clean.assign(year=clean["date"].dt.year)
    cnt = pd.DataFrame({f"≥{c} °C": d[d["tmax"] >= c].groupby("year").size() for c in (40, 42, 45)}).reindex(sorted(d["year"].unique())).fillna(0)
    show(px.bar(cnt, barmode="group", title="Hot days per year in the data (45 °C days are very rare, so the model has little to learn from)"), 340)
    st.caption("Hyper-parameters: " + json.dumps(meta["hyper_parameters"]))
    st.caption("Inputs: " + ", ".join(FEATS))
    st.caption("Limits: one ERA5 grid point; live inputs come from Open-Meteo's forecast service while training used the archive; "
               "the peak adjustment is a post-hoc correction, not a retrained model.")
