"""Delhi Heat-Stress Risk Dashboard (Streamlit). Run from project root: streamlit run app/app.py"""
import os
import sys

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))
from predict import (ADVICE, CLASSES, COLORS, COLS, LOCATIONS, heat_index_at,  # noqa: E402
                     live_prediction, load_artifacts, predict_from_df)

st.set_page_config(page_title="Delhi Heat-Stress Risk", page_icon="🌡️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.3rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#111827,#1f2937);border-radius:18px;padding:20px 26px;color:#fff;margin-bottom:12px}
.pill{display:inline-block;padding:3px 12px;border-radius:99px;font-weight:600;font-size:13px;vertical-align:middle}
[data-testid="stMetric"]{background:rgba(127,127,127,.09);border-radius:12px;padding:8px 14px}
</style>""", unsafe_allow_html=True)
CMAP = {n: COLORS[i] for i, n in enumerate(CLASSES)}
fg = lambda c: "#000" if c == 1 else "#fff"  # noqa: E731


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data(ttl=1800, show_spinner=False)
def get_live(name, lat, lon):
    cfg, meta, model = get_artifacts()
    return live_prediction(lat, lon, cfg, meta, model)


@st.cache_data(show_spinner=False)
def load_csv(name):
    p = os.path.join(ROOT, "data", "processed", name)
    return pd.read_csv(p, index_col="time", parse_dates=True) if os.path.exists(p) else None


def risk_card(label, cls, sub=""):
    st.markdown(f"""<div style="background:{COLORS[cls]};padding:16px 18px;border-radius:14px;color:{fg(cls)}">
    <div style="font-size:13px;opacity:.9">{label}</div><div style="font-size:30px;font-weight:800">{CLASSES[cls]}</div>
    <div style="font-size:12px">{sub}</div></div>""", unsafe_allow_html=True)


cfg, meta, model = get_artifacts()

# ------------------------------------------------------------ sidebar
st.sidebar.title("🎛️ Controls")
mode = st.sidebar.radio("Data mode", ["Live (Open-Meteo)", "Demo (stored 2023-24 data)"])
layer = st.sidebar.radio("Map layer", ["Predicted (24 h)", "Current"])
show = st.sidebar.multiselect("Show risk levels on map", CLASSES, default=CLASSES)
if st.sidebar.button("🔄 Refresh live data"):
    st.cache_data.clear()
st.sidebar.markdown("---")
for i, n in enumerate(CLASSES):
    st.sidebar.markdown(f"<span style='color:{COLORS[i]};font-size:18px'>●</span> {n}", unsafe_allow_html=True)

# ------------------------------------------------------------ data
results = {}
if mode.startswith("Live"):
    errs = []
    with st.spinner("Fetching latest weather and predicting..."):
        for nm, (la, lo) in LOCATIONS.items():
            try:
                results[nm] = get_live(nm, la, lo)
            except Exception as e:
                errs.append(f"{nm}: {e}")
    if not results:
        st.error("Could not fetch live data. Check internet or switch to Demo mode.\n\n" + "\n".join(errs[:2]))
        st.stop()
    if errs:
        st.warning(f"{len(errs)} location(s) failed to load.")
else:
    test = load_csv("test.csv")
    if test is None:
        st.error("data/processed/test.csv not found.")
        st.stop()
    d = st.sidebar.date_input("Demo date", value=pd.Timestamp("2024-05-25").date(),
                              min_value=test.index.min().date() + pd.Timedelta(days=3),
                              max_value=(test.index.max() - pd.Timedelta(days=1)).date())
    hr = st.sidebar.slider("Hour of day", 0, 23, 14)
    res = predict_from_df(test.loc[:pd.Timestamp(d) + pd.Timedelta(hours=hr), COLS].tail(120), cfg, meta, model)
    tg = res["target_time"]
    res["actual_future"] = int(test.loc[tg, "risk_class"]) if tg in test.index else None
    results["Central Delhi"] = res

first = next(iter(results.values()))
worst = max(r["pred_class"] for r in results.values())
n_hi = sum(r["pred_class"] >= 2 for r in results.values())

# ------------------------------------------------------------ hero
st.markdown(f"""<div class="hero" style="border-left:10px solid {COLORS[worst]}">
<div style="opacity:.7;font-size:13px">🌡️ Delhi heat-stress outlook · next 24 hours</div>
<div style="font-size:34px;font-weight:800">{CLASSES[worst]} risk
<span class="pill" style="background:{COLORS[worst]};color:{fg(worst)}">{n_hi} of {len(results)} points High or above</span></div>
<div style="opacity:.8">Valid for {first['target_time']:%a %d %b, %H:%M} IST · built from data up to {first['time']:%d %b, %H:%M} IST</div>
</div>""", unsafe_allow_html=True)
if worst >= 2:
    st.error(f"⚠️ Heat alert: {n_hi} location(s) forecast **{CLASSES[worst]}** risk. {ADVICE[worst]}")
if mode.startswith("Demo"):
    st.info("Demo mode: one point (Central Delhi) from stored data; the actual outcome is shown for comparison.")

t_map, t_24, t_sim, t_hist, t_about = st.tabs(
    ["🗺️ Risk map", "⏱️ Next 24 hours", "🧪 What-if simulator", "📊 History & model", "ℹ️ About"])

# ------------------------------------------------------------ map tab
with t_map:
    if st.session_state.get("sel") not in results:
        st.session_state["sel"] = list(results)[0]
    clicked = st.session_state.pop("map_click", None)
    if clicked in results:
        st.session_state["sel"] = clicked
    key = "pred_class" if layer.startswith("Pred") else "class_now"
    left, right = st.columns([3, 2])
    with right:
        sel = st.selectbox("📍 Location (or click a marker)", list(results), key="sel")
        r = results[sel]
        risk_card(f"{sel}: predicted risk in 24 h", r["pred_class"], f"for {r['target_time']:%d %b, %H:%M} IST")
        st.markdown(f"**What it means:** {ADVICE[r['pred_class']]}")
        st.caption("Generic guidance by risk category, not medical advice.")
        if r.get("actual_future") is not None:
            st.success(f"Actual risk at that time: **{CLASSES[r['actual_future']]}**")
        pf = px.bar(pd.DataFrame({"Risk": CLASSES, "Probability": r["proba"]}), x="Risk", y="Probability",
                    color="Risk", color_discrete_map=CMAP, text_auto=".0%")
        pf.update_layout(showlegend=False, height=240, margin=dict(l=0, r=0, t=8, b=0), yaxis_range=[0, 1.05])
        st.plotly_chart(pf, use_container_width=True)
    with left:
        m = folium.Map(location=[28.66, 77.17], zoom_start=10, tiles=None)
        folium.TileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                 attr="© OpenStreetMap contributors © CARTO", name="Light map").add_to(m)
        folium.TileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                 attr="© OpenStreetMap contributors", name="OpenStreetMap").add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)
        geo = os.path.join(ROOT, "data", "delhi_boundary.geojson")
        if os.path.exists(geo):
            folium.GeoJson(geo, style_function=lambda _: {"fillOpacity": .04, "color": "#555", "weight": 2}).add_to(m)
        for nm, rr in results.items():
            c = rr[key]
            if CLASSES[c] not in show:
                continue
            la, lo = LOCATIONS[nm]
            tip = (f"<b>{nm}</b><br>Predicted (24 h): {CLASSES[rr['pred_class']]}<br>Now: {CLASSES[rr['class_now']]}<br>"
                   f"{rr['now']['temp_c']:.1f} °C · RH {rr['now']['rh_pct']:.0f}% · wind {rr['now']['wind_ms']:.1f} m/s")
            folium.CircleMarker([la, lo], radius=20 if nm == sel else 14, color="#111", weight=3 if nm == sel else 1,
                                fill=True, fill_color=COLORS[c], fill_opacity=.88, tooltip=nm,
                                popup=folium.Popup(tip, max_width=260)).add_to(m)
        m.get_root().html.add_child(folium.Element(
            "<div style='position:fixed;bottom:18px;left:18px;z-index:9999;background:#fff;padding:8px 12px;"
            "border-radius:8px;font:12px sans-serif;box-shadow:0 1px 6px #0004;color:#222'>"
            + "".join(f"<div><span style='color:{COLORS[i]}'>●</span> {n}</div>" for i, n in enumerate(CLASSES)) + "</div>"))
        out = st_folium(m, height=500, use_container_width=True, returned_objects=["last_object_clicked_tooltip"])
        tipc = (out or {}).get("last_object_clicked_tooltip")
        if tipc and tipc.strip() in results and tipc.strip() != st.session_state["sel"]:
            st.session_state["map_click"] = tipc.strip()
            st.rerun()
        st.caption(f"Layer: **{layer}**. Markers are coarse weather-grid points (~10 km grid), not neighbourhood data.")
    tbl = pd.DataFrame([{"Location": n, "Temp °C": round(x["now"]["temp_c"], 1), "Index now": round(x["now"]["risk_index"], 1),
                         "Risk now": CLASSES[x["class_now"]], "Risk in 24 h": CLASSES[x["pred_class"]],
                         "P(High+)": round(sum(x["proba"][2:]), 2)} for n, x in results.items()])
    with st.expander("📋 All locations (table + download)"):
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Download CSV", tbl.to_csv(index=False), "delhi_heat_risk.csv", "text/csv")

# ------------------------------------------------------------ next 24 h tab
with t_24:
    r = results[st.session_state["sel"]]
    st.subheader(f"{st.session_state['sel']}: current weather and next-24-hour outlook")
    a, b, c, d_, e = st.columns(5)
    a.metric("Temperature", f"{r['now']['temp_c']:.1f} °C")
    b.metric("Humidity", f"{r['now']['rh_pct']:.0f} %")
    c.metric("Wind", f"{r['now']['wind_ms']:.1f} m/s")
    d_.metric("Solar", f"{r['now']['solar_wm2']:.0f} W/m²")
    e.metric("Heat-stress index", f"{r['now']['risk_index']:.1f}", CLASSES[r["class_now"]], delta_color="off")
    tl = r["timeline"].copy()
    tl["Risk"] = tl.pred_class.map(lambda x: CLASSES[x])
    fig = px.bar(tl, x="target_time", y="p_high_plus", color="Risk", color_discrete_map=CMAP,
                 labels={"target_time": "Hour (IST)", "p_high_plus": "Probability of High or Very High"},
                 title="Hour-by-hour risk for the next 24 hours (each bar = a 24 h-ahead forecast)")
    fig.update_layout(height=340, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    hi = tl[tl.pred_class >= 2]
    st.info(f"First High+ hour: **{hi.target_time.iloc[0]:%a %H:%M}**, peak class **{CLASSES[int(tl.pred_class.max())]}**."
            if len(hi) else "No High or Very High hours predicted in the next 24 hours.")
    h = r["history"].reset_index().rename(columns={"index": "time"})
    g1, g2 = st.columns(2)
    g1.plotly_chart(px.line(h, x="time", y=["temp_c", "rh_pct"], title="Temperature (°C) and humidity (%), last 72 h")
                    .update_layout(legend_title_text="", height=300), use_container_width=True)
    f2 = px.line(h, x="time", y="risk_index", title="Heat-stress index, last 72 h")
    for th, nm in zip(cfg["thresholds"], CLASSES[1:]):
        f2.add_hline(y=th, line_dash="dot", annotation_text=nm, annotation_position="top left")
    g2.plotly_chart(f2.update_layout(height=300), use_container_width=True)

# ------------------------------------------------------------ what-if tab
with t_sim:
    st.subheader("What-if simulator: how do the four weather factors combine?")
    st.caption("Uses the same heat-stress index as the model labels (current conditions, not the 24 h forecast).")
    S = {"w_t": 38.0, "w_h": 40, "w_w": 2.0, "w_s": 800}

    def preset(t, h_, w, s):
        st.session_state.update(w_t=float(t), w_h=h_, w_w=float(w), w_s=s)

    for k, v in S.items():
        st.session_state.setdefault(k, v)
    p1, p2, p3, p4 = st.columns(4)
    p1.button("☀️ Dry May afternoon", on_click=preset, args=(42, 25, 3, 900), use_container_width=True)
    p2.button("🌧️ Humid monsoon day", on_click=preset, args=(34, 80, 1.5, 450), use_container_width=True)
    p3.button("🌙 Winter night", on_click=preset, args=(9, 85, 1.5, 0), use_container_width=True)
    p4.button("🔥 Extreme heatwave", on_click=preset, args=(46, 30, 1, 1000), use_container_width=True)
    s1, s2 = st.columns(2)
    T = s1.slider("Temperature (°C)", 0.0, 48.0, key="w_t", step=0.5)
    H = s1.slider("Relative humidity (%)", 0, 100, key="w_h")
    W = s2.slider("Wind speed (m/s)", 0.0, 10.0, key="w_w", step=0.1)
    Q = s2.slider("Solar radiation (W/m²)", 0, 1100, key="w_s", step=10)
    idx = float(heat_index_at(T, H, W, Q, cfg["a"]))
    cls = sum(idx >= th for th in cfg["thresholds"])
    cc, cg = st.columns([1, 2])
    with cc:
        risk_card("Estimated heat-stress category", cls, f"index {idx:.1f} (air temp {T:.1f} °C)")
        st.write(f"Feels like **{idx - T:+.1f} °C** vs air temperature. " + ADVICE[cls])
    gf = go.Figure(go.Indicator(mode="gauge+number", value=idx, gauge={
        "axis": {"range": [-5, 60]}, "bar": {"color": "#111"},
        "steps": [{"range": [-5, cfg["thresholds"][0]], "color": COLORS[0]},
                  {"range": cfg["thresholds"][:2], "color": COLORS[1]},
                  {"range": cfg["thresholds"][1:], "color": COLORS[2]},
                  {"range": [cfg["thresholds"][2], 60], "color": COLORS[3]}]}))
    gf.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=0))
    cg.plotly_chart(gf, use_container_width=True)

# ------------------------------------------------------------ history tab
with t_hist:
    full = load_csv("delhi_clean.csv")
    if full is not None:
        st.subheader("Delhi 2015-2024 (Open-Meteo reanalysis)")
        h1, h2 = st.columns(2)
        mm = full.groupby(full.index.month)[["temp_c", "risk_index"]].mean().reset_index()
        mm.columns = ["month", "temp_c", "risk_index"]
        h1.plotly_chart(px.line(mm, x="month", y=["temp_c", "risk_index"], markers=True,
                                title="Average temperature and index by month"), use_container_width=True)
        yy = (full.risk_class >= 2).groupby(full.index.year).sum().reset_index()
        yy.columns = ["year", "hours"]
        h2.plotly_chart(px.bar(yy, x="year", y="hours", title="Hours per year at High or Very High"),
                        use_container_width=True)
    st.subheader("Evaluation on untouched real test data (2023-24)")
    cp = os.path.join(ROOT, "reports", "test_model_comparison.csv")
    if os.path.exists(cp):
        st.dataframe(pd.read_csv(cp), hide_index=True, use_container_width=True)
    i1, i2 = st.columns(2)
    for col, fn in [(i1, "test_confusion_matrix.png"), (i2, "feature_importance.png")]:
        pth = os.path.join(ROOT, "reports", fn)
        if os.path.exists(pth):
            col.image(pth, caption=fn.replace("_", " ").replace(".png", "").title())
    st.caption("Warning level (High or Very High): recall 0.91, precision 0.77.")

# ------------------------------------------------------------ about tab
with t_about:
    st.markdown("""
**How it works.** Recent hourly weather is turned into a heat-stress index (apparent temperature with a solar term).
An XGBoost classifier trained on 2015-2022 Delhi data predicts the risk category 24 hours ahead.

**Limitations**
- Estimated *environmental heat-stress category*, not a validated health-outcome prediction.
- The label comes from a formula; the model learns to forecast that indicator 24 h ahead.
- Trained on one Delhi point; other map points reuse the model without spatial validation, on a ~10 km weather grid.
- Live inputs come from Open-Meteo's forecast service, slightly different from the training archive.
- Fixed 24 h horizon; weaker in monsoon. Synthetic data did not meaningfully help, so the final model is real-only.
""")
