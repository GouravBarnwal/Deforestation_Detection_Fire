# =============================================================================
# 🔥 India Fire Type Classifier — MODIS Satellite Data
# Project: Deforestation & Fire Detection — India
# Modes: Historical (2022–2024) + Live NASA FIRMS API
# =============================================================================

import os
import requests
import numpy as np
import pandas as pd
import joblib
import gdown
import folium
import streamlit as st
from datetime import datetime
from io import StringIO
from geopy.distance import geodesic
from streamlit_folium import st_folium


# =============================================================================
# CONFIGURATION
# =============================================================================

PAGE_TITLE      = "🔥 India Fire Type Classifier"
MODEL_URL       = "https://drive.google.com/uc?id=1JvbfPY6bq4HD4oOVyCUEMoya--eSB7Qd"
SCALER_URL      = "https://drive.google.com/uc?id=18FjzK0oepVCJ43hUOuXK79bzUEY6bOk_"
MODEL_PATH      = "best_fire_detection_model.pkl"
SCALER_PATH     = "scaler.pkl"

# NASA Firms Map Key for accessing live data
FIRMS_API_KEY   = "b6895fba06f08209b6825f549a1210a0"
FIRMS_BASE_URL  = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
INDIA_BBOX      = "68.7,8.4,97.25,37.6"   # west,south,east,north

# Alert thresholds
FRP_CRITICAL    = 50.0   # MW — dispatch immediately
FRP_WARNING     = 20.0   # MW — monitor closely
CONF_HIGH       = 70     # confidence % for high alert

# India bounding box
INDIA_LAT_MIN, INDIA_LAT_MAX = 8.4,  37.6
INDIA_LON_MIN, INDIA_LON_MAX = 68.7, 97.25

FIRE_TYPES = {
    0: ("🌳 Vegetation Fire",          "#28a745"),
    1: ("🔥 Industrial / Urban Fire",  "#dc3545"),
    2: ("🏭 Other Static Land Source", "#6c757d"),
    3: ("🌊 Offshore Fire",            "#17a2b8"),
}

CONFIDENCE_MAP = {"low": 0, "nominal": 1, "high": 2}

DATA_FILES = [
    "modis_2022_India.csv",
    "modis_2023_India.csv",
    "modis_2024_India.csv",
]


# =============================================================================
# MODEL LOADING
# =============================================================================

def _download_if_missing(url, path, label):
    if os.path.exists(path):
        return True
    try:
        with st.spinner(f"📦 Downloading {label} …"):
            gdown.download(url, path, quiet=False)
        return True
    except Exception as exc:
        st.sidebar.error(f"❌ Failed to download {label}: {exc}")
        return False


@st.cache_resource
def load_model():
    if not _download_if_missing(MODEL_URL, MODEL_PATH, "model"):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"❌ Could not load model: {exc}")
        return None


@st.cache_resource
def load_scaler():
    if not _download_if_missing(SCALER_URL, SCALER_PATH, "scaler"):
        return None
    try:
        return joblib.load(SCALER_PATH)
    except Exception as exc:
        st.error(f"❌ Could not load scaler: {exc}")
        return None


@st.cache_data
def load_historical_data():
    frames = [pd.read_csv(f) for f in DATA_FILES]
    df = pd.concat(frames, ignore_index=True)
    df.dropna(subset=["latitude", "longitude"], inplace=True)
    df = df[
        df["latitude"].between(INDIA_LAT_MIN, INDIA_LAT_MAX) &
        df["longitude"].between(INDIA_LON_MIN, INDIA_LON_MAX)
    ].reset_index(drop=True)
    return df


# =============================================================================
# FIRMS LIVE DATA
# =============================================================================

def fetch_live_fires(day_range=1):
    """Pull today's MODIS NRT fire detections for India from NASA FIRMS."""
    url = f"{FIRMS_BASE_URL}/{FIRMS_API_KEY}/MODIS_NRT/{INDIA_BBOX}/{day_range}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        if df.empty or "latitude" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["latitude", "longitude"])
        return df
    except Exception as exc:
        st.error(f"❌ Could not fetch live data: {exc}")
        return pd.DataFrame()


def alert_level(row):
    """Return alert level string based on FRP and confidence."""
    frp  = float(row.get("frp", 0))
    conf = float(row.get("confidence", 0))
    if frp >= FRP_CRITICAL and conf >= CONF_HIGH:
        return "🚨 CRITICAL"
    elif frp >= FRP_WARNING or conf >= CONF_HIGH:
        return "⚠️ WARNING"
    else:
        return "ℹ️ MONITOR"


def alert_color(level):
    return {"🚨 CRITICAL": "#dc3545", "⚠️ WARNING": "#ffc107", "ℹ️ MONITOR": "#17a2b8"}.get(level, "#888")


# =============================================================================
# PREDICTION
# =============================================================================

def get_fire_label(pred, lat, lon):
    label, color = FIRE_TYPES.get(pred, ("Unknown", "gray"))
    if label == "🌳 Vegetation Fire" and (lat <= 10.0 or lon >= 92.0):
        label, color = FIRE_TYPES[3]
    return label, color


def predict_row(row, model, scaler):
    try:
        conf_val = CONFIDENCE_MAP.get(str(row.get("confidence", "nominal")).lower(), 1)
        # FIRMS confidence is numeric (0-100), map to low/nominal/high
        if isinstance(row.get("confidence"), (int, float)):
            c = float(row["confidence"])
            conf_val = 0 if c < 40 else (2 if c >= 70 else 1)
        # Feature order MUST match notebook training:
        # ['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']
        X = np.array([[
            float(row["brightness"]), float(row["scan"]),       float(row["track"]),
            conf_val,                 float(row["bright_t31"]), float(row["frp"])
        ]])
        X_scaled = scaler.transform(X)
        pred     = model.predict(X_scaled)[0]
        return get_fire_label(pred, float(row["latitude"]), float(row["longitude"]))
    except Exception:
        return ("Unknown", "#888")


def classify_live_fires(df, model, scaler):
    """Add predicted fire type and alert level to live dataframe."""
    results = df.apply(lambda r: predict_row(r, model, scaler), axis=1)
    df["fire_type"]   = [r[0] for r in results]
    df["fire_color"]  = [r[1] for r in results]
    df["alert_level"] = df.apply(alert_level, axis=1)
    return df


# =============================================================================
# HISTORICAL MATCH
# =============================================================================

def find_closest_match(df, brightness, bright_t31, frp, scan, track, confidence_str):
    conf_val = CONFIDENCE_MAP.get(confidence_str.lower(), 1)
    ranges   = {"brightness": (200, 500), "bright_t31": (250, 350),
                "frp": (0, 100), "scan": (0.5, 10), "track": (0.5, 10)}

    def norm(val, col):
        lo, hi = ranges[col]
        return (val - lo) / (hi - lo) if hi != lo else 0.0

    user = np.array([
        norm(brightness, "brightness"), norm(bright_t31, "bright_t31"),
        norm(frp, "frp"), norm(scan, "scan"), norm(track, "track"),
        conf_val / 2.0,
    ])

    sample = df.sample(min(5000, len(df)), random_state=42).copy()

    def row_vec(r):
        return np.array([
            norm(r["brightness"], "brightness"), norm(r["bright_t31"], "bright_t31"),
            norm(r["frp"], "frp"), norm(r["scan"], "scan"), norm(r["track"], "track"),
            CONFIDENCE_MAP.get(str(r["confidence"]).lower(), 1) / 2.0,
        ])

    sample["_dist"] = sample.apply(lambda r: np.linalg.norm(row_vec(r) - user), axis=1)
    return sample.loc[sample["_dist"].idxmin()]


# =============================================================================
# MAP RENDERING
# =============================================================================

def render_historical_map(lat, lon, label, color, nearby=None):
    m = folium.Map(location=[lat, lon], zoom_start=7)

    folium.Marker(
        [lat, lon],
        popup=folium.Popup(f"<b>{label}</b><br>📍 {lat:.4f}, {lon:.4f}", max_width=200),
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
        tooltip=label,
    ).add_to(m)

    folium.Circle([lat, lon], radius=40_000,
                  color=color, fill=True, fill_color=color, fill_opacity=0.15).add_to(m)

    if nearby is not None and not nearby.empty:
        for _, r in nearby.iterrows():
            folium.CircleMarker(
                [r["latitude"], r["longitude"]], radius=4,
                color="#888", fill=True, fill_opacity=0.4,
                tooltip=f"Nearby — {r.get('acq_date','')}",
            ).add_to(m)

    m.get_root().html.add_child(folium.Element(
        f"<div style='position:absolute;top:10px;left:50%;transform:translateX(-50%);"
        f"z-index:9999;background:{color};color:#fff;padding:6px 18px;"
        f"border-radius:8px;font-size:14px;font-weight:bold;"
        f"box-shadow:0 2px 6px rgba(0,0,0,0.3);'>{label}</div>"
    ))
    st_folium(m, width="100%", height=460, returned_objects=[])


def render_live_map(df_live):
    """Render all live fire detections on a single map."""
    center_lat = df_live["latitude"].mean()
    center_lon = df_live["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    for _, row in df_live.iterrows():
        level = row.get("alert_level", "ℹ️ MONITOR")
        color = alert_color(level)
        label = row.get("fire_type", "Unknown")
        lat, lon = row["latitude"], row["longitude"]

        popup_html = (
            f"<b>{label}</b><br>"
            f"<b style='color:{color}'>{level}</b><br>"
            f"📍 {lat:.4f}, {lon:.4f}<br>"
            f"🔥 FRP: {row.get('frp','N/A')} MW<br>"
            f"📅 {row.get('acq_date','N/A')} {row.get('acq_time','')}<br>"
            f"📡 Confidence: {row.get('confidence','N/A')}%"
        )

        # Size marker by FRP intensity
        frp_val  = float(row.get("frp", 5))
        radius   = max(6, min(20, frp_val / 3))

        folium.CircleMarker(
            [lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{level} — {label}",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed;bottom:30px;left:20px;z-index:9999;
         background:white;padding:12px;border-radius:8px;
         box-shadow:0 2px 8px rgba(0,0,0,0.2);font-size:13px;'>
        <b>Alert Levels</b><br>
        <span style='color:#dc3545'>●</span> 🚨 CRITICAL (FRP≥50, Conf≥70%)<br>
        <span style='color:#ffc107'>●</span> ⚠️ WARNING  (FRP≥20 or Conf≥70%)<br>
        <span style='color:#17a2b8'>●</span> ℹ️ MONITOR  (low intensity)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width="100%", height=520, returned_objects=[])


# =============================================================================
# TABS
# =============================================================================

def tab_live(model, scaler):
    st.subheader("🔴 Live Fire Detections — India (Today)")
    st.caption("Pulling real-time data from NASA FIRMS MODIS NRT satellite feed.")

    col1, col2 = st.columns([2, 1])
    with col1:
        day_range = st.radio("Data range", [1, 2, 3],
                             format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}",
                             horizontal=True)
    with col2:
        refresh = st.button("🔄 Refresh Live Data", use_container_width=True)

    # Load live data
    if "live_df" not in st.session_state or refresh:
        with st.spinner("🛰 Fetching live satellite data from NASA FIRMS …"):
            df_live = fetch_live_fires(day_range)
        if df_live.empty:
            st.warning("No fire detections found for India in this period.")
            return
        df_live = classify_live_fires(df_live, model, scaler)
        st.session_state.live_df = df_live

    df_live = st.session_state.live_df

    # ── Summary cards ────────────────────────────────────────────────────────
    total    = len(df_live)
    critical = len(df_live[df_live["alert_level"] == "🚨 CRITICAL"])
    warning  = len(df_live[df_live["alert_level"] == "⚠️ WARNING"])
    monitor  = len(df_live[df_live["alert_level"] == "ℹ️ MONITOR"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔥 Total Fires",    total)
    c2.metric("🚨 Critical",       critical, delta="Dispatch now"   if critical > 0 else None, delta_color="inverse")
    c3.metric("⚠️ Warning",        warning,  delta="Monitor closely" if warning  > 0 else None, delta_color="off")
    c4.metric("ℹ️ Monitor",        monitor)

    st.markdown("---")

    # ── Map ──────────────────────────────────────────────────────────────────
    st.subheader("🗺 Live Fire Map")
    render_live_map(df_live)

    # ── Critical alerts ───────────────────────────────────────────────────────
    critical_df = df_live[df_live["alert_level"] == "🚨 CRITICAL"].sort_values("frp", ascending=False)
    if not critical_df.empty:
        st.markdown("---")
        st.subheader("🚨 Critical Alerts — Immediate Dispatch Recommended")
        for i, (_, row) in enumerate(critical_df.iterrows(), 1):
            with st.container():
                st.markdown(
                    f"<div style='background:#2d0a0a;border-left:4px solid #dc3545;"
                    f"padding:14px;border-radius:6px;margin-bottom:10px;'>"
                    f"<b style='color:#dc3545'>Alert #{i} — {row['fire_type']}</b><br>"
                    f"📍 <b>Location:</b> {row['latitude']:.4f}°N, {row['longitude']:.4f}°E<br>"
                    f"🔥 <b>FRP:</b> {row['frp']} MW &nbsp;|&nbsp; "
                    f"📊 <b>Confidence:</b> {row['confidence']}% &nbsp;|&nbsp; "
                    f"🌡 <b>Brightness:</b> {row['brightness']} K<br>"
                    f"📅 <b>Detected:</b> {row['acq_date']} at {str(row['acq_time']).zfill(4)[:2]}:{str(row['acq_time']).zfill(4)[2:]} UTC &nbsp;|&nbsp; "
                    f"🛰 <b>Satellite:</b> {row.get('satellite','N/A')}<br>"
                    f"<span style='color:#dc3545;font-weight:bold;'>⚠️ Contact nearest fire department immediately</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Warning alerts ────────────────────────────────────────────────────────
    warning_df = df_live[df_live["alert_level"] == "⚠️ WARNING"].sort_values("frp", ascending=False)
    if not warning_df.empty:
        st.markdown("---")
        st.subheader("⚠️ Warning Alerts — Monitor Closely")
        for i, (_, row) in enumerate(warning_df.head(5).iterrows(), 1):
            st.markdown(
                f"<div style='background:#2d2200;border-left:4px solid #ffc107;"
                f"padding:12px;border-radius:6px;margin-bottom:8px;'>"
                f"<b style='color:#ffc107'>Warning #{i} — {row['fire_type']}</b><br>"
                f"📍 {row['latitude']:.4f}°N, {row['longitude']:.4f}°E &nbsp;|&nbsp; "
                f"🔥 FRP: {row['frp']} MW &nbsp;|&nbsp; "
                f"📊 Confidence: {row['confidence']}% &nbsp;|&nbsp; "
                f"📅 {row['acq_date']}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Full data table ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📋 View All Detected Fires (raw data)"):
        show_cols = ["acq_date", "acq_time", "latitude", "longitude",
                     "brightness", "frp", "confidence", "satellite",
                     "fire_type", "alert_level"]
        show_cols = [c for c in show_cols if c in df_live.columns]
        st.dataframe(df_live[show_cols].sort_values("frp", ascending=False),
                     use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        "⬇️ Download Full Alert Report",
        df_live.to_csv(index=False).encode("utf-8"),
        f"india_fire_alerts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True,
    )


def tab_historical(model, scaler, df):
    st.subheader("📂 Historical Fire Explorer (2022–2024)")
    st.caption("Match sensor values to the closest real MODIS fire event in the dataset.")

    st.sidebar.title("⚙️ Sensor Controls")
    brightness = st.sidebar.slider("🌡 Brightness (K)",      200.0, 500.0, 320.0, step=1.0)
    bright_t31 = st.sidebar.slider("🌡 Brightness T31 (K)",  250.0, 350.0, 295.0, step=1.0)
    frp        = st.sidebar.slider("🔥 FRP (MW)",              0.0, 100.0,  20.0, step=0.5)
    scan       = st.sidebar.number_input("📡 Scan",  0.5, 10.0, 1.0, step=0.1)
    track      = st.sidebar.number_input("📡 Track", 0.5, 10.0, 1.0, step=0.1)
    confidence = st.sidebar.selectbox("📊 Confidence", list(CONFIDENCE_MAP))

    col1, col2 = st.columns([1, 4])
    with col1:
        predict = st.button("🔍 Find Match", use_container_width=True)
    with col2:
        if st.button("🎲 Random Fire Event"):
            row = df.sample(1).iloc[0]
            label, color = predict_row(row, model, scaler)
            st.session_state.hist_result = {"row": row, "label": label, "color": color}

    if predict:
        best = find_closest_match(df, brightness, bright_t31, frp, scan, track, confidence)
        label, color = predict_row(best, model, scaler)
        st.session_state.hist_result = {"row": best, "label": label, "color": color}

    if "hist_result" in st.session_state:
        res   = st.session_state.hist_result
        row   = res["row"]
        label = res["label"]
        color = res["color"]
        lat   = row["latitude"]
        lon   = row["longitude"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔥 Fire Type", label.split(" ", 1)[-1])
        c2.metric("📍 Latitude",  f"{lat:.4f}°")
        c3.metric("📍 Longitude", f"{lon:.4f}°")
        c4.metric("📅 Date",      str(row.get("acq_date", "N/A")))

        st.markdown("---")
        left, right = st.columns([3, 2])

        with left:
            st.subheader(label)
            margin = 1.5
            nearby = df[
                df["latitude"].between(lat - margin, lat + margin) &
                df["longitude"].between(lon - margin, lon + margin)
            ].head(50)
            render_historical_map(lat, lon, label, color, nearby)

        with right:
            st.subheader("📋 Event Details")
            details = {
                "Date":            row.get("acq_date",    "N/A"),
                "Brightness (K)":  f"{float(row.get('brightness', 0)):.1f}",
                "Bright T31 (K)":  f"{float(row.get('bright_t31', 0)):.1f}",
                "FRP (MW)":        f"{float(row.get('frp', 0)):.2f}",
                "Confidence":      str(row.get("confidence", "N/A")),
                "Satellite":       str(row.get("satellite",  "N/A")),
                "Day / Night":     str(row.get("daynight",   "N/A")),
            }
            for k, v in details.items():
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:7px 0;border-bottom:1px solid #333;'>"
                    f"<span style='color:#aaa'>{k}</span>"
                    f"<span style='font-weight:600'>{v}</span></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🗺 Fire Types")
            for _, (lbl, clr) in FIRE_TYPES.items():
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0;'>"
                    f"<div style='width:12px;height:12px;border-radius:50%;background:{clr};'>"
                    f"</div><span>{lbl}</span></div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("👈 Adjust sensor values and click **Find Match** to explore historical fire events.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.markdown(
        "Real-time fire detection & classification for India using NASA MODIS satellite data. "
        "Switch between **Live Mode** for active fire alerts and **Historical Mode** to explore past events."
    )
    st.markdown("---")

    model  = load_model()
    scaler = load_scaler()

    if model is None or scaler is None:
        st.error("❌ Model or scaler could not be loaded.")
        st.stop()

    df_hist = load_historical_data()

    live_tab, hist_tab = st.tabs(["🔴 Live Fire Alerts", "📂 Historical Explorer"])

    with live_tab:
        tab_live(model, scaler)

    with hist_tab:
        tab_historical(model, scaler, df_hist)

    st.markdown(
        "<div style='text-align:center;margin-top:3em;color:#555;'>"
        "🛰 Powered by NASA FIRMS MODIS NRT & Streamlit</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()