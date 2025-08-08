import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from geopy.distance import geodesic
import os
import gdown

# Google Drive file URLs
MODEL_URL = "https://drive.google.com/uc?id=1ZcohAUBMQTB9hG3YGYNedtxzDN_5cHZF"
SCALER_URL = "https://drive.google.com/uc?id=18FjzK0oepVCJ43hUOuXK79bzUEY6bOk_"
MODEL_PATH = "best_fire_detection_model.pkl"
SCALER_PATH = "scaler.pkl"

# Load model and scaler from Google Drive
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        gdown.download(SCALER_URL, SCALER_PATH, quiet=False)
        return joblib.load(SCALER_PATH)

@st.cache_data
def load_all_years():
    df_2021 = pd.read_csv("modis_2021_India.csv")
    df_2022 = pd.read_csv("modis_2022_India.csv")
    df_2023 = pd.read_csv("modis_2023_India.csv")
    df = pd.concat([df_2021, df_2022, df_2023], ignore_index=True)
    df.dropna(subset=["latitude", "longitude"], inplace=True)
    return df

fire_types = {
    0: ("🌳 Vegetation Fire", "#28a745"),
    1: ("🔥 Industrial/Urban Fire", "#ffc107"),
    2: ("🏭 Other Static Land Source", "#6c757d"),
    3: ("🌊 Offshore Fire", "#17a2b8")
}
confidence_map = {"low": 0, "nominal": 1, "high": 2}

def infer_fire_type_from_row(row, model, scaler):
    try:
        X = np.array([[row['brightness'], row['bright_t31'], row['frp'], row['scan'], row['track'], confidence_map.get(str(row['confidence']).lower(), 1)]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        label, _ = fire_types.get(pred, ("Unknown", "gray"))
        if label == "🌳 Vegetation Fire" and (row["latitude"] <= 10.0 or row["longitude"] >= 92.0):
            label = "🌊 Offshore Fire"
        return label
    except:
        return "Prediction Error"

def find_closest_fire(df, lat, lon, model=None, scaler=None, radius_km=20):
    lat_min, lat_max = lat - 0.2, lat + 0.2
    lon_min, lon_max = lon - 0.2, lon + 0.2
    nearby = df[(df["latitude"].between(lat_min, lat_max)) & (df["longitude"].between(lon_min, lon_max))].copy()
    if nearby.empty:
        return pd.DataFrame()
    nearby["distance_km"] = nearby.apply(lambda row: geodesic((lat, lon), (row["latitude"], row["longitude"])).km, axis=1)
    nearby = nearby[nearby["distance_km"] <= radius_km].sort_values("distance_km")
    if model and scaler:
        nearby["inferred_fire_type"] = nearby.apply(lambda row: infer_fire_type_from_row(row, model, scaler), axis=1)
    return nearby

def show_map(lat, lon, fire_label, color):
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker([lat, lon], popup=f"{fire_label} at ({lat:.4f}, {lon:.4f})", icon=folium.Icon(color="red", icon="fire", prefix="fa")).add_to(m)
    folium.Circle([lat, lon], radius=50000, color=color, fill=True, fill_color=color, opacity=0.2).add_to(m)
    title_html = f"""
        <div style="position: absolute; top: 10px; left: 50%; transform: translateX(-50%); z-index:9999; background-color:{color}; color:white; padding:6px 16px; border-radius:8px; font-size:14px; font-weight:bold;">
            {fire_label}
        </div>"""
    m.get_root().html.add_child(folium.Element(title_html))
    st.markdown("""<style>.map-container {margin-bottom:-40px;}</style>""", unsafe_allow_html=True)
    with st.container():
        st_folium(m, width="100%", height=420, returned_objects=[])
        st.markdown(f"<div style='margin-top:-15px; text-align:center; font-size:14px; color:white;'>📍 <b>Location:</b> {lat:.4f}, {lon:.4f}</div>", unsafe_allow_html=True)

st.set_page_config(page_title="🔥 Fire Type Classifier", layout="wide")
st.title("🔥 Fire Type Classifier")
st.markdown("Predict fire types using MODIS satellite readings and compare with real data.")

model = load_model()
scaler = load_scaler()

# Show where the files came from
st.toast("✅ Model loaded from " + ("disk cache" if os.path.exists(MODEL_PATH) else "Google Drive"))
st.toast("✅ Scaler loaded from " + ("disk cache" if os.path.exists(SCALER_PATH) else "Google Drive"))
df_all_years = load_all_years()

if model is None or scaler is None:
    st.error("❌ Model or scaler could not be loaded. Prediction disabled.")
    st.stop()

st.sidebar.title("⚙️ Controls")
brightness = st.sidebar.slider("Brightness", 200.0, 500.0, 300.0)
bright_t31 = st.sidebar.slider("Brightness T31", 250.0, 350.0, 290.0)
frp = st.sidebar.slider("FRP", 0.0, 100.0, 15.0)
scan = st.sidebar.number_input("Scan", 0.5, 10.0, 1.0)
track = st.sidebar.number_input("Track", 0.5, 10.0, 1.0)
confidence = st.sidebar.selectbox("Confidence", list(confidence_map))

if "coords" not in st.session_state:
    sample_row = df_all_years.sample(1).iloc[0]
    st.session_state.coords = {"use_random": True, "lat": sample_row["latitude"], "lon": sample_row["longitude"]}

st.session_state.coords["use_random"] = st.sidebar.checkbox("Use Random Coordinates", value=st.session_state.coords["use_random"])

if st.session_state.coords["use_random"]:
    lat, lon = st.session_state.coords["lat"], st.session_state.coords["lon"]
    st.sidebar.write(f"Random location: ({lat:.4f}, {lon:.4f})")
else:
    lat = st.sidebar.number_input("Latitude", -90.0, 90.0, st.session_state.coords.get("lat", 20.5937))
    lon = st.sidebar.number_input("Longitude", -180.0, 180.0, st.session_state.coords.get("lon", 78.9629))
    st.session_state.coords.update({"lat": lat, "lon": lon})

compare_real = st.sidebar.checkbox("📍 Compare with real fire data (2021–2023)", value=True)

if st.button("🔍 Predict Fire Type"):
    if not (8.4 <= lat <= 37.6 and 68.7 <= lon <= 97.25):
        st.error("❌ Coordinates outside India, please enter coordinates within India.")
        st.stop()

    confidence_val = confidence_map[confidence]
    X_input = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]
    label, color = fire_types.get(pred, ("Unknown", "gray"))

    if label == "🌳 Vegetation Fire" and (lat <= 10.0 or lon >= 92.0):
        label, color = fire_types[3]

    st.session_state.prediction = {
        "label": label,
        "color": color,
        "lat": lat,
        "lon": lon,
        "inputs": {
            "Brightness": brightness,
            "Brightness T31": bright_t31,
            "FRP": frp,
            "Scan": scan,
            "Track": track,
            "Confidence": confidence
        }
    }

if "prediction" in st.session_state and (8.4 <= st.session_state.prediction["lat"] <= 37.6 and 68.7 <= st.session_state.prediction["lon"] <= 97.25):
    p = st.session_state.prediction
    st.success(f"✅ Fire predicted at location: **{p['lat']:.4f}, {p['lon']:.4f}** — {p['label']}")

    nearby_fires = pd.DataFrame()
    if compare_real:
        st.subheader("🔎 Real Fire Events Nearby (2021–2023)")
        nearby_fires = find_closest_fire(df_all_years, p["lat"], p["lon"], model, scaler)
        if not nearby_fires.empty:
            st.dataframe(nearby_fires[["acq_date", "latitude", "longitude", "brightness", "bright_t31", "frp", "confidence", "distance_km", "inferred_fire_type"]].head(5), use_container_width=True)
            show_map(p["lat"], p["lon"], p["label"], p["color"])
        else:
            st.warning("No recorded fire events within 20 km.")
            st.markdown("""
                <div style="border: 1px dashed #ccc; padding: 20px; text-align: center; border-radius: 8px; background-color: #f9f9f9;">
                    🔍 No nearby MODIS fire data found for this location and time period.
                </div>
            """, unsafe_allow_html=True)

    result_df = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **p["inputs"],
        "Latitude": p["lat"],
        "Longitude": p["lon"],
        "Predicted Fire Type": p["label"]
    }])
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Prediction Report", csv, "fire_prediction.csv", "text/csv")

st.markdown("<div style='text-align: center; margin-top: 1em; color: gray'>🔥 Powered by MODIS & Streamlit | @ Gourav Barnwal</div>", unsafe_allow_html=True)
