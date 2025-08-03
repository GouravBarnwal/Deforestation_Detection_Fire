import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random
from datetime import datetime
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from geopy.distance import geodesic
import io
import os
import requests
from pathlib import Path

# Load secrets
GDRIVE_BASE_URL = "https://drive.google.com/uc?export=download&id="

# === 🔽 Helper: Download large files from Google Drive ===
def download_if_missing(url, filename):
    try:
        if not os.path.exists(filename):
            # Create directory if it doesn't exist
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # For Google Drive links, handle the virus scan warning
            if 'drive.google.com' in url:
                response = requests.get(url, stream=True, timeout=30)
                
                # Handle Google Drive's virus scan warning
                for key, value in response.cookies.items():
                    if 'download_warning' in key:
                        params = {'confirm': value, 'id': url.split('id=')[-1]}
                        response = requests.get('https://drive.google.com/uc', params=params, stream=True)
                        break
                
                response.raise_for_status()
                
                # Save the file with progress
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                progress_bar = st.progress(0)
                downloaded_size = 0
                
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                progress = min(100, int((downloaded_size / total_size) * 100))
                                progress_bar.progress(progress)
                
                progress_bar.empty()
                return True
            
            # For direct URLs
            else:
                with requests.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                return True
        return False
    except Exception as e:
        st.error(f"Error downloading {filename}: {str(e)}")
        return False

def get_gdrive_url(file_id):
    return f"{GDRIVE_BASE_URL}{file_id}"

@st.cache_resource
def ensure_dependencies():
    try:
        # Get file IDs from secrets
        secrets = st.secrets.get("gdrive", {})
        
        # Define files to download with their paths and IDs
        files_to_download = [
            ("models/best_fire_detection_model.pkl", secrets.get("model_id")),
            ("models/scaler.pkl", secrets.get("scaler_id")),
            ("data/modis_2021_India.csv", secrets.get("data_2021_id")),
            ("data/modis_2022_India.csv", secrets.get("data_2022_id")),
            ("data/modis_2023_India.csv", secrets.get("data_2023_id"))
        ]
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Download files with progress
        for filename, file_id in files_to_download:
            if not file_id:
                st.error(f"Missing file ID for {filename} in secrets")
                continue
                
            url = get_gdrive_url(file_id)
            with st.spinner(f"Downloading {os.path.basename(filename)}..."):
                if not download_if_missing(url, filename):
                    st.error(f"Failed to download {filename}")
    except Exception as e:
        st.error(f"Error in ensure_dependencies: {str(e)}")

# Initialize dependencies
ensure_dependencies()

@st.cache_resource
def load_model():
    model_path = "models/best_fire_detection_model.pkl"
    try:
        if not os.path.exists(model_path):
            st.warning("Model file not found. Attempting to download...")
            ensure_dependencies()
            if not os.path.exists(model_path):
                st.error("Failed to download the model file. Please check your internet connection and try again.")
                return None
        
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    scaler_path = "models/scaler.pkl"
    try:
        if not os.path.exists(scaler_path):
            st.warning("Scaler file not found. Attempting to download...")
            ensure_dependencies()
            if not os.path.exists(scaler_path):
                st.error("Failed to download the scaler file. Please check your internet connection and try again.")
                return None
        return joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Error loading the scaler: {str(e)}")
        return None

@st.cache_resource
def load_all_years():
    data_files = [
        "data/modis_2021_India.csv",
        "data/modis_2022_India.csv",
        "data/modis_2023_India.csv"
    ]
    
    # Check if all data files exist
    missing_files = [f for f in data_files if not os.path.exists(f)]
    if missing_files:
        st.warning("Some data files are missing. Attempting to download...")
        ensure_dependencies()
        
        # Check again after attempting download
        missing_files = [f for f in data_files if not os.path.exists(f)]
        if missing_files:
            st.error(f"Failed to download the following data files: {', '.join(missing_files)}")
            return None
    
    try:
        df = pd.concat([pd.read_csv(f) for f in data_files], ignore_index=True)
        df.dropna(subset=["latitude", "longitude"], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        return None

fire_types = {
    0: ("🌳 Vegetation Fire", "#28a745"),
    1: ("🔥 Industrial/Urban Fire", "#ffc107"),
    2: ("🌍 Other Static Land Source", "#6c757d"),
    3: ("🌊 Offshore Fire", "#17a2b8")
}
confidence_map = {"low": 0, "nominal": 1, "high": 2}

def is_offshore_region(lat, lon):
    return (
        (8.4 <= lat <= 23.0 and 68.7 <= lon <= 77.0) or
        (8.4 <= lat <= 22.0 and 77.0 <= lon <= 92.0) or
        (10.0 <= lat <= 15.0 and 92.0 <= lon <= 97.25)
    )

def predict_fire_type(model, scaler, features, lat, lon):
    X_input = np.array([features])
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]

    if is_offshore_region(lat, lon):
        if features[1] >= 290 and features[2] >= 15:
            pred = 3

    return pred

def find_closest_fire(df, lat, lon, radius_km=20):
    lat_min, lat_max = lat - 0.2, lat + 0.2
    lon_min, lon_max = lon - 0.2, lon + 0.2
    subset = df[(df["latitude"].between(lat_min, lat_max)) & (df["longitude"].between(lon_min, lon_max))].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["distance_km"] = subset.apply(lambda row: geodesic((lat, lon), (row["latitude"], row["longitude"]))
        .km, axis=1)
    subset["lat_rounded"] = subset["latitude"].round(2)
    subset["lon_rounded"] = subset["longitude"].round(2)
    return subset[subset["distance_km"] <= radius_km].sort_values("distance_km")

def show_map(lat, lon, fire_label, color):
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker(
        [lat, lon],
        popup=f"{fire_label} at ({lat:.4f}, {lon:.4f})",
        icon=folium.Icon(color="red", icon="fire", prefix="fa")
    ).add_to(m)
    folium.Circle([lat, lon], radius=50000, color=color, fill=True, fill_color=color, opacity=0.2).add_to(m)
    st_folium(m, width="100%", height=400)
    st.markdown(f"**📍 Location:** `{lat:.4f}, {lon:.4f}`")

st.set_page_config(page_title="🔥 Fire Type Classifier", layout="wide")
st.title("🔥 Fire Type Classifier")
st.markdown("Predict fire types using MODIS satellite readings and compare with real data.")

model = load_model()
scaler = load_scaler()
df_all_years = load_all_years()

if model is None or scaler is None:
    st.error("⚠️ Model or scaler not found.")
    st.stop()

st.sidebar.title("⚙️ Controls")
brightness = st.sidebar.slider("Brightness", 200.0, 500.0, 300.0)
bright_t31 = st.sidebar.slider("Brightness T31", 250.0, 350.0, 290.0)
frp = st.sidebar.slider("FRP", 0.0, 100.0, 15.0)
scan = st.sidebar.number_input("Scan", 0.5, 10.0, 1.0)
track = st.sidebar.number_input("Track", 0.5, 10.0, 1.0)
confidence = st.sidebar.selectbox("Confidence", list(confidence_map))

if "coords" not in st.session_state:
    st.session_state.coords = {
        "use_random": True,
        "lat": round(random.uniform(8.4, 37.6), 4),
        "lon": round(random.uniform(68.7, 97.25), 4)
    }

st.session_state.coords["use_random"] = st.sidebar.checkbox("Use Random Coordinates", value=st.session_state.coords["use_random"])

if st.session_state.coords["use_random"]:
    lat = st.session_state.coords["lat"]
    lon = st.session_state.coords["lon"]
    st.sidebar.write(f"📍 Random location: ({lat}, {lon})")
else:
    lat = st.sidebar.number_input("Latitude", -90.0, 90.0, st.session_state.coords.get("lat", 20.5937))
    lon = st.sidebar.number_input("Longitude", -180.0, 180.0, st.session_state.coords.get("lon", 78.9629))
    st.session_state.coords.update({"lat": lat, "lon": lon})

predict = st.button("🔍 Predict Fire Type")

if not (8.4 <= lat <= 37.6 and 68.7 <= lon <= 97.25):
    st.error("❌ Location is outside India.")
    if st.button("🔁 Reset to Random Coordinates"):
        st.session_state.coords = {
            "use_random": True,
            "lat": round(random.uniform(8.4, 37.6), 4),
            "lon": round(random.uniform(68.7, 97.25), 4)
        }
    st.stop()

compare_real = st.sidebar.checkbox("📍 Compare with real fire data (2021–2023)", value=True)

if predict:
    features = [brightness, bright_t31, frp, scan, track, confidence_map[confidence]]
    pred = predict_fire_type(model, scaler, features, lat, lon)
    label, color = fire_types.get(pred, ("Unknown", "gray"))

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

if "prediction" in st.session_state:
    p = st.session_state.prediction
    st.subheader(f"📌 Predicted Fire Type: {p['label']}")
    show_map(p["lat"], p["lon"], p["label"], p["color"])

    if compare_real:
        st.subheader("🔎 Real Fire Events Nearby")
        nearby_fires = find_closest_fire(df_all_years, p["lat"], p["lon"])

        if not nearby_fires.empty:
            real_fires = nearby_fires.copy()
            real_fires["lat_rounded"] = real_fires["latitude"].round(2)
            real_fires["lon_rounded"] = real_fires["longitude"].round(2)
            pred_lat, pred_lon = round(p["lat"], 2), round(p["lon"], 2)
            match = real_fires[(real_fires["lat_rounded"] == pred_lat) & (real_fires["lon_rounded"] == pred_lon)]

            if not match.empty:
                st.markdown("✅ **Matching historical fire events found within 2 decimal places.**")
                st.dataframe(match[["acq_date", "latitude", "longitude", "brightness", "bright_t31", "frp", "confidence", "type"]])
            else:
                st.warning("⚠️ The predicted fire type does **not match** any real fire events recorded at this location (to 2 decimal precision).")
                st.markdown(
                    "> 🔍 _Model prediction is based on input features and general patterns in data, "
                    "which may not always align with sparse MODIS historical data._")
        else:
            st.info("📭 No recorded fire events within 20 km in 2021–2023.")

st.markdown("<hr><center>🔥 Powered by MODIS, Streamlit, and Machine Learning @ Project by Gourav Barnwal </center>", unsafe_allow_html=True)
