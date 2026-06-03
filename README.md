# 🔥 FireWatch India — Real-Time Fire Detection & Classification using NASA MODIS Satellite Data

> Real-time fire detection and classification for India using NASA MODIS satellite data and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square)
![NASA FIRMS](https://img.shields.io/badge/Data-NASA%20FIRMS%20MODIS-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/Model-Random%20Forest-brightgreen?style=flat-square)

---

## 🌐 Live App

👉 **[Open on Streamlit Cloud](https://grv-firewatch-india.streamlit.app/)**

---

## 📌 What It Does

This project has two modes:

### 🔴 Live Fire Alerts
- Fetches today's fire detections across India from the **NASA FIRMS MODIS NRT API** (updated every few hours)
- Runs each satellite detection through a trained **Random Forest classifier** to identify the fire type
- Assigns a **3-tier alert level** based on Fire Radiative Power (FRP) and confidence:

| Alert | Condition | Action |
|-------|-----------|--------|
| 🚨 CRITICAL | FRP ≥ 50 MW **and** Confidence ≥ 70% | Dispatch fire department immediately(yet to be added) |
| ⚠️ WARNING | FRP ≥ 20 MW **or** Confidence ≥ 70% | Monitor closely |
| ℹ️ MONITOR | Low intensity | Keep watching |

- Interactive map with markers sized by FRP intensity
- Download full alert report as CSV

### 📂 Historical Explorer
- Explore 233,981 real fire events from 2022–2024
- Adjust sensor sliders → finds the closest matching real MODIS fire event
- Shows location on map with fire type, date, brightness, FRP, and satellite details

---

## 🔥 Fire Types Classified

| Code | Type | Occurrence in Dataset |
|------|------|-----------------------|
| 0 | 🌳 Vegetation Fire | ~81% |
| 1 | 🔥 Industrial / Urban Fire | Rare (model trained via SMOTE) |
| 2 | 🏭 Other Static Land Source | ~15% |
| 3 | 🌊 Offshore Fire | ~4% |

---

## 📊 Dataset

| Year | Fire Detections | Date Range |
|------|----------------|------------|
| 2022 | 81,525 | Jan 2022 – Dec 2022 |
| 2023 | 78,425 | Jan 2023 – Dec 2023 |
| 2024 | 74,029 | Jan 2024 – Dec 2024 |
| **Total** | **233,979** | **3 full years** |

- **Source:** [NASA FIRMS MODIS Collection 6.1](https://firms.modaps.eosdis.nasa.gov/)
- **Region:** India (68.7°E – 97.25°E, 8.4°N – 37.6°N)
- **Features used:** `brightness`, `scan`, `track`, `confidence`, `bright_t31`, `frp`
- **Satellites:** Terra & Aqua (MODIS instrument)

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | scikit-learn — Random Forest Classifier |
| Class Imbalance | SMOTE (imbalanced-learn) |
| Live Data Feed | NASA FIRMS NRT API |
| Frontend | Streamlit |
| Maps | Folium + streamlit-folium |
| Deployment | Streamlit Cloud |

---

## 🚀 Run Locally

```bash
git clone https://github.com/GouravBarnwal/Deforestation_Detection_Fire.git
cd Deforestation_Detection_Fire
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** `best_fire_detection_model.pkl` is 439 MB — too large for GitHub.
> The app auto-downloads it from Google Drive on first launch.
> To download manually: [Google Drive Link](https://drive.google.com/drive/folders/1w2fvB54FvNihaIXJABpRGmffVX5vYCH0?usp=sharing)

---

## 📁 Project Structure

```
Deforestation_Detection_Fire/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── scaler.pkl                      # Feature scaler (saved from training)
│
├── modis_2022_India.csv            # MODIS fire data — 2022
├── modis_2023_India.csv            # MODIS fire data — 2023
├── modis_2024_India.csv            # MODIS fire data — 2024
│
├── Classification_of_Fire_Types_in_India_Using_MODIS_Satellite_Data.ipynb
│                                   # Full training notebook
├── evaluate_model.py               # Model evaluation script
├── predict_from_csv.py             # Batch prediction script
└── README.md
```

---

## 🧠 Model Details

- **Algorithm:** Random Forest Classifier
- **Training data:** MODIS 2022–2024 India fire detections
- **Class imbalance handling:** SMOTE oversampling (Industrial/Urban fires are rare in real data)
- **Feature order:** `brightness → scan → track → confidence → bright_t31 → frp`
- **Saved as:** `best_fire_detection_model.pkl` (439 MB)

---

## 👤 Author

**Gourav Barnwal**
[GitHub](https://github.com/GouravBarnwal)

---

## 📄 Data Credit

Fire data sourced from **NASA FIRMS (Fire Information for Resource Management System)**
— [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov/)
