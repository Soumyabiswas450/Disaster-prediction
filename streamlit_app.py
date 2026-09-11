import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Optional workaround for some Anaconda/Windows OpenMP conflicts.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Multi-Hazard Weather & Hazard Monitor",
    page_icon="🌍",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FLOOD_DIR = os.path.join(BASE_DIR, "Flood train")

FLOOD_MODEL_PATH = os.path.join(FLOOD_DIR, "live_flood_model.pkl")
FLOOD_SCALER_PATH = os.path.join(FLOOD_DIR, "live_flood_scaler.pkl")
FLOOD_FEATURES_PATH = os.path.join(FLOOD_DIR, "live_flood_features.pkl")
CYCLONE_DATASET_PATH = os.path.join(BASE_DIR, "cyclone_dataset.csv")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


@st.cache_resource
def load_models_and_data():
    f_model = joblib.load(FLOOD_MODEL_PATH)
    f_scaler = joblib.load(FLOOD_SCALER_PATH)
    f_features = joblib.load(FLOOD_FEATURES_PATH)

    df_cyc = pd.read_csv(CYCLONE_DATASET_PATH)
    X_cyc = df_cyc.drop(columns=["Cyclone", "Pre_existing_Disturbance"])
    y_cyc = df_cyc["Cyclone"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_cyc, y_cyc, test_size=0.2, random_state=42, stratify=y_cyc
    )

    c_scaler = StandardScaler()
    X_train_scaled = c_scaler.fit_transform(X_train)

    c_model = RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
    )
    c_model.fit(X_train, y_train)

    return (
        f_model,
        f_scaler,
        f_features,
        c_model,
        c_scaler,
        list(X_cyc.columns),
    )


def geocode_city(city):
    r = requests.get(
        GEOCODING_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=15,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise ValueError(f"Location not found: {city}")
    return results[0]


def get_weather(lat, lon):
    r = requests.get(
        WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,precipitation,"
                "pressure_msl,wind_speed_10m,wind_gusts_10m,wind_direction_10m,weather_code"
            ),
            "hourly": (
                "precipitation,rain,temperature_2m,relative_humidity_2m,"
                "pressure_msl,wind_speed_10m,wind_gusts_10m"
            ),
            "past_hours": 48,
            "forecast_hours": 48,
            "timezone": "auto",
        },
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def weather_values(weather, location):
    current = weather["current"]
    hourly = weather["hourly"]

    times = pd.to_datetime(hourly["time"])
    rain = np.asarray(hourly["precipitation"], dtype=float)
    rain = np.nan_to_num(rain, nan=0.0, posinf=0.0, neginf=0.0)

    now = pd.Timestamp(current["time"])

    past = rain[times <= now]
    future = rain[times > now]

    def total(arr, n=None):
        if n is None:
            return float(arr.sum())
        return float(arr[-n:].sum()) if len(arr) else 0.0

    lat = float(location["latitude"])
    lon = float(location["longitude"])
    temp = float(current["temperature_2m"])
    wind_speed = float(current["wind_speed_10m"])

    dyn_wind_shear = max(5.0, 30.0 - (wind_speed * 0.3))
    dyn_vorticity = 0.000001 + (max(0.0, wind_speed - 15) * 0.000003)

    return {
        "temperature": temp,
        "sea_surface_temperature": temp + 1.5,
        "humidity": float(current["relative_humidity_2m"]),
        "pressure": float(current["pressure_msl"]),
        "wind_speed": wind_speed,
        "wind_gust": float(current["wind_gusts_10m"]),
        "wind_direction": float(current["wind_direction_10m"]),
        "wind_shear": dyn_wind_shear,
        "vorticity": dyn_vorticity,
        "latitude": lat,
        "ocean_depth": (
            300.0 if (70 <= lon <= 95 and 5 <= lat <= 25) else 2000.0
        ),
        "proximity_to_coastline": 0.85,
        "current_rain": float(current["precipitation"]),
        "rainfall_1h": total(past, 1),
        "rainfall_3h": total(past, 3),
        "rainfall_6h": total(past, 6),
        "rainfall_24h": total(past, 24),
        "rainfall_48h": total(past, 48),
        "forecast_1h": total(future[:1]),
        "forecast_3h": total(future[:3]),
        "forecast_6h": total(future[:6]),
        "forecast_24h": total(future[:24]),
        "weather_code": int(current["weather_code"]),
        "time": current["time"],
    }


def calculate_thunderstorm_probability(w):
    code = w["weather_code"]
    gust = w["wind_gust"]
    humidity = w["humidity"]
    current_rain = w["current_rain"]

    if code in [95, 96, 99]:
        base_prob = 0.85
    elif code in [80, 81, 82]:
        base_prob = 0.50
    elif code in [63, 65]:
        base_prob = 0.35
    else:
        base_prob = 0.10

    gust_factor = min(0.25, max(0.0, (gust - 20.0) / 60.0))
    humidity_factor = 0.15 if humidity > 80 else 0.0

    total_prob = min(0.99, base_prob + gust_factor + humidity_factor)
    if current_rain > 5.0:
        total_prob = min(0.99, total_prob + 0.1)

    return float(total_prob)


def get_realistic_cyclone_probability(w, ml_probability):
    wind = w["wind_speed"]
    pressure = w["pressure"]

    if pressure > 1008 and wind < 25:
        return float(ml_probability * 0.15)

    return float(np.clip(ml_probability, 0.01, 0.99))


def render_weather_map(lat, lon, city_name, temp):
    map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
    st.map(map_data, zoom=10, use_container_width=True)


def norm(x):
    return (
        str(x)
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "percent")
        .replace("°", "")
        .replace("/", "")
    )


def map_feature(feature, w):
    n = norm(feature)
    exact = {
        "seasurfacetemperature": w["sea_surface_temperature"],
        "atmosphericpressure": w["pressure"],
        "pressure": w["pressure"],
        "pressurehpa": w["pressure"],
        "humidity": w["humidity"],
        "humiditypercent": w["humidity"],
        "relativehumidity": w["humidity"],
        "windshear": w["wind_shear"],
        "vorticity": w["vorticity"],
        "latitude": w["latitude"],
        "oceandepth": w["ocean_depth"],
        "proximitytocoastline": w["proximity_to_coastline"],
        "rainfall(mm)": w["rainfall_24h"],
        "rainfallmm": w["rainfall_24h"],
        "rainfall": w["rainfall_24h"],
        "temperature(°c)": w["temperature"],
        "temperaturec": w["temperature"],
        "temperature": w["temperature"],
        "windspeed": w["wind_speed"],
        "windspeedkmh": w["wind_speed"],
        "windspeed10m": w["wind_speed"],
        "windgust": w["wind_gust"],
        "windgustkmh": w["wind_gust"],
        "winddirection": w["wind_direction"],
        "currentrain": w["current_rain"],
        "currentprecipitation": w["current_rain"],
        "rainfall1h": w["rainfall_1h"],
        "rainfall3h": w["rainfall_3h"],
        "rainfall6h": w["rainfall_6h"],
        "rainfall24h": w["rainfall_24h"],
        "rainfall48h": w["rainfall_48h"],
    }
    if n in exact:
        return exact[n]

    if "seasurface" in n or "sst" in n:
        return w["sea_surface_temperature"]
    if "temperature" in n:
        return w["temperature"]
    if "humidity" in n:
        return w["humidity"]
    if "pressure" in n:
        return w["pressure"]
    if "shear" in n:
        return w["wind_shear"]
    if "vorticity" in n:
        return w["vorticity"]
    if "latitude" in n:
        return w["latitude"]
    if "depth" in n:
        return w["ocean_depth"]
    if "coast" in n or "proximity" in n:
        return w["proximity_to_coastline"]
    if "gust" in n:
        return w["wind_gust"]
    if "wind" in n and "direction" in n:
        return w["wind_direction"]
    if "wind" in n:
        return w["wind_speed"]
    if "rain" in n or "precip" in n:
        if "48" in n:
            return w["rainfall_48h"]
        if "24" in n:
            return w["rainfall_24h"]
        if "6" in n:
            return w["rainfall_6h"]
        if "3" in n:
            return w["rainfall_3h"]
        if "1" in n:
            return w["rainfall_1h"]
        return w["rainfall_24h"]
    return None


def make_input(features, w):
    values = []
    missing = []
    for f in features:
        value = map_feature(f, w)
        if value is None:
            missing.append(f)
        values.append(value)
    if missing:
        return None, missing
    return pd.DataFrame([values], columns=list(features)), []


def predict(model, scaler, features, w):
    x, missing = make_input(features, w)
    if missing:
        return None, missing, x
    xs = scaler.transform(x)
    probs = model.predict_proba(xs)[0]
    if 1 in model.classes_:
        idx = list(model.classes_).index(1)
    else:
        idx = int(np.argmax(probs))
    return float(probs[idx]), [], x


def weather_code_text(code):
    mapping = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        51: "Light drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return mapping.get(int(code), "Unknown")


st.title("🌍 Multi-Hazard Weather & Disaster Monitor")
st.write(
    "Live weather dashboard integrating Flood, Cyclone, and Thunderstorm risk"
    " analysis."
)

required_files = [
    FLOOD_MODEL_PATH,
    FLOOD_SCALER_PATH,
    FLOOD_FEATURES_PATH,
    CYCLONE_DATASET_PATH,
]
missing = [p for p in required_files if not os.path.exists(p)]
if missing:
    st.error("Missing required model or dataset files:\n\n" + "\n".join(missing))
    st.stop()

try:
    (
        flood_model,
        flood_scaler,
        flood_features,
        cyclone_model,
        cyclone_scaler,
        cyclone_features,
    ) = load_models_and_data()
except Exception as e:
    st.error(f"Could not load models or dataset: {e}")
    st.stop()

st.sidebar.header("📍 Location")
city = st.sidebar.text_input("City", "Kolkata")
run = st.sidebar.button(
    "🌐 Get Live Data & Predict", type="primary", use_container_width=True
)

if run:
    try:
        with st.spinner("Fetching live weather & atmospheric data..."):
            location = geocode_city(city)
            weather = get_weather(location["latitude"], location["longitude"])
            w = weather_values(weather, location)

        st.subheader(f"📍 {location['name']}, {location.get('country', '')}")
        st.caption(
            f"Coordinates: {location['latitude']:.4f}, {location['longitude']:.4f} |"
            f" API time: {w['time']} | Condition:"
            f" {weather_code_text(w['weather_code'])}"
        )

        st.markdown("### 🌤️ Current Conditions")
        a, b, c, d, e = st.columns(5)
        a.metric("Temperature", f"{w['temperature']:.1f} °C")
        b.metric("Humidity", f"{w['humidity']:.0f}%")
        c.metric("Current Rain", f"{w['current_rain']:.1f} mm")
        d.metric("Pressure", f"{w['pressure']:.0f} hPa")
        e.metric("Wind Speed", f"{w['wind_speed']:.1f} km/h")

        f1, f2 = st.columns(2)
        f1.metric("Wind Gusts", f"{w['wind_gust']:.1f} km/h")
        f2.metric("Weather Summary", weather_code_text(w["weather_code"]))

        st.markdown("### 🗺️ Interactive Weather Map")
        render_weather_map(
            location["latitude"], location["longitude"], location["name"], w["temperature"]
        )

        st.markdown("### 🌧️ Rainfall History")
        a, b, c, d, e = st.columns(5)
        a.metric("Last 1h", f"{w['rainfall_1h']:.1f} mm")
        b.metric("Last 3h", f"{w['rainfall_3h']:.1f} mm")
        c.metric("Last 6h", f"{w['rainfall_6h']:.1f} mm")
        d.metric("Last 24h", f"{w['rainfall_24h']:.1f} mm")
        e.metric("Last 48h", f"{w['rainfall_48h']:.1f} mm")

        st.markdown("### 🔮 Rainfall Forecast")
        a, b, c, d = st.columns(4)
        a.metric("Next 1h", f"{w['forecast_1h']:.1f} mm")
        b.metric("Next 3h", f"{w['forecast_3h']:.1f} mm")
        c.metric("Next 6h", f"{w['forecast_6h']:.1f} mm")
        d.metric("Next 24h", f"{w['forecast_24h']:.1f} mm")

        st.divider()
        st.subheader("🚨 Hazard Predictions (Flood, Cyclone & Thunderstorm)")

        flood_p, flood_missing, flood_input = predict(
            flood_model, flood_scaler, flood_features, w
        )

        raw_cyclone_p, cyclone_missing, cyclone_input = predict(
            cyclone_model, cyclone_scaler, cyclone_features, w
        )
        if raw_cyclone_p is not None:
            cyclone_p = get_realistic_cyclone_probability(w, raw_cyclone_p)
        else:
            cyclone_p = None

        thunderstorm_p = calculate_thunderstorm_probability(w)

        col_f, col_c, col_t = st.columns(3)

        with col_f:
            st.markdown("#### 🌧️ Flood")
            if flood_p is None:
                st.warning("Flood prediction unavailable.")
            else:
                st.metric("Flood Probability", f"{flood_p * 100:.2f}%")
                st.progress(flood_p)

        with col_c:
            st.markdown("#### 🌀 Cyclone")
            if cyclone_p is None:
                st.warning("Cyclone prediction unavailable.")
            else:
                st.metric("Cyclone Probability", f"{cyclone_p * 100:.2f}%")
                st.progress(cyclone_p)

        with col_t:
            st.markdown("#### ⚡ Thunderstorm")
            st.metric("Thunderstorm Probability", f"{thunderstorm_p * 100:.2f}%")
            st.progress(thunderstorm_p)

        available = {}
        if flood_p is not None:
            available["Flood"] = flood_p
        if cyclone_p is not None:
            available["Cyclone"] = cyclone_p
        available["Thunderstorm"] = thunderstorm_p

        if available:
            dominant = max(available, key=available.get)
            st.divider()
            st.subheader("🎯 Highest Model Signal")
            st.metric("Hazard", dominant)
            st.metric("Model Probability", f"{available[dominant] * 100:.2f}%")
            st.caption(
                "This is the highest individual hazard output among Flood, Cyclone,"
                " and Thunderstorm evaluations."
            )

    except requests.RequestException as e:
        st.error(f"Weather API request failed: {e}")
    except Exception as e:
        st.error(f"Could not complete prediction: {e}")
else:
    st.markdown("### How to use")
    st.write("Enter a city on the left and click **Get Live Data & Predict**.")