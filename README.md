# 🌍 Multi-Hazard Weather & Disaster Prediction System

A machine-learning-powered **Multi-Hazard Weather & Disaster Monitoring System** developed during the **Venomix Hackathon**.

The project combines live weather data with machine learning and rule-based analysis to estimate the risk of **Floods, Cyclones, and Thunderstorms** for a selected location.

> 🚀 Built during an intense **12-hour overnight hackathon** and selected as **one of the 15 finalist teams**.

---

## 🚨 What Does It Do?

The application allows users to enter a city and retrieve live weather information before generating hazard-risk estimates.

### Supported Hazards

* 🌧️ **Flood Risk**
* 🌀 **Cyclone Risk**
* ⚡ **Thunderstorm Risk**

The dashboard also provides:

* 🌡️ Current temperature
* 💧 Humidity
* 🌧️ Current precipitation
* 💨 Wind speed and gusts
* 🧭 Wind direction
* 🌡️ Atmospheric pressure
* 🗺️ Interactive location map
* 📊 Recent rainfall history
* 🔮 Rainfall forecast
* 🚨 Hazard probability estimates
* 🎯 Highest model signal among the evaluated hazards

---

## 🧠 System Overview

```text
                 User
                   │
                   ▼
          Enter City / Location
                   │
                   ▼
          Geocoding API
                   │
                   ▼
        Live Weather Data API
                   │
                   ▼
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
     Flood      Cyclone    Thunderstorm
     Model       Model       Analysis
       │           │           │
       └───────────┼───────────┘
                   ▼
          Risk Probability
                   │
                   ▼
        Streamlit Dashboard
```

---

## 🤖 Machine Learning

### 🌧️ Flood Prediction

The flood model uses weather-related features including:

* Rainfall
* Temperature
* Humidity

A **Random Forest Classifier** is trained to estimate flood occurrence probability.

The trained artifacts include:

* `live_flood_model.pkl`
* `live_flood_scaler.pkl`
* `live_flood_features.pkl`

---

### 🌀 Cyclone Prediction

The cyclone model uses atmospheric and geographical features such as:

* Sea Surface Temperature
* Atmospheric Pressure
* Humidity
* Wind Shear
* Vorticity
* Latitude
* Ocean Depth
* Proximity to Coastline

The project uses a **Random Forest Classifier** for cyclone-risk estimation.

The training experiments also compare:

* Logistic Regression
* Random Forest
* Gradient Boosting

---

### ⚡ Thunderstorm Analysis

Thunderstorm probability is calculated using a combination of current weather conditions, including:

* Weather condition code
* Wind gusts
* Humidity
* Current rainfall

The system combines these signals to produce an estimated thunderstorm probability.

---

## 🌐 Live Weather Integration

The application retrieves real-time weather information using the **Open-Meteo API**.

The system obtains:

* Current weather conditions
* Historical rainfall for the previous 48 hours
* Rainfall forecasts for the next 48 hours
* Temperature
* Humidity
* Pressure
* Wind speed
* Wind gusts
* Wind direction
* Weather condition codes

City names are converted into geographic coordinates using the Open-Meteo Geocoding API.

---

## 🖥️ Dashboard

The application is built using **Streamlit** and provides an interactive dashboard.

### Main sections

**📍 Location**

Enter a city to retrieve its geographical and weather information.

**🌤️ Current Conditions**

Displays temperature, humidity, rainfall, pressure and wind information.

**🗺️ Interactive Weather Map**

Displays the selected location on an interactive map.

**🌧️ Rainfall History**

Shows rainfall accumulated over:

* 1 hour
* 3 hours
* 6 hours
* 24 hours
* 48 hours

**🔮 Rainfall Forecast**

Shows expected rainfall over:

* 1 hour
* 3 hours
* 6 hours
* 24 hours

**🚨 Hazard Predictions**

Displays estimated probabilities for:

* Flood
* Cyclone
* Thunderstorm

---

## 🛠️ Tech Stack

### Programming

* Python

### Machine Learning

* Scikit-learn
* Random Forest
* Logistic Regression
* Gradient Boosting
* StandardScaler

### Data Processing

* Pandas
* NumPy

### Model Persistence

* Joblib

### Web Application

* Streamlit

### Weather & Location Data

* Open-Meteo Weather API
* Open-Meteo Geocoding API

### Visualization

* Streamlit Maps
* Matplotlib
* Seaborn

---

## 📁 Project Structure

```text
Disaster-prediction/
│
├── streamlit_app.py
├── requirements.txt
├── cyclone_dataset.csv
│
├── Flood code/
│   └── flood.ipynb
│
├── Cyclone Code/
│   └── Cyclone.ipynb
│
├── landslide code/
│   └── landslide.ipynb
│
├── Flood train/
│   ├── live_flood_model.pkl
│   ├── live_flood_scaler.pkl
│   └── live_flood_features.pkl
│
├── landslide train/
│   └── offline_landslide_model.json
│
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Soumyabiswas450/Disaster-prediction.git
cd Disaster-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser.

---

## 📊 Dataset

The project contains datasets for different hazard prediction tasks.

### Cyclone Dataset

The cyclone dataset contains atmospheric and geographical variables including:

```text
Sea_Surface_Temperature
Atmospheric_Pressure
Humidity
Wind_Shear
Vorticity
Latitude
Ocean_Depth
Proximity_to_Coastline
Pre_existing_Disturbance
Cyclone
```

### Flood Dataset

The flood model was developed using weather variables such as:

```text
Rainfall
Temperature
Humidity
```

---

## 🎯 Hackathon Context

This project was developed as part of the **Venomix Hackathon**, an intensive **12-hour overnight development challenge**.

Despite the limited development time, our team built an end-to-end prototype combining:

* Machine Learning
* Live weather APIs
* Data processing
* Risk analysis
* Interactive visualization
* Streamlit deployment

🏆 **Selected as one of the 15 finalist teams.**

---

## 👥 Team

- [**Soumya Biswas**](https://github.com/Soumyabiswas450)
- [**Soumya Sharma**](https://github.com/SoumyaSha2003)
- [**Saptarsha Nandan**](https://github.com/chele7)

> 🫂 Couldn't have done it without the team's collaboration, support, and late-night problem solving.

---

## 🔮 Future Improvements

Potential future improvements include:

* Integration of more disaster types
* Better calibrated probability estimates
* Larger and more diverse real-world datasets
* Satellite and remote-sensing data integration
* Historical disaster-event data
* Geographic risk heatmaps
* Automated weather alerts
* Time-series forecasting
* Explainable AI for model predictions
* Improved regional models for different geographical areas

---

## ⚠️ Disclaimer

This project is a **hackathon prototype for educational and experimental purposes**.

The displayed probabilities should not be treated as official disaster warnings or as a replacement for information from government agencies, meteorological departments, or emergency services.

---

## 📜 License

This project is intended for educational and research purposes.
