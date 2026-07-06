import streamlit as st
import pandas as pd
import joblib
import time
from math import radians, cos, sin, asin, sqrt
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Road Accident Risk Predictor",
    page_icon="🚦",
    layout="centered"
)

# ------------------------------------------------------------------
# STYLING
# ------------------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        text-align: center;
    }
    .risk-high {
        background-color: #3a1414;
        border: 1px solid #ff4b4b;
        color: #ff8080;
    }
    .risk-medium {
        background-color: #3a2f14;
        border: 1px solid #ffb84b;
        color: #ffcf80;
    }
    .risk-low {
        background-color: #143a1c;
        border: 1px solid #4bff6a;
        color: #80ffa0;
    }
    .risk-title {
        font-size: 1.1rem;
        opacity: 0.85;
        margin-bottom: 0.3rem;
    }
    .risk-value {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .stat-box {
        background-color: #1a1d24;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# LOAD MODELS + DATA (cached so it only loads once)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("models/query_risk_model.pkl")
    encoder = joblib.load("models/risk_level_encoder.pkl")
    season_encoder = joblib.load("models/season_encoder.pkl")
    return model, encoder, season_encoder

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/cleaned_accident_data.csv")
    return df

@st.cache_resource
def get_geolocator():
    return Nominatim(user_agent="accident_risk_project", timeout=10)

query_model, risk_encoder, season_encoder = load_models()
df_query = load_data()
geolocator = get_geolocator()

# ------------------------------------------------------------------
# HELPER FUNCTIONS (same logic as the notebook)
# ------------------------------------------------------------------
def geocode_road(road, city):
    try:
        location = geolocator.geocode(f"{road}, {city}, India")
        time.sleep(1)
        if location:
            return location.latitude, location.longitude
        return None
    except (GeocoderTimedOut, GeocoderUnavailable):
        return None
    except Exception:
        return None

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def find_nearby_accidents(df, lat, lon, radius_km=2):
    df = df.copy()
    df["distance_km"] = df.apply(
        lambda row: haversine(lat, lon, row["latitude"], row["longitude"]), axis=1
    )
    return df[df["distance_km"] <= radius_km]

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

def predict_risk_for_road(road, city, df_query, model, encoder, season_encoder, query_time=None):
    coords = geocode_road(road, city)
    if coords is None:
        return {"error": "Location not found. Please check the road and city name."}

    lat, lon = coords
    if query_time is None:
        query_time = pd.Timestamp.now()

    hour = query_time.hour
    weekday = query_time.weekday()
    month = query_time.month
    is_weekend = int(weekday in [5, 6])
    is_peak_hour = int(hour in [8, 9, 18, 19, 20])
    season_str = get_season(month)
    season_encoded = season_encoder.transform([season_str])[0]

    nearby_all = find_nearby_accidents(df_query, lat, lon, radius_km=2)
    is_hotspot = (
        int(nearby_all["is_hotspot"].mode()[0])
        if not nearby_all.empty and "is_hotspot" in nearby_all.columns
        else 0
    )

    features = pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "hour": hour,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "month": month,
        "weekday": weekday,
        "is_hotspot": is_hotspot,
        "season_encoded": season_encoded
    }])

    pred_encoded = model.predict(features)[0]
    risk_level = encoder.inverse_transform([pred_encoded])[0]

    nearby = find_nearby_accidents(df_query, lat, lon)
    accident_count = len(nearby)
    peak_hours = (
        nearby["hour"].value_counts().head(3).to_dict()
        if not nearby.empty else {}
    )

    return {
        "road": road,
        "city": city,
        "lat": lat,
        "lon": lon,
        "predicted_risk_level": risk_level,
        "nearby_accident_count": accident_count,
        "peak_hours_from_history": peak_hours
    }

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🚦 Road Accident Risk Predictor")
st.markdown(
    "Enter a road and city in India to get a predicted accident risk level, "
    "based on a Random Forest model trained on historical accident data."
)

with st.form("risk_form"):
    col1, col2 = st.columns(2)
    with col1:
        road = st.text_input("Road name", placeholder="e.g. MG Road")
    with col2:
        city = st.text_input("City", placeholder="e.g. Bangalore")
    submitted = st.form_submit_button("Check Risk", use_container_width=True)

if submitted:
    if not road or not city:
        st.warning("Please enter both a road name and a city.")
    else:
        with st.spinner("Locating road and calculating risk..."):
            result = predict_risk_for_road(
                road, city, df_query, query_model, risk_encoder, season_encoder
            )

        if "error" in result:
            st.error(result["error"])
        else:
            risk = result["predicted_risk_level"]
            risk_class = {
                "High": "risk-high",
                "Medium": "risk-medium",
                "Low": "risk-low"
            }.get(risk, "risk-medium")

            st.markdown(f"""
                <div class="risk-card {risk_class}">
                    <div class="risk-title">Predicted Risk Level for {result['road']}, {result['city']}</div>
                    <div class="risk-value">{risk}</div>
                </div>
            """, unsafe_allow_html=True)

            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                    <div class="stat-box">
                        <div style="font-size:0.85rem;opacity:0.7;">Nearby Historical Accidents</div>
                        <div style="font-size:1.6rem;font-weight:700;">{result['nearby_accident_count']}</div>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                if result["peak_hours_from_history"]:
                    peak_str = ", ".join(f"{h}:00" for h in result["peak_hours_from_history"].keys())
                else:
                    peak_str = "No data"
                st.markdown(f"""
                    <div class="stat-box">
                        <div style="font-size:0.85rem;opacity:0.7;">Peak Accident Hours (History)</div>
                        <div style="font-size:1.3rem;font-weight:700;">{peak_str}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.map(pd.DataFrame({"lat": [result["lat"]], "lon": [result["lon"]]}), zoom=13)

            with st.expander("How is this calculated?"):
                st.markdown("""
                This prediction uses a Random Forest model trained only on
                features that are knowable *before* an accident happens —
                location, time of day, day of week, month, season, and
                whether the area is a known accident hotspot (from DBSCAN
                clustering on historical data). Historical accident counts
                and peak hours shown above are drawn from nearby records
                within a 2 km radius, for context.
                """)

st.markdown("---")
st.caption("Built with Streamlit · Random Forest · OpenStreetMap Nominatim geocoding")
