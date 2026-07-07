# Traffic Accident Hotspot Prediction

This is a data science project that analyzes historical road accident data from Indian cities. It identifies accident hotspots and predicts the risk level for any specified road and city, all through a live Streamlit web app.

🔗 **Live demo:** [https://roadtrafficpredict.streamlit.app]

## Project Overview

This project covers the entire process:

1. **Data cleaning & feature engineering**: This includes parsing dates and times, and creating features such as `hour`, `day_of_week`, `is_weekend`, `is_peak_hour`, `season`, and others from raw accident records.
2. **Exploratory data analysis**: This step involves understanding how accidents are distributed by time, location, and conditions.
3. **Hotspot detection**: DBSCAN clustering is used on accident coordinates to identify areas with a high density of past accidents (`is_hotspot`).
4. **Risk classification**: Two Random Forest models are created:
   - A **full model** that uses all available accident-scene features (traffic density, visibility, temperature, etc.) to understand which factors most impact accident risk.
   - A **query model** that uses only features knowable in advance (location, time, day, season, hotspot status). This model powers live predictions because scene-specific details aren’t available for a hypothetical road or time query.
5. **Live prediction app**: This Streamlit app allows a user to enter a road and city. The app geocodes the input and returns a predicted risk level along with historical accident context.

## Why Two Models?

The full model achieves around 91% accuracy but relies on features like traffic density or visibility that are known only after an accident occurs. Therefore, it cannot be used for real-time risk queries. The query model, while limited to features that can be known beforehand, is better suited for this purpose. However, its accuracy is lower, around 43%. Both models are included in the notebook for transparency about this tradeoff.

## Results

### Full Model (all accident-scene features)
- **Accuracy:** ~91%
- **Top features by importance:**
  1. `visibility` — 0.446
  2. `traffic_density` — 0.164
  3. `is_peak_hour` — 0.105
  4. `hour` — 0.057
  5. `longitude` / `latitude` — ~0.050 each

This indicates that accident risk in this dataset is influenced more by road and weather conditions than by location or time of day alone.

### Query Model (location/time features only)
- **Accuracy:** ~43%
- Trained on: `latitude`, `longitude`, `hour`, `is_weekend`, `is_peak_hour`, `month`, `weekday`, `is_hotspot`, `season_encoded`
- Adding `is_hotspot` and `season` improved accuracy only slightly (43.78% → 43.1%). This is primarily because DBSCAN currently classifies almost all locations as hotspots, which is a known limitation mentioned below.

### Sample Prediction

Querying **"MG Road", "Bangalore"** produces:

```json
{
  "road": "MG Road",
  "city": "Bangalore",
  "predicted_risk_level": "Medium",
  "nearby_accident_count": 16,
  "peak_hours_from_history": {"19": 4, "13": 3, "11": 2}
}
```

## Repository Structure
traffic_hotspot_project/
├── final_project.ipynb          # Main analysis + modeling notebook
├── app.py                       # Streamlit web app
├── requirements.txt             # Python dependencies
├── models/                      # Saved trained models
├── data/processed/               # Cleaned dataset
└── notebooks/                   # Exploratory / scratch notebooks

## Running the App Locally ```bash git clone https://github.com/khannashivansh03coder/traffic-accident-hotspot-prediction.git cd traffic-accident-hotspot-prediction python3 -m pip install -r requirements.txt python3 -m streamlit run app.py ```

## Technology Stack
- **Python** - pandas, scikit-learn, geopy
- **Classification** - Random Forest Classifier, DBSCAN clustering
- **Geocoding** - Nominatim / OpenCage**App/UI** — Streamlit
- **Deployment** – Streamlit Community Cloud

### Limitations

Accuracy of query model (~43%) shows difficulty to predict risk from location/time only, without scene-specific conditions.
- Geocoding may not work sometimes for less common road names.
- The data set is restricted to certain Indian cities / states and time periods.

# Future Improvement

- Hotspot detection tuning by DBSCAN parameters
- Increase the number of queryable features for better accuracy of query model. - Add cross-validation and hyperparameter tuning.