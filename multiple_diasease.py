import streamlit as st
import numpy as np
import pandas as pd
import pickle

# === Load model with caching ===
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("weather_model.sav", "rb"))
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'weather_model.sav' not found in the directory.")
        st.stop()

weather_model = load_model()

# === Encoding Maps ===
cloud_cover_map = {"clear": 0, "partly cloudy": 1, "overcast": 2}
season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
location_map = {'inland': 0, 'coastal': 1, 'mountain': 2}
weather_map = {0: "Cloudy", 1: "Rainy", 2: "Sunny", 3: "Overcast", 4: "Snowy"}

# === Page Configuration ===
st.set_page_config(page_title="🌦️ Smart Weather Predictor", layout="centered", page_icon="🌤️")
st.title("🌦️ Smart Weather Type Prediction")

# === Sidebar ===
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("This app uses an AI model to predict the weather type.")
    st.markdown("Built by [Your Name](https://github.com/yourprofile)")
    uploaded_file = st.file_uploader("📁 Upload CSV for Bulk Prediction", type=["csv"])

# === Input Sliders ===
st.header("🔧 Enter Weather Parameters")

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature (°C)", -30.0, 50.0, 25.0)
    wind_speed = st.slider("Wind Speed (km/h)", 0.0, 150.0, 10.0)
    atmospheric_pressure = st.slider("Pressure (hPa)", 900.0, 1100.0, 1013.0)
    visibility = st.slider("Visibility (km)", 0.0, 50.0, 10.0)
with col2:
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
    precipitation = st.slider("Precipitation (%)", 0.0, 100.0, 10.0)
    uv_index = st.slider("UV Index", 0.0, 15.0, 5.0)

col3, col4, col5 = st.columns(3)
with col3:
    cloud_cover_input = st.selectbox("☁️ Cloud Cover", list(cloud_cover_map.keys()))
with col4:
    season_input = st.selectbox("🌱 Season", list(season_map.keys()))
with col5:
    location_input = st.selectbox("📍 Location", list(location_map.keys()))

# === Prediction Function ===
def predict_weather(data):
    arr = np.array(data).reshape(1, -1)
    prediction = weather_model.predict(arr)[0]
    try:
        confidence = weather_model.predict_proba(arr)[0]
        top_conf = np.max(confidence)
    except:
        top_conf = None
    return prediction, top_conf

# === Prediction Trigger ===
if st.button("🔍 Predict Weather Type"):
    input_data = [
        temperature,
        humidity,
        wind_speed,
        precipitation,
        cloud_cover_map[cloud_cover_input],
        atmospheric_pressure,
        uv_index,
        season_map[season_input],
        visibility,
        location_map[location_input]
    ]

    label, conf = predict_weather(input_data)
    label_text = weather_map.get(label, f"Label {label}")

    st.success(f"🌤️ Predicted Weather Type: **{label_text}**")
    if conf:
        st.progress(int(conf * 100))
        st.caption(f"Confidence: **{conf * 100:.2f}%**")

    # Live Input Summary
    with st.expander("📊 Input Summary"):
        st.write({
            "Temperature": temperature,
            "Humidity": humidity,
            "Wind Speed": wind_speed,
            "Precipitation": precipitation,
            "Cloud Cover": cloud_cover_input,
            "Pressure": atmospheric_pressure,
            "UV Index": uv_index,
            "Season": season_input,
            "Visibility": visibility,
            "Location": location_input
        })

    # CSV Report Download
    result_df = pd.DataFrame([{
        "Prediction": label_text,
        "Confidence": f"{conf * 100:.2f}%" if conf else "N/A",
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind Speed": wind_speed,
        "Precipitation": precipitation,
        "Cloud Cover": cloud_cover_input,
        "Pressure": atmospheric_pressure,
        "UV Index": uv_index,
        "Season": season_input,
        "Visibility": visibility,
        "Location": location_input
    }])
    csv = result_df.to_csv(index=False).encode()
    st.download_button("📥 Download Prediction Report", csv, "weather_prediction.csv", mime="text/csv")

# === Bulk CSV Prediction ===
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        df["Cloud Cover"] = df["Cloud Cover"].map(cloud_cover_map)
        df["Season"] = df["Season"].map(season_map)
        df["Location"] = df["Location"].map(location_map)

        features = [
            "Temperature", "Humidity", "Wind Speed", "Precipitation",
            "Cloud Cover", "Pressure", "UV Index", "Season", "Visibility", "Location"
        ]

        predictions = weather_model.predict(df[features])
        try:
            probs = weather_model.predict_proba(df[features])
            confs = np.max(probs, axis=1)
        except:
            confs = [None] * len(predictions)

        df["Prediction"] = [weather_map.get(p, p) for p in predictions]
        df["Confidence"] = [f"{c*100:.2f}%" if c else "N/A" for c in confs]

        st.subheader("📋 Bulk Prediction Results")
        st.dataframe(df)

        csv_bulk = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Bulk Results", csv_bulk, "bulk_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
