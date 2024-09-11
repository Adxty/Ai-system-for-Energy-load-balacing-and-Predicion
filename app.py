import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from streamlit_lottie import st_lottie
import requests

# Load the trained model
model = load_model("energy_consumption_model.h5")

# Load dataset for scaling
df = pd.read_csv("AEP_hourly_dataset_final.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df.set_index("Datetime", inplace=True)

# Feature Scaling (same as used during training)
sc = MinMaxScaler(feature_range=(0, 1))
sc.fit(df[["AEP_MW"]])

# Custom CSS for the app layout
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom, #f0f2f6, #ffffff);
        }
        .header-text {
            font-size: 2.5rem;
            color: #013A6B;
            font-weight: 700;
            text-align: center;
        }
        .info-text {
            font-size: 1.2rem;
            margin-top: 20px;
            text-align: center;
        }
        .prediction-result {
            font-size: 1.5rem;
            color: #4CAF50;
            font-weight: bold;
            margin-top: 20px;
            text-align: center;
        }
        .card {
            padding: 15px;
            background: #f5f5f5;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Header section
st.markdown("<div class='header-text'>Delhi Energy Consumption Prediction</div>", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class='info-text'>
This app predicts the energy consumption in Delhi for a specific day and time. Select the date and time, and we'll provide the predicted energy consumption.
</div>
""", unsafe_allow_html=True)

# Optional: Adding a Lottie animation to make the UI more engaging
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottie_url('https://assets9.lottiefiles.com/packages/lf20_w51pcehl.json')

st_lottie(lottie_animation, speed=1, height=200, key="energy_animation")

# Create a card-like layout for inputs
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Date input from user in yyyy/mm/dd format
    date_input = st.text_input("📅 Enter a date (YYYY/MM/DD)", datetime.today().strftime("%Y/%m/%d"))

    # Parse and validate the date input
    try:
        date_input = datetime.strptime(date_input, "%Y/%m/%d")
        st.success(f"Date entered: {date_input.strftime('%Y/%m/%d')}")
    except ValueError:
        st.error("Please enter a valid date in YYYY/MM/DD format.")

    # Optional sliders for more features
    hour_input = st.slider("⏰ Select the hour of the day (0-23)", 0, 23, 12)
    day_input = st.selectbox("📆 Select the day of the week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    st.markdown("</div>", unsafe_allow_html=True)

# Function to preprocess the input date for the model
def preprocess_input_date(input_date, hour):
    # Generate the past 60 days of data before the input date
    date_range = pd.date_range(end=input_date, periods=60, freq="D")
    
    # Get the recent data from the dataset
    recent_data = df.loc[df.index < pd.Timestamp(input_date)].tail(60)[["AEP_MW"]].values
    
    # If there's not enough data, fill with random or average values (fallback logic)
    if len(recent_data) < 60:
        st.warning("Not enough past data available, filling with average values.")
        recent_data = np.array([df["AEP_MW"].mean()] * 60).reshape(-1, 1)
    
    # Scale the data
    scaled_recent_data = sc.transform(recent_data)
    
    # Reshape to match the input format for the model
    X_test = np.array([scaled_recent_data])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_test

# Predict energy consumption
if st.button("🔮 Predict"):
    if isinstance(date_input, datetime):
        # Preprocess the date for model prediction
        input_data = preprocess_input_date(date_input, hour_input)
        
        # Predict using the model
        predicted_consumption = model.predict(input_data)
        
        # Inverse transform the result to get actual MW values
        predicted_consumption = sc.inverse_transform(predicted_consumption)[0][0]
        
        # Display the prediction in a visually striking way
        st.markdown(f"<div class='prediction-result'>Predicted Energy Consumption: **{predicted_consumption:.2f} MW** at {hour_input}:00 on {date_input.strftime('%Y/%m/%d')}</div>", unsafe_allow_html=True)
        
        # Optional: Retrieve and display actual energy consumption if available
        actual_data = df.loc[df.index.date == date_input.date()]
        if not actual_data.empty:
            actual_consumption = actual_data["AEP_MW"].mean()
            st.write(f"Actual Energy Consumption on {date_input.strftime('%Y/%m/%d')}: **{actual_consumption:.2f} MW**")
        else:
            st.write("No actual data available for the selected date.")
        
        # Plot the predicted vs actual (if available)
        st.subheader("Energy Consumption Prediction vs Actual (if available)")
        actual_vals = [actual_consumption] if 'actual_consumption' in locals() else []
        pred_vals = [predicted_consumption]
        labels = ['Predicted'] + (['Actual'] if actual_vals else [])
        values = pred_vals + actual_vals
        
        # Enhance the bar chart appearance
        st.bar_chart(pd.DataFrame(values, index=labels, columns=['MW']))
    else:
        st.error("Please enter a valid date to proceed.")
