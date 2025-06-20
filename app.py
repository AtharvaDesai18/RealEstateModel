import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os
import random

# --- Page Setup ---
st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")
st.title("🏠 Real Estate Price Predictor")
st.write("🚀 Fill in property details to estimate the price.")

# --- Load Trained Model ---
model = None

if os.path.exists("xgboost.pkl"):
    try:
        with open("xgboost.pkl", "rb") as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.warning("⚠️ Model file `xgboost.pkl` not found.")

# --- Input Form ---
st.subheader("📋 Enter Property Details")

with st.form("predict_form"):
    city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Pune", "Chennai", "Kolkata"])
    area = st.number_input("Area (in sq ft)", min_value=100, max_value=10000, value=1000)
    bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2)
    submit = st.form_submit_button("Predict Price")

# --- Prediction ---
if submit:
    if model:
        input_df = pd.DataFrame([{
            "city": city.title().strip(),
            "area": area,
            "bhk": bhk
        }])
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"💸 Estimated Price: ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
    else:
        st.error("🚫 Model not loaded. Please check `xgboost.pkl`")

# --- Heatmap (Dummy Sample) ---
st.subheader("📍 Heatmap of Property Prices (Sample)")

sample_data = pd.DataFrame({
    "lat": [19.076, 28.6139, 12.9716, 17.3850, 18.5204, 13.0827, 22.5726],
    "lon": [72.8777, 77.2090, 77.5946, 78.4867, 73.8567, 80.2707, 88.3639],
    "city": ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Pune", "Chennai", "Kolkata"],
    "price": [random.randint(40, 100) * 1e5 for _ in range(7)]
})

fig = px.density_mapbox(
    sample_data, lat="lat", lon="lon", z="price", radius=30,
    center={"lat": 20.5937, "lon": 78.9629}, zoom=4,
    mapbox_style="carto-positron", hover_name="city"
)
st.plotly_chart(fig)

# --- EMI Calculator ---
st.subheader("🏦 EMI Calculator")

loan_amount = st.number_input("Loan Amount (₹)", value=5000000)
interest_rate = st.slider("Interest Rate (%)", 5.0, 15.0, 8.5)
loan_years = st.slider("Loan Tenure (Years)", 5, 30, 20)

if st.button("Calculate EMI"):
    r = interest_rate / 1200
    n = loan_years * 12
    try:
        emi = loan_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)
        st.info(f"💳 Monthly EMI: ₹{emi:,.2f}")
    except Exception as e:
        st.error(f"EMI Calculation Error: {e}")


# --- ROI Calculator ---
st.subheader("📈 ROI Calculator (Rental Return)")

col1, col2 = st.columns(2)
with col1:
    investment = st.number_input("Total Investment (₹)", min_value=100000, value=5000000, step=50000)
with col2:
    rent = st.number_input("Monthly Rental Income (₹)", min_value=0, value=20000, step=1000)

if st.button("Calculate ROI"):
    try:
        annual_income = rent * 12
        roi = (annual_income / investment) * 100
        st.success(f"📊 Annual ROI: {roi:.2f}%")
    except Exception as e:
        st.error(f"ROI Calculation Error: {e}")
