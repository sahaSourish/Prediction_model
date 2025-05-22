
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load datasets
agri_df = pd.read_csv("agriculture_dataset.csv")
crop_df = pd.read_csv("Crop_recommendation.csv")

# Preprocess agriculture dataset
agri_df.drop(['Farm_ID', 'Irrigation_Type', 'Water_Usage(cubic meters)'], axis=1, inplace=True)

# Add default values from crop dataset to agriculture dataset for matching columns
agri_df['N'] = crop_df['N'][:len(agri_df)]
agri_df['P'] = crop_df['P'][:len(agri_df)]
agri_df['K'] = crop_df['K'][:len(agri_df)]
agri_df['ph'] = crop_df['ph'][:len(agri_df)]
agri_df['temperature'] = crop_df['temperature'][:len(agri_df)]
agri_df['humidity'] = crop_df['humidity'][:len(agri_df)]
agri_df['rainfall'] = crop_df['rainfall'][:len(agri_df)]

# Encode categorical columns
label_encoders = {}
for col in ['Crop_Type', 'Season', 'Soil_Type']:
    if col in agri_df.columns:
        le = LabelEncoder()
        agri_df[col] = le.fit_transform(agri_df[col])
        label_encoders[col] = le

# Define features and targets
features = ['Crop_Type', 'Farm_Area(acres)', 'Season', 'Soil_Type', 'N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall']
X = agri_df[features]
y = agri_df[['Fertilizer_Used(tons)', 'Pesticide_Used(kg)', 'Yield(tons)']]

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Smart Farm Planner")
st.title("Smart Farm Resource, Yield & Investment Predictor")

# Input selections
crop = st.selectbox("Select Crop Type", label_encoders['Crop_Type'].classes_)
season = st.selectbox("Select Season", label_encoders['Season'].classes_)
soil = st.selectbox("Select Soil Type", label_encoders['Soil_Type'].classes_)
area = st.number_input("Enter Farm Area (in acres)", min_value=0.1, step=0.1)

st.subheader("Soil and Environment Parameters")
N = st.number_input("Nitrogen content (N)", min_value=0.0, value=50.0)
P = st.number_input("Phosphorus content (P)", min_value=0.0, value=50.0)
K = st.number_input("Potassium content (K)", min_value=0.0, value=50.0)
ph = st.number_input("pH level of soil", min_value=0.0, max_value=14.0, value=6.5)
temperature = st.number_input("Average Temperature (°C)", value=25.0)
humidity = st.number_input("Average Humidity (%)", value=70.0)
rainfall = st.number_input("Rainfall (mm)", value=100.0)

st.subheader("Market Prices")
fertilizer_price = st.number_input("Fertilizer cost (Rs/kg)", min_value=0.0, value=25.0)
pesticide_price = st.number_input("Pesticide cost (Rs/kg)", min_value=0.0, value=40.0)
crop_price = st.number_input("Expected selling price of crop (Rs/kg)", min_value=0.0, value=20.0)

if st.button("Predict"):
    crop_encoded = label_encoders['Crop_Type'].transform([crop])[0]
    season_encoded = label_encoders['Season'].transform([season])[0]
    soil_encoded = label_encoders['Soil_Type'].transform([soil])[0]

    input_data = pd.DataFrame([[crop_encoded, area, season_encoded, soil_encoded, N, P, K, ph, temperature, humidity, rainfall]],
                              columns=features)
    prediction = model.predict(input_data)[0]
    fert_needed, pest_needed, yield_pred = prediction

    # Investment calculations
    fert_cost = fert_needed * 1000 * fertilizer_price
    pest_cost = pest_needed * pesticide_price
    income = yield_pred * 1000 * crop_price
    profit = income - (fert_cost + pest_cost)

    # Results
    st.subheader("Predicted Results:")
    st.write(f"**Fertilizer Needed:** {round(fert_needed, 2)} tons")
    st.write(f"**Pesticide Needed:** {round(pest_needed, 2)} kg")
    st.write(f"**Expected Yield:** {round(yield_pred, 2)} tons")

    st.subheader("Investment and Income Estimate:")
    st.write(f"Estimated Fertilizer Cost: ₹{fert_cost:,.2f}")
    st.write(f"Estimated Pesticide Cost: ₹{pest_cost:,.2f}")
    st.write(f"Expected Income from Crop Sale: ₹{income:,.2f}")
    st.write(f"**Net Profit:** ₹{profit:,.2f}")
