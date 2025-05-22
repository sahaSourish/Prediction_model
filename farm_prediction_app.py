import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


# Crop conditions reference data
crop_conditions = [
    {"crop": "Rice", "soil": "Clayey loam, alluvial, moisture-retentive", "temperature_c": [25, 35], "rainfall_cm": [100, 300], "season": "Kharif", "notes": "Requires standing water; best on level land."},
    {"crop": "Wheat", "soil": "Well-drained loamy or clay loam", "temperature_c": [10, 25], "rainfall_cm": [50, 100], "season": "Rabi", "notes": "Cool growing season with dry harvesting conditions."},
    {"crop": "Maize", "soil": "Fertile, well-drained sandy loam", "temperature_c": [21, 27], "rainfall_cm": [50, 100], "season": "Kharif", "notes": "Sensitive to waterlogging; needs full sun."},
    {"crop": "Sugarcane", "soil": "Deep, well-drained loamy soil with neutral pH", "temperature_c": [20, 30], "rainfall_cm": [100, 250], "season": "Annual", "notes": "Requires long frost-free period and plenty of water."},
    {"crop": "Cotton", "soil": "Black cotton soil or well-drained loam", "temperature_c": [21, 30], "rainfall_cm": [50, 100], "season": "Kharif", "notes": "Warm, dry weather needed for maturity."},
    {"crop": "Groundnut", "soil": "Sandy loam or loamy soil with good drainage", "temperature_c": [25, 30], "rainfall_cm": [50, 100], "season": "Kharif", "notes": "Susceptible to waterlogging; needs well-aerated soil."},
    {"crop": "Soybean", "soil": "Fertile, well-drained alluvial soil", "temperature_c": [20, 30], "rainfall_cm": [60, 100], "season": "Kharif", "notes": "Fixes nitrogen; suited to moderate rainfall regions."},
    {"crop": "Barley", "soil": "Well-drained loamy to sandy soil", "temperature_c": [12, 25], "rainfall_cm": [30, 90], "season": "Rabi", "notes": "Grows in dry, cool climate with light irrigation."},
    {"crop": "Bajra", "soil": "Sandy loam and black soil", "temperature_c": [20, 35], "rainfall_cm": [30, 65], "season": "Kharif", "notes": "Tolerant to drought and high temperatures."},
    {"crop": "Sorghum", "soil": "Loamy to clay soils, tolerates salinity", "temperature_c": [12, 34], "rainfall_cm": [30, 100], "season": "Kharif", "notes": "Drought-resistant; suitable for semi-arid zones."},
    {"crop": "Cumin", "soil": "Well-drained sandy loam", "temperature_c": [20, 30], "rainfall_cm": [30, 50], "season": "Rabi", "notes": "Requires dry climate during flowering and maturity."},
    {"crop": "Cowpea", "soil": "Light sandy to loamy soil", "temperature_c": [20, 30], "rainfall_cm": [40, 60], "season": "Kharif", "notes": "Can grow in poor soils; drought-tolerant."},
    {"crop": "Coffee", "soil": "Loamy soil rich in organic matter", "temperature_c": [15, 28], "rainfall_cm": [150, 250], "season": "Kharif", "notes": "Requires shade and well-distributed rainfall."},
    {"crop": "Tea", "soil": "Acidic, well-drained loamy soil", "temperature_c": [20, 30], "rainfall_cm": [200, 300], "season": "Year-round", "notes": "Prefers humid, high-rainfall regions with shade."},
    {"crop": "Banana", "soil": "Rich loamy soil with good drainage", "temperature_c": [26, 30], "rainfall_cm": [100, 200], "season": "Year-round", "notes": "Requires high humidity and well-distributed rainfall."},
    {"crop": "Papaya", "soil": "Well-drained sandy loam soil", "temperature_c": [22, 35], "rainfall_cm": [100, 150], "season": "Kharif", "notes": "Sensitive to frost; needs protection from strong winds."},
    {"crop": "Pineapple", "soil": "Light, well-drained sandy loam", "temperature_c": [20, 30], "rainfall_cm": [100, 150], "season": "Kharif", "notes": "Requires good drainage and high humidity."},
    {"crop": "Tomato", "soil": "Sandy loam rich in organic matter", "temperature_c": [20, 27], "rainfall_cm": [50, 100], "season": "Kharif/Rabi", "notes": "Requires moderate climate and good sunlight."}
]


def display_crop_conditions(selected_crop):
    for item in crop_conditions:
        if item["crop"].lower() == selected_crop.lower():
            st.subheader("Ideal Conditions for Selected Crop")
            st.write(f"**Crop:** {item['crop']}")
            st.write(f"**Preferred Soil:** {item['soil']}")
            st.write(f"**Ideal Temperature Range:** {item['temperature_c'][0]}–{item['temperature_c'][1]} °C")
            st.write(f"**Ideal Rainfall Range:** {item['rainfall_cm'][0]}–{item['rainfall_cm'][1]} cm/year")
            st.write(f"**Season:** {item['season']}")
            st.write(f"**Notes:** {item['notes']}")
            break

# Load datasets
agri_df = pd.read_csv("agriculture_dataset.csv")
crop_df = pd.read_csv("Crop_recommendation.csv")

# Normalize crop label for comparison
crop_df['label'] = crop_df['label'].str.strip().str.lower()

# Drop unused columns
agri_df.drop(['Farm_ID', 'Irrigation_Type', 'Water_Usage(cubic meters)'], axis=1, inplace=True)

# Add default values from crop dataset to agriculture dataset for matching columns
for col in ['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall']:
    agri_df[col] = crop_df[col][:len(agri_df)]

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

# Train separate stacking models for each target
models = {}
targets = ['Fertilizer_Used(tons)', 'Pesticide_Used(kg)', 'Yield(tons)']
for target in targets:
    stacking = StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(random_state=42)),
            ('gb', GradientBoostingRegressor(random_state=42))
        ],
        final_estimator=Ridge()
    )
    stacking.fit(X, y[target])
    models[target] = stacking

# Streamlit UI
st.set_page_config(page_title="Smart Farm Planner")
st.title("Smart Farm Resource, Yield & Investment Predictor")

# Input selections
crop = st.selectbox("Select Crop Type", label_encoders['Crop_Type'].classes_)

display_crop_conditions(crop)
season = st.selectbox("Select Season", label_encoders['Season'].classes_)
soil = st.selectbox("Select Soil Type", label_encoders['Soil_Type'].classes_)
area = st.number_input("Enter Farm Area (in acres)", min_value=0.1, step=0.1)

st.subheader("Soil and Environment Parameters")
N = st.number_input("Nitrogen content (N)", min_value=0.0, value=90.0)
P = st.number_input("Phosphorus content (P)", min_value=0.0, value=42.0)
K = st.number_input("Potassium content (K)", min_value=0.0, value=43.0)
ph = st.number_input("pH level of soil", min_value=0.0, max_value=14.0, value=6.5)
temperature = st.number_input("Average Temperature (°C)", value=27.5)
humidity = st.number_input("Average Humidity (%)", value=80.0)
rainfall = st.number_input("Rainfall (mm)", value=210.0)

st.subheader("Market Prices")
fertilizer_price = st.number_input("Fertilizer cost (Rs/kg)", min_value=0.0, value=25.0)
pesticide_price = st.number_input("Pesticide cost (Rs/kg)", min_value=0.0, value=40.0)
crop_price = st.number_input("Expected selling price of crop (Rs/kg)", min_value=0.0, value=20.0)

if st.button("Predict"):
    # Encode user inputs
    crop_encoded = label_encoders['Crop_Type'].transform([crop])[0]
    season_encoded = label_encoders['Season'].transform([season])[0]
    soil_encoded = label_encoders['Soil_Type'].transform([soil])[0]

    input_data = pd.DataFrame([[crop_encoded, area, season_encoded, soil_encoded, N, P, K, ph, temperature, humidity, rainfall]],
                              columns=features)
    fert_needed = models['Fertilizer_Used(tons)'].predict(input_data)[0]
    pest_needed = models['Pesticide_Used(kg)'].predict(input_data)[0]
    yield_pred = models['Yield(tons)'].predict(input_data)[0]

    fert_cost = fert_needed * 1000 * fertilizer_price
    pest_cost = pest_needed * pesticide_price
    income = yield_pred * 1000 * crop_price
    profit = income - (fert_cost + pest_cost)

    st.subheader("Predicted Results:")
    st.write(f"Fertilizer Needed: {round(fert_needed, 2)} tons")
    st.write(f"Pesticide Needed: {round(pest_needed, 2)} kg")
    st.write(f"Expected Yield: {round(yield_pred, 2)} tons")

    st.subheader("Investment and Income Estimate:")
    st.write(f"Estimated Fertilizer Cost: ₹{fert_cost:,.2f}")
    st.write(f"Estimated Pesticide Cost: ₹{pest_cost:,.2f}")
    st.write(f"Expected Income from Crop Sale: ₹{income:,.2f}")
    st.write(f"Net Profit: ₹{profit:,.2f}")

    st.subheader("Crop Input Suitability Check")
    crop_normalized = crop.strip().lower()
    crop_match = crop_df[crop_df['label'] == crop_normalized]
    if not crop_match.empty:
        ideal_values = crop_match[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean()
        input_values = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        similarity = cosine_similarity([ideal_values], input_values)[0][0] * 100

        st.write(f"Input similarity with ideal conditions for {crop}: {similarity:.2f}%")

        if similarity >= 80:
            st.success("Input conditions are well-aligned with recommended crop requirements.")
        elif similarity >= 60:
            st.warning("Input conditions are moderately aligned. Consider minor adjustments.")
        else:
            st.error("Input conditions differ significantly from ideal crop parameters.")
    else:
        st.warning("Could not find ideal parameters for the selected crop.")

    # Suggest best crops based on profitability and yield
    st.subheader("Recommended Crops Based on Profitability and Yield on Provided Soil and Other Environmental Parameters")
    crop_suggestions = []
    for crop_name in label_encoders['Crop_Type'].classes_:
        c_encoded = label_encoders['Crop_Type'].transform([crop_name])[0]
        row = pd.DataFrame([[c_encoded, area, season_encoded, soil_encoded, N, P, K, ph, temperature, humidity, rainfall]],
                           columns=features)
        yld = models['Yield(tons)'].predict(row)[0]
        fert = models['Fertilizer_Used(tons)'].predict(row)[0]
        pest = models['Pesticide_Used(kg)'].predict(row)[0]
        inc = yld * 1000 * crop_price
        cost = fert * 1000 * fertilizer_price + pest * pesticide_price
        prof = inc - cost

        crop_norm = crop_name.strip().lower()
        match = crop_df[crop_df['label'] == crop_norm]
        if not match.empty:
            ideal_vals = match[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean()
            sim = cosine_similarity([ideal_vals], [[N, P, K, temperature, humidity, ph, rainfall]])[0][0] * 100
        else:
            sim = 0

        crop_suggestions.append({
            'Crop': crop_name,
            'Yield': round(yld, 2),
            'Profit': round(prof, 2),
            'Similarity': round(sim, 1)
        })

    sorted_suggestions = sorted(crop_suggestions, key=lambda x: (x['Profit'], x['Yield'], x['Similarity']), reverse=True)[:5]
    st.table(pd.DataFrame(sorted_suggestions))
