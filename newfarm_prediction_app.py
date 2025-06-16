import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from streamlit import cache_resource

# Load datasets
agri_df = pd.read_csv("agriculture_dataset.csv")
crop_df = pd.read_csv("Crop_recommendation.csv")

# Page and Value Setup
st.set_page_config(page_title="Smart Farm Planner")
DEFAULTS = {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "ph": 6.5,
    "temperature": 27.5,
    "humidity": 80.0,
    "rainfall": 110.0,
    "area": 1.0,
    "fertilizer_price": 25.0,
    "pesticide_price": 40.0,
    "crop_price": 20.0
}
# Initialize Streamlit session state values with defaults (must run before widgets)
if 'initialized' not in st.session_state:
    st.session_state.update(DEFAULTS)
    st.session_state.initialized = True

# Normalize crop names
crop_df['Crop_Type'] = crop_df['Crop_Type'].str.strip().str.lower()
agri_df['Crop_Type'] = agri_df['Crop_Type'].str.strip().str.lower()

# Crop conditions reference data
crop_conditions = [
  {"Crop": "Apple", "Soil Type": "Loamy", "Temperature Range (°C)": [6, 16], "Rainfall Range (cm)": [100, 125], "Season": "Rabi", "Notes": "Requires Chilling Hours; Suited to Temperate Regions."},
  {"Crop": "Banana", "Soil Type": "Loamy", "Temperature Range (°C)": [26, 30], "Rainfall Range (cm)": [100, 200], "Season": "Kharif", "Notes": "Requires High Humidity and Well-Distributed Rainfall."},
  {"Crop": "Barley", "Soil Type": "Sandy", "Temperature Range (°C)": [12, 25], "Rainfall Range (cm)": [30, 90], "Season": "Rabi", "Notes": "Grows in Dry, Cool Climate with Light Irrigation."},
  {"Crop": "Blackgram", "Soil Type": "Loamy", "Temperature Range (°C)": [25, 35], "Rainfall Range (cm)": [60, 75], "Season": "Kharif", "Notes": "Needs Moist Soil During Vegetative Stage."},
  {"Crop": "Carrot", "Soil Type": "Sandy", "Temperature Range (°C)": [16, 20], "Rainfall Range (cm)": [70, 100], "Season": "Rabi", "Notes": "Requires Cool Climate and Well-Prepared Seedbed."},
  {"Crop": "Chickpea", "Soil Type": "Loamy", "Temperature Range (°C)": [10, 25], "Rainfall Range (cm)": [65, 100], "Season": "Rabi", "Notes": "Needs Cool Climate for Vegetative Growth and Dry Climate for Maturity."},
  {"Crop": "Coconut", "Soil Type": "Sandy", "Temperature Range (°C)": [25, 30], "Rainfall Range (cm)": [100, 150], "Season": "Kharif", "Notes": "Requires Humid Climate; Frost Sensitive."},
  {"Crop": "Coffee", "Soil Type": "Loamy", "Temperature Range (°C)": [15, 28], "Rainfall Range (cm)": [150, 250], "Season": "Kharif", "Notes": "Requires Shade and Well-Distributed Rainfall."},
  {"Crop": "Cotton", "Soil Type": "Loamy", "Temperature Range (°C)": [21, 30], "Rainfall Range (cm)": [50, 100], "Season": "Kharif", "Notes": "Warm, Dry Weather Needed for Maturity."},
  {"Crop": "Grapes", "Soil Type": "Sandy", "Temperature Range (°C)": [15, 35], "Rainfall Range (cm)": [50, 80], "Season": "Rabi", "Notes": "Needs Dry Climate at Ripening; Sensitive to Waterlogging."},
  {"Crop": "Jute", "Soil Type": "Silty", "Temperature Range (°C)": [20, 30], "Rainfall Range (cm)": [150, 200], "Season": "Kharif", "Notes": "Requires Warm and Humid Climate."},
  {"Crop": "Kidneybeans", "Soil Type": "Loamy", "Temperature Range (°C)": [18, 25], "Rainfall Range (cm)": [60, 100], "Season": "Kharif", "Notes": "Moderate Climate Required for Growth."},
  {"Crop": "Lentil", "Soil Type": "Loamy", "Temperature Range (°C)": [15, 25], "Rainfall Range (cm)": [40, 85], "Season": "Rabi", "Notes": "Prefers Cool Growing Conditions with Low Humidity."},
  {"Crop": "Maize", "Soil Type": "Sandy", "Temperature Range (°C)": [21, 27], "Rainfall Range (cm)": [50, 100], "Season": "Kharif", "Notes": "Sensitive to Waterlogging; Needs Full Sun."},
  {"Crop": "Mango", "Soil Type": "Loamy", "Temperature Range (°C)": [24, 30], "Rainfall Range (cm)": [75, 250], "Season": "Kharif", "Notes": "Requires Dry Weather During Flowering."},
  {"Crop": "Mothbeans", "Soil Type": "Sandy", "Temperature Range (°C)": [25, 35], "Rainfall Range (cm)": [20, 40], "Season": "Kharif", "Notes": "Requires Very Low Water; Good for Arid Regions."},
  {"Crop": "Mungbean", "Soil Type": "Sandy", "Temperature Range (°C)": [25, 35], "Rainfall Range (cm)": [60, 90], "Season": "Kharif", "Notes": "Grows Best in Warm Humid Climate."},
  {"Crop": "Muskmelon", "Soil Type": "Sandy", "Temperature Range (°C)": [24, 30], "Rainfall Range (cm)": [50, 75], "Season": "Zaid", "Notes": "Warm Season Crop with Low Humidity Needs."},
  {"Crop": "Orange", "Soil Type": "Loamy", "Temperature Range (°C)": [15, 35], "Rainfall Range (cm)": [100, 200], "Season": "Kharif", "Notes": "Prefers Dry Climate During Fruiting."},
  {"Crop": "Papaya", "Soil Type": "Sandy", "Temperature Range (°C)": [22, 35], "Rainfall Range (cm)": [100, 150], "Season": "Kharif", "Notes": "Sensitive to Frost; Needs Protection from Strong Winds."},
  {"Crop": "Pigeonpeas", "Soil Type": "Loamy", "Temperature Range (°C)": [26, 30], "Rainfall Range (cm)": [60, 100], "Season": "Kharif", "Notes": "Tolerant to Dry Conditions."},
  {"Crop": "Pomegranate", "Soil Type": "Loamy", "Temperature Range (°C)": [18, 32], "Rainfall Range (cm)": [50, 100], "Season": "Kharif", "Notes": "Prefers Semi-Arid Regions; Tolerates Drought."},
  {"Crop": "Potato", "Soil Type": "Loamy", "Temperature Range (°C)": [15, 20], "Rainfall Range (cm)": [100, 150], "Season": "Rabi", "Notes": "Requires Cool Climate, Sensitive to Frost During Growth."},
  {"Crop": "Rice", "Soil Type": "Clay", "Temperature Range (°C)": [25, 35], "Rainfall Range (cm)": [100, 300], "Season": "Kharif", "Notes": "Requires Standing Water; Best on Level Land."},
  {"Crop": "Soybean", "Soil Type": "Loamy", "Temperature Range (°C)": [20, 30], "Rainfall Range (cm)": [60, 100], "Season": "Kharif", "Notes": "Fixes Nitrogen; Suited to Moderate Rainfall Regions."},
  {"Crop": "Sugarcane", "Soil Type": "Loamy", "Temperature Range (°C)": [20, 30], "Rainfall Range (cm)": [100, 250], "Season": "Kharif", "Notes": "Requires Long Frost-Free Period and Plenty of Water."},
  {"Crop": "Tomato", "Soil Type": "Sandy", "Temperature Range (°C)": [20, 27], "Rainfall Range (cm)": [50, 100], "Season": "Rabi", "Notes": "Requires Moderate Climate and Good Sunlight."},
  {"Crop": "Watermelon", "Soil Type": "Sandy", "Temperature Range (°C)": [25, 30], "Rainfall Range (cm)": [40, 70], "Season": "Zaid", "Notes": "Requires Hot Climate and Moderate Irrigation."},
  {"Crop": "Wheat", "Soil Type": "Loamy", "Temperature Range (°C)": [10, 25], "Rainfall Range (cm)": [50, 100], "Season": "Rabi", "Notes": "Cool Growing Season with Dry Harvesting Conditions."}
]
def display_crop_conditions(selected_crop):
    crop_lower = selected_crop.strip().lower()
    crop_match = [item for item in crop_conditions if item['crop'] == crop_lower]
    if crop_match:
        item = crop_match[0]
        st.subheader("Ideal Conditions for Selected Crop")
        st.write(f"**Crop:** {item['crop'].title()}")
        st.write(f"**Preferred Soil:** {item['soil']}")
        st.write(f"**Ideal Temperature Range:** {item['temperature_c'][0]}–{item['temperature_c'][1]} °C")
        st.write(f"**Ideal Rainfall Range:** {item['rainfall_cm'][0]}–{item['rainfall_cm'][1]} cm/year")
        st.write(f"**Season:** {item['season']}")
        st.write(f"**Notes:** {item['notes']}")

# Generate dropdown crops list
dropdown_crops = sorted([item['crop'] for item in crop_conditions])

# Encode categorical columns
label_encoders = {}
for col in ['Crop_Type', 'Season', 'Soil_Type']:
    if col in agri_df.columns:
        le = LabelEncoder()
        agri_df[col] = le.fit_transform(agri_df[col])
        label_encoders[col] = le

# Add default values from crop dataset to agriculture dataset for matching columns
for col in ['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall']:
    agri_df[col] = crop_df[col][:len(agri_df)]

# Define Features and targets
features = ['Crop_Type', 'Farm_Area(acres)', 'Season', 'Soil_Type', 'N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall']
X = agri_df[features]
y = agri_df[['Fertilizer_Used(tons)', 'Pesticide_Used(kg)', 'Yield(tons)']]

@cache_resource
def train_final_models(X, y):
    from sklearn.model_selection import GridSearchCV

    target_columns = ['Fertilizer_Used(tons)', 'Pesticide_Used(kg)', 'Yield(tons)']
    final_models = {}
    best_model_names = {}

    model_grid = {
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "n_estimators": [100, 150],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        },
        "KNN": {
            "model": KNeighborsRegressor(),
            "params": {
                "n_neighbors": [3, 5, 7]
            }
        },
        "SVM": {
            "model": SVR(),
            "params": {
                "C": [0.5, 1, 10],
                "gamma": ["scale", "auto"]
            }
        }
    }

    for target in target_columns:
        best_model = None
        best_score = float('inf')
        best_name = None

        for name, config in model_grid.items():
            grid = GridSearchCV(config['model'], config['params'], scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
            grid.fit(X, y[target])
            score = -grid.best_score_

            if score < best_score:
                best_model = grid.best_estimator_
                best_score = score
                best_name = name

        final_models[target] = best_model
        best_model_names[target] = best_name

    return final_models, best_model_names

# Calling the cached function here
final_models, best_model_names = train_final_models(X, y)

# Streamlit UI
st.title("Smart Farm Resource, Yield & Investment Predictor")

crop = st.selectbox("Select Crop Type", [c.title() for c in dropdown_crops])
display_crop_conditions(crop)
if st.checkbox("Show ideal conditions for all crops"):
    st.subheader("Ideal Conditions by Crop")
    for item in crop_conditions:
        st.markdown(f"""
        **{item['crop']}**  
        - Soil: {item['soil']}  
        - Temperature: {item['temperature_c'][0]}–{item['temperature_c'][1]} °C  
        - Rainfall: {item['rainfall_cm'][0]}–{item['rainfall_cm'][1]} cm/year  
        - Season: {item['season']}  
        - Notes: {item['notes']}  
        """)


season = st.selectbox("Select Season", label_encoders['Season'].classes_)
soil = st.selectbox("Select Soil Type", label_encoders['Soil_Type'].classes_)
area = st.number_input("Enter Farm Area (in acres)", min_value=0.1, step=0.1, key="area")

st.subheader("Soil and Environment Parameters")
st.markdown("Enter the current soil nutrient levels and weather conditions for your farm to get accurate predictions.")
N = st.number_input("Nitrogen content (N)", min_value=0.0, key="N")
P = st.number_input("Phosphorus content (P)", min_value=0.0, key="P")
K = st.number_input("Potassium content (K)", min_value=0.0, key="K")
ph = st.number_input("pH level of soil", min_value=0.0, max_value=14.0, key="ph")
temperature = st.number_input("Average Temperature (°C)", key="temperature")
humidity = st.number_input("Average Humidity (%)", key="humidity")
rainfall = st.number_input("Rainfall (mm)", key="rainfall")

st.subheader("Market Prices")
st.markdown("Provide the expected market prices for fertilizers, pesticides, and crops to estimate your profit margins.")
fertilizer_price = st.number_input("Fertilizer cost (Rs/kg)", min_value=0.0, value=25.0, key="fertilizer_price")
pesticide_price = st.number_input("Pesticide cost (Rs/kg)", min_value=0.0, value=40.0, key="pesticide_price")
crop_price = st.number_input("Expected selling price of crop (Rs/kg)", min_value=0.0, value=20.0, key="crop_price")

if st.button("Predict"):
    crop_encoded = label_encoders['Crop_Type'].transform([crop.strip().lower()])[0]
    season_encoded = label_encoders['Season'].transform([season])[0]
    soil_encoded = label_encoders['Soil_Type'].transform([soil])[0]

    input_data = pd.DataFrame([[crop_encoded, area, season_encoded, soil_encoded, N, P, K, ph, temperature, humidity, rainfall]],
                              columns=features)
    fert_needed = final_models['Fertilizer_Used(tons)'].predict(input_data)[0]
    pest_needed = final_models['Pesticide_Used(kg)'].predict(input_data)[0]
    yield_pred = final_models['Yield(tons)'].predict(input_data)[0]

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
    crop_match = [item for item in crop_conditions if item['crop'] == crop.strip().lower()]
    if crop_match:
        ideal_values = crop_df[crop_df['Crop_Type'] == crop.strip().lower()][['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean()
        input_values = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        similarity = cosine_similarity([ideal_values], input_values)[0][0] * 100
        st.write(f"Your farm conditions match ideal {crop} requirements by {similarity:.2f}%.")
        if similarity >= 80:
            st.success("Input conditions are well-aligned with recommended crop requirements.")
        elif similarity >= 60:
            st.warning("Input conditions are moderately aligned. Consider minor adjustments.")
        else:
            st.error("Input conditions differ significantly from ideal crop parameters.")
    else:
        st.warning("Could not find ideal parameters for the selected crop.")

    st.subheader("Recommended Crops Based on Profitability and Yield")
    crop_suggestions = []
    for crop_name in dropdown_crops:
        c_encoded = label_encoders['Crop_Type'].transform([crop_name])[0]
        row = pd.DataFrame([[c_encoded, area, season_encoded, soil_encoded, N, P, K, ph, temperature, humidity, rainfall]],
                           columns=features)
        yld = final_models['Yield(tons)'].predict(row)[0]
        fert = final_models['Fertilizer_Used(tons)'].predict(row)[0]
        pest = final_models['Pesticide_Used(kg)'].predict(row)[0]
        inc = yld * 1000 * crop_price
        cost = fert * 1000 * fertilizer_price + pest * pesticide_price
        prof = inc - cost

        match = crop_df[crop_df['Crop_Type'] == crop_name]
        if not match.empty:
            ideal_vals = match[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean()
            sim = cosine_similarity([ideal_vals], [[N, P, K, temperature, humidity, ph, rainfall]])[0][0] * 100
        else:
            sim = 0

        crop_suggestions.append({
            'Crop': crop_name.title(),
            'Yield': round(yld, 2),
            'Profit': round(prof, 2),
            'Similarity': round(sim, 1)
        })

    filtered_suggestions = [c for c in crop_suggestions if c['Similarity'] >= 50]
    sorted_suggestions = sorted(filtered_suggestions, key=lambda x: (x['Similarity'], x['Profit'], x['Yield']), reverse=True)[:5]
    st.table(pd.DataFrame(sorted_suggestions))

if st.button("Reset Inputs"):
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
        st.session_state.initialized = False  # Will trigger re-initialization on rerun
    st.experimental_rerun()
