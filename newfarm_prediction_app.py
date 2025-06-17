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
# Handle reset flag BEFORE widgets are rendered
if 'reset' in st.session_state and st.session_state.reset:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.reset = False
    st.rerun()

# Normalize crop names
crop_df['Crop_Type'] = crop_df['Crop_Type'].str.strip().str.lower()
agri_df['Crop_Type'] = agri_df['Crop_Type'].str.strip().str.lower()

# Crop conditions reference data
crop_conditions = [
  {"crop": "apple", "soil": "loamy", "temperature_c": [6, 16], "rainfall_cm": [100, 125], "season": "Rabi", "notes": "Requires chilling hours; suited to temperate regions."},
  {"crop": "banana", "soil": "loamy", "temperature_c": [26, 30], "rainfall_cm": [100, 200], "season": "Kharif", "notes": "Requires high humidity and well-distributed rainfall."},
  {"crop": "barley", "soil": "sandy", "temperature_c": [12, 25], "rainfall_cm": [30, 90], "season": "Rabi", "notes": "Grows in dry, cool climate with light irrigation."},
  {"crop": "blackgram", "soil": "loamy", "temperature_c": [25, 35], "rainfall_cm": [60, 75], "season": "Kharif", "notes": "Needs moist soil during vegetative stage."},
  {"crop": "carrot", "soil": "sandy", "temperature_c": [16, 20], "rainfall_cm": [70, 100], "season": "Rabi", "notes": "Requires cool climate and well-prepared seedbed."},
  {"crop": "chickpea", "soil": "loamy", "temperature_c": [10, 25], "rainfall_cm": [65, 100], "season": "Rabi", "notes": "Needs cool climate for vegetative growth and dry climate for maturity."},
  {"crop": "coconut", "soil": "sandy", "temperature_c": [25, 30], "rainfall_cm": [100, 150], "season": "Kharif", "notes": "Requires humid climate; frost sensitive."},
  {"crop": "coffee", "soil": "loamy", "temperature_c": [15, 28], "rainfall_cm": [150, 250], "season": "Kharif", "notes": "Requires shade and well-distributed rainfall."},
  {"crop": "cotton", "soil": "loamy", "temperature_c": [21, 30], "rainfall_cm": [50, 100], "season": "Kharif", "notes": "Warm, dry weather needed for maturity."},
  {"crop": "grapes", "soil": "sandy", "temperature_c": [15, 35], "rainfall_cm": [50, 80], "season": "Rabi", "notes": "Needs dry climate at ripening; sensitive to waterlogging."},
  {"crop": "jute", "soil": "silty", "temperature_c": [20, 30], "rainfall_cm": [150, 200], "season": "Kharif", "notes": "Requires warm and humid climate."},
  {"crop": "kidneybeans", "soil": "loamy", "temperature_c": [18, 25], "rainfall_cm": [60, 100], "season": "Kharif", "notes": "Moderate climate required for growth."},
  {"crop": "lentil", "soil": "loamy", "temperature_c": [15, 25], "rainfall_cm": [40, 85], "season": "Rabi", "notes": "Prefers cool growing conditions with low humidity."},
  {"crop": "maize", "soil": "sandy", "temperature_c": [21, 27], "rainfall_cm": [50, 100], "season": "Kharif", "notes": "Sensitive to waterlogging; needs full sun."},
  {"crop": "mango", "soil": "loamy", "temperature_c": [24, 30], "rainfall_cm": [75, 250], "season": "Kharif", "notes": "Requires dry weather during flowering."},
  {"crop": "mothbeans", "soil": "sandy", "temperature_c": [25, 35], "rainfall_cm": [20, 40], "season": "Kharif", "notes": "Requires very low water; good for arid regions."},
  {"crop": "mungbean", "soil": "sandy", "temperature_c": [25, 35], "rainfall_cm": [60, 90], "season": "Kharif", "notes": "Grows best in warm humid climate."},
  {"crop": "muskmelon", "soil": "sandy", "temperature_c": [24, 30], "rainfall_cm": [50, 75], "season": "Zaid", "notes": "Warm season crop with low humidity needs."},
  {"crop": "orange", "soil": "loamy", "temperature_c": [15, 35], "rainfall_cm": [100, 200], "season": "Kharif", "notes": "Prefers dry climate during fruiting."},
  {"crop": "papaya", "soil": "sandy", "temperature_c": [22, 35], "rainfall_cm": [100, 150], "season": "Kharif", "notes": "Sensitive to frost; needs protection from strong winds."},
  {"crop": "pigeonpeas", "soil": "loamy", "temperature_c": [26, 30], "rainfall_cm": [60, 100], "season": "Kharif", "notes": "Tolerant to dry conditions."},
  {"crop": "pomegranate", "soil": "loamy", "temperature_c": [18, 32], "rainfall_cm": [50, 100], "season": "Kharif", "notes": "Prefers semi-arid regions; tolerates drought."},
  {"crop": "potato", "soil": "loamy", "temperature_c": [15, 20], "rainfall_cm": [100, 150], "season": "Rabi", "notes": "Requires cool climate, sensitive to frost during growth."},
  {"crop": "rice", "soil": "clay", "temperature_c": [25, 35], "rainfall_cm": [100, 300], "season": "Kharif", "notes": "Requires standing water; best on level land."},
  {"crop": "soybean", "soil": "loamy", "temperature_c": [20, 30], "rainfall_cm": [60, 100], "season": "Kharif", "notes": "Fixes nitrogen; suited to moderate rainfall regions."},
  {"crop": "sugarcane", "soil": "loamy", "temperature_c": [20, 30], "rainfall_cm": [100, 250], "season": "Kharif", "notes": "Requires long frost-free period and plenty of water."},
  {"crop": "tomato", "soil": "sandy", "temperature_c": [20, 27], "rainfall_cm": [50, 100], "season": "Rabi", "notes": "Requires moderate climate and good sunlight."},
  {"crop": "watermelon", "soil": "sandy", "temperature_c": [25, 30], "rainfall_cm": [40, 70], "season": "Zaid", "notes": "Requires hot climate and moderate irrigation."},
  {"crop": "wheat", "soil": "loamy", "temperature_c": [10, 25], "rainfall_cm": [50, 100], "season": "Rabi", "notes": "Cool growing season with dry harvesting conditions."}
]

def title_case_excluding_prepositions(text):
    prepositions = {'for', 'of', 'and', 'in', 'to', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into', 'over', 'after'}
    words = text.split()
    titled = [words[0].capitalize()] + [
        word if word in prepositions else word.capitalize() for word in words[1:]
    ]
    return ' '.join(titled)

def display_crop_conditions(selected_crop):
    crop_lower = selected_crop.strip().lower()
    crop_match = [item for item in crop_conditions if item['crop'] == crop_lower]
    if crop_match:
        item = crop_match[0]
        st.subheader("Ideal Conditions for Selected Crop")
        st.write(f"**Crop:** {item['crop'].title()}")
        st.write(f"**Season:** {item['season'].capitalize()}")
        st.write(f"**Preferred Soil:** {item['soil'].capitalize()}")
        st.write(f"**Ideal Temperature Range:** {item['temperature_c'][0]}–{item['temperature_c'][1]} °C")
        st.write(f"**Ideal Rainfall Range:** {item['rainfall_cm'][0]}–{item['rainfall_cm'][1]} cm/year")
        st.write(f"**Notes:** {title_case_excluding_prepositions(item['notes'])}")

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
fertilizer_price = st.number_input("Fertilizer cost (Rs/kg)", min_value=0.0, key="fertilizer_price")
pesticide_price = st.number_input("Pesticide cost (Rs/kg)", min_value=0.0, key="pesticide_price")
crop_price = st.number_input("Expected selling price of crop (Rs/kg)", min_value=0.0, key="crop_price")

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
    st.session_state.reset = True
    st.rerun()
