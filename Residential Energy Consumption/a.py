import streamlit as st
import pandas as pd
import joblib
import datetime

# Page Configuration
st.set_page_config(page_title="Energy Consumption Prediction")
st.title("Energy Consumption Prediction")

# Load the model (change this to your actual model file)
model = joblib.load(r"C:\2monthTraining\Project-3\Random_forest_model (2)\Random_forest_model (2).pkl")

#Background & Styling
st.markdown("""
<style>
/* App background */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
}

/* Blurred overlay effect */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    backdrop-filter: blur(4px);
    background: rgba(255, 255, 255, 0.3);
    z-index: -1;
}

/* Form styling */
[data-testid="stForm"] {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Input styles */
input, select, textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 6px !important;
    padding: 8px !important;
    border: 1px solid #bbb !important;
}

/* Label & text color */
label, .stRadio > label, .stCheckbox, .css-1cpxqw2, .st-bf, .st-c9 {
    color: #000000 !important;
    font-size: 16px !important;
}

/* Headings */
h1, h2, h3 {
    color: #000000 !important;
    font-weight: 700 !important;
}

/* Buttons */
button[kind="primary"] {
    background-color: #1a73e8 !important;
    color: white !important;
    border: none;
    border-radius: 6px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


model_columns = [
    'num_occupants', 'house_size_sqft', 'monthly_income', 'outside_temp_celsius', 'year', 'month', 'day', 'season',
    'heating_type_Electric', 'heating_type_Gas', 'heating_type_None',
    'cooling_type_AC', 'cooling_type_Fan', 'cooling_type_None',
    'manual_override_Y', 'manual_override_N', 'is_weekend', 'temp_above_avg',
    'income_per_person', 'square_feet_per_person', 'high_income_flag', 'low_temp_flag',
    'season_spring', 'season_summer', 'season_fall', 'season_winter',
    'day_of_week_0', 'day_of_week_6', 'energy_star_home'
]

with st.form("User_inputs"):
    st.header("Enter Inputs (info)")
    col1, col2 = st.columns(2)

    with col1:
        num_occupants = st.number_input("Number of Occupants", min_value=1)
        house_size = st.number_input("House size (sqft)", min_value=100, max_value=10000)
        income = st.number_input("Monthly Income", min_value=100, max_value=100000)
        temp = st.number_input("Outside Temperature (°C)", value=25.0)

    with col2:
        date = st.date_input("Date", value=datetime.date.today())
        heating = st.selectbox("Heating Type", ["Gas", "Electric", "None"])
        cooling = st.selectbox("Cooling Type", ["AC", "Fan", "None"])
        manual = st.radio("Manual Override", ["Yes", "No"])
        energy_star = st.checkbox("Energy Star Home")

    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            day_of_week = date.weekday()  # 0=Monday, 6=Sunday

            # Determine season
            season_label = {
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall'
            }.get(date.month, 'fall')

            features = {
                'num_occupants': num_occupants,
                'house_size_sqft': house_size,
                'monthly_income': income,
                'outside_temp_celsius': temp,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'season': {'spring': 2, 'summer': 3, 'fall': 4, 'winter': 1}[season_label],
                'heating_type_Electric': int(heating == 'Electric'),
                'heating_type_Gas': int(heating == 'Gas'),
                'heating_type_None': int(heating == 'None'),
                'cooling_type_AC': int(cooling == 'AC'),
                'cooling_type_Fan': int(cooling == 'Fan'),
                'cooling_type_None': int(cooling == 'None'),
                'manual_override_Y': int(manual == 'Yes'),
                'manual_override_N': int(manual == 'No'),
                'is_weekend': int(day_of_week >= 5),
                'temp_above_avg': int(temp > 22),
                'income_per_person': income / num_occupants,
                'square_feet_per_person': house_size / num_occupants,
                'high_income_flag': int(income > 10000),
                'low_temp_flag': int(temp < 22),
                'season_spring': int(season_label == 'spring'),
                'season_summer': int(season_label == 'summer'),
                'season_fall': int(season_label == 'fall'),
                'season_winter': int(season_label == 'winter'),
                'day_of_week_0': int(day_of_week == 0),
                'day_of_week_6': int(day_of_week == 6),
                'energy_star_home': int(energy_star),
            }

            df = pd.DataFrame([{col: features.get(col, 0) for col in model_columns}])
            prediction = model.predict(df)[0]

            st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")

        except Exception as e:
            st.error(f"An error occurred: {e}")