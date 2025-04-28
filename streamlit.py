import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
MODEL_PATH = "./models/voting_classifier_model.joblib"
model = joblib.load(MODEL_PATH)

# Label mapping
LABEL_MAPPING = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

# BMI calculation
def calculate_bmi(height, weight):
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)

# Ideal weight calculation
def calculate_ideal_weight(gender, height):
    if gender.lower() == "male":
        return round((height - 100) - (0.1 * (height - 100)), 2)
    else:
        return round((height - 100) - (0.15 * (height - 100)), 2)

# CSS styling
def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: white;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .header {
        text-align: center;
        color: #4B6EAF;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "records" not in st.session_state:
    st.session_state.records = []

# Apply CSS
local_css()

# App title
st.markdown("<h1 class='header'>BMI Prediction App</h1>", unsafe_allow_html=True)

# Layout: Two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Data")
    with st.form("prediction_form"):
        nama = st.text_input("Name")
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=50.0, max_value=300.0, step=0.1)
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, step=0.1)
        submit = st.form_submit_button("Predict")

    if submit:
        gender_val = 0 if gender == "Male" else 1
        input_data = np.array([[gender_val, height, weight]])

        pred_label = model.predict(input_data)[0]
        pred_text = LABEL_MAPPING.get(pred_label, "Unknown")

        bmi = calculate_bmi(height, weight)
        ideal_weight = calculate_ideal_weight(gender, height)

        record = {
            "Name": nama,
            "Gender": gender,
            "Height (cm)": height,
            "Weight (kg)": weight,
            "BMI": bmi,
            "Status": pred_text,
            "Ideal Weight (kg)": ideal_weight
        }
        st.session_state.records.append(record)

with col2:
    if submit:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Prediction Result")
        st.success(f"**Status:** {pred_text}")
        st.info(f"**BMI:** {bmi}")
        st.info(f"**Ideal Weight:** {ideal_weight} kg")
        st.markdown("</div>", unsafe_allow_html=True)

# Display prediction records
st.subheader("Prediction History")

if st.session_state.records:
    df = pd.DataFrame(st.session_state.records)
    st.dataframe(df.style.set_properties(**{'text-align': 'center'}), use_container_width=True)

    clear_button = st.button("Clear All Records", type="primary")
    if clear_button:
        st.session_state.records = []
else:
    st.info("No predictions yet. Fill the form and submit to start.")
