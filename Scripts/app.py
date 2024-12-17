import pandas as pd
import numpy as np 
import streamlit as st
import joblib 

# Mapping Gender dan Class
gender_mapping = {"Laki-laki": 1, "Perempuan": 0}
class_mapping = {
    0: "Extreme Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

# Load Model dari Joblib
@st.cache_resource
def load_model():
    return joblib.load("../Notebook/voting_classifier_model.joblib")  # Ganti dengan nama file model Anda

# Aplikasi Streamlit
def main():
    st.title("Aplikasi Prediksi BMI dengan Voting Classifier")
    st.write("Masukkan data berikut untuk mendapatkan prediksi BMI:")
    
    # Input Gender
    gender_input = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    gender = gender_mapping[gender_input]  # Konversi ke nilai numerik

    # Input Height
    height = st.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=250.0, step=0.1)

    # Input Weight
    weight = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=200.0, step=0.1)

    # Tombol Prediksi
    if st.button("Prediksi"):
        if height > 0 and weight > 0:
            model = load_model()  # Load model
            input_data = np.array([[gender, height, weight]])  # Bentuk data input
            prediction = model.predict(input_data)[0]  # Prediksi model
            class_name = class_mapping[prediction]  # Dapatkan nama kelas

            # Tampilkan hasil
            st.success(f"Hasil Prediksi: {class_name}")
        else:
            st.error("Harap masukkan tinggi dan berat badan dengan benar.")

# Jalankan aplikasi
if __name__ == "__main__":
    main()
