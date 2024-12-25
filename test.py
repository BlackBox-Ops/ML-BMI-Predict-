from flask import Flask, request, jsonify, render_template, redirect, url_for
from dotenv import load_dotenv

import joblib
import numpy as np 
import os 
import psycopg2
import logging

# memuat variabel lingkungan dari file .env 
load_dotenv()

# Mengatur logging
logging.basicConfig(level=logging.DEBUG)

# Load model machine learning 
MODEL_PATH = "voting_classifier_model.joblib"

# membuat koneksi ke database 
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')


# Define the label mapping untuk hasil prediksi model machine learning 
LABEL_MAPPING = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

app = Flask(__name__)

# buat function untuk load model machine learning 
def load_model(path):
    # test apabila model berhasil dibaca
    try:
        return joblib.load(path)
    except Exception as e:
        # test apabila model tidak berhasil dibaca 
        raise RuntimeError(f'Failed to load model : {e}')

# load the model & inisialisasi postgresql connection 
model = load_model(MODEL_PATH)

# buat fungsi untuk koneksi ke database 
def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

# buat function untuk menghitung nilai bmi 
def calculate_bmi(height, weight):
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)

# buat function untuk menampilkan isi dari tabel bmi_predict 
@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM bmi_predict;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('test.html', rows=[{
    # tampilkan isi dari tabel 
        'id': r[0],     
        'gender': r[1],
        'height': r[2],
        'weight': r[3],
        'bmi': r[4],
        'status': r[5]
    } for r in rows])

# buat function untuk prediksi dan perhitungan bmi lalu simpan ke database 
@app.route('/predict', methods=['POST'])
def predict_bmi():
    # handle bmi prediction and store results in the database 
    try:
        data = request.get_json()
        gender = "Male" if data["gender"] == 0 else "Female"
        height = data["height"]
        weight = data["weight"]

        # calculate bmi 
        bmi = calculate_bmi(weight, height)

        # prepare data for prediction 
        input_data = np.array([{data['gender'], height, weight}])
        prediction = model.predict(input_data)

        # map prediction to label 
        prediction_label = LABEL_MAPPING.get(prediction[0], "Unknown")

        # store prediction in the database 
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (gender, height, weight, bmi, result) VALUES (%s, %s, %s, %s, %s)",
            (gender, height, weight, bmi, prediction_label)
        )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'prediction':prediction_label, 'bmi':bmi})

    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500
    
if __name__ == '__main__':
    app.run(debug=True)