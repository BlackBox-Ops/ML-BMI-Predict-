from flask import Flask, request, jsonify, render_template, redirect, url_for
from dotenv import load_dotenv
import joblib
import numpy as np
import os
import psycopg2
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load machine learning model
MODEL_PATH = "voting_classifier_model.joblib"

# Database connection settings from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')

# Label mapping for BMI prediction results
LABEL_MAPPING = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

# Initialize Flask app
app = Flask(__name__)

# Load machine learning model function
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Load the machine learning model
model = load_model(MODEL_PATH)

# Database connection function
def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

# Function to calculate BMI
def calculate_bmi(height, weight):
    height_m = height / 100  # Convert height to meters
    return round(weight / (height_m ** 2), 2)

# Function to calculate ideal weight
def calculate_ideal_weight(gender, height):
    if gender.lower() == "male":
        return round((height - 100) - ((height - 100) * 0.1), 2)
    else:  # Female
        return round((height - 100) - ((height - 100) * 0.15), 2)

# Function to reorder IDs starting from 1
def reorder_ids():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            ALTER SEQUENCE bmi_predict_id_seq RESTART WITH 1;
            UPDATE bmi_predict SET id = DEFAULT;
        """)
        conn.commit()
    finally:
        cur.close()
        conn.close()

# Route to display all records from the database
@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM bmi_predict;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Convert query result to a list of dictionaries
    rows = [
        {
            'id': row[0],
            'nama': row[1],
            'gender': row[2],
            'height': row[3],
            'weight': row[4],
            'status': row[5],
            'bmi': row[6],
            'ideal_weight': row[7]
        }
        for row in rows
    ]
    return render_template('app.html', rows=rows)

# Route for BMI prediction and storing the result in the database
@app.route('/predict', methods=['POST'])
def predict_bmi():
    try:
        data = request.get_json()
        nama = data["nama"]
        gender = "Male" if data["gender"] == 0 else "Female"
        height = data["height"]
        weight = data["weight"]

        # Calculate BMI
        bmi = calculate_bmi(height, weight)

        # Calculate ideal weight
        ideal_weight = calculate_ideal_weight(gender, height)

        # Prepare data for prediction
        input_data = np.array([[data['gender'], height, weight]])
        prediction = model.predict(input_data)

        # Map prediction result to label
        prediction_label = LABEL_MAPPING.get(prediction[0], "Unknown")

        # Store prediction in the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO bmi_predict (nama, gender, height, weight, bmi, status, ideal_weight) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (nama, gender, height, weight, bmi, prediction_label, ideal_weight)
        )
        conn.commit()
        cur.close()
        conn.close()

        # Reorder IDs after inserting new data
        reorder_ids()

        return jsonify({'prediction': prediction_label, 'bmi': bmi, 'ideal_weight': ideal_weight})

    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

# Route to delete a record from the database
@app.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM bmi_predict WHERE id = %s", (id,))
        conn.commit()
    finally:
        cur.close()
        conn.close()

    # Reorder IDs after deletion
    reorder_ids()

    return redirect(url_for('index'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
