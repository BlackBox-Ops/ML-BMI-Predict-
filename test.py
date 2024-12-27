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

# Function to reorder IDs starting from 3
def reorder_ids():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            WITH reordered AS (
                SELECT id, ROW_NUMBER() OVER () AS new_id
                FROM bmi_predict
            )
            UPDATE bmi_predict
            SET id = reordered.new_id
            FROM reordered
            WHERE bmi_predict.id = reordered.id;
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
            'id': row[5],
            'gender': row[0],
            'height': row[1],
            'weight': row[2],
            'bmi': row[3],
            'status': row[4],
        }
        for row in rows
    ]
    return render_template('app.html', rows=rows)

# Route for BMI prediction and storing the result in the database
@app.route('/predict', methods=['POST'])
def predict_bmi():
    try:
        data = request.get_json()
        gender = "Male" if data["gender"] == 0 else "Female"
        height = data["height"]
        weight = data["weight"]

        # Calculate BMI
        bmi = calculate_bmi(height, weight)

        # Prepare data for prediction
        input_data = np.array([[data['gender'], height, weight]])
        prediction = model.predict(input_data)

        # Map prediction result to label
        prediction_label = LABEL_MAPPING.get(prediction[0], "Unknown")

        # Store prediction in the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO bmi_predict (gender, height, weight, bmi, status) VALUES (%s, %s, %s, %s, %s)",
            (gender, height, weight, bmi, prediction_label)
        )
        conn.commit()
        cur.close()
        conn.close()

        # Reorder IDs after inserting new data
        reorder_ids()

        return jsonify({'prediction': prediction_label, 'bmi': bmi})

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

# Function to reset the ID sequence
def reset_sequence():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT setval('bmi_predict_id_seq', (SELECT MAX(id) FROM bmi_predict));
        """)
        conn.commit()
    finally:
        cur.close()
        conn.close()

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
