from flask import Flask, request, render_template, redirect, url_for, jsonify
from dotenv import load_dotenv
import psycopg2
import joblib
import numpy as np
import os
import math

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Database configuration
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Load ML model
MODEL_PATH = "voting_classifier_model.joblib"
model = joblib.load(MODEL_PATH)

# Label mapping for BMI categories
LABEL_MAPPING = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
    )

# BMI calculation
def calculate_bmi(height, weight):
    height_m = height / 100  # Convert to meters
    return round(weight / (height_m ** 2), 2)

# Ideal weight calculation
def calculate_ideal_weight(gender, height):
    if gender.lower() == "male":
        return round((height - 100) - (0.1 * (height - 100)), 2)
    else:
        return round((height - 100) - (0.15 * (height - 100)), 2)

# Reorder IDs
def reorder_ids():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        ALTER SEQUENCE bmi_predict_id_seq RESTART WITH 1;
        UPDATE bmi_predict SET id = DEFAULT;
    """)
    conn.commit()
    cur.close()
    conn.close()

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize default prediction results
    prediction = None
    bmi = None
    ideal_weight = None
    success = False

    # Pagination parameters
    page = int(request.args.get("page", 1))
    per_page = 6
    offset = (page - 1) * per_page

    # Handle prediction request
    if request.method == "POST":
        try:
            # Get form data
            nama = request.form.get("nama")
            gender = int(request.form.get("gender"))
            height = float(request.form.get("height"))
            weight = float(request.form.get("weight"))

            # Calculate BMI and ideal weight
            bmi = calculate_bmi(height, weight)
            ideal_weight = calculate_ideal_weight("Male" if gender == 0 else "Female", height)

            # Make prediction
            input_data = np.array([[gender, height, weight]])
            prediction = LABEL_MAPPING.get(model.predict(input_data)[0], "Unknown")

            # Store result in database
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO bmi_predict (nama, gender, height, weight, bmi, status, ideal_weight)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (nama, "Male" if gender == 0 else "Female", height, weight, bmi, prediction, ideal_weight),
            )
            conn.commit()
            cur.close()
            conn.close()
            success = True
        except Exception as e:
            success = False

    # Fetch records with pagination
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM bmi_predict")
    total_rows = cur.fetchone()[0]
    total_pages = math.ceil(total_rows / per_page)

    cur.execute("SELECT * FROM bmi_predict ORDER BY id LIMIT %s OFFSET %s", (per_page, offset))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Convert rows to dictionary format and re-index IDs
    rows = [
        {
            "index": idx + 1 + offset,
            "id": row[0],
            "nama": row[1],
            "gender": row[2],
            "height": row[3],
            "weight": row[4],
            "bmi": row[5],
            "status": row[6],
            "ideal_weight": row[7],
        }
        for idx, row in enumerate(rows)
    ]

    # Render template with prediction and table data
    return render_template(
        "Dashboard.html",
        prediction=prediction,
        bmi=bmi,
        ideal_weight=ideal_weight,
        success=success,
        rows=rows,
        page=page,
        total_pages=total_pages,
    )

# Delete record
@app.route("/delete/<int:id>", methods=["POST"])
def delete(id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM bmi_predict WHERE id = %s", (id,))
    conn.commit()
    cur.close()
    conn.close()
    reorder_ids()
    return redirect(url_for("index"))

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
