from flask import Flask, render_template, request, redirect, url_for, jsonify
import psycopg2
import joblib
import numpy as np 

# Load model joblib 
MODEL_PATH = "voting_classifier_model.joblib"

# PostgreSQL connection setup
DATABASE_URI = {
    'dbname': "predictions",
    'user': "postgres",
    'password': "admin",
    'host': "localhost",
    'port': "5432"
}

# Define the label mapping
LABEL_MAPPING = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

app = Flask(__name__)

# Load model
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
# Initialize PostgreSQL connection
def get_db_connection():
    return psycopg2.connect(**DATABASE_URI)

# Load the model
model = load_model(MODEL_PATH)

@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('test.html', rows=[{
        'id': r[0], 'gender': r[1], 'height': r[2], 'weight': r[3], 'result': r[4]
    } for r in rows])

@app.route('/predict', methods=["POST"])
def predict_bmi():
    """Handle BMI prediction and store results in the database."""
    try:
        data = request.get_json()
        gender = "Male" if data["gender"] == 0 else "Female"
        height = data["height"]
        weight = data["weight"]

        # Prepare data for prediction
        input_data = np.array([[data["gender"], height, weight]])
        prediction = model.predict(input_data)

        # Map prediction to label
        prediction_label = LABEL_MAPPING.get(prediction[0], "Unknown")

        # Store prediction in the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (gender, height, weight, result) VALUES (%s, %s, %s, %s)",
            (gender, height, weight, prediction_label)
        )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"prediction": prediction_label})

    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

@app.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    """Delete a prediction from the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE id = %s", (id,))
    conn.commit()
    cur.close()
    conn.close()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
