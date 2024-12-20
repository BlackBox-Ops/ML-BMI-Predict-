# Import library yang diperlukan
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np

# Load model joblib
MODEL_PATH = "voting_classifier_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Initialize Flask app
app = Flask(__name__)

# Konfigurasi database PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:admin@localhost:5432/predictions'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Define the label mapping
LABEL_MAPPING = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

# Model database untuk menyimpan prediksi
class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    gender = db.Column(db.String(6), nullable=False)
    height = db.Column(db.Float, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    result = db.Column(db.String(50), nullable=False)

# Endpoint untuk halaman utama
@app.route("/")
def home():
    return render_template("index.html")

# Endpoint untuk melakukan prediksi
@app.route("/predict", methods=["POST"])
def predict_bmi():
    try:
        # Parse JSON request
        data = request.get_json()
        gender = data["gender"]
        height = data["height"]
        weight = data["weight"]

        # Convert input data to numpy array
        input_data = np.array([[gender, height, weight]])

        # Make prediction
        prediction = model.predict(input_data)

        # Map prediction to label
        prediction_label = LABEL_MAPPING.get(prediction[0], "Unknown")

        # Simpan hasil ke database
        gender_str = "Male" if gender == 0 else "Female"
        new_prediction = Prediction(
            gender=gender_str,
            height=height,
            weight=weight,
            result=prediction_label
        )
        db.session.add(new_prediction)
        db.session.commit()

        return jsonify({"prediction": prediction_label})

    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

# Endpoint untuk mendapatkan riwayat prediksi
@app.route("/history", methods=["GET"])
def get_history():
    try:
        # Ambil semua data prediksi dari database
        predictions = Prediction.query.order_by(Prediction.predicted_at.desc()).all()
        results = [
            {
                "gender": prediction.gender,
                "height": prediction.height,
                "weight": prediction.weight,
                "result": prediction.result,
            }
            for prediction in predictions
        ]
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": f"Error fetching history: {e}"}), 500

if __name__ == "__main__":
    # Buat tabel jika belum ada
    with app.app_context():
        db.create_all()
    app.run(debug=True)
