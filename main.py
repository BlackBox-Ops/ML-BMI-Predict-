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

# Konfigurasi database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:admin@localhost:5432/predictions'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
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

# Define database model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    gender = db.Column(db.String(10), nullable=False)
    height = db.Column(db.Float, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    result = db.Column(db.String(20), nullable=False)

    def __repr__(self):
        return f"<Prediction {self.id} - {self.result}>"

# Buat database
with app.app_context():
    db.create_all()

@app.route("/")
def home():
    # Render halaman HTML utama
    return render_template("prediction.html")

@app.route("/predict", methods=["POST"])
def predict_bmi():
    try:
        # Parse JSON request
        data = request.get_json()
        gender = "Male" if data["gender"] == 0 else "Female"
        height = data["height"]
        weight = data["weight"]

        # Convert input data to numpy array
        input_data = np.array([[data["gender"], height, weight]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Map prediction to label
        prediction_label = LABEL_MAPPING.get(prediction[0], "Unknown")

        # Simpan hasil prediksi ke database
        new_prediction = Prediction(
            gender=gender,
            height=height,
            weight=weight,
            result=prediction_label
        )
        db.session.add(new_prediction)
        db.session.commit()

        return jsonify({"prediction": prediction_label})
    
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

@app.route("/history", methods=["GET"])
def history():
    # Ambil semua prediksi dari database
    predictions = Prediction.query.order_by(Prediction.predicted_at.desc()).all()
    return jsonify([
        {
            "id": p.id,
            "gender": p.gender,
            "height": p.height,
            "weight": p.weight,
            "result": p.result,
        } for p in predictions
    ])

if __name__ == "__main__":
    app.run(debug=True)
