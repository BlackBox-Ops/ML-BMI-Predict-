# import library yang diperlukan 
from flask import Flask, request, jsonify
import joblib
import numpy as np 

# load model joblib 
MODEL_PATH = "../models/voting_classifier_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model : {e}")

# initialize flask app
app = Flask(__name__)

# Define the label mapping 
LABEL_MAPPING = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"}

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the BMI Prediction API!"})

@app.route("/predict", methods=["POST"])
def predict_bmi():
    try:
        # parse json request 
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
        
        return jsonify({"prediction": prediction_label})
    
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)