from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import uvicorn
import numpy as np

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="API Model Klasifikasi BMI dengan ALgoritma Voting Classifier", 
    description="API untuk prediksi BMI menggunakan Voting Classifier", version="1.0")

# Load model yang telah disimpan
try:
    model_path = "../models/voting_classifier_model.joblib"
    classifier = load(model_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Schema untuk input data
class InputData(BaseModel):
    gender: int  # 0 untuk Male, 1 untuk Female
    height: float  # Tinggi dalam cm
    weight: float  # Berat dalam kg

# Mapping untuk hasil prediksi dan gender
label_mapping = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}
gender_mapping = {
    0: "Male",
    1: "Female"
}

# Endpoint utama untuk prediksi
@app.post("/predict", summary="Prediksi BMI")
async def predict(data: InputData):
    try:
        # Mengonversi input menjadi format numpy
        input_features = np.array([[data.gender, data.height, data.weight]])
        
        # Melakukan prediksi
        prediction = classifier.predict(input_features)
        probabilities = classifier.predict_proba(input_features)
        
        # Mapping hasil prediksi
        predicted_label = label_mapping.get(int(prediction[0]), "Unknown")
        gender_label = gender_mapping.get(data.gender, "Unknown")

        # Hasil prediksi dengan label
        return {
            "input": {
                "gender": gender_label,
                "height": data.height,
                "weight": data.weight
            },
            "prediction": {
                "label": predicted_label,
                "class": int(prediction[0])
            },
            "probabilities": probabilities[0].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Menjalankan aplikasi FastAPI dengan Uvicorn
if __name__ == "__main__":
    uvicorn.run("api_model:app", host="127.0.0.1", port=8000, log_level="info")
