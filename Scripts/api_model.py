from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import uvicorn
import numpy as np

# Inisialisasi aplikasi FastAPI
app = FastAPI(title="Voting Classifier BMI API", description="API untuk prediksi BMI menggunakan Voting Classifier", version="1.0")

# Load model yang telah disimpan
model_path = "../models/voting_classifier_model.joblib"
classifier = load(model_path)

# Schema untuk input data
class InputData(BaseModel):
    gender: int  # 0 untuk Male, 1 untuk Female
    height: float  # Tinggi dalam cm
    weight: float  # Berat dalam kg

# Endpoint utama untuk prediksi
@app.post("/predict", summary="Prediksi BMI")
async def predict(data: InputData):
    try:
        # Mengonversi input menjadi format numpy
        input_features = np.array([[data.gender, data.height, data.weight]])
        
        # Melakukan prediksi
        prediction = classifier.predict(input_features)
        probabilities = classifier.predict_proba(input_features)

        # Hasil prediksi
        return {
            "input": {
                "gender": data.gender,
                "height": data.height,
                "weight": data.weight
            },
            "prediction": int(prediction[0]),
            "probabilities": probabilities[0].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Menjalankan aplikasi FastAPI dengan Unicorn
if __name__ == "__main__":
    uvicorn.run("api_model:app", host="127.0.0.1", port=8000, log_level="info")