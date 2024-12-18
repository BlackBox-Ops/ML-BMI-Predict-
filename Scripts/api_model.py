from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import joblib 
import numpy as np

model_path = '../models/voting_classifier_model.joblib'
model = joblib.load(model_path)