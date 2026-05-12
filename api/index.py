import sys
import os

# Add the project root to sys.path so 'src' can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import pandas as pd

app = FastAPI(title="Advanced ML Pipeline API", description="API for predicting with trained models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models to prevent reloading on every request
MODELS = {}

def load_model(dataset: str, model_type: str = "random_forest"):
    model_key = f"{dataset}_{model_type}"
    if model_key in MODELS:
        return MODELS[model_key]
        
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f"{model_key}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")
        
    try:
        model = joblib.load(model_path)
        MODELS[model_key] = model
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "message": "ML Pipeline API is running!"}

@app.post("/api/predict/{dataset}")
def predict(dataset: str, features: dict):
    """
    Expects JSON body like:
    {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }
    """
    model = load_model(dataset)
    
    try:
        # Create DataFrame from input
        df = pd.DataFrame([features])
        
        # Predict
        prediction = model.predict(df)[0]
        
        # Some scikit-learn models return numpy types
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
            
        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            probability = proba.tolist()
            
        return {
            "dataset": dataset,
            "prediction": prediction,
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
