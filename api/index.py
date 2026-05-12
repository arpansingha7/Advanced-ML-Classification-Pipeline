import sys
import os

# Add the project root to sys.path so 'src' can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import joblib
import pandas as pd
import numpy as np

# Explicitly import src modules so Vercel's python builder bundles them
import src.preprocessor
import src.feature_engineer

app = FastAPI(title="FIFA World Cup Predictor API", description="API for predicting World Cup Winners")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to prevent reloading on every request
MODELS = {}
PREPROCESSOR = None
FEATURE_ENGINEER = None
FIFA_2026_DATA = None

def load_pipeline_components(dataset: str = "fifa", model_type: str = "xgboost"):
    global PREPROCESSOR, FEATURE_ENGINEER, FIFA_2026_DATA
    
    model_key = f"{dataset}_{model_type}"
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    
    # Load preprocessor
    if PREPROCESSOR is None:
        prep_path = os.path.join(models_dir, f"{dataset}_preprocessor.pkl")
        if os.path.exists(prep_path): PREPROCESSOR = joblib.load(prep_path)
            
    # Load feature engineer
    if FEATURE_ENGINEER is None:
        fe_path = os.path.join(models_dir, f"{dataset}_feature_engineer.pkl")
        if os.path.exists(fe_path): FEATURE_ENGINEER = joblib.load(fe_path)

    # Load 2026 dataset
    if FIFA_2026_DATA is None:
        data_path = os.path.join(data_dir, f"fifa_2026.csv")
        if os.path.exists(data_path):
            FIFA_2026_DATA = pd.read_csv(data_path)
    
    # Load model
    if model_key in MODELS:
        return MODELS[model_key]
        
    model_path = os.path.join(models_dir, f"{model_key}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")
        
    model = joblib.load(model_path)
    MODELS[model_key] = model
    return model

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "message": "ML Pipeline API is running!"}

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")

@app.get("/api/teams")
def get_teams():
    load_pipeline_components()
    if FIFA_2026_DATA is not None:
        return {"teams": sorted(FIFA_2026_DATA['team'].tolist())}
    return {"teams": []}

@app.post("/api/predict/team")
def predict_team(body: dict):
    team_name = body.get("team")
    if not team_name:
        raise HTTPException(status_code=400, detail="Missing 'team' in request body")
        
    model = load_pipeline_components("fifa")
    
    if FIFA_2026_DATA is None: 
        raise HTTPException(status_code=500, detail="2026 Data not found")
    
    team_data = FIFA_2026_DATA[FIFA_2026_DATA['team'] == team_name]
    if team_data.empty: 
        raise HTTPException(status_code=404, detail="Team not found in 2026 dataset")
    
    # Drop targets and info cols
    df = team_data.drop(columns=['winner', 'finalist', 'semi_finalist', 'quarter_finalist', 'version', 'team', 'tournament_stage'], errors='ignore')
    
    # Transform
    if PREPROCESSOR: df = PREPROCESSOR.transform(df)
    if FEATURE_ENGINEER: df = FEATURE_ENGINEER.transform(df)
    
    prediction = int(model.predict(df)[0])
    return {"team": team_name, "prediction": prediction}

@app.get("/api/predict/all")
def predict_all():
    model = load_pipeline_components("fifa")
    if FIFA_2026_DATA is None: 
        raise HTTPException(status_code=500, detail="2026 Data not found")
    
    df = FIFA_2026_DATA.drop(columns=['winner', 'finalist', 'semi_finalist', 'quarter_finalist', 'version', 'team', 'tournament_stage'], errors='ignore')
    
    if PREPROCESSOR: df = PREPROCESSOR.transform(df)
    if FEATURE_ENGINEER: df = FEATURE_ENGINEER.transform(df)
    
    predictions = model.predict(df)
    results = []
    for team, pred in zip(FIFA_2026_DATA['team'], predictions):
        results.append({"team": team, "prediction": int(pred)})
        
    results = sorted(results, key=lambda x: x['prediction'], reverse=True)
    return {"leaderboard": results}
