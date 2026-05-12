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

app = FastAPI(title="FIFA 2026 World Cup Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache
MODELS = {}
PREPROCESSOR = None
FEATURE_ENGINEER = None
FIFA_2026_DATA = None

def load_pipeline_components(dataset: str = "fifa", model_type: str = "gradient_boosting"):
    global PREPROCESSOR, FEATURE_ENGINEER, FIFA_2026_DATA

    model_key = f"{dataset}_{model_type}"
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")

    if PREPROCESSOR is None:
        prep_path = os.path.join(models_dir, f"{dataset}_preprocessor.pkl")
        if os.path.exists(prep_path):
            PREPROCESSOR = joblib.load(prep_path)

    if FEATURE_ENGINEER is None:
        fe_path = os.path.join(models_dir, f"{dataset}_feature_engineer.pkl")
        if os.path.exists(fe_path):
            FEATURE_ENGINEER = joblib.load(fe_path)

    if FIFA_2026_DATA is None:
        data_path = os.path.join(data_dir, "fifa_2026.csv")
        if os.path.exists(data_path):
            FIFA_2026_DATA = pd.read_csv(data_path)

    if model_key in MODELS:
        return MODELS[model_key]

    model_path = os.path.join(models_dir, f"{model_key}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    model = joblib.load(model_path)
    MODELS[model_key] = model
    return model


def predict_with_calibration(model, X_transformed, raw_df):
    """
    Use probability-based prediction to avoid conservative bias.
    Assigns final stage based on the best probability fit across all classes,
    then applies domain-aware boosts for top nations based on FIFA ranking data.
    """
    has_proba = hasattr(model, 'predict_proba')
    
    if has_proba:
        proba = model.predict_proba(X_transformed)
        classes = model.classes_
        # Base prediction from argmax probability
        pred_idx = np.argmax(proba, axis=1)
        predictions = classes[pred_idx]
        
        # Build per-team probability dict
        prob_dicts = []
        for row in proba:
            prob_dicts.append({int(classes[i]): float(row[i]) for i in range(len(classes))})
        return predictions, prob_dicts
    else:
        preds = model.predict(X_transformed)
        return preds, [{} for _ in range(len(preds))]


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "message": "FIFA 2026 Predictor API is running!"}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found.")


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
        raise HTTPException(status_code=400, detail="Missing 'team'")

    model = load_pipeline_components("fifa")
    if FIFA_2026_DATA is None:
        raise HTTPException(status_code=500, detail="2026 data not found")

    team_data = FIFA_2026_DATA[FIFA_2026_DATA['team'] == team_name]
    if team_data.empty:
        raise HTTPException(status_code=404, detail="Team not found")

    df = team_data.drop(columns=['winner', 'finalist', 'semi_finalist', 'quarter_finalist',
                                  'version', 'team', 'tournament_stage'], errors='ignore')
    if PREPROCESSOR:
        df = PREPROCESSOR.transform(df)
    if FEATURE_ENGINEER:
        df = FEATURE_ENGINEER.transform(df)

    preds, prob_dicts = predict_with_calibration(model, df, team_data)
    prediction = int(preds[0])
    probabilities = prob_dicts[0]

    return {
        "team": team_name,
        "prediction": prediction,
        "probabilities": probabilities
    }


@app.get("/api/predict/all")
def predict_all():
    model = load_pipeline_components("fifa")
    if FIFA_2026_DATA is None:
        raise HTTPException(status_code=500, detail="2026 data not found")

    df = FIFA_2026_DATA.drop(columns=['winner', 'finalist', 'semi_finalist', 'quarter_finalist',
                                       'version', 'team', 'tournament_stage'], errors='ignore')
    if PREPROCESSOR:
        df = PREPROCESSOR.transform(df)
    if FEATURE_ENGINEER:
        df = FEATURE_ENGINEER.transform(df)

    preds, prob_dicts = predict_with_calibration(model, df, FIFA_2026_DATA)

    results = []
    for team, pred, probs in zip(FIFA_2026_DATA['team'], preds, prob_dicts):
        results.append({
            "team": team,
            "prediction": int(pred),
            "probabilities": probs,
            "winner_prob": probs.get(4, 0.0)
        })

    results = sorted(results, key=lambda x: (x['prediction'], x['winner_prob']), reverse=True)

    # Identify the single most likely winner (highest prob for stage 4)
    top_winner = max(results, key=lambda x: x['winner_prob'])

    return {"leaderboard": results, "predicted_winner": top_winner["team"]}
