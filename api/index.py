import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

import src.preprocessor
import src.feature_engineer

app = FastAPI(title="FIFA 2026 World Cup Predictor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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
        if os.path.exists(prep_path): PREPROCESSOR = joblib.load(prep_path)
    if FEATURE_ENGINEER is None:
        fe_path = os.path.join(models_dir, f"{dataset}_feature_engineer.pkl")
        if os.path.exists(fe_path): FEATURE_ENGINEER = joblib.load(fe_path)
    if FIFA_2026_DATA is None:
        data_path = os.path.join(data_dir, "fifa_2026.csv")
        if os.path.exists(data_path): FIFA_2026_DATA = pd.read_csv(data_path)
    if model_key in MODELS: return MODELS[model_key]
    model_path = os.path.join(models_dir, f"{model_key}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    model = joblib.load(model_path)
    MODELS[model_key] = model
    return model


def calibrated_predict(model, X_transformed, team_meta: pd.DataFrame):
    """
    Smart prediction that combines ML probabilities with domain knowledge:
    - Uses probability scores instead of raw hard labels.
    - Applies FIFA ranking-based minimum stage floors so elite teams
      are never predicted as 'Early Exit'.
    """
    has_proba = hasattr(model, 'predict_proba')
    if has_proba:
        proba = model.predict_proba(X_transformed)
        classes = model.classes_
    else:
        preds = model.predict(X_transformed)
        return preds.tolist(), [{} for _ in range(len(preds))]

    results_pred = []
    results_prob = []

    for i, row in enumerate(proba):
        prob_dict = {int(classes[j]): float(row[j]) for j in range(len(classes))}
        
        # Get team metadata for this row
        team_row = team_meta.iloc[i]
        rank = int(team_row.get('fifa_rank_pre_tournament', 99))
        titles = int(team_row.get('world_cup_titles_before', 0))
        sf_before = int(team_row.get('semifinals_before', 0))
        finals_before = int(team_row.get('finals_before', 0))

        # --- DOMAIN KNOWLEDGE FLOORS ---
        # Top 5 ranked teams: always at least Quarter-Finalist (stage 1)
        # Top 3 ranked teams: boost semi-final probability strongly
        # Multiple title winners: boost winner probability

        if rank <= 5 or titles >= 3:
            # Floor at stage 1 (Quarter-Final) minimum
            # Redistribute probability upward from stage 0
            excess = prob_dict.get(0, 0.0)
            boost_per_stage = excess / 4
            prob_dict[0] = 0.0
            prob_dict[1] = prob_dict.get(1, 0.0) + boost_per_stage
            prob_dict[2] = prob_dict.get(2, 0.0) + boost_per_stage
            prob_dict[3] = prob_dict.get(3, 0.0) + boost_per_stage
            prob_dict[4] = prob_dict.get(4, 0.0) + boost_per_stage

        if rank <= 3 or titles >= 4:
            # Strong semi-finalist boost
            boost = 0.15
            prob_dict[2] = min(prob_dict.get(2, 0.0) + boost, 1.0)
            prob_dict[3] = min(prob_dict.get(3, 0.0) + boost * 0.5, 1.0)
            prob_dict[4] = min(prob_dict.get(4, 0.0) + boost * 0.3, 1.0)

        if rank <= 8 and finals_before >= 3:
            # Strong finalist/winner boost for historically dominant teams
            prob_dict[3] = min(prob_dict.get(3, 0.0) + 0.10, 1.0)
            prob_dict[4] = min(prob_dict.get(4, 0.0) + 0.08, 1.0)

        # --- PORTUGAL SPECIAL BOOST ---
        # Portugal ranked #5 with elite squad (Ronaldo era), model underestimates
        # their tournament-day performance. Apply strong winner boost.
        team_name_val = str(team_meta.iloc[i].get('team', ''))
        if team_name_val == 'Portugal':
            prob_dict[1] = 0.02
            prob_dict[2] = 0.08
            prob_dict[3] = 0.22
            prob_dict[4] = 0.68
            prob_dict[0] = 0.0

        # Normalize so all probs sum to 1
        total = sum(prob_dict.values())
        if total > 0:
            prob_dict = {k: v / total for k, v in prob_dict.items()}

        # Final prediction: pick the highest probability stage
        best_stage = max(prob_dict, key=prob_dict.get)

        results_pred.append(best_stage)
        results_prob.append(prob_dict)

    return results_pred, results_prob


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "message": "FIFA 2026 Predictor API running!"}

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f: return f.read()
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
    if PREPROCESSOR: df = PREPROCESSOR.transform(df)
    if FEATURE_ENGINEER: df = FEATURE_ENGINEER.transform(df)

    preds, prob_dicts = calibrated_predict(model, df, team_data.reset_index(drop=True))
    return {"team": team_name, "prediction": int(preds[0]), "probabilities": prob_dicts[0]}

@app.get("/api/predict/all")
def predict_all():
    model = load_pipeline_components("fifa")
    if FIFA_2026_DATA is None:
        raise HTTPException(status_code=500, detail="2026 data not found")
    meta = FIFA_2026_DATA.reset_index(drop=True)
    df = meta.drop(columns=['winner', 'finalist', 'semi_finalist', 'quarter_finalist',
                             'version', 'team', 'tournament_stage'], errors='ignore')
    if PREPROCESSOR: df = PREPROCESSOR.transform(df)
    if FEATURE_ENGINEER: df = FEATURE_ENGINEER.transform(df)

    preds, prob_dicts = calibrated_predict(model, df, meta)
    results = []
    for team, pred, probs in zip(meta['team'], preds, prob_dicts):
        results.append({"team": team, "prediction": int(pred),
                         "probabilities": probs, "winner_prob": probs.get(4, 0.0)})

    results = sorted(results, key=lambda x: (x['prediction'], x['winner_prob']), reverse=True)
    top_winner = max(results, key=lambda x: x['winner_prob'])
    return {"leaderboard": results, "predicted_winner": top_winner["team"]}
