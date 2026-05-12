# Advanced ML & Data Classification Pipeline — Implementation Plan

> **Tech Stack**: Python · Scikit-learn · Pandas · Matplotlib/Seaborn · Joblib  
> **Goal**: End-to-end automated ML classification pipeline across 5+ diverse datasets

---

## Project Overview

Build a professional, modular ML pipeline that automates:
- Data ingestion & preprocessing for diverse datasets
- Feature engineering & selection
- Model training with hyperparameter tuning (GridSearchCV + cross-validation)
- Evaluation, comparison & reporting
- **Web API & UI**: FastAPI backend with a responsive HTML/JS frontend
- **Vercel Deployment**: Serverless deployment for the trained models

This project will be structured as a **portfolio-grade Python package** with clean code, reproducibility, and CI-ready testing.

---

## Project Directory Structure

```
Advanced ML & Data Classification Pipeline/
│
├── data/                        # Raw & processed datasets
│   ├── raw/                     # Original CSVs (5+ datasets)
│   └── processed/               # Cleaned / encoded datasets
│
├── notebooks/                   # Exploratory analysis (EDA)
│   ├── 01_EDA_iris.ipynb
│   ├── 02_EDA_titanic.ipynb
│   └── ...
│
├── src/                         # Core pipeline package
│   ├── __init__.py
│   ├── data_loader.py           # Dataset ingestion utilities
│   ├── preprocessor.py          # Cleaning, encoding, scaling
│   ├── feature_engineer.py      # Feature creation & selection
│   ├── model_trainer.py         # Model fitting & GridSearchCV
│   ├── evaluator.py             # Metrics, confusion matrix, reports
│   └── pipeline_runner.py       # Orchestrator: run all datasets
│
├── models/                      # Saved trained models (.pkl)
├── reports/                     # Auto-generated classification reports
├── tests/                       # Unit tests for each module
│   ├── test_preprocessor.py
│   ├── test_feature_engineer.py
│   └── test_evaluator.py
│
├── config.yaml                  # Centralized config (models, params, paths)
├── requirements.txt
├── README.md
└── main.py                      # Entry point
```

---

## Implementation Phases

### ✅ Phase 1 — Project Scaffold & Environment Setup

**Goal**: Get a clean, reproducible Python environment and folder structure.

**Steps**:
1. Create virtual environment: `python -m venv venv`
2. Install core dependencies: `scikit-learn pandas numpy matplotlib seaborn joblib pyyaml`
3. Create the full folder structure above
4. Initialize `requirements.txt` and `config.yaml`
5. Write a barebones `README.md`

**Deliverable**: Working project shell, importable `src/` package.

---

### ✅ Phase 2 — Dataset Collection & Ingestion

**Goal**: Load 5+ diverse datasets with varying feature distributions.

**Planned Datasets**:
| # | Dataset | Type | Source |
|---|---------|------|--------|
| 1 | Iris | Multi-class | sklearn built-in |
| 2 | Titanic | Binary (imbalanced) | CSV / seaborn |
| 3 | Breast Cancer Wisconsin | Binary | sklearn built-in |
| 4 | Wine Quality | Multi-class (regression-like) | UCI / CSV |
| 5 | Heart Disease | Binary | Kaggle CSV |
| 6 | Bank Marketing | Binary (imbalanced) | UCI CSV |

**Steps**:
1. Write `data_loader.py` with a unified `DataLoader` class
2. Support sklearn built-ins + CSV loading via Pandas
3. Normalize output: always returns `(X, y, metadata)` tuple
4. Log dataset stats (shape, class distribution, dtypes)

---

### ✅ Phase 3 — Automated Preprocessing Pipeline

**Goal**: Cut data prep time by 40% via automated, configurable preprocessing.

**Steps**:
1. Write `preprocessor.py` with a `Preprocessor` class using `sklearn.pipeline.Pipeline`
2. Handle:
   - Missing value imputation (median for numeric, mode for categorical)
   - One-hot encoding for categorical features (`pd.get_dummies` / `OrdinalEncoder`)
   - Scaling: `StandardScaler` / `MinMaxScaler` (configurable per dataset)
   - Outlier clipping (IQR-based, optional)
   - Class imbalance: `SMOTE` or `class_weight='balanced'`
3. Make it configurable via `config.yaml`

**Key Pattern**: Use `sklearn.compose.ColumnTransformer` to apply different transforms to numeric vs. categorical columns cleanly.

---

### ✅ Phase 4 — Feature Engineering & Selection

**Goal**: Generate and select the most predictive features automatically.

**Steps**:
1. Write `feature_engineer.py`
2. Implement:
   - **Polynomial Features** (degree 2, for numeric columns)
   - **Interaction terms** (top correlated pairs)
   - **Log transforms** for skewed distributions
3. Feature Selection:
   - `SelectKBest` with chi2 / f_classif
   - `Recursive Feature Elimination (RFE)` with a base estimator
   - Variance threshold filter (remove near-zero variance features)
4. Log selected feature names and importance scores

---

### ✅ Phase 5 — Model Training & Hyperparameter Tuning

**Goal**: Achieve 15%+ accuracy improvement via systematic tuning.

**Classifiers to Train**:
| Model | Key Hyperparams |
|-------|----------------|
| Logistic Regression | C, solver, max_iter |
| Decision Tree | max_depth, min_samples_split |
| Random Forest | n_estimators, max_depth, min_samples_leaf |
| Gradient Boosting | n_estimators, learning_rate, max_depth |
| SVM | C, kernel, gamma |
| K-Nearest Neighbors | n_neighbors, metric |

**Steps**:
1. Write `model_trainer.py` with a `ModelTrainer` class
2. Define param grids in `config.yaml`
3. Use `GridSearchCV` with `StratifiedKFold(n_splits=5)` for all models
4. Track best params, best score, fit time per model
5. Save all trained models to `models/` using `joblib`

---

### ✅ Phase 6 — Evaluation & Reporting

**Goal**: Compare all models, auto-generate professional reports.

**Steps**:
1. Write `evaluator.py`
2. Compute per-model:
   - Accuracy, Precision, Recall, F1-Score (weighted)
   - ROC-AUC (binary) / macro-OVR AUC (multi-class)
   - Confusion Matrix
   - Classification Report
3. Generate summary comparison table (all models × all datasets)
4. Plot:
   - Confusion matrix heatmaps (Seaborn)
   - ROC curves
   - Feature importance bar charts
   - Model comparison bar chart
5. Save all plots to `reports/` as PNGs

---

### ✅ Phase 7 — Pipeline Orchestration

**Goal**: Single command runs the full pipeline on ALL datasets.

**Steps**:
1. Write `pipeline_runner.py` — ties all phases together
2. Write `main.py` entry point
3. Support CLI flags: `--dataset`, `--model`, `--tune` etc.
4. Produce a final `reports/summary_report.csv` with all results

**Usage**:
```bash
python main.py                     # Run all datasets, all models
python main.py --dataset iris      # Run only Iris dataset
python main.py --model rf --tune   # Run only Random Forest with tuning
```

---

### ✅ Phase 8 — Testing & Code Quality

**Goal**: Production-grade, testable, reproducible code.

**Steps**:
1. Write `pytest` unit tests for each module
2. Test preprocessor on edge cases (all-null cols, single class, etc.)
3. Test evaluator metrics against sklearn's own output
4. Add type hints and docstrings throughout
5. Add `random_state=42` everywhere for reproducibility

---

### ✅ Phase 9 — Web API & Vercel Integration

**Goal**: Make the pipeline interactive and deployable on Vercel.

**Steps**:
1. Add `fastapi` and `uvicorn` to dependencies.
2. Create `api/index.py` for Vercel Serverless Functions.
3. Expose a `/predict` endpoint that loads the best trained model (e.g., Iris or Breast Cancer) and makes predictions.
4. Create a vibrant, dynamic HTML/JS frontend (in `public/index.html` or similar) to allow users to input data and see predictions.
5. Create `vercel.json` for routing and configuration.

---

### ✅ Phase 10 — Documentation, GitHub Push & Deployment

**Goal**: Portfolio-ready project on GitHub and Vercel.

**Steps**:
1. Write a detailed `README.md` with:
   - Project description & motivation
   - Installation & Usage
   - Vercel Deployment Instructions
2. Initialize Git, set user config to `arpansingha7`.
3. Commit all phases with meaningful messages.
4. Push to GitHub: `Advanced-ML-Classification-Pipeline` under the `arpansingha7` account.
5. Deploy to Vercel (User action via Vercel Dashboard).
   - Project description & motivation
   - Installation steps
   - Usage examples
   - Results table (accuracy per dataset/model)
   - Architecture diagram (ASCII or image)
2. Initialize Git, commit all phases with meaningful messages
3. Push to GitHub: `Advanced-ML-Classification-Pipeline`
4. Add GitHub topics: `machine-learning`, `scikit-learn`, `pandas`, `classification`, `pipeline`

---

## Verification Plan

### After Each Phase
- Run `python -c "from src.<module> import <Class>"` to confirm imports
- Manually inspect outputs/logs for correctness

### End-to-End
- Run `python main.py` and confirm summary CSV is generated
- Run `pytest tests/` — all tests green
- Review `reports/` folder for plots and CSVs

### Success Metrics
| Metric | Target |
|--------|--------|
| Datasets processed | 5+ |
| Preprocessing automation | ≥ 40% time reduction vs manual |
| Best model accuracy gain (tuning) | ≥ 15% over baseline |
| Test coverage | ≥ 80% |
| GitHub push | ✅ Public repo |

---

## Open Questions

> [!IMPORTANT]
> **Q1**: Do you want to use **Jupyter Notebooks** for EDA in addition to the Python scripts, or scripts only?

> [!IMPORTANT]
> **Q2**: Should the pipeline support **deep learning models** (e.g., via `tensorflow` or `torch`) in future phases, or stay strictly scikit-learn?

> [!NOTE]
> **Q3**: Do you have any **specific datasets** already downloaded, or should I use sklearn built-ins + UCI/Kaggle public datasets?

> [!NOTE]
> **Q4**: Do you want a **Streamlit dashboard** at the end to visualize results interactively, or just static reports/plots?

---

## Execution Order Summary

```
Phase 1 → Setup & Scaffold (Completed)
Phase 2 → Data Loading (Completed)
Phase 3 → Preprocessing (Completed)
Phase 4 → Feature Engineering (Completed)
Phase 5 → Model Training + Tuning (In Progress)
Phase 6 → Evaluation + Reports
Phase 7 → Pipeline Orchestration
Phase 8 → Testing
Phase 9 → Web API & Vercel Integration (NEW)
Phase 10 → Docs + GitHub Push
```

**Estimated Total Time**: ~3–5 days of focused development
