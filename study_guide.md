# 📚 Study Guide — Advanced ML & Data Classification Pipeline

> Learn each topic **as you build**. Every phase = theory + code + practice.

---

## Phase 1 — Python Environment & Project Structure

### 📖 What to Learn
- Virtual environments (`venv`)
- Python package structure (`__init__.py`)
- YAML config files
- `requirements.txt` best practices

### 🧠 Core Concepts
| Concept | Why It Matters |
|---------|---------------|
| `venv` | Isolates project dependencies |
| `packages` | Makes `src/` importable across files |
| `config.yaml` | Centralize settings instead of hardcoding |

### 💻 Key Code Patterns
```python
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows

# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn joblib pyyaml

# Read config
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)
```

### ✏️ Mini Practice
- Create a `config.yaml` with dataset paths and a model list
- Import it in a test script and print values

---

## Phase 2 — Data Loading with Pandas

### 📖 What to Learn
- `pandas.read_csv()`, `DataFrame`, `Series`
- sklearn built-in datasets (`load_iris`, `load_breast_cancer`)
- Exploratory Data Analysis (EDA)

### 🧠 Core Concepts
| Concept | Description |
|---------|-------------|
| DataFrame | 2D labeled table (rows = samples, cols = features) |
| `.info()` | Data types, null counts |
| `.describe()` | Stats (mean, std, min, max) |
| `.value_counts()` | Class distribution |

### 💻 Key Code Patterns
```python
import pandas as pd
from sklearn.datasets import load_iris

# Load sklearn dataset into DataFrame
data = load_iris(as_frame=True)
X, y = data.data, data.target
print(X.shape, y.value_counts())

# Load CSV
df = pd.read_csv("data/raw/titanic.csv")
print(df.info())
print(df.isnull().sum())        # Count nulls per column
```

### ✏️ Mini Practice
- Load Iris, print shape and class distribution
- Load Titanic CSV, find columns with missing values

---

## Phase 3 — Automated Preprocessing

### 📖 What to Learn
- Handling missing values (imputation)
- Encoding categorical variables
- Feature scaling
- `sklearn.pipeline.Pipeline` & `ColumnTransformer`

### 🧠 Core Concepts
| Technique | When to Use |
|-----------|------------|
| `SimpleImputer(strategy='median')` | Numeric nulls |
| `SimpleImputer(strategy='most_frequent')` | Categorical nulls |
| `OneHotEncoder` | Nominal categories (no order) |
| `OrdinalEncoder` | Ordinal categories (have order) |
| `StandardScaler` | When features have very different ranges |
| `MinMaxScaler` | When you need values in [0, 1] range |

### 💻 Key Code Patterns
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ['age', 'fare']
categorical_features = ['sex', 'embarked']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
```

### ✏️ Mini Practice
- Apply the above preprocessor to Titanic data
- Print the shape before and after transformation

---

## Phase 4 — Feature Engineering & Selection

### 📖 What to Learn
- Creating new features from existing ones
- Log transforms for skewed data
- Polynomial features
- Feature selection methods

### 🧠 Core Concepts
| Technique | Purpose |
|-----------|---------|
| `PolynomialFeatures(degree=2)` | Captures non-linear relationships |
| `np.log1p(col)` | Reduces right-skewed distributions |
| `SelectKBest(f_classif, k=10)` | Keep top K statistically significant features |
| `VarianceThreshold` | Remove near-zero-variance (useless) features |
| `RFE` | Recursively eliminate weakest features |

### 💻 Key Code Patterns
```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures

# Log transform skewed column
df['log_fare'] = np.log1p(df['fare'])

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Select top 10 features
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
print("Selected features:", selector.get_support())
```

### ✏️ Mini Practice
- Apply `SelectKBest` to Breast Cancer dataset
- Compare model accuracy with all features vs. selected features

---

## Phase 5 — Model Training & Hyperparameter Tuning

### 📖 What to Learn
- Scikit-learn estimator API (`.fit()`, `.predict()`, `.score()`)
- Train/test split & cross-validation
- `GridSearchCV` for hyperparameter tuning
- `StratifiedKFold` for imbalanced datasets

### 🧠 Core Concepts
| Concept | Description |
|---------|-------------|
| Overfitting | Model memorizes training data, fails on new data |
| Cross-validation | Evaluate on multiple train/test splits — more reliable |
| `GridSearchCV` | Tries every param combination, picks best |
| `StratifiedKFold` | Preserves class ratio in each fold |

### 💻 Key Code Patterns
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)
```

### ✏️ Mini Practice
- Train a Logistic Regression baseline, note accuracy
- Run GridSearchCV, compare improved accuracy

---

## Phase 6 — Model Evaluation & Reporting

### 📖 What to Learn
- Classification metrics (Accuracy, Precision, Recall, F1)
- Confusion Matrix interpretation
- ROC curve & AUC score
- When to use which metric

### 🧠 Core Concepts — Metrics Cheat Sheet
| Metric | Formula | Best When |
|--------|---------|-----------|
| Accuracy | Correct / Total | Balanced classes |
| Precision | TP / (TP + FP) | False positives are costly |
| Recall | TP / (TP + FN) | False negatives are costly |
| F1-Score | 2 × (P × R) / (P + R) | Imbalanced classes |
| ROC-AUC | Area under ROC curve | Binary classification |

### 💻 Key Code Patterns
```python
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

y_pred = best_model.predict(X_test)

# Full report
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("reports/confusion_matrix.png")
plt.show()

# ROC AUC (binary)
y_prob = best_model.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, y_prob))
```

### ✏️ Mini Practice
- Train on Heart Disease dataset
- Print classification report, plot confusion matrix
- Calculate ROC-AUC and interpret it

---

## Phase 7 — Pipeline Orchestration

### 📖 What to Learn
- Writing a master runner script
- Python `argparse` for CLI arguments
- Saving models with `joblib`
- Logging with Python's `logging` module

### 💻 Key Code Patterns
```python
import argparse, joblib, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='all')
parser.add_argument('--model', type=str, default='all')
parser.add_argument('--tune', action='store_true')
args = parser.parse_args()

# Save model
joblib.dump(best_model, f"models/{dataset_name}_{model_name}.pkl")

# Load model
model = joblib.load("models/iris_rf.pkl")
```

### ✏️ Mini Practice
- Write `main.py` that accepts `--dataset iris` and runs the full pipeline
- Save and reload a trained model, verify predictions match

---

## Phase 8 — Testing with Pytest

### 📖 What to Learn
- Writing `pytest` unit tests
- `assert` statements for validation
- Fixtures for reusable test data
- Edge case testing

### 💻 Key Code Patterns
```python
# tests/test_preprocessor.py
import pytest
import pandas as pd
from src.preprocessor import Preprocessor

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'age': [25, None, 35],
        'sex': ['male', 'female', None],
        'survived': [1, 0, 1]
    })

def test_no_nulls_after_preprocessing(sample_df):
    prep = Preprocessor()
    X = sample_df.drop('survived', axis=1)
    X_transformed = prep.fit_transform(X)
    assert pd.isnull(X_transformed).sum() == 0  # No nulls remain

def test_output_shape(sample_df):
    prep = Preprocessor()
    X = sample_df.drop('survived', axis=1)
    X_out = prep.fit_transform(X)
    assert X_out.shape[0] == 3   # Same number of rows
```

**Run tests:**
```bash
pytest tests/ -v
```

### ✏️ Mini Practice
- Write a test that verifies `StandardScaler` output has mean ≈ 0
- Write a test for a model that checks output shape of `.predict()`

---

## Phase 9 — Git & GitHub Workflow

### 📖 What to Learn
- Git fundamentals (init, add, commit, push)
- Writing good commit messages
- `.gitignore` for ML projects
- GitHub README best practices

### 💻 Key Commands
```bash
git init
git add .
git commit -m "Phase 1: Project scaffold and environment setup"

git remote add origin https://github.com/arpansingha7/Advanced-ML-Classification-Pipeline.git
git push -u origin main
```

### 📄 `.gitignore` for ML Projects
```
venv/
__pycache__/
*.pkl
*.pyc
data/raw/       # Don't commit large datasets
.env
reports/*.png   # Optional: regenerate plots
```

### ✏️ Mini Practice
- Write 3 meaningful commit messages for Phases 1–3
- Create a GitHub repo and push your scaffold

---

## 🗺️ Study Path Summary

```
Week 1:  Phase 1 (Setup) → Phase 2 (Pandas + EDA)
Week 2:  Phase 3 (Preprocessing) → Phase 4 (Feature Engineering)
Week 3:  Phase 5 (Model Training + GridSearchCV)
Week 4:  Phase 6 (Evaluation) → Phase 7 (Orchestration)
Week 5:  Phase 8 (Testing) → Phase 9 (GitHub + README)
```

---

## 📌 Key Resources (Free)

| Topic | Resource |
|-------|----------|
| Pandas | [pandas.pydata.org/docs](https://pandas.pydata.org/docs/) |
| Scikit-learn | [scikit-learn.org/stable/user_guide](https://scikit-learn.org/stable/user_guide.html) |
| ML Theory | [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) |
| GridSearchCV | [sklearn GridSearchCV docs](https://scikit-learn.org/stable/modules/grid_search.html) |
| Pytest | [docs.pytest.org](https://docs.pytest.org/en/stable/) |
| Git | [git-scm.com/book](https://git-scm.com/book/en/v2) |
