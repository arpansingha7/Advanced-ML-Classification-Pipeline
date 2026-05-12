import pandas as pd
import joblib
import os
import yaml
import logging
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Automates model training and hyperparameter tuning using GridSearchCV.
    """
    
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.random_state = self.config.get('random_state', 42)
        self.models_dir = self.config.get('directories', {}).get('models', 'models')
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000, class_weight='balanced'),
            'random_forest': RandomForestClassifier(random_state=self.random_state, class_weight='balanced'),
            'svm': SVC(random_state=self.random_state, probability=True, class_weight='balanced'),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state, class_weight='balanced'),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'knn': KNeighborsClassifier()
        }
        
        # Minimal default grids if not in config
        self.param_grids = self.config.get('param_grids', {
            'logistic_regression': {'C': [0.1, 1.0, 10.0]},
            'random_forest': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
            'svm': {'C': [0.1, 1.0], 'kernel': ['linear', 'rbf']},
            'decision_tree': {'max_depth': [None, 10, 20]},
            'gradient_boosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            'knn': {'n_neighbors': [3, 5, 7]}
        })

    def _load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def train_and_tune(self, X: pd.DataFrame, y: pd.Series, dataset_name: str, models_to_run=None):
        logger.info(f"Starting training for dataset: {dataset_name}")
        best_estimators = {}
        results = []
        
        models_to_run = models_to_run or list(self.models.keys())
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for model_name in models_to_run:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not found. Skipping.")
                continue
                
            logger.info(f"Tuning {model_name}...")
            model = self.models[model_name]
            param_grid = self.param_grids.get(model_name, {})
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_
            
            logger.info(f"Best score for {model_name}: {best_score:.4f}")
            logger.info(f"Best params: {grid_search.best_params_}")
            
            # Save the model
            model_path = os.path.join(self.models_dir, f"{dataset_name}_{model_name}.pkl")
            joblib.dump(best_model, model_path)
            
            best_estimators[model_name] = best_model
            results.append({
                'dataset': dataset_name,
                'model': model_name,
                'best_cv_accuracy': best_score,
                'best_params': str(grid_search.best_params_)
            })
            
        return best_estimators, pd.DataFrame(results)
