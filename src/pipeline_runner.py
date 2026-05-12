import logging
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator

logger = logging.getLogger(__name__)

class PipelineRunner:
    """
    Orchestrates the entire ML pipeline end-to-end.
    """
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        # Less aggressive feature engineering for faster runs
        self.feature_engineer = FeatureEngineer(degree=1, k_best=10)
        self.model_trainer = ModelTrainer()
        self.evaluator = Evaluator()

    def run_all(self):
        all_results = []
        
        for dataset_name in self.data_loader.available_datasets:
            try:
                results = self.run_single(dataset_name)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Failed pipeline for {dataset_name}: {e}", exc_info=True)
                
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            self.evaluator.generate_summary_report(final_df)
            logger.info("Full pipeline execution complete.")

    def run_single(self, dataset_name, models_to_run=None):
        logger.info(f"=== Starting Pipeline for {dataset_name} ===")
        
        # 1. Load Data
        X, y, metadata = self.data_loader.load_dataset(dataset_name)
        task_type = metadata.get('task', 'binary')
        
        # Train-test split before preprocessing to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 2. Preprocessing
        logger.info("Preprocessing...")
        X_train_prep = self.preprocessor.fit_transform(X_train, y_train)
        X_test_prep = self.preprocessor.transform(X_test)
        joblib.dump(self.preprocessor, os.path.join('models', f'{dataset_name}_preprocessor.pkl'))
        
        # 3. Feature Engineering
        logger.info("Feature Engineering...")
        X_train_eng = self.feature_engineer.fit_transform(X_train_prep, y_train)
        X_test_eng = self.feature_engineer.transform(X_test_prep)
        joblib.dump(self.feature_engineer, os.path.join('models', f'{dataset_name}_feature_engineer.pkl'))
        
        # 4. Model Training & Tuning
        best_models, cv_results = self.model_trainer.train_and_tune(X_train_eng, y_train, dataset_name, models_to_run)
        
        # 5. Evaluation
        eval_results = self.evaluator.evaluate_models(best_models, X_test_eng, y_test, dataset_name, task_type)
        
        logger.info(f"=== Completed Pipeline for {dataset_name} ===")
        return eval_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    runner = PipelineRunner()
    runner.run_single('fifa')
