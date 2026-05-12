import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import os
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Evaluates trained models and generates performance metrics and plots.
    """
    
    def __init__(self, config_path="config.yaml"):
        self.reports_dir = 'reports'
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def evaluate_models(self, models_dict, X_test, y_test, dataset_name, task_type='binary'):
        logger.info(f"Evaluating models for dataset: {dataset_name}")
        results = []
        
        for model_name, model in models_dict.items():
            y_pred = model.predict(X_test)
            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                
            metrics = self._calculate_metrics(y_test, y_pred, y_proba, task_type)
            metrics['model'] = model_name
            metrics['dataset'] = dataset_name
            results.append(metrics)
            
            self._plot_confusion_matrix(y_test, y_pred, dataset_name, model_name)
            
        return pd.DataFrame(results)

    def _calculate_metrics(self, y_true, y_pred, y_proba, task_type):
        if task_type == 'binary':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
            }
            if y_proba is not None:
                # Handle binary classification roc_auc where y_proba might have 2 columns
                if y_proba.shape[1] == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        else: # multiclass
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            if y_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                except ValueError:
                    metrics['roc_auc'] = np.nan # If classes don't match
                    
        return metrics

    def _plot_confusion_matrix(self, y_true, y_pred, dataset_name, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        
        plot_path = os.path.join(self.reports_dir, f"{dataset_name}_{model_name}_cm.png")
        plt.savefig(plot_path)
        plt.close()
        
    def generate_summary_report(self, results_df):
        if results_df.empty:
            logger.warning("No results to generate summary.")
            return
            
        csv_path = os.path.join(self.reports_dir, "summary_report.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Summary report saved to {csv_path}")
        
        # Plot model comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='dataset', y='accuracy', hue='model')
        plt.title('Model Accuracy Comparison across Datasets')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "model_comparison.png"))
        plt.close()
