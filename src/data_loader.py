import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_openml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Unified class to load and return datasets in a standard format:
    (X, y, metadata)
    """

    def __init__(self):
        self.available_datasets = ['iris', 'breast_cancer', 'wine', 'titanic', 'heart']

    def load_dataset(self, name: str):
        """
        Loads a specific dataset by name.
        """
        if name not in self.available_datasets:
            raise ValueError(f"Dataset {name} not found. Available: {self.available_datasets}")

        logger.info(f"Loading dataset: {name}")

        if name == 'iris':
            data = load_iris(as_frame=True)
            X, y = data.data, data.target
            metadata = {'name': 'iris', 'task': 'multiclass', 'classes': data.target_names.tolist()}
        elif name == 'breast_cancer':
            data = load_breast_cancer(as_frame=True)
            X, y = data.data, data.target
            metadata = {'name': 'breast_cancer', 'task': 'binary', 'classes': data.target_names.tolist()}
        elif name == 'wine':
            data = load_wine(as_frame=True)
            X, y = data.data, data.target
            metadata = {'name': 'wine', 'task': 'multiclass', 'classes': data.target_names.tolist()}
        elif name == 'titanic':
            # Titanic dataset from OpenML
            data = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
            X, y = data.data, data.target
            metadata = {'name': 'titanic', 'task': 'binary'}
        elif name == 'heart':
            # Heart disease dataset from OpenML
            # Note: Fetching might return different formats, we use ID 1565 or similar, but name 'heart' works usually.
            data = fetch_openml('heart-statlog', version=1, as_frame=True, parser='auto')
            X, y = data.data, data.target
            metadata = {'name': 'heart', 'task': 'binary'}

        # Ensure X is DataFrame, y is Series
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self._log_stats(name, X, y)
        return X, y, metadata

    def _log_stats(self, name: str, X: pd.DataFrame, y: pd.Series):
        logger.info(f"--- Stats for {name} ---")
        logger.info(f"Shape of X: {X.shape}")
        logger.info(f"Shape of y: {y.shape}")
        logger.info(f"Missing values in X: \n{X.isnull().sum()[X.isnull().sum() > 0]}")
        logger.info(f"Class distribution in y: \n{y.value_counts()}")
        logger.info("-" * 25)

if __name__ == "__main__":
    loader = DataLoader()
    X, y, meta = loader.load_dataset('iris')
