import pandas as pd
import kagglehub
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads the FIFA World Cup Team Dataset from kagglehub for Multi-Class Classification.
    """

    def __init__(self):
        self.available_datasets = ['fifa']

    def load_dataset(self, name: str):
        if name != 'fifa':
            raise ValueError(f"Dataset {name} not supported. Only 'fifa' is allowed.")

        logger.info(f"Downloading/loading FIFA dataset via kagglehub...")
        path = kagglehub.dataset_download("harrachimustapha/fifa-world-cup-team-dataset")
        train_path = os.path.join(path, "train.csv")
        
        df = pd.read_csv(train_path)
        
        # Construct Multi-Class Target: tournament_stage
        # 0: Eliminated Early, 1: Quarter-Finals, 2: Semi-Finals, 3: Finalist, 4: Winner
        conditions = [
            (df['winner'] == 1),
            (df['finalist'] == 1) & (df['winner'] == 0),
            (df['semi_finalist'] == 1) & (df['finalist'] == 0),
            (df['quarter_finalist'] == 1) & (df['semi_finalist'] == 0)
        ]
        choices = [4, 3, 2, 1]
        df['tournament_stage'] = np.select(conditions, choices, default=0)
        
        y = df['tournament_stage']
        
        # Features: We use all relevant features, including categorical (continent)
        drop_cols = ['winner', 'finalist', 'semi_finalist', 'quarter_finalist', 'version', 'team', 'tournament_stage']
        X = df.drop(columns=drop_cols)
        
        metadata = {'name': 'fifa', 'task': 'multiclass', 'classes': [0, 1, 2, 3, 4]}

        self._log_stats(name, X, y)
        return X, y, metadata

    def _log_stats(self, name: str, X: pd.DataFrame, y: pd.Series):
        logger.info(f"--- Stats for {name} ---")
        logger.info(f"Shape of X: {X.shape}")
        logger.info(f"Shape of y: {y.shape}")
        logger.info(f"Class distribution in y: \n{y.value_counts().sort_index()}")
        logger.info("-" * 25)

if __name__ == "__main__":
    loader = DataLoader()
    X, y, meta = loader.load_dataset('fifa')
