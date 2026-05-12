import pandas as pd
import kagglehub
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads the FIFA World Cup Team Dataset from kagglehub.
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
        
        # Target: winner (0 or 1)
        y = df['winner']
        
        # We will select only a subset of features to make the UI and API simpler.
        selected_features = [
            'fifa_rank_pre_tournament',
            'squad_total_market_value_eur',
            'wins_last_4y',
            'world_cup_titles_before',
            'is_host'
        ]
        
        X = df[selected_features]
        
        metadata = {'name': 'fifa', 'task': 'binary', 'classes': [0, 1]}

        self._log_stats(name, X, y)
        return X, y, metadata

    def _log_stats(self, name: str, X: pd.DataFrame, y: pd.Series):
        logger.info(f"--- Stats for {name} ---")
        logger.info(f"Shape of X: {X.shape}")
        logger.info(f"Shape of y: {y.shape}")
        logger.info(f"Class distribution in y: \n{y.value_counts()}")
        logger.info("-" * 25)

if __name__ == "__main__":
    loader = DataLoader()
    X, y, meta = loader.load_dataset('fifa')
