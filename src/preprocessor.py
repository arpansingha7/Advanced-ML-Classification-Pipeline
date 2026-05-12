import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Automated preprocessing pipeline builder using ColumnTransformer.
    """

    def __init__(self, numeric_imputation='median', categorical_imputation='most_frequent'):
        self.numeric_imputation = numeric_imputation
        self.categorical_imputation = categorical_imputation
        self.pipeline = None
        self.numeric_features = []
        self.categorical_features = []

    def fit_transform(self, X: pd.DataFrame, y=None):
        self._build_pipeline(X)
        logger.info("Fitting and transforming data through preprocessing pipeline.")
        X_processed = self.pipeline.fit_transform(X, y)
        
        # Get feature names if possible (scikit-learn >= 1.0 supports get_feature_names_out)
        try:
            feature_names = self.pipeline.get_feature_names_out()
            return pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        except Exception as e:
            logger.warning(f"Could not retrieve feature names: {e}")
            return pd.DataFrame(X_processed, index=X.index)

    def transform(self, X: pd.DataFrame):
        if self.pipeline is None:
            raise ValueError("Pipeline has not been fitted yet.")
        logger.info("Transforming data through preprocessing pipeline.")
        X_processed = self.pipeline.transform(X)
        try:
            feature_names = self.pipeline.get_feature_names_out()
            return pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        except:
            return pd.DataFrame(X_processed, index=X.index)

    def _build_pipeline(self, X: pd.DataFrame):
        logger.info("Building preprocessing pipeline...")
        # Infer column types
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        logger.info(f"Numeric features identified: {len(self.numeric_features)}")
        logger.info(f"Categorical features identified: {len(self.categorical_features)}")

        # Numeric Transformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.numeric_imputation)),
            ('scaler', StandardScaler())
        ])

        # Categorical Transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.categorical_imputation)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine
        transformers = []
        if self.numeric_features:
            transformers.append(('num', numeric_transformer, self.numeric_features))
        if self.categorical_features:
            transformers.append(('cat', categorical_transformer, self.categorical_features))

        self.pipeline = ColumnTransformer(transformers=transformers, remainder='drop')
