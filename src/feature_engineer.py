import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Automates feature creation (polynomials/interactions) and feature selection.
    """

    def __init__(self, degree=2, k_best=10, variance_threshold=0.0):
        self.degree = degree
        self.k_best = k_best
        self.variance_threshold = variance_threshold
        self.pipeline = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Fitting and transforming data through feature engineering pipeline.")
        self._build_pipeline()
        X_engineered = self.pipeline.fit_transform(X, y)
        
        try:
            # Need to traverse pipeline steps to get feature names correctly
            # VarianceThreshold -> PolynomialFeatures -> SelectKBest (if order is changed)
            # Actually standard pipeline get_feature_names_out does this
            feature_names = self.pipeline.get_feature_names_out(X.columns)
            return pd.DataFrame(X_engineered, columns=feature_names, index=X.index)
        except Exception as e:
            logger.warning(f"Could not retrieve feature names after engineering: {e}")
            return pd.DataFrame(X_engineered, index=X.index)

    def transform(self, X: pd.DataFrame):
        if self.pipeline is None:
            raise ValueError("Pipeline has not been fitted yet.")
        logger.info("Transforming data through feature engineering pipeline.")
        X_engineered = self.pipeline.transform(X)
        try:
            feature_names = self.pipeline.get_feature_names_out(X.columns)
            return pd.DataFrame(X_engineered, columns=feature_names, index=X.index)
        except:
            return pd.DataFrame(X_engineered, index=X.index)

    def _build_pipeline(self):
        logger.info("Building feature engineering pipeline...")
        steps = []
        
        # 1. Variance Threshold (remove constant features)
        if self.variance_threshold is not None:
            steps.append(('variance_filter', VarianceThreshold(threshold=self.variance_threshold)))

        # 2. Polynomial & Interaction Features (only numeric, but since preprocessor outputs all numeric after encoding)
        # Note: applying polynomial features to one-hot encoded variables can create many redundant features
        # we will apply it generally but rely on SelectKBest to prune
        if self.degree > 1:
            steps.append(('poly_features', PolynomialFeatures(degree=self.degree, interaction_only=False, include_bias=False)))

        # 3. Feature Selection
        if self.k_best is not None and self.k_best != 'all':
            steps.append(('select_k_best', SelectKBest(score_func=f_classif, k=self.k_best)))

        self.pipeline = Pipeline(steps=steps)
