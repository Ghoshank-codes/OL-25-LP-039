import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class Label_Map_Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_map):
        self.encoding_map = encoding_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # Apply map to each column individually
            return X.apply(lambda col: col.map(self.encoding_map).fillna(0))
        elif isinstance(X, pd.Series):
            return X.map(self.encoding_map).fillna(0).to_frame()
        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")