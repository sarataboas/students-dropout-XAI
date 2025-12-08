import pandas as pd
import xgboost as xgb   
 

class RobustXGBModel:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        # Force conversion of object/string columns to numeric
        X_numeric = X.apply(pd.to_numeric, errors='coerce')
        return self.model.predict(X_numeric)

    def predict_proba(self, X):
        X_numeric = X.apply(pd.to_numeric, errors='coerce')
        return self.model.predict_proba(X_numeric)