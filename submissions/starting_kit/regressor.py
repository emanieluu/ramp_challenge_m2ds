import numpy as np
from sklearn.base import BaseEstimator
import random

class Regressor(BaseEstimator):
    def __init__(self):
        return

    def fit(self, X, y):
        return

    def predict(self, X):
        # Generate random predictions
        y_pred = np.random.randint(0, 101, size=X.shape[0])
        return y_pred