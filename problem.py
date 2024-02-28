import os
import pandas as pd
import rampwf as rw
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle

problem_title = 'Spotify Popularity Prediction'

features = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "playlist_genre",
    "playlist_subgenre",
]

_target_column_name = 'track_popularity'

Predictions = rw.prediction_types.make_regression()
workflow = rw.workflows.Regressor()

score_types = [
    rw.score_types.RMSE(name='rmse'),
    #rw.score_types.MARE(name='mare'),
]

def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, "data", f_name))
    y_array = data[_target_column_name].values
    X_array = data[features].values
    return X_array, y_array

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=57)
    return cv.split(X, y)
