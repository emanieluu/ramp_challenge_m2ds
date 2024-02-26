import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


class Regressor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = [
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

        numeric_features = self.features[
            :-2
        ]  # Assuming the last two are categorical
        categorical_features = self.features[-2:]

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(n_estimators=100, random_state=42),
                ),
            ]
        )

    def _convert_array_to_df(self, X):
        """Converts a numpy array to a pandas DataFrame with the correct column names."""
        return pd.DataFrame(X, columns=self.features)

    def fit(self, X, y):
        X_df = self._convert_array_to_df(X)
        self.model.fit(X_df, y)

    def predict(self, X):
        X_df = self._convert_array_to_df(X)
        return self.model.predict(X_df)
