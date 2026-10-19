
# src/knn_regression.py

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)


class KNNRegression:
    """
    Train and evaluate K-Nearest Neighbors regression
    models for predicting diabetes progression.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        features
    ):
        self.features = list(features)

        self.X_train = X_train[self.features].copy()
        self.y_train = y_train.copy()

        self.X_val = X_val[self.features].copy()
        self.y_val = y_val.copy()

        self.models = {}

    def train_models(self, k_values=(3, 9)):
        """Train KNN regression models with different K values."""

        for k in k_values:

            if k < 1 or k > len(self.X_train):
                raise ValueError(
                    "K must be between 1 and the training sample count."
                )

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(
                    n_neighbors=k,
                    weights="uniform",
                    metric="minkowski",
                    p=2
                ))
            ])

            model.fit(self.X_train, self.y_train)

            self.models[k] = model

        return self.models

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate the required regression metrics."""

        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE (%)": (
                mean_absolute_percentage_error(
                    y_true, y_pred
                ) * 100
            )
        }

    def compare_models(self):
        """Compare training and validation performance."""

        if not self.models:
            raise ValueError(
                "Train the KNN models before comparing them."
            )

        results = []

        for k, model in self.models.items():

            train_predictions = model.predict(self.X_train)
            val_predictions = model.predict(self.X_val)

            train_metrics = self.calculate_metrics(
                self.y_train,
                train_predictions
            )

            val_metrics = self.calculate_metrics(
                self.y_val,
                val_predictions
            )

            results.append({
                "Model": f"KNN Regression K={k}",
                "K": k,
                "Weights": "uniform",
                "Features": len(self.features),
                "Train R2": train_metrics["R2"],
                "Validation R2": val_metrics["R2"],
                "Train MAE": train_metrics["MAE"],
                "Validation MAE": val_metrics["MAE"],
                "Train MAPE (%)": train_metrics["MAPE (%)"],
                "Validation MAPE (%)": val_metrics["MAPE (%)"]
            })

        return pd.DataFrame(results)
