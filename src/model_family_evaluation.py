
# src/model_family_evaluation.py

import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)


class RegressionFamilyEvaluator:
    """
    Evaluate already-trained regression models
    and generate example predictions.

    No retraining or hyperparameter tuning occurs.
    """

    def __init__(self, X_train, X_test, y_test, features):
        self.features = list(features)

        self.X_test = X_test[self.features].copy()
        self.y_test = y_test.copy()

        # Representative example from training data only
        self.example_patient = (
            X_train[self.features]
            .median()
            .to_frame()
            .T
        )

        if not self.X_test.index.equals(self.y_test.index):
            raise ValueError(
                "Test features and targets are misaligned."
            )

    def evaluate_model(self, model_name, model):
        """Calculate test metrics for a fitted model."""

        predictions = model.predict(self.X_test)

        return {
            "Model": model_name,
            "Test R2": r2_score(
                self.y_test, predictions
            ),
            "Test MAE": mean_absolute_error(
                self.y_test, predictions
            ),
            "Test MAPE (%)": (
                mean_absolute_percentage_error(
                    self.y_test, predictions
                ) * 100
            )
        }

    def example_prediction(self, model):
        """
        Predict progression for a representative
        training-median patient profile.
        """

        return float(
            model.predict(self.example_patient)[0]
        )
