
# src/logistic_regression.py

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)


class DiabetesLogisticRegression:
    """
    Classify lower/higher diabetes progression and
    derive approximate continuous progression scores.

    All thresholds and group means are determined
    using training data only.
    """

    def __init__(
        self, X_train, y_train, X_val, y_val, features
    ):
        self.features = list(features)

        self.X_train = X_train[self.features].copy()
        self.X_val = X_val[self.features].copy()

        self.y_train = y_train.copy()
        self.y_val = y_val.copy()

        # Calculate threshold from training data only
        self.threshold = float(self.y_train.median())

        self.y_train_class = (
            self.y_train >= self.threshold
        ).astype(int)

        self.y_val_class = (
            self.y_val >= self.threshold
        ).astype(int)

        # Class-specific average progression values
        self.group_means = {
            category: float(
                self.y_train[self.y_train_class == category].mean()
            )
            for category in (0, 1)
        }

        self.models = {}

    def train_models(self, c_values=(0.1, 10.0)):
        """Train two Logistic Regression classifiers."""

        for c in c_values:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    C=c,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42
                ))
            ])

            model.fit(
                self.X_train,
                self.y_train_class
            )

            self.models[c] = model

        return self.models

    def predict_progression_score(self, model, X):
        """
        Derive an approximate continuous score using
        class probabilities and training group means.
        """

        X_selected = X[self.features]

        probabilities = model.predict_proba(X_selected)

        class_labels = model.named_steps[
            "classifier"
        ].classes_

        positive_index = np.where(
            class_labels == 1
        )[0][0]

        p_high = probabilities[:, positive_index]

        low_mean = self.group_means[0]
        high_mean = self.group_means[1]

        return (
            (1 - p_high) * low_mean
            + p_high * high_mean
        )

    def compare_models(self):
        """Return classification and score comparison tables."""

        if not self.models:
            raise ValueError("Train models first.")

        classification_rows = []
        regression_rows = []

        datasets = [
            (
                "Train", self.X_train,
                self.y_train, self.y_train_class
            ),
            (
                "Validation", self.X_val,
                self.y_val, self.y_val_class
            )
        ]

        for c, model in self.models.items():
            class_row = {"Model": f"Logistic C={c}"}
            reg_row = {"Model": f"Logistic C={c} (Score Proxy)"}

            for name, X, y, y_class in datasets:

                # Native classification predictions
                predicted_class = model.predict(X)

                class_row[f"{name} Accuracy"] = accuracy_score(
                    y_class, predicted_class
                )
                class_row[f"{name} Precision"] = precision_score(
                    y_class, predicted_class,
                    zero_division=0
                )
                class_row[f"{name} Recall"] = recall_score(
                    y_class, predicted_class,
                    zero_division=0
                )
                class_row[f"{name} F1"] = f1_score(
                    y_class, predicted_class,
                    zero_division=0
                )

                # Derived continuous score predictions
                predicted_score = self.predict_progression_score(
                    model, X
                )

                reg_row[f"{name} R2"] = r2_score(
                    y, predicted_score
                )
                reg_row[f"{name} MAE"] = mean_absolute_error(
                    y, predicted_score
                )
                reg_row[f"{name} MAPE (%)"] = (
                    mean_absolute_percentage_error(
                        y, predicted_score
                    ) * 100
                )

            classification_rows.append(class_row)
            regression_rows.append(reg_row)

        return (
            pd.DataFrame(classification_rows),
            pd.DataFrame(regression_rows)
        )


    
    def evaluate_on_test(self, c, X_test, y_test):
        """
        Evaluate an already-trained Logistic Regression model
        on held-out test data.

        Returns classification metrics and derived score metrics.
        No retraining or tuning is performed.
        """

        if c not in self.models:
            raise ValueError("Requested model is not trained.")

        X_selected = X_test[self.features].copy()
        y_actual = y_test.copy()

        if not X_selected.index.equals(y_actual.index):
            raise ValueError(
                "Test features and targets are misaligned."
            )

        model = self.models[c]

        # Classification evaluation
        y_test_class = (
            y_actual >= self.threshold
        ).astype(int)

        predicted_classes = model.predict(X_selected)

        classification_results = pd.DataFrame([{
            "Model": f"Logistic C={c}",
            "Test Accuracy": accuracy_score(
                y_test_class, predicted_classes
            ),
            "Test Precision": precision_score(
                y_test_class, predicted_classes,
                zero_division=0
            ),
            "Test Recall": recall_score(
                y_test_class, predicted_classes,
                zero_division=0
            ),
            "Test F1": f1_score(
                y_test_class, predicted_classes,
                zero_division=0
            )
        }])

        # Derived continuous score evaluation
        predicted_scores = self.predict_progression_score(
            model, X_selected
        )

        score_results = pd.DataFrame([{
            "Model": f"Logistic C={c} (Score Proxy)",
            "Test R2": r2_score(
                y_actual, predicted_scores
            ),
            "Test MAE": mean_absolute_error(
                y_actual, predicted_scores
            ),
            "Test MAPE (%)": (
                mean_absolute_percentage_error(
                    y_actual, predicted_scores
                ) * 100
            )
        }])

        return classification_results, score_results
