
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)


class DecisionTreeRegression:
    """
    Train and evaluate Decision Tree regression models
    using the selected Diabetes dataset features.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        features,
        random_state=42
    ):
        self.features = list(features)
        self.X_train = X_train[self.features].copy()
        self.y_train = y_train.copy()
        self.X_val = X_val[self.features].copy()
        self.y_val = y_val.copy()
        self.random_state = random_state

        self.models = {}

    def train_models(self, depths=(3, 6)):
        """Train trees with different maximum depths."""

        for depth in depths:
            model = DecisionTreeRegressor(
                max_depth=depth,
                min_samples_leaf=5,
                random_state=self.random_state
            )

            model.fit(self.X_train, self.y_train)
            self.models[depth] = model

        return self.models

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate the required regression metrics."""

        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE (%)": (
                mean_absolute_percentage_error(y_true, y_pred)
                * 100
            )
        }

    def compare_models(self):
        """Compare training and validation results."""

        if not self.models:
            raise ValueError("Train the models first.")

        results = []

        for depth, model in self.models.items():

            train_predictions = model.predict(self.X_train)
            val_predictions = model.predict(self.X_val)

            train_metrics = self.calculate_metrics(
                self.y_train, train_predictions
            )

            val_metrics = self.calculate_metrics(
                self.y_val, val_predictions
            )

            results.append({
                "Model": f"Decision Tree Depth {depth}",
                "Max Depth": depth,
                "Actual Depth": model.get_depth(),
                "Leaf Nodes": model.get_n_leaves(),
                "Total Nodes": model.tree_.node_count,
                "Train R2": train_metrics["R2"],
                "Validation R2": val_metrics["R2"],
                "Train MAE": train_metrics["MAE"],
                "Validation MAE": val_metrics["MAE"],
                "Train MAPE (%)": train_metrics["MAPE (%)"],
                "Validation MAPE (%)": val_metrics["MAPE (%)"]
            })

        return pd.DataFrame(results)
