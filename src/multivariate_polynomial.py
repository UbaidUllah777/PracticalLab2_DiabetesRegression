
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)


class MultivariatePolynomialRegression:
    """
    Train and compare multivariate polynomial
    regression models using selected features.
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

        if len(self.features) < 2:
            raise ValueError(
                "Multivariate models require at least two features."
            )

        self.X_train = X_train[self.features].copy()
        self.y_train = y_train.copy()

        self.X_val = X_val[self.features].copy()
        self.y_val = y_val.copy()

        self.models = {}

    def train_models(self, degrees=(2, 3)):
        """Train multivariate polynomial models."""

        for degree in degrees:

            if degree <= 1:
                raise ValueError(
                    "Polynomial degree must be greater than 1."
                )

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("polynomial", PolynomialFeatures(
                    degree=degree,
                    include_bias=True
                )),
                ("regression", LinearRegression(
                    fit_intercept=False
                ))
            ])

            model.fit(self.X_train, self.y_train)

            self.models[degree] = model

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
        """Compare training and validation performance."""

        if not self.models:
            raise ValueError("Train the models first.")

        results = []

        for degree, model in self.models.items():

            train_predictions = model.predict(self.X_train)
            val_predictions = model.predict(self.X_val)

            train_metrics = self.calculate_metrics(
                self.y_train, train_predictions
            )

            val_metrics = self.calculate_metrics(
                self.y_val, val_predictions
            )

            parameter_count = len(
                model.named_steps["regression"].coef_
            )

            results.append({
                "Model": f"Multivariate Polynomial Degree {degree}",
                "Degree": degree,
                "Parameters": parameter_count,
                "Train R2": train_metrics["R2"],
                "Validation R2": val_metrics["R2"],
                "Train MAE": train_metrics["MAE"],
                "Validation MAE": val_metrics["MAE"],
                "Train MAPE (%)": train_metrics["MAPE (%)"],
                "Validation MAPE (%)": val_metrics["MAPE (%)"]
            })

        return pd.DataFrame(results)


    
    def evaluate_on_test(self, degree, X_test, y_test):
        """
        Evaluate an already-trained multivariate polynomial
        model on the held-out test dataset.

        No retraining or parameter tuning is performed.
        """

        if degree not in self.models:
            raise ValueError("Selected model has not been trained.")

        # Use the same selected features
        X_test_selected = X_test[self.features].copy()

        # Verify features and targets are aligned
        if not X_test_selected.index.equals(y_test.index):
            raise ValueError("Test features and targets are misaligned.")

        # Retrieve the trained model
        model = self.models[degree]

        # Generate test predictions
        predictions = model.predict(X_test_selected)

        # Calculate metrics using the existing OOP method
        metrics = self.calculate_metrics(y_test, predictions)

        # Create final evaluation table
        results = pd.DataFrame([{
            "Model": f"Multivariate Polynomial Degree {degree}",
            "Test R2": metrics["R2"],
            "Test MAE": metrics["MAE"],
            "Test MAPE (%)": metrics["MAPE (%)"]
        }])

        return results

    
    def get_model_equation_terms(self, degree):
        """Return polynomial terms and their fitted coefficients."""

        if degree not in self.models:
            raise ValueError("Model has not been trained.")

        model = self.models[degree]

        polynomial = model.named_steps["polynomial"]
        regression = model.named_steps["regression"]

        # z indicates a standardized input feature
        scaled_names = [
            f"z_{feature}" for feature in self.features
        ]

        terms = polynomial.get_feature_names_out(
            scaled_names
        )

        return pd.DataFrame({
            "Term": terms,
            "Coefficient": regression.coef_
        })

    def predict_patient(self, degree, patient_data):
        """Predict progression from a dictionary of feature values."""

        if degree not in self.models:
            raise ValueError("Model has not been trained.")

        patient_df = pd.DataFrame([patient_data])

        patient_df = patient_df[self.features]

        prediction = self.models[degree].predict(
            patient_df
        )

        return float(prediction[0])


    def get_test_error_report(self, degree, X_test, y_test):
        """
        Examine individual test prediction errors without
        retraining or modifying the selected model.
        """

        if degree not in self.models:
            raise ValueError("Selected model is not trained.")

        X_selected = X_test[self.features].copy()

        if not X_selected.index.equals(y_test.index):
            raise ValueError("Features and targets are misaligned.")

        # Predict using the existing trained model
        predictions = self.models[degree].predict(X_selected)

        # Build patient-level error report
        errors = pd.DataFrame({
            "BMI": X_selected["bmi"].to_numpy(),
            "Actual": y_test.to_numpy(),
            "Predicted": predictions
        }, index=y_test.index)

        errors["Residual"] = (
            errors["Actual"] - errors["Predicted"]
        )

        errors["Absolute Error"] = errors["Residual"].abs()

        errors["Absolute Percentage Error (%)"] = (
            errors["Absolute Error"]
            / errors["Actual"].abs()
            * 100
        )

        return errors


