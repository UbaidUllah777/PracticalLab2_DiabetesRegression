
# src/univariate_polynomial.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)


class UnivariatePolynomialRegression:
    """
    Train and evaluate polynomial regression models
    of degrees 0 through 5 using BMI as the predictor.
    """

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train[["bmi"]].copy()
        self.y_train = y_train.copy()

        self.X_val = X_val[["bmi"]].copy()
        self.y_val = y_val.copy()

        self.models = {}

    def train_models(self):
        """Train six polynomial regression models."""

        for degree in range(6):

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("polynomial", PolynomialFeatures(
                    degree=degree,
                    include_bias=True
                )),
                ("regression", LinearRegression(
                    fit_intercept=False
                ))
            ])

            pipeline.fit(self.X_train, self.y_train)

            self.models[degree] = pipeline

        return self.models

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate the required regression metrics."""

        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE (%)": (
                mean_absolute_percentage_error(y_true, y_pred) * 100
            )
        }

    def compare_models(self):
        """Compare training and validation performance."""

        if not self.models:
            raise ValueError(
                "Train the models before comparing them."
            )

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

            results.append({
                "Degree": degree,
                "Train R2": train_metrics["R2"],
                "Validation R2": val_metrics["R2"],
                "Train MAE": train_metrics["MAE"],
                "Validation MAE": val_metrics["MAE"],
                "Train MAPE (%)": train_metrics["MAPE (%)"],
                "Validation MAPE (%)": val_metrics["MAPE (%)"]
            })

        return pd.DataFrame(results)


    def plot_model_across_splits(self, degree, X_test, y_test):
        """
        Plot the same trained polynomial model against
        training, validation, and test observations.

        No model retraining is performed.
        """

        if degree not in self.models:
            raise ValueError("The selected model is not trained.")

        model = self.models[degree]

        X_test_bmi = X_test[["bmi"]].copy()

        datasets = [
            ("Training", self.X_train, self.y_train, "steelblue"),
            ("Validation", self.X_val, self.y_val, "darkorange"),
            ("Test", X_test_bmi, y_test, "forestgreen")
        ]

        # Use one shared BMI range for all three plots
        all_bmi = np.concatenate([
            features["bmi"].to_numpy()
            for _, features, _, _ in datasets
        ])

        bmi_grid = np.linspace(
            all_bmi.min(),
            all_bmi.max(),
            300
        )

        grid_df = pd.DataFrame({"bmi": bmi_grid})

        # Predict using the already-trained model
        curve_predictions = model.predict(grid_df)

        fig, axes = plt.subplots(
            1, 3,
            figsize=(17, 5),
            sharex=True,
            sharey=True
        )

        for ax, (name, features, target, color) in zip(
            axes, datasets
        ):
            ax.scatter(
                features["bmi"],
                target,
                alpha=0.6,
                s=30,
                color=color,
                label=f"{name} observations"
            )

            ax.plot(
                bmi_grid,
                curve_predictions,
                color="red",
                linewidth=2.5,
                label=f"Degree {degree} fit"
            )

            ax.set_title(f"{name} Set (n={len(features)})")
            ax.set_xlabel("BMI (Original Scale)")
            ax.set_ylabel("Diabetes Progression Score")
            ax.legend()
            ax.grid(alpha=0.3)

        fig.suptitle(
            f"Polynomial Regression (Degree {degree}) "
            "- Training, Validation and Test",
            fontsize=15
        )

        plt.tight_layout()
        plt.show()

 
    def get_quadratic_coefficients(self, degree=2):
        """
        Convert the fitted quadratic model's coefficients
        from standardized BMI to original BMI units.
        """
        if degree != 2:
            raise ValueError(
                "This method is designed for Degree 2."
            )

        model = self.models[degree]

        scaler = model.named_steps["scaler"]
        regression = model.named_steps["regression"]

        a0, a1, a2 = regression.coef_

        mean = scaler.mean_[0]
        scale = scaler.scale_[0]

        # Convert from z = (BMI - mean) / scale
        b0 = a0 - (a1 * mean / scale) + (
            a2 * mean**2 / scale**2
        )

        b1 = (a1 / scale) - (
            2 * a2 * mean / scale**2
        )

        b2 = a2 / scale**2

        return {
            "Intercept": float(b0),
            "BMI": float(b1),
            "BMI Squared": float(b2)
        }

    def predict_for_bmi(self, degree, bmi_value):
        """Predict progression for an original-scale BMI."""

        model = self.models[degree]

        bmi_input = pd.DataFrame({
            "bmi": [float(bmi_value)]
        })

        return float(model.predict(bmi_input)[0])

    def parameter_summary(self):
        """Count fitted coefficients for each polynomial degree."""

        rows = []

        for degree, model in self.models.items():
            polynomial = model.named_steps["polynomial"]
            regression = model.named_steps["regression"]

            feature_names = polynomial.get_feature_names_out(
                ["bmi_scaled"]
            )

            rows.append({
                "Degree": degree,
                "Polynomial Terms": ", ".join(feature_names),
                "Trainable Parameters": len(regression.coef_)
            })

        return pd.DataFrame(rows)

    
    def get_test_error_report(self, degree, X_test, y_test):
        """
        Analyze prediction errors on the held-out test set.

        Residual = Actual - Predicted
        Positive residual means underprediction.
        Negative residual means overprediction.
        """

        if degree not in self.models:
            raise ValueError("Selected model is not trained.")

        model = self.models[degree]
        X_test_bmi = X_test[["bmi"]].copy()

        predictions = model.predict(X_test_bmi)

        errors = pd.DataFrame(
            {
                "BMI": X_test_bmi["bmi"].to_numpy(),
                "Actual": y_test.loc[X_test_bmi.index].to_numpy(),
                "Predicted": predictions
            },
            index=X_test_bmi.index
        )

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

    def plot_test_error_diagnostics(self, errors):
        """Visualize prediction errors for the test set."""

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))

        # 1. Actual versus predicted
        axes[0].scatter(
            errors["Actual"],
            errors["Predicted"],
            alpha=0.7,
            color="steelblue"
        )

        lower = min(
            errors["Actual"].min(),
            errors["Predicted"].min()
        )

        upper = max(
            errors["Actual"].max(),
            errors["Predicted"].max()
        )

        axes[0].plot(
            [lower, upper],
            [lower, upper],
            "r--",
            label="Perfect prediction"
        )

        axes[0].set_title("Actual vs Predicted")
        axes[0].set_xlabel("Actual Progression")
        axes[0].set_ylabel("Predicted Progression")
        axes[0].legend()

        # 2. Residuals versus BMI
        axes[1].scatter(
            errors["BMI"],
            errors["Residual"],
            alpha=0.7,
            color="darkorange"
        )

        axes[1].axhline(
            y=0,
            color="red",
            linestyle="--"
        )

        axes[1].set_title("Residuals vs BMI")
        axes[1].set_xlabel("BMI")
        axes[1].set_ylabel("Residual (Actual - Predicted)")

        # 3. Residual distribution
        axes[2].hist(
            errors["Residual"],
            bins=12,
            edgecolor="black",
            color="seagreen",
            alpha=0.8
        )

        axes[2].axvline(
            x=0,
            color="red",
            linestyle="--"
        )

        axes[2].set_title("Distribution of Test Residuals")
        axes[2].set_xlabel("Residual")
        axes[2].set_ylabel("Frequency")

        fig.suptitle(
            "Degree 2 Polynomial Regression - Test Error Analysis",
            fontsize=15
        )

        plt.tight_layout()
        plt.show()






    
