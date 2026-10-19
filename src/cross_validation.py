
# src/cross_validation.py

import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression


class DiabetesCrossValidation:
    """
    Perform leakage-safe K-Fold Cross-Validation
    for multivariate polynomial regression.
    """

    def __init__(
        self,
        X_train,
        y_train,
        n_splits=5,
        random_state=42
    ):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate(self, degree=2, k_features=4):
        """
        Evaluate a polynomial modeling procedure.

        Feature selection and scaling are fitted
        independently within each training fold.
        """

        pipeline = Pipeline([
            ("feature_selection", SelectKBest(
                score_func=f_regression,
                k=k_features
            )),
            ("scaler", StandardScaler()),
            ("polynomial", PolynomialFeatures(
                degree=degree,
                include_bias=True
            )),
            ("regression", LinearRegression(
                fit_intercept=False
            ))
        ])

        cv = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "mape": "neg_mean_absolute_percentage_error"
        }

        scores = cross_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            error_score="raise"
        )

        fold_results = pd.DataFrame({
            "Fold": range(1, self.n_splits + 1),

            "Train R2": scores["train_r2"],
            "Validation R2": scores["test_r2"],

            "Train MAE": -scores["train_mae"],
            "Validation MAE": -scores["test_mae"],

            "Train MAPE (%)": (
                -scores["train_mape"] * 100
            ),
            "Validation MAPE (%)": (
                -scores["test_mape"] * 100
            )
        })

        summary = (
            fold_results.drop(columns="Fold")
            .agg(["mean", "std"])
            .rename(index={
                "mean": "Mean",
                "std": "Standard Deviation"
            })
        )

        return fold_results, summary
