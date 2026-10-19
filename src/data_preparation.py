
# src/data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split


class DiabetesDataPreparation:
    """
    Prepare the Diabetes dataset for machine learning.

    Splits data into:
    - Training: approximately 75%
    - Validation: approximately 10%
    - Testing: approximately 15%
    """

    def __init__(self, dataframe, random_state=42):
        self.df = dataframe.copy()
        self.random_state = random_state

        if "target" not in self.df.columns:
            raise ValueError("Target column is missing.")

    def split_data(self):
        """Create reproducible train/validation/test sets."""

        total_samples = len(self.df)

        # Calculate integer sample counts
        test_count = round(total_samples * 0.15)
        validation_count = round(total_samples * 0.10)

        if total_samples - test_count - validation_count < 1:
            raise ValueError("Not enough samples for splitting.")

        # First split: training + validation versus test
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_count,
            random_state=self.random_state,
            shuffle=True
        )

        # Second split: separate training and validation
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=validation_count,
            random_state=self.random_state,
            shuffle=True
        )

        # Separate features (X) and target (y)
        return {
            "X_train": train_df.drop(columns="target"),
            "X_val": val_df.drop(columns="target"),
            "X_test": test_df.drop(columns="target"),
            "y_train": train_df["target"],
            "y_val": val_df["target"],
            "y_test": test_df["target"]
        }
