# src/diabetes_eda.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class DiabetesEDA:
    """
    Reusable exploratory data analysis tools
    for the Scikit-Learn Diabetes dataset.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

        if "target" not in self.df.columns:
            raise ValueError("Dataset must contain a target column.")

    def descriptive_statistics(self):
        """Return descriptive statistics."""
        return self.df.describe().T

    def data_quality_report(self):
        """Summarize missing values, types and unique values."""
        return pd.DataFrame({
            "Data Type": self.df.dtypes.astype(str),
            "Missing Values": self.df.isna().sum(),
            "Unique Values": self.df.nunique()
        })

    def plot_scatter_relationships(self):
        """Plot selected predictors against diabetes progression."""

        features = ["age", "bmi", "bp", "s5"]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        for ax, feature in zip(axes.flat, features):
            sns.regplot(
                data=self.df,
                x=feature,
                y="target",
                ax=ax,
                scatter_kws={"alpha": 0.5, "s": 25},
                line_kws={"color": "red"}
            )

            ax.set_title(f"{feature.upper()} vs Diabetes Progression")
            ax.set_xlabel(feature.upper())
            ax.set_ylabel("Progression Score")

        fig.suptitle(
            "Relationships Between Selected Features and Diabetes Progression",
            fontsize=14
        )

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self):
        """Visualize Pearson correlations between numeric variables."""

        correlation = self.df.corr(numeric_only=True)

        plt.figure(figsize=(12, 9))

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5
        )

        plt.title("Diabetes Dataset - Correlation Matrix")
        plt.tight_layout()
        plt.show()

    def get_target_correlations(self):
        """Return feature correlations with the target."""

        correlation = self.df.corr(numeric_only=True)

        return (
            correlation["target"]
            .drop("target")
            .sort_values(ascending=False)
        )

    
    def outlier_report(self):
        """
        Detect potential outliers using the 1.5*IQR rule.

        The encoded categorical feature 'sex' is excluded.
        The target is included for descriptive inspection only.
        """
        numeric_columns = self.df.select_dtypes(
            include="number"
        ).columns.drop("sex", errors="ignore")

        results = []

        for column in numeric_columns:
            q1 = self.df[column].quantile(0.25)
            q3 = self.df[column].quantile(0.75)

            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = (
                (self.df[column] < lower_bound) |
                (self.df[column] > upper_bound)
            )

            results.append({
                "Feature": column,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower Bound": lower_bound,
                "Upper Bound": upper_bound,
                "Outlier Count": int(outliers.sum()),
                "Outlier %": 100 * outliers.mean()
            })

        return pd.DataFrame(results).set_index("Feature")

    def plot_boxplots(self):
        """Visualize potential outliers in selected variables."""

        features = ["bmi", "bp", "s3", "s5", "target"]

        fig, axes = plt.subplots(
            1, len(features), figsize=(15, 5)
        )

        for ax, feature in zip(axes, features):
            sns.boxplot(
                y=self.df[feature],
                ax=ax,
                color="skyblue"
            )

            ax.set_title(feature.upper())
            ax.set_ylabel("Value")

        fig.suptitle(
            "Boxplots of Selected Diabetes Variables",
            fontsize=15
        )

        plt.tight_layout()
        plt.show()



    
