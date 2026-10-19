# Practical Lab 2 — Diabetes Regression and Cross-Validation

**Course:** CSCN8010 — Foundations of Machine Learning  
**Project:** Multivariate Linear Regression, Non-Parametric Models and Cross-Validation  
**Language:** Python 3.13.7

## Project Overview

This project investigates machine learning approaches for predicting diabetes disease progression using the Diabetes dataset provided by Scikit-Learn.

The lab covers exploratory data analysis, data preparation, univariate and multivariate polynomial regression, Decision Tree Regression, K-Nearest Neighbors Regression, Logistic Regression Classification, model evaluation, and five-fold cross-validation.

The implementation uses Object-Oriented Programming (OOP), with reusable Python classes stored in the `src/` directory and experiments documented in a Jupyter notebook.

## Dataset

The dataset is loaded directly through Scikit-Learn using `load_diabetes(as_frame=True, scaled=False)`.

- **Observations:** 442
- **Input features:** 10
- **Target:** Continuous diabetes disease progression score
- **Missing values:** None
- **Duplicate records:** None

The dataset was divided reproducibly using `random_state=42`:

| Dataset | Observations | Approximate Percentage |
|---|---:|---:|
| Training | 332 | 75% |
| Validation | 44 | 10% |
| Testing | 66 | 15% |

## Project Implementation

### Part 1 — Exploratory Data Analysis and Preparation

- Defined the supervised regression problem.
- Examined dataset structure, descriptive statistics, and data quality.
- Visualized feature distributions, scatter relationships, and correlations.
- Investigated potential outliers using the interquartile range (IQR) method.
- Documented cleaning decisions and retained the original observations.
- Created reproducible training, validation, and testing datasets.

### Part 2 — Univariate Polynomial Regression

Six polynomial regression models, with degrees 0 through 5, were trained using BMI as the predictor.

Models were compared using R², MAE, and MAPE on the training and validation datasets.

Degree 2 was selected for final testing.

**Final Test Results:**

| Metric | Result |
|---|---:|
| R² | 0.215 |
| MAE | 54.842 |
| MAPE | 48.386% |

The notebook includes polynomial visualizations, model equations, fitted parameter counts, example predictions, and detailed error analysis.

### Part 3 — Multivariate and Non-Parametric Models

Four predictors were selected using correlations calculated exclusively from training observations:

`bmi`, `s5`, `bp`, `s4`

Eight models were evaluated:

- Multivariate Polynomial Regression: Degrees 2 and 3
- Decision Tree Regression: Maximum depths 3 and 6
- KNN Regression: K = 3 and K = 9
- Logistic Regression Classification: C = 0.1 and C = 10.0

Logistic Regression requires categorical targets. Therefore, the continuous progression target was converted into lower and higher progression groups using the training-set median. Separate classification metrics and probability-derived continuous score approximations are reported.

**Best Validation-Selected Native Regression Model:** Multivariate Polynomial Degree 2.

**Final Test Results:**

| Model | R² | MAE | MAPE |
|---|---:|---:|---:|
| Multivariate Polynomial Degree 2 | 0.450 | 45.301 | 40.336% |
| Decision Tree Depth 6 | 0.293 | 50.873 | 43.586% |
| KNN Regression K=9 | 0.420 | 45.281 | 41.182% |
| Logistic C=10.0 (derived score proxy) | 0.414 | 47.526 | 43.458% |

Although KNN achieved a marginally lower test MAE, the polynomial model remained the selected model because selection was based on validation performance rather than test results.

### Part 4 — Five-Fold Cross-Validation

Supplementary five-fold cross-validation was performed using only the original training data.

Feature selection and scaling were repeated independently within each fold to prevent data leakage.

**Cross-Validation Results:**

| Metric | Mean Validation Score | Standard Deviation |
|---|---:|---:|
| R² | 0.371 | 0.136 |
| MAE | 48.089 | 3.584 |
| MAPE | 42.916% | 6.842 |

The results demonstrate variation in predictive performance across different validation partitions.

## Project Structure

- `PracticalLab2_DiabetesRegression.ipynb` — Complete notebook with experiments, visualizations, results, and conclusions.
- `src/diabetes_eda.py` — Exploratory data analysis.
- `src/data_preparation.py` — Dataset splitting and preparation.
- `src/univariate_polynomial.py` — BMI-based polynomial regression.
- `src/multivariate_polynomial.py` — Multivariate polynomial regression.
- `src/decision_tree_regression.py` — Decision Tree Regression.
- `src/knn_regression.py` — KNN Regression.
- `src/logistic_regression.py` — Logistic Regression Classification and derived scores.
- `src/model_family_evaluation.py` — Additional model-family test evaluation.
- `src/cross_validation.py` — Five-fold cross-validation.
- `requirements.txt` — Python dependencies.
- `.gitignore` — Excluded local environments and generated files.

## Installation and Execution

**1. Clone the repository**

`git clone https://github.com/UbaidUllah777/PracticalLab2_DiabetesRegression.git`

**2. Open the project directory**

`cd PracticalLab2_DiabetesRegression`

**3. Create a virtual environment**

`python -m venv .venv`

**4. Activate the environment (Windows PowerShell)**

`.\.venv\Scripts\Activate.ps1`

**5. Install dependencies**

`python -m pip install -r requirements.txt`

**6. Open the notebook**

Launch VS Code or JupyterLab, select the Python virtual environment as the notebook kernel, and open `PracticalLab2_DiabetesRegression.ipynb`.

Run all cells sequentially from top to bottom.

The dataset is included with Scikit-Learn, so no separate CSV download is required.

## Key Findings

- Using multiple relevant patient features improved predictive performance compared with using BMI alone.
- More complex polynomial models did not necessarily generalize better.
- Deeper Decision Trees improved training performance but showed mixed validation results.
- KNN performance depended on the number of neighbors.
- Logistic Regression Classification required a categorical target and was evaluated separately from native regression.
- Cross-validation revealed meaningful variation between data partitions.
- Validation-based model selection helped avoid choosing the final model using test performance.

## Limitations

The dataset is relatively small, and substantial prediction errors remain for individual observations. The models demonstrate machine learning techniques rather than clinically validated prediction systems.

The results must not be used for medical diagnosis, screening, or treatment decisions.

## Conclusion

This practical lab demonstrates an end-to-end machine learning workflow using reproducible data preparation, object-oriented model implementations, multiple evaluation metrics, validation-based model selection, and supplementary cross-validation.

Multivariate Polynomial Degree 2 achieved the strongest validation MAE among the native regression models, improving test performance compared with the univariate BMI-only approach.

The project highlights the importance of model complexity, generalization, careful evaluation, and transparent reporting of limitations.