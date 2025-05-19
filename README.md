# Multivariable Linear Regression - CSOC Assignment

**Author:** Yogesh Kumar  
**Roll Number:** 23124052  
**Course:** CSOC-IG

## 📌 Objective

To implement and compare three different approaches to multivariable linear regression, focusing on convergence speed and predictive accuracy using the California Housing Price Dataset.

## 🧠 Methods Implemented

### Part 1: Pure Python
- Gradient Descent using core Python (no external libraries).
- Educational implementation focused on mathematical clarity.

### Part 2: NumPy
- Optimized version using NumPy for vectorized operations.
- Significantly faster with identical results.

### Part 3: Scikit-learn
- Uses `LinearRegression` from `sklearn`.
- Fastest and most concise implementation.

## 📊 Evaluation Metrics

Compared all methods using:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

Also compared convergence speeds via training time and plotted MSE vs. Epochs.

## 📁 Contents

- `pure_python.py` – Part 1
- `numpy_implementation.py` – Part 2
- `report_LINEAR_REGRESSION.tex` – Final LaTeX report
- `README.md` – This file

## 📝 Declaration

I used AI tools only to improve readability, reduce redundancy, and handle small errors—not to generate code logic.
