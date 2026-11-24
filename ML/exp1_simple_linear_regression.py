"""
=============================================================================
Machine Learning Experiment 1: Implementation of Simple Linear Regression
Objective:
    - Implement Simple Linear Regression from scratch and using scikit-learn.
    - Model relationship between a single independent variable (X) and dependent
      variable (y) (e.g., Years of Experience vs. Salary).
    - Estimate parameters: Slope (m / beta_1) and Intercept (c / beta_0).
    - Evaluate performance using MSE, RMSE, MAE, and R-squared (R2) score.
    - Visualize data points, regression line, and residuals.
=============================================================================
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_output_dir():
    """Ensure the outputs directory exists and return its path."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_salary_data():
    """
    Generates a realistic dataset modeling Years of Experience vs Annual Salary.
    """
    np.random.seed(42)
    experience = np.round(np.random.uniform(1.0, 12.0, 60), 1)
    # Salary = Base ($38,000) + $9,200 * Experience + Gaussian Noise
    salary = 38000 + (9200 * experience) + np.random.normal(0, 4800, len(experience))
    salary = np.round(salary, 2)

    df = pd.DataFrame({
        "Years_Experience": experience,
        "Salary": salary
    }).sort_values(by="Years_Experience").reset_index(drop=True)

    print(f"[INFO] Dataset created: {len(df)} records.")
    print(df.head())
    return df


def train_and_evaluate(df):
    """Splits data, trains Linear Regression model, and computes evaluation metrics."""
    X = df[["Years_Experience"]]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Fit Scikit-Learn Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Model Parameters
    slope = model.coef_[0]
    intercept = model.intercept_

    # Metrics
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)

    print("\n" + "=" * 60)
    print(" MODEL PARAMETERS & EQUATION ")
    print("=" * 60)
    print(f"Regression Equation: Salary = {intercept:,.2f} + ({slope:,.2f} * Experience)")
    print(f"Slope (beta_1)    : {slope:,.2f}")
    print(f"Intercept (beta_0): {intercept:,.2f}")

    print("\n" + "=" * 60)
    print(" PERFORMANCE EVALUATION ON TEST SET ")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE) : ${mae:,.2f}")
    print(f"Mean Squared Error (MSE)  : ${mse:,.2f}")
    print(f"Root Mean Squared Error   : ${rmse:,.2f}")
    print(f"R-squared (R2) Score      : {r2:.4f} ({r2 * 100:.2f}% variance explained)")

    return model, X_train, X_test, y_train, y_test, y_test_pred, slope, intercept, r2, rmse


def plot_regression_results(X_train, X_test, y_train, y_test, model, slope, intercept, r2, rmse, output_dir):
    """Plots training points, test points, fitted line, and residual differences."""
    plt.figure(figsize=(10, 6))

    # Scatter points
    plt.scatter(X_train, y_train, color="#2980b9", alpha=0.7, s=55, label="Train Data Points")
    plt.scatter(X_test, y_test, color="#e74c3c", s=70, edgecolor="black", label="Test Data Points (Actual)")

    # Fitted Regression Line
    x_vals = np.linspace(X_train.min().values[0], X_train.max().values[0], 100)
    x_line = pd.DataFrame({"Years_Experience": x_vals})
    y_line = model.predict(x_line)
    plt.plot(x_vals, y_line, color="#2c3e50", linewidth=2.5, label=f"Fitted Line: y = {slope:,.0f}x + {intercept:,.0f}")

    # Plot residual lines on test points
    y_test_pred = model.predict(X_test)
    for x_val, y_act, y_pr in zip(X_test.values.flatten(), y_test.values, y_test_pred):
        plt.plot([x_val, x_val], [y_act, y_pr], color="#e74c3c", linestyle=":", alpha=0.6)

    plt.title("Simple Linear Regression: Experience vs Salary", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Years of Professional Experience", fontsize=11, labelpad=8)
    plt.ylabel("Annual Salary (USD)", fontsize=11, labelpad=8)
    plt.gca().yaxis.set_major_formatter("${x:,.0f}")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Annotation box
    metrics_text = f"Model Performance:\nR2 Score = {r2:.3f}\nRMSE = ${rmse:,.0f}\nSlope = ${slope:,.0f}/yr"
    plt.text(
        0.05, 0.78, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", edgecolor="#bdc3c7")
    )

    plt.legend(loc="lower right", frameon=True)

    output_path = os.path.join(output_dir, "exp1_simple_linear_regression.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Regression plot saved to: {output_path}")


def main():
    print("Starting ML Experiment 1: Simple Linear Regression...")
    output_dir = get_output_dir()
    df = generate_salary_data()
    model, X_train, X_test, y_train, y_test, y_test_pred, slope, intercept, r2, rmse = train_and_evaluate(df)
    plot_regression_results(X_train, X_test, y_train, y_test, model, slope, intercept, r2, rmse, output_dir)
    print("\n[SUCCESS] Experiment 1 completed successfully.")


if __name__ == "__main__":
    main()
