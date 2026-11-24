"""
=============================================================================
Machine Learning Experiment 2: Implementation of Multiple Linear Regression
Objective:
    - Implement Multiple Linear Regression with multiple explanatory features
      (Square Footage, Bedrooms, Bathrooms, Property Age, Distance to Center).
    - Analyze multicollinearity and feature impact via model coefficients.
    - Compute performance metrics: MAE, MSE, RMSE, R2, and Adjusted R2.
    - Visualize Actual vs. Predicted values and Residual Distribution.
=============================================================================
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_output_dir():
    """Ensure the outputs directory exists and return its path."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_housing_dataset():
    """
    Generates a realistic multi-variable housing price dataset.
    Features:
        - Sq_Footage: 800 - 4,200 sq ft
        - Bedrooms: 1 - 5 rooms
        - Bathrooms: 1 - 4 baths
        - Property_Age: 1 - 40 years
        - Dist_City_Center: 1 - 25 miles
    """
    np.random.seed(42)
    n = 250

    sq_ft = np.random.uniform(900, 3800, n).round()
    bedrooms = np.random.randint(1, 6, n)
    bathrooms = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], n)
    age = np.random.uniform(1, 35, n).round()
    dist = np.random.uniform(1.5, 22.0, n).round(1)

    # Price = Base ($90,000) + 145*SqFt + 18,000*Beds + 22,000*Baths - 1,800*Age - 3,200*Dist + Noise
    price = (
        90000 +
        (145 * sq_ft) +
        (18000 * bedrooms) +
        (22000 * bathrooms) -
        (1800 * age) -
        (3200 * dist) +
        np.random.normal(0, 18000, n)
    )
    price = np.round(price, 2)

    df = pd.DataFrame({
        "Sq_Footage": sq_ft,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Property_Age": age,
        "Dist_City_Center": dist,
        "Price": price
    })

    print(f"[INFO] Multi-variable housing dataset created: {len(df)} records with {df.shape[1]-1} features.")
    print(df.head())
    return df


def train_and_evaluate_mlr(df):
    """Fits multiple linear regression model and reports metrics."""
    feature_cols = ["Sq_Footage", "Bedrooms", "Bathrooms", "Property_Age", "Dist_City_Center"]
    X = df[feature_cols]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation Metrics
    n = len(y_test)
    p = X.shape[1]
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

    # Coefficients Table
    coeff_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient (Beta)": model.coef_.round(2)
    })

    print("\n" + "=" * 60)
    print(" MODEL COEFFICIENTS & INTERCEPT ")
    print("=" * 60)
    print(f"Intercept (Beta_0): ${model.intercept_:,.2f}")
    print(coeff_df.to_string(index=False))

    print("\n" + "=" * 60)
    print(" PERFORMANCE EVALUATION ON TEST SET ")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE) : ${mae:,.2f}")
    print(f"Root Mean Squared Error   : ${rmse:,.2f}")
    print(f"R-squared (R2) Score      : {r2:.4f}")
    print(f"Adjusted R-squared Score  : {adj_r2:.4f}")

    return model, X_test, y_test, y_pred, coeff_df, r2, rmse


def plot_mlr_diagnostics(y_test, y_pred, coeff_df, r2, rmse, output_dir):
    """Generates a 2-panel diagnostic figure: Actual vs. Predicted and Residuals Distribution."""
    residuals = y_test - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Actual vs. Predicted
    ax1.scatter(y_test, y_pred, color="#2980b9", alpha=0.75, s=60, edgecolor="black", linewidth=0.6)
    # Perfect prediction line y = x
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=2.2, label="Ideal Fit (y = x)")

    ax1.set_title("Actual vs. Predicted Property Prices", fontsize=12, weight="bold")
    ax1.set_xlabel("Actual Price ($)", fontsize=10)
    ax1.set_ylabel("Predicted Price ($)", fontsize=10)
    ax1.xaxis.set_major_formatter("${x:,.0f}")
    ax1.yaxis.set_major_formatter("${x:,.0f}")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    ax1.text(0.05, 0.85, f"R2 = {r2:.3f}\nRMSE = ${rmse:,.0f}", transform=ax1.transAxes,
             fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", edgecolor="#bdc3c7"))

    # Panel 2: Residuals Distribution
    sns.histplot(residuals, kde=True, ax=ax2, color="#8e44ad", bins=15, edgecolor="black", alpha=0.7)
    ax2.axvline(x=0, color="#e74c3c", linestyle="--", linewidth=2, label="Zero Error Line")
    ax2.set_title("Residuals Error Distribution (Homoscedasticity)", fontsize=12, weight="bold")
    ax2.set_xlabel("Residual Error ($)", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.xaxis.set_major_formatter("${x:,.0f}")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.suptitle("Multiple Linear Regression Model Diagnostics", fontsize=14, weight="bold", y=1.02)
    output_path = os.path.join(output_dir, "exp2_multiple_linear_regression.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Diagnostics plot saved to: {output_path}")


def main():
    print("Starting ML Experiment 2: Multiple Linear Regression...")
    output_dir = get_output_dir()
    df = generate_housing_dataset()
    model, X_test, y_test, y_pred, coeff_df, r2, rmse = train_and_evaluate_mlr(df)
    plot_mlr_diagnostics(y_test, y_pred, coeff_df, r2, rmse, output_dir)
    print("\n[SUCCESS] Experiment 2 completed successfully.")


if __name__ == "__main__":
    main()
