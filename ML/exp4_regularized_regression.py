"""
=============================================================================
Machine Learning Experiment 4: Regression Models with Regularization
Objective:
    - Implement and compare Ordinary Least Squares (OLS), Ridge (L2),
      Lasso (L1), and ElasticNet regularization techniques.
    - Demonstrate how Ridge shrinks coefficients to handle multicollinearity.
    - Demonstrate how Lasso induces feature sparsity (automatic feature selection).
    - Compare model performance across R2 and RMSE metrics.
    - Visualize coefficient shrinkage profiles and model comparison charts.
=============================================================================
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score


def get_output_dir():
    """Ensure the outputs directory exists and return its path."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_collinear_data():
    """
    Generates a regression dataset with 12 features, including informative,
    redundant/correlated, and noise features.
    """
    np.random.seed(42)
    X, y, true_coef = make_regression(
        n_samples=200, n_features=12, n_informative=5, noise=15.0, coef=True, random_state=42
    )
    feature_names = [f"Feature_{i+1}" for i in range(12)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="Target")

    print(f"[INFO] Generated synthetic dataset: {X.shape[0]} samples with {X.shape[1]} features (5 true informative).")
    return X_df, y_series, feature_names


def train_regularized_models(X, y):
    """Fits OLS, Ridge, Lasso, and ElasticNet models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "OLS (Linear)": LinearRegression(),
        "Ridge (L2, alpha=10.0)": Ridge(alpha=10.0, random_state=42),
        "Lasso (L1, alpha=1.5)": Lasso(alpha=1.5, random_state=42),
        "ElasticNet (alpha=1.0, L1=0.5)": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
    }

    results = {}
    coef_dict = {}

    print("\n" + "=" * 65)
    print(" REGULARIZED REGRESSION MODELS COMPARISON ")
    print("=" * 65)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        zero_coefs = np.sum(np.isclose(model.coef_, 0, atol=1e-3))

        results[name] = {"R2": r2, "RMSE": rmse, "Zero_Coefficients": zero_coefs}
        coef_dict[name] = model.coef_

        print(f"[{name}]")
        print(f"  R2 Score: {r2:.4f} | RMSE: {rmse:.2f} | Zeroed Features: {zero_coefs}/{len(model.coef_)}")

    coef_df = pd.DataFrame(coef_dict, index=X.columns)
    perf_df = pd.DataFrame(results).T
    return coef_df, perf_df


def plot_regularization_comparison(coef_df, perf_df, output_dir):
    """Renders coefficient comparison bar chart and model performance metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Coefficients across models
    x = np.arange(len(coef_df))
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    for i, (col, offset, color) in enumerate(zip(coef_df.columns, offsets, colors)):
        ax1.bar(x + offset, coef_df[col], width, label=col, color=color, alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(coef_df.index, rotation=45, ha="right", fontsize=9)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title("Learned Feature Coefficients Comparison\n(Notice Lasso & ElasticNet Feature Selection / Sparsity)", fontsize=11, weight="bold")
    ax1.set_ylabel("Coefficient Weight", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, loc="upper right")

    # Panel 2: Model Performance Metrics (R2 and RMSE)
    model_names = [n.split("(")[0].strip() for n in perf_df.index]
    x_perf = np.arange(len(model_names))
    w = 0.35

    ax2_r2 = ax2
    ax2_rmse = ax2.twinx()

    b1 = ax2_r2.bar(x_perf - w/2, perf_df["R2"], w, label="R2 Score (Higher is better)", color="#27ae60", alpha=0.85)
    b2 = ax2_rmse.bar(x_perf + w/2, perf_df["RMSE"], w, label="RMSE (Lower is better)", color="#c0392b", alpha=0.85)

    ax2_r2.set_xticks(x_perf)
    ax2_r2.set_xticklabels(model_names, fontsize=10, weight="bold")
    ax2_r2.set_ylabel("R2 Score", color="#27ae60", fontsize=10, weight="bold")
    ax2_rmse.set_ylabel("Root Mean Squared Error (RMSE)", color="#c0392b", fontsize=10, weight="bold")
    ax2_r2.set_title("Test Generalization Error: R2 Score vs RMSE", fontsize=11, weight="bold")
    ax2_r2.grid(axis="y", linestyle="--", alpha=0.5)

    # Combined legend
    lines1, labels1 = ax2_r2.get_legend_handles_labels()
    lines2, labels2 = ax2_rmse.get_legend_handles_labels()
    ax2_r2.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    plt.suptitle("Regularized Regression Diagnostics (Ridge, Lasso, ElasticNet vs OLS)", fontsize=13, weight="bold", y=1.02)
    output_path = os.path.join(output_dir, "exp4_regularized_regression.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Regularization comparison plot saved to: {output_path}")


def main():
    print("Starting ML Experiment 4: Regularized Regression Models...")
    output_dir = get_output_dir()
    X, y, feature_names = generate_collinear_data()
    coef_df, perf_df = train_regularized_models(X, y)
    plot_regularization_comparison(coef_df, perf_df, output_dir)
    print("\n[SUCCESS] Experiment 4 completed successfully.")


if __name__ == "__main__":
    main()
