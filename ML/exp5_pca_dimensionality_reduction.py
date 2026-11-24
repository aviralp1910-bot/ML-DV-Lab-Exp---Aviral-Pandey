"""
=============================================================================
Machine Learning Experiment 5: Dimensionality Reduction using PCA
Objective:
    - Implement Principal Component Analysis (PCA) for unsupervised
      dimensionality reduction on the multi-attribute Wine dataset (13 features).
    - Standardize features and compute Eigenvectors / Principal Axes.
    - Analyze Explained Variance Ratio & Cumulative Variance (Scree Plot).
    - Project 13-dimensional feature space onto 2 principal components (PC1, PC2).
    - Compare downstream classifier accuracy before vs. after dimensionality reduction.
    - Visualize 2D PCA projections with class separation.
=============================================================================
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_output_dir():
    """Ensure the outputs directory exists and return its path."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_wine_data():
    """Loads scikit-learn Wine recognition dataset (13 chemical attributes)."""
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="Cultivar")
    target_names = wine.target_names

    print(f"[INFO] Wine dataset loaded: {X.shape[0]} samples with {X.shape[1]} chemical features.")
    print(f"[INFO] Target Classes: {list(target_names)}")
    return X, y, target_names


def perform_pca_analysis(X, y):
    """Executes full PCA decomposition and extracts 2-component projection."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full PCA for Scree Plot
    pca_full = PCA()
    pca_full.fit(X_scaled)
    exp_var = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    # 2-Component PCA for 2D Projection
    pca_2d = PCA(n_components=2)
    X_pca2 = pca_2d.fit_transform(X_scaled)
    var_pc1 = pca_2d.explained_variance_ratio_[0] * 100
    var_pc2 = pca_2d.explained_variance_ratio_[1] * 100

    print("\n" + "=" * 65)
    print(" PRINCIPAL COMPONENT ANALYSIS VARIANCE RATIOS ")
    print("=" * 65)
    print(f"PC1 Variance Explained: {var_pc1:.2f}%")
    print(f"PC2 Variance Explained: {var_pc2:.2f}%")
    print(f"Total 2D Variance Captured: {(var_pc1 + var_pc2):.2f}%")

    # Benchmark: Classification before vs after PCA
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    clf_full = LogisticRegression().fit(X_train_raw, y_train)
    acc_full = accuracy_score(y_test, clf_full.predict(X_test_raw))

    X_train_pca, X_test_pca, _, _ = train_test_split(X_pca2, y, test_size=0.3, random_state=42)
    clf_pca = LogisticRegression().fit(X_train_pca, y_train)
    acc_pca = accuracy_score(y_test, clf_pca.predict(X_test_pca))

    print(f"Classifier Accuracy (All 13 features) : {acc_full * 100:.2f}%")
    print(f"Classifier Accuracy (Only 2 PCA comps): {acc_pca * 100:.2f}%")

    return exp_var, cum_var, X_pca2, var_pc1, var_pc2, acc_full, acc_pca


def plot_pca_results(exp_var, cum_var, X_pca2, y, target_names, var_pc1, var_pc2, output_dir):
    """Renders Scree Plot and 2D PCA Class Projection Scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Scree Plot
    components = np.arange(1, len(exp_var) + 1)
    ax1.bar(components, exp_var * 100, color="#3498db", alpha=0.8, label="Individual Variance (%)")
    ax1.plot(components, cum_var * 100, color="#e74c3c", marker="o", linewidth=2.2, label="Cumulative Variance (%)")
    ax1.axhline(y=80, color="#27ae60", linestyle="--", alpha=0.7, label="80% Threshold")

    ax1.set_title("Scree Plot: Explained Variance by Components", fontsize=12, weight="bold")
    ax1.set_xlabel("Principal Component Number", fontsize=10)
    ax1.set_ylabel("Explained Variance Ratio (%)", fontsize=10)
    ax1.set_xticks(components)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="center right", fontsize=9)

    # Panel 2: 2D PCA Class Projection
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    markers = ["o", "s", "^"]
    for i, name in enumerate(target_names):
        mask = (y == i)
        ax2.scatter(
            X_pca2[mask, 0], X_pca2[mask, 1],
            c=colors[i], marker=markers[i], s=60, edgecolors="black", linewidth=0.6,
            label=f"Class {i+1}: {name.capitalize()}"
        )

    ax2.set_title(f"2D Projection via PCA (Total Explained: {(var_pc1+var_pc2):.1f}%)", fontsize=12, weight="bold")
    ax2.set_xlabel(f"Principal Component 1 ({var_pc1:.1f}% Variance)", fontsize=10)
    ax2.set_ylabel(f"Principal Component 2 ({var_pc2:.1f}% Variance)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="best", frameon=True)

    plt.suptitle("Principal Component Analysis (PCA) Dimensionality Reduction", fontsize=14, weight="bold", y=1.02)
    output_path = os.path.join(output_dir, "exp5_pca_dimensionality_reduction.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] PCA plots saved to: {output_path}")


def main():
    print("Starting ML Experiment 5: PCA Dimensionality Reduction...")
    output_dir = get_output_dir()
    X, y, target_names = load_wine_data()
    exp_var, cum_var, X_pca2, var_pc1, var_pc2, acc_full, acc_pca = perform_pca_analysis(X, y)
    plot_pca_results(exp_var, cum_var, X_pca2, y, target_names, var_pc1, var_pc2, output_dir)
    print("\n[SUCCESS] Experiment 5 completed successfully.")


if __name__ == "__main__":
    main()
