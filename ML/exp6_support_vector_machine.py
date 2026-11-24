"""
=============================================================================
Machine Learning Experiment 6: Implementation of Support Vector Machines (SVM)
Objective:
    - Implement Support Vector Machine (SVM) classifiers using scikit-learn.
    - Evaluate different kernel functions: Linear, Polynomial, and RBF.
    - Analyze the influence of hyperparameters: Margin Penalty (C) and Gamma.
    - Identify and highlight Support Vectors and maximum margin boundaries.
    - Visualize decision regions and hyperplanes for non-linear classification.
=============================================================================
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


def get_output_dir():
    """Ensure the outputs directory exists and return its path."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_nonlinear_dataset():
    """Generates non-linearly separable interleaving half-circles (Moons)."""
    X, y = make_moons(n_samples=240, noise=0.22, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[INFO] Generated non-linear Moons dataset: {len(X)} samples with 2 classes.")
    return X_scaled, y


def train_svm_kernels(X, y):
    """Trains SVM with Linear, Polynomial, and RBF kernels."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    kernel_configs = [
        {"name": "Linear Kernel (C=1.0)", "kernel": "linear", "C": 1.0, "gamma": "scale", "degree": 3},
        {"name": "Polynomial Kernel (deg=3, C=1.0)", "kernel": "poly", "C": 1.0, "gamma": "scale", "degree": 3},
        {"name": "RBF Kernel (C=1.0, gamma=1.0)", "kernel": "rbf", "C": 1.0, "gamma": 1.0, "degree": 3},
        {"name": "RBF Kernel (C=10.0, gamma=2.5)", "kernel": "rbf", "C": 10.0, "gamma": 2.5, "degree": 3}
    ]

    trained_models = []

    print("\n" + "=" * 65)
    print(" SVM KERNEL PERFORMANCE COMPARISON ")
    print("=" * 65)

    for cfg in kernel_configs:
        clf = SVC(
            kernel=cfg["kernel"],
            C=cfg["C"],
            gamma=cfg["gamma"],
            degree=cfg["degree"],
            random_state=42
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        n_sv = len(clf.support_vectors_)

        print(f"[{cfg['name']}]")
        print(f"  Accuracy: {acc * 100:.2f}% | F1: {f1:.4f} | Total Support Vectors: {n_sv}")

        trained_models.append((cfg["name"], clf, acc, f1))

    return X_train, X_test, y_train, y_test, trained_models


def plot_svm_decision_surfaces(X_train, y_train, trained_models, output_dir):
    """Plots 2x2 comparison grid of decision boundaries and support vectors."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    # Create meshgrid
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

    for i, (name, clf, acc, f1) in enumerate(trained_models):
        ax = axes[i]

        # Predict over grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision contour and margins (levels = -1, 0, 1)
        ax.contourf(xx, yy, Z > 0, alpha=0.25, cmap="coolwarm")
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"], colors=["#2c3e50", "#c0392b", "#2c3e50"], linewidths=[1.2, 2.2, 1.2])

        # Plot training points
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="black", s=40, alpha=0.85)

        # Highlight Support Vectors
        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none", edgecolors="#27ae60", linewidth=1.6, label=f"Support Vectors ({len(sv)})")

        ax.set_title(f"{name}\nAcc: {acc*100:.1f}% | Support Vectors: {len(sv)}", fontsize=11, weight="bold")
        ax.set_xlabel("Feature 1 (Standardized)", fontsize=9)
        ax.set_ylabel("Feature 2 (Standardized)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Support Vector Machine (SVM): Kernel Comparison & Decision Hyperplanes", fontsize=14, weight="bold", y=0.99)
    output_path = os.path.join(output_dir, "exp6_support_vector_machine.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] SVM decision boundaries saved to: {output_path}")


def main():
    print("Starting ML Experiment 6: Support Vector Machines...")
    output_dir = get_output_dir()
    X, y = generate_nonlinear_dataset()
    X_train, X_test, y_train, y_test, trained_models = train_svm_kernels(X, y)
    plot_svm_decision_surfaces(X_train, y_train, trained_models, output_dir)
    print("\n[SUCCESS] Experiment 6 completed successfully.")


if __name__ == "__main__":
    main()
