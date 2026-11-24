"""
=============================================================================
Machine Learning Experiment 8: Implementation of Naive Bayes Classifier
Objective:
    - Implement Gaussian Naive Bayes (GaussianNB) for multi-class classification.
    - Apply Bayes' Theorem with conditional feature independence assumptions.
    - Analyze class prior probabilities P(Y) and feature likelihood parameters
      (Gaussian mean and variance per feature/class).
    - Evaluate performance using Accuracy, Precision, Recall, F1, and Confusion Matrix.
    - Visualize Confusion Matrix and Predicted Posterior Probabilities.
=============================================================================
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def get_output_dir():
    """Ensure the outputs directory exists and return its path."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_dataset():
    """Loads Wine dataset with continuous chemical features."""
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="Class")
    target_names = [f"Cultivar {i+1}" for i in range(len(wine.target_names))]

    print(f"[INFO] Loaded Wine dataset: {len(X)} records across {len(X.columns)} continuous features.")
    return X, y, target_names


def train_naive_bayes(X, y, target_names):
    """Fits Gaussian Naive Bayes and extracts priors and parameters."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)
    y_prob = gnb.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print(" NAIVE BAYES MODEL PARAMETERS & PRIORS ")
    print("=" * 60)
    for i, name in enumerate(target_names):
        print(f"Class Prior P({name}): {gnb.class_prior_[i]:.4f} ({gnb.class_count_[i]:.0f} training samples)")

    print("\n" + "=" * 60)
    print(" EVALUATION METRICS ON TEST SET ")
    print("=" * 60)
    print(f"Test Accuracy    : {acc * 100:.2f}%")
    print(f"Weighted Precision: {prec:.4f}")
    print(f"Weighted Recall   : {rec:.4f}")
    print(f"Weighted F1-Score : {f1:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return gnb, y_test, y_pred, y_prob, cm, acc, f1


def plot_naive_bayes_results(y_test, y_prob, cm, target_names, acc, output_dir):
    """Renders Confusion Matrix and Posterior Probabilities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Confusion Matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax1,
        xticklabels=target_names,
        yticklabels=target_names,
        annot_kws={"size": 13, "weight": "bold"},
        cbar=False
    )
    ax1.set_title("Naive Bayes Confusion Matrix", fontsize=12, weight="bold")
    ax1.set_xlabel("Predicted Class", fontsize=10)
    ax1.set_ylabel("True Class", fontsize=10)

    # Panel 2: Predicted Class Probability Distributions
    sample_indices = np.arange(min(20, len(y_test)))
    sample_probs = y_prob[:len(sample_indices)]

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bottom = np.zeros(len(sample_indices))

    for c_idx in range(sample_probs.shape[1]):
        ax2.bar(
            sample_indices,
            sample_probs[:, c_idx],
            bottom=bottom,
            label=target_names[c_idx],
            color=colors[c_idx],
            alpha=0.85
        )
        bottom += sample_probs[:, c_idx]

    ax2.set_title("Predicted Posterior Probability Distribution P(Y|X)\n(Sample of 20 Test Instances)", fontsize=12, weight="bold")
    ax2.set_xlabel("Test Instance Index", fontsize=10)
    ax2.set_ylabel("Cumulative Probability", fontsize=10)
    ax2.set_xticks(sample_indices)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(f"Gaussian Naive Bayes Classifier (Test Accuracy: {acc*100:.1f}%)", fontsize=14, weight="bold", y=1.02)
    output_path = os.path.join(output_dir, "exp8_naive_bayes.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Naive Bayes plots saved to: {output_path}")


def main():
    print("Starting ML Experiment 8: Naive Bayes Classifier...")
    output_dir = get_output_dir()
    X, y, target_names = load_dataset()
    gnb, y_test, y_pred, y_prob, cm, acc, f1 = train_naive_bayes(X, y, target_names)
    plot_naive_bayes_results(y_test, y_prob, cm, target_names, acc, output_dir)
    print("\n[SUCCESS] Experiment 8 completed successfully.")


if __name__ == "__main__":
    main()
