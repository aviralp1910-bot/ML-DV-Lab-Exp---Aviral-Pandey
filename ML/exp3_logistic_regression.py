"""
=============================================================================
Machine Learning Experiment 3: Implementation of Logistic Regression
Objective:
    - Implement Logistic Regression for Binary Classification using scikit-learn.
    - Train on the real-world Breast Cancer Wisconsin diagnostic dataset.
    - Apply feature standardization using StandardScaler.
    - Predict binary class outcomes and class probability estimates (Sigmoid).
    - Evaluate using Confusion Matrix, Accuracy, Precision, Recall, F1, and ROC-AUC.
    - Visualize Confusion Matrix heatmap and ROC Curve.
=============================================================================
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, classification_report
)


def get_output_dir():
    """Ensure the outputs directory exists and return its path."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_classification_data():
    """Loads the Breast Cancer Wisconsin Diagnostic dataset from scikit-learn."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="Diagnosis")
    class_names = data.target_names  # ['malignant', 'benign']

    print(f"[INFO] Loaded Breast Cancer dataset: {X.shape[0]} samples with {X.shape[1]} biological features.")
    print(f"[INFO] Target Classes: 0 = {class_names[0]} ({sum(y == 0)}), 1 = {class_names[1]} ({sum(y == 1)})")
    return X, y, class_names


def train_logistic_regression(X, y):
    """Scales data and trains Logistic Regression model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print(" LOGISTIC REGRESSION EVALUATION METRICS ")
    print("=" * 60)
    print(f"Accuracy Score  : {acc * 100:.2f}%")
    print(f"Precision Score : {prec:.4f}")
    print(f"Recall Score    : {rec:.4f}")
    print(f"F1-Score        : {f1:.4f}")
    print(f"ROC-AUC Score   : {auc:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

    return model, y_test, y_pred, y_prob, cm, acc, f1, auc


def plot_classification_results(y_test, y_prob, cm, class_names, acc, auc, output_dir):
    """Renders Confusion Matrix and ROC Curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: Confusion Matrix Heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax1,
        xticklabels=[f"Pred {c.capitalize()}" for c in class_names],
        yticklabels=[f"Actual {c.capitalize()}" for c in class_names],
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax1.set_title("Confusion Matrix", fontsize=13, weight="bold")
    ax1.set_xlabel("Predicted Label", fontsize=10)
    ax1.set_ylabel("True Label", fontsize=10)

    # Panel 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax2.plot(fpr, tpr, color="#2980b9", linewidth=2.5, label=f"Logistic Regression (AUC = {auc:.3f})")
    ax2.plot([0, 1], [0, 1], color="#e74c3c", linestyle="--", linewidth=1.8, label="Random Classifier (AUC = 0.500)")
    ax2.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, weight="bold")
    ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax2.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.suptitle(f"Logistic Regression Performance (Overall Accuracy: {acc*100:.1f}%)", fontsize=14, weight="bold", y=1.02)
    output_path = os.path.join(output_dir, "exp3_logistic_regression.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Classification plots saved to: {output_path}")


def main():
    print("Starting ML Experiment 3: Logistic Regression...")
    output_dir = get_output_dir()
    X, y, class_names = load_classification_data()
    model, y_test, y_pred, y_prob, cm, acc, f1, auc = train_logistic_regression(X, y)
    plot_classification_results(y_test, y_prob, cm, class_names, acc, auc, output_dir)
    print("\n[SUCCESS] Experiment 3 completed successfully.")


if __name__ == "__main__":
    main()
