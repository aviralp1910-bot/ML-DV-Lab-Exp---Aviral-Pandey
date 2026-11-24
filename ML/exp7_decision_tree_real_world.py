"""
=============================================================================
Machine Learning Experiment 7: Decision Tree on Real-World Dataset
Objective:
    - Train a Decision Tree Classifier on the real-world Iris benchmark dataset.
    - Analyze split criteria (Gini Impurity vs. Information Gain / Entropy).
    - Prevent overfitting using pre-pruning techniques (e.g., max_depth).
    - Evaluate classification metrics: Accuracy, Precision, Recall, Confusion Matrix.
    - Extract and visualize Feature Importances.
    - Render the complete Decision Tree graphical flowchart using plot_tree.
=============================================================================
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def get_output_dir():
    """Ensure the outputs directory exists and return its path."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_iris_dataset():
    """Loads the benchmark Iris flower dataset."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="Species")
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"[INFO] Iris dataset loaded: {len(X)} samples across {len(feature_names)} morphological features.")
    print(f"[INFO] Target Classes: {list(target_names)}")
    return X, y, feature_names, target_names


def train_decision_tree(X, y):
    """Trains Decision Tree model with controlled maximum depth."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        min_samples_split=4,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print(" DECISION TREE CLASSIFICATION REPORT ")
    print("=" * 60)
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Setosa", "Versicolor", "Virginica"]))

    # Feature Importance
    feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature Importances:")
    for feat, imp in feat_imp.items():
        print(f"  {feat:<25}: {imp:.4f} ({imp*100:.1f}%)")

    return clf, feat_imp, acc, cm


def plot_tree_and_importance(clf, feat_imp, feature_names, target_names, acc, output_dir):
    """Renders visual tree structure and feature importance bar plot."""
    fig = plt.figure(figsize=(16, 7))

    # Grid: Tree on left (span 2 cols), Feature importance on right
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((1, 3), (0, 2))

    # Panel 1: Tree Graph
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax1
    )
    ax1.set_title(f"Decision Tree Flowchart (Max Depth = 3, Test Acc = {acc*100:.1f}%)", fontsize=12, weight="bold")

    # Panel 2: Feature Importances
    y_pos = np.arange(len(feat_imp))
    ax2.barh(y_pos, feat_imp.values[::-1], color="#27ae60", edgecolor="black", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feat_imp.index[::-1], fontsize=9)
    ax2.set_xlabel("Relative Importance (Gini)", fontsize=10)
    ax2.set_title("Feature Importances Breakdown", fontsize=12, weight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    # Add data annotations
    for i, v in enumerate(feat_imp.values[::-1]):
        ax2.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=9, weight="bold")

    plt.suptitle("Decision Tree Classifier on Real-World Iris Dataset", fontsize=14, weight="bold", y=1.02)
    output_path = os.path.join(output_dir, "exp7_decision_tree.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Decision Tree visualizations saved to: {output_path}")


def main():
    print("Starting ML Experiment 7: Decision Tree on Real-World Data...")
    output_dir = get_output_dir()
    X, y, feature_names, target_names = load_iris_dataset()
    clf, feat_imp, acc, cm = train_decision_tree(X, y)
    plot_tree_and_importance(clf, feat_imp, feature_names, target_names, acc, output_dir)
    print("\n[SUCCESS] Experiment 7 completed successfully.")


if __name__ == "__main__":
    main()
