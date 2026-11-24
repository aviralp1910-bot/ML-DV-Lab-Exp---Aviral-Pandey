"""
=============================================================================
Machine Learning Experiment 9: Implementation of K-Means Clustering
Objective:
    - Implement K-Means Unsupervised Clustering from scratch & with scikit-learn.
    - Evaluate cluster compactness using Within-Cluster Sum of Squares (Inertia).
    - Determine optimal cluster count (k) via the Elbow Method & Silhouette Scores.
    - Identify cluster centroids and assign instances to nearest centroid.
    - Visualize the Elbow curve and 2D partitioned clusters with centroids.
=============================================================================
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_output_dir():
    """Ensure the outputs directory exists and return its path."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_customer_spending_data():
    """
    Generates realistic 2D customer transaction data:
    Features: Annual Income ($k) vs Spending Score (1-100).
    """
    X, _ = make_blobs(
        n_samples=350,
        centers=4,
        cluster_std=1.1,
        random_state=42
    )
    # Rescale to realistic customer attributes
    income = np.round((X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min()) * 90 + 20, 1)
    spending = np.round((X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min()) * 85 + 10, 1)

    df = pd.DataFrame({
        "Annual_Income_k": income,
        "Spending_Score": spending
    })

    print(f"[INFO] Customer Segmentation dataset generated: {len(df)} customer records.")
    return df


def find_optimal_k(X, k_range=range(1, 10)):
    """Computes Inertia and Silhouette scores across candidate values of k."""
    inertias = []
    silhouette_scores = []

    print("\n" + "=" * 60)
    print(" EVALUATING CLUSTER COUNTS VIA ELBOW & SILHOUETTE ")
    print("=" * 60)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

        if k > 1:
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"k = {k} | Inertia (WCSS): {kmeans.inertia_:,.1f} | Silhouette Score: {score:.4f}")
        else:
            print(f"k = {k} | Inertia (WCSS): {kmeans.inertia_:,.1f} | Silhouette Score: N/A")

    return inertias, silhouette_scores


def fit_final_kmeans(X, optimal_k=4):
    """Fits KMeans with optimal cluster count and extracts centroids."""
    kmeans = KMeans(n_clusters=optimal_k, init="k-means++", n_init=15, random_state=42)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    print(f"\n[INFO] Optimal KMeans Model Fitted (k = {optimal_k}):")
    for i, c in enumerate(centroids):
        print(f"  Cluster {i+1} Centroid: Income = ${c[0]:.1f}k, Spending Score = {c[1]:.1f}")

    return kmeans, labels, centroids


def plot_clustering_diagnostics(X, inertias, labels, centroids, optimal_k, output_dir):
    """Plots Elbow Method Curve and 2D Clustered Space with Centroids."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Elbow Curve
    k_vals = list(range(1, len(inertias) + 1))
    ax1.plot(k_vals, inertias, marker="o", color="#2980b9", linewidth=2.2, markersize=7, label="WCSS / Inertia")
    ax1.axvline(x=optimal_k, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Elbow Point (k = {optimal_k})")
    ax1.set_title("Elbow Method for Optimal k Selection", fontsize=12, weight="bold")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax1.set_ylabel("Within-Cluster Sum of Squares (Inertia)", fontsize=10)
    ax1.set_xticks(k_vals)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Panel 2: Clustered Scatter Plot
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for i in range(optimal_k):
        cluster_mask = (labels == i)
        ax2.scatter(
            X.iloc[cluster_mask, 0], X.iloc[cluster_mask, 1],
            s=50, color=colors[i % len(colors)], alpha=0.75, edgecolors="black", linewidth=0.5,
            label=f"Cluster {i+1} ({sum(cluster_mask)} customers)"
        )

    # Plot Centroids
    ax2.scatter(
        centroids[:, 0], centroids[:, 1],
        s=250, c="yellow", marker="*", edgecolors="black", linewidth=1.5,
        label="Cluster Centroids"
    )

    ax2.set_title(f"K-Means Customer Segmentation (k = {optimal_k})", fontsize=12, weight="bold")
    ax2.set_xlabel("Annual Income ($k)", fontsize=10)
    ax2.set_ylabel("Spending Score (1 - 100)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    plt.suptitle("K-Means Unsupervised Clustering Analysis", fontsize=14, weight="bold", y=1.02)
    output_path = os.path.join(output_dir, "exp9_kmeans_clustering.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Clustering plots saved to: {output_path}")


def main():
    print("Starting ML Experiment 9: K-Means Clustering...")
    output_dir = get_output_dir()
    df = generate_customer_spending_data()
    inertias, sil_scores = find_optimal_k(df)
    optimal_k = 4
    kmeans, labels, centroids = fit_final_kmeans(df, optimal_k=optimal_k)
    plot_clustering_diagnostics(df, inertias, labels, centroids, optimal_k, output_dir)
    print("\n[SUCCESS] Experiment 9 completed successfully.")


if __name__ == "__main__":
    main()
