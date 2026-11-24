# Machine Learning (ML) Lab Experiments

**Author:** Aviral Pandey  
**Coursework:** Machine Learning Lab (CS / AIML)

This directory contains complete, standalone implementations for all **9 Machine Learning Lab Experiments**. Each experiment is self-contained, using real-world benchmark datasets or high-fidelity synthetic generators from `scikit-learn` with zero missing CSV dependencies.

---

## 📋 Experiments Overview

| # | Script Name | Topic & Algorithms | Dataset Used | Key Metrics & Evaluations | Output Visualizations |
|---|---|---|---|---|---|
| **1** | [`exp1_simple_linear_regression.py`](exp1_simple_linear_regression.py) | **Simple Linear Regression** | Experience vs. Salary | $R^2$, MAE, MSE, RMSE, Slope & Intercept | `exp1_simple_linear_regression.png` |
| **2** | [`exp2_multiple_linear_regression.py`](exp2_multiple_linear_regression.py) | **Multiple Linear Regression** | Multi-Variable Housing Prices | $R^2$, Adjusted $R^2$, RMSE, Homoscedasticity | `exp2_multiple_linear_regression.png` |
| **3** | [`exp3_logistic_regression.py`](exp3_logistic_regression.py) | **Logistic Regression** | Breast Cancer Wisconsin Diagnostic | Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC | `exp3_logistic_regression.png` |
| **4** | [`exp4_regularized_regression.py`](exp4_regularized_regression.py) | **Regularized Regression** (Ridge, Lasso, ElasticNet) | High-Dimensional Collinear Data | Coefficient shrinkage, Feature Sparsity, Test RMSE | `exp4_regularized_regression.png` |
| **5** | [`exp5_pca_dimensionality_reduction.py`](exp5_pca_dimensionality_reduction.py) | **Dimensionality Reduction via PCA** | Wine Recognition Dataset (13 features) | Scree Plot, Explained Variance %, 2D projection | `exp5_pca_dimensionality_reduction.png` |
| **6** | [`exp6_support_vector_machine.py`](exp6_support_vector_machine.py) | **Support Vector Machines (SVM)** | Non-linear Interleaving Moons | Linear, Poly, RBF Kernels, Support Vector Count | `exp6_support_vector_machine.png` |
| **7** | [`exp7_decision_tree_real_world.py`](exp7_decision_tree_real_world.py) | **Decision Tree Classifier** | Iris Flower Benchmark | Gini Impurity, Tree Flowchart, Feature Importance | `exp7_decision_tree.png` |
| **8** | [`exp8_naive_bayes_classifier.py`](exp8_naive_bayes_classifier.py) | **Gaussian Naive Bayes** | Wine Chemical Dataset | Prior $P(Y)$, Class Likelihoods, Posterior $P(Y\|X)$ | `exp8_naive_bayes.png` |
| **9** | [`exp9_kmeans_clustering.py`](exp9_kmeans_clustering.py) | **K-Means Clustering** | Customer Annual Income & Spend | Elbow Method (WCSS/Inertia), Silhouette Score, Centroids | `exp9_kmeans_clustering.png` |

---

## 🚀 Execution Guide

Run any experiment individually from the repository root:

```bash
# Experiment 1: Simple Linear Regression
python ML/exp1_simple_linear_regression.py

# Experiment 2: Multiple Linear Regression
python ML/exp2_multiple_linear_regression.py

# Experiment 3: Logistic Regression
python ML/exp3_logistic_regression.py

# Experiment 4: Regularized Regression (Ridge, Lasso, ElasticNet)
python ML/exp4_regularized_regression.py

# Experiment 5: Dimensionality Reduction using PCA
python ML/exp5_pca_dimensionality_reduction.py

# Experiment 6: Support Vector Machines (SVM)
python ML/exp6_support_vector_machine.py

# Experiment 7: Decision Tree on Real-World Data
python ML/exp7_decision_tree_real_world.py

# Experiment 8: Naive Bayes Classifier
python ML/exp8_naive_bayes_classifier.py

# Experiment 9: K-Means Clustering
python ML/exp9_kmeans_clustering.py
```

All generated high-resolution diagnostic plots (`.png`) are saved in the `ML/outputs/` directory.
