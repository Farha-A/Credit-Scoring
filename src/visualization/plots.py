"""Reusable plotting helpers for model performance visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve


def plot_learning_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    title: str = "Learning Curve",
    cv: int = 5,
    n_jobs: int = -1,
) -> None:
    """Plot training and cross-validation accuracy as a function of training set size.

    Args:
        estimator: A fitted or unfitted scikit-learn estimator.
        X: Feature matrix.
        y: Target vector.
        title: Plot title.
        cv: Number of cross-validation folds.
        n_jobs: Parallel jobs for learning_curve computation.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=n_jobs,
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, "o-", label="Training score")
    plt.plot(train_sizes, val_mean, "o-", label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> None:
    """Render a confusion matrix as an annotated heatmap.

    Args:
        cm: 2×2 confusion matrix array from :func:`sklearn.metrics.confusion_matrix`.
        title: Plot title.
    """
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> None:
    """Plot a Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr: False positive rates from :func:`sklearn.metrics.roc_curve`.
        tpr: True positive rates from :func:`sklearn.metrics.roc_curve`.
        roc_auc: Area under the ROC curve.
    """
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_feature_importances(
    importances: np.ndarray,
    feature_names: list[str],
    title: str = "Feature Importances",
) -> None:
    """Horizontal bar chart of feature importances.

    Args:
        importances: Array of importance scores, one per feature.
        feature_names: Corresponding feature name for each score.
        title: Plot title.
    """
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_lofo_bar(
    cols: list[str],
    metric_values: list[float],
    metric_name: str = "Accuracy",
) -> None:
    """Bar chart for Leave-One-Feature-Out (LOFO) importance analysis.

    The last entry in *cols* should be "None" (all features present) and its
    corresponding value serves as the baseline.

    Args:
        cols: Feature names; last element is "None" (baseline).
        metric_values: Metric score when each feature is removed.
        metric_name: Name of the metric for the x-axis label.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(y=cols, x=metric_values)
    plt.xlabel(metric_name)
    plt.title(f"LOFO – {metric_name} when each feature is removed")
    plt.tight_layout()
    plt.show()
