"""Logistic Regression model training and evaluation for credit scoring."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.visualization.plots import plot_confusion_matrix, plot_roc_curve


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs,
) -> LogisticRegression:
    """Train a Logistic Regression classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        max_iter: Maximum number of solver iterations (default 1 000).
        random_state: Random seed for reproducibility.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~sklearn.linear_model.LogisticRegression`.

    Returns:
        Fitted :class:`~sklearn.linear_model.LogisticRegression` model.
    """
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_logistic_regression(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a trained Logistic Regression model and display diagnostic plots.

    Prints the accuracy and classification report, then renders a confusion
    matrix heatmap and a ROC curve.

    Args:
        model: Fitted :class:`~sklearn.linear_model.LogisticRegression` model.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Dictionary with keys ``accuracy``, ``roc_auc``, and ``y_pred``.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, title="Logistic Regression – Confusion Matrix")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plot_roc_curve(fpr, tpr, roc_auc)

    return {"accuracy": accuracy, "roc_auc": roc_auc, "y_pred": y_pred}


def interpret_coefficients(
    model: LogisticRegression,
    feature_names: list[str],
) -> pd.DataFrame:
    """Build a coefficient interpretation table with odds ratios.

    Args:
        model: Fitted :class:`~sklearn.linear_model.LogisticRegression` model.
        feature_names: Names of the input features in the same order as the
            training columns.

    Returns:
        DataFrame with columns ``Feature``, ``Coefficient``, and ``Odds Ratio``,
        sorted by ``Odds Ratio`` descending.
    """
    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": model.coef_[0],
            # e^coef gives multiplicative change in odds per unit increase
            "Odds Ratio": np.exp(model.coef_[0]),
        }
    ).sort_values(by="Odds Ratio", ascending=False)

    return coef_df
