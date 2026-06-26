"""Stacking ensemble model: TabNet + XGBoost base learners, Random Forest meta-model.

Architecture (Nour):
    1. TabNet and XGBoost are trained independently on the training data.
    2. Their predictions on the training set are stacked with the original features
       to form a meta-feature matrix.
    3. A Random Forest is trained on this meta-feature matrix.
    4. At inference time the same stacking is applied.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


def _to_numpy(arr):
    """Convert DataFrame or Series to ndarray if needed."""
    return arr.values if hasattr(arr, "values") else arr


def train_stacked_ensemble(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tabnet_verbose: int = 0,
) -> tuple:
    """Train the stacking ensemble and return the fitted components.

    TabNet and XGBoost serve as base models.  Their predictions on the
    *training* set are concatenated with the original features to build a
    richer representation for the Random Forest meta-model.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: True test labels.
        tabnet_verbose: TabNet verbosity level (default 0 – silent).

    Returns:
        Tuple of (meta_model, tabnet, xgb_model, metrics_dict).
    """
    from pytorch_tabnet.tab_model import TabNetClassifier

    X_train_np = _to_numpy(X_train)
    X_test_np = _to_numpy(X_test)
    y_train_np = _to_numpy(y_train)
    y_test_np = _to_numpy(y_test)

    # --- Base learners ---
    tabnet = TabNetClassifier(verbose=tabnet_verbose)
    tabnet.fit(X_train_np, y_train_np)

    xgb_model = XGBClassifier(eval_metric="logloss")
    xgb_model.fit(X_train_np, y_train_np)

    # --- Build meta-features by stacking base predictions ---
    tabnet_train_preds = tabnet.predict(X_train_np)
    xgb_train_preds = xgb_model.predict(X_train_np)
    X_train_meta = np.column_stack((X_train_np, tabnet_train_preds, xgb_train_preds))

    tabnet_test_preds = tabnet.predict(X_test_np)
    xgb_test_preds = xgb_model.predict(X_test_np)
    X_test_meta = np.column_stack((X_test_np, tabnet_test_preds, xgb_test_preds))

    # --- Meta-model ---
    meta_model = RandomForestClassifier()
    meta_model.fit(X_train_meta, y_train_np)

    final_preds = meta_model.predict(X_test_meta)
    accuracy = accuracy_score(y_test_np, final_preds)
    metrics = {"accuracy": accuracy}

    print(f"Stacked ensemble accuracy: {accuracy:.4f}")
    return meta_model, tabnet, xgb_model, metrics


def predict_with_stack(
    meta_model: RandomForestClassifier,
    tabnet,
    xgb_model: XGBClassifier,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate predictions from the stacking ensemble.

    Args:
        meta_model: Fitted Random Forest meta-model.
        tabnet: Fitted TabNet base learner.
        xgb_model: Fitted XGBoost base learner.
        X: Feature matrix for inference.

    Returns:
        1-D array of predicted class labels.
    """
    X_np = _to_numpy(X)
    tabnet_preds = tabnet.predict(X_np)
    xgb_preds = xgb_model.predict(X_np)
    X_meta = np.column_stack((X_np, tabnet_preds, xgb_preds))
    return meta_model.predict(X_meta)


def evaluate_stacked_ensemble(
    meta_model: RandomForestClassifier,
    tabnet,
    xgb_model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate the stacking ensemble and print comprehensive metrics.

    Args:
        meta_model: Fitted Random Forest meta-model.
        tabnet: Fitted TabNet base learner.
        xgb_model: Fitted XGBoost base learner.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Dictionary with ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``roc_auc``, and confusion matrix ``cm``.
    """
    y_pred = predict_with_stack(meta_model, tabnet, xgb_model, X_test)
    y_test_np = _to_numpy(y_test)

    metrics = {
        "accuracy": accuracy_score(y_test_np, y_pred),
        "precision": precision_score(y_test_np, y_pred),
        "recall": recall_score(y_test_np, y_pred),
        "f1": f1_score(y_test_np, y_pred),
        "roc_auc": roc_auc_score(y_test_np, y_pred),
        "cm": confusion_matrix(y_test_np, y_pred),
    }

    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key.capitalize():10}: {metrics[key]:.4f}")

    # Confusion matrix heatmap
    plt.figure()
    sns.heatmap(metrics["cm"], annot=True, fmt="d")
    plt.title("Stacked Ensemble – Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    return metrics


def plot_tabnet_feature_importance(tabnet, feature_names: list[str]) -> None:
    """Horizontal bar chart of TabNet's built-in feature importances.

    Args:
        tabnet: Fitted TabNet model.
        feature_names: Feature names in the same order as the training columns.
    """
    importances = tabnet.feature_importances_
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 5))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("TabNet Feature Importances")
    plt.tight_layout()
    plt.show()


def plot_meta_model_feature_importance(
    meta_model: RandomForestClassifier,
    feature_names: list[str],
) -> None:
    """Horizontal bar chart of the Random Forest meta-model's feature importances.

    The feature list should include the two extra columns added by the stacking
    step: ``"tabnet_pred"`` and ``"xgb_pred"``.

    Args:
        meta_model: Fitted Random Forest meta-model.
        feature_names: Feature names including ``"tabnet_pred"`` and ``"xgb_pred"``.
    """
    importances = meta_model.feature_importances_
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 5))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Meta-Model (Random Forest) Feature Importances")
    plt.tight_layout()
    plt.show()
