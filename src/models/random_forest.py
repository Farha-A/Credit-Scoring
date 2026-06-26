"""Random Forest classifier training and evaluation for credit scoring."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.tree import _tree, plot_tree
import matplotlib.pyplot as plt

from src.visualization.plots import plot_confusion_matrix


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    criterion: str = "gini",
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_estimators: Number of trees in the forest (default 100).
        criterion: Impurity measure – ``"gini"`` or ``"entropy"`` (default ``"gini"``).
        random_state: Random seed for reproducibility.

    Returns:
        Fitted :class:`~sklearn.ensemble.RandomForestClassifier`.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_random_forest(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a trained Random Forest and print key metrics.

    Args:
        model: Fitted :class:`~sklearn.ensemble.RandomForestClassifier`.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Dictionary with ``accuracy``, ``roc_auc``, ``sensitivity``,
        ``specificity``, and ``y_pred``.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity = recall for the positive class
    sensitivity = tp / (tp + fn)
    # Specificity = recall for the negative class
    specificity = tn / (tn + fp)

    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")

    plot_confusion_matrix(cm, title="Random Forest – Confusion Matrix")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "y_pred": y_pred,
    }


def get_gini_reductions(tree_estimator, feature_names: list[str]) -> None:
    """Print the Gini impurity reduction at each split of a single decision tree.

    Args:
        tree_estimator: A single :class:`~sklearn.tree.DecisionTreeClassifier`
            extracted from ``rf_model.estimators_[i]``.
        feature_names: Feature names in the same order as the training columns.
    """
    tree_ = tree_estimator.tree_
    print("Node   | Feature            | Gini Reduction")
    print("---------------------------------------------")
    for node in range(tree_.node_count):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            continue  # leaf node – no split to report
        left = tree_.children_left[node]
        right = tree_.children_right[node]
        n_total = tree_.n_node_samples[node]
        weighted_child_impurity = (
            tree_.n_node_samples[left] / n_total * tree_.impurity[left]
            + tree_.n_node_samples[right] / n_total * tree_.impurity[right]
        )
        gini_reduction = tree_.impurity[node] - weighted_child_impurity
        print(f"{node:<6} | {feature_names[tree_.feature[node]]:<18} | {gini_reduction:.4f}")


def visualize_forest_trees(
    model: RandomForestClassifier,
    feature_names: list[str],
    class_names: list[str] | None = None,
    max_trees: int = 3,
    max_depth: int = 3,
) -> None:
    """Visualize the first *max_trees* trees in the forest.

    For each tree the best split (highest Gini reduction) is printed and a
    shallow tree diagram is shown.

    Args:
        model: Fitted :class:`~sklearn.ensemble.RandomForestClassifier`.
        feature_names: Feature names in the same order as the training columns.
        class_names: Optional list of human-readable class labels.
        max_trees: Number of trees to display (default 3).
        max_depth: Maximum depth to render in the tree diagram (default 3).
    """
    for i, estimator in enumerate(model.estimators_[:max_trees]):
        print(f"\n--- Tree {i + 1} ---")
        get_gini_reductions(estimator, feature_names)

        plt.figure(figsize=(12, 6))
        plot_tree(
            estimator,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            max_depth=max_depth,
        )
        plt.title(f"Tree {i + 1} (top {max_depth} levels)")
        plt.tight_layout()
        plt.show()
