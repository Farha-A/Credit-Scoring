"""Decision Tree classifier training and evaluation for credit scoring."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    criterion: str = "gini",
    max_depth: int = 5,
    max_features: float = 0.8,
    max_leaf_nodes: int | None = None,
    min_samples_leaf: int = 7,
    min_samples_split: int = 3,
    min_weight_fraction_leaf: float = 0.0,
) -> DecisionTreeClassifier:
    """Train a Decision Tree classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        criterion: Split quality measure – ``"gini"`` or ``"entropy"`` (default ``"gini"``).
        max_depth: Maximum tree depth; ``None`` grows until leaves are pure (default 5).
        max_features: Fraction of features considered at each split (default 0.8).
        max_leaf_nodes: Maximum number of leaf nodes; ``None`` means unlimited.
        min_samples_leaf: Minimum samples required at a leaf node (default 7).
        min_samples_split: Minimum samples required to split a node (default 3).
        min_weight_fraction_leaf: Minimum weighted fraction of samples at a leaf (default 0).

    Returns:
        Fitted :class:`~sklearn.tree.DecisionTreeClassifier`.
    """
    dt = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
    )
    dt.fit(X_train, y_train)
    return dt


def evaluate_decision_tree(
    model: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a fitted Decision Tree and report standard metrics.

    Args:
        model: Fitted :class:`~sklearn.tree.DecisionTreeClassifier`.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Dictionary with ``accuracy``, ``f1``, and ``roc_auc``.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    return {"accuracy": accuracy, "f1": f1, "roc_auc": roc_auc}


def visualize_tree(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str] | None = None,
    figsize: tuple = (30, 25),
) -> None:
    """Render the full decision tree diagram.

    Args:
        model: Fitted :class:`~sklearn.tree.DecisionTreeClassifier`.
        feature_names: Feature names matching the training columns.
        class_names: Human-readable class labels (e.g. ``["Bad", "Good"]``).
        figsize: Figure size in inches (default ``(30, 25)``).
    """
    plt.figure(figsize=figsize)
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.show()


def lofo_importance(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **dt_kwargs,
) -> dict:
    """Run Leave-One-Feature-Out importance analysis using the Decision Tree.

    Retrains the tree with each feature removed in turn and records accuracy,
    F1 score, and ROC AUC.  The last entry (key ``"None"``) is the full-model
    baseline.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: True test labels.
        **dt_kwargs: Forwarded to :func:`train_decision_tree`.

    Returns:
        Dictionary mapping feature name (or ``"None"``) to
        ``{"accuracy": …, "f1": …, "roc_auc": …}``.
    """
    results: dict = {}
    for col in X_train.columns:
        dt_tmp = train_decision_tree(X_train.drop(col, axis=1), y_train, **dt_kwargs)
        y_pred = dt_tmp.predict(X_test.drop(col, axis=1))
        results[col] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred),
        }

    # Baseline: all features present
    dt_full = train_decision_tree(X_train, y_train, **dt_kwargs)
    y_pred_full = dt_full.predict(X_test)
    results["None"] = {
        "accuracy": accuracy_score(y_test, y_pred_full),
        "f1": f1_score(y_test, y_pred_full),
        "roc_auc": roc_auc_score(y_test, y_pred_full),
    }
    return results


def plot_lofo_results(lofo_results: dict, metric: str = "accuracy") -> None:
    """Bar chart for LOFO importance results.

    Args:
        lofo_results: Output of :func:`lofo_importance`.
        metric: Which metric to plot – ``"accuracy"``, ``"f1"``, or ``"roc_auc"``.
    """
    cols = list(lofo_results.keys())
    values = [lofo_results[c][metric] for c in cols]

    plt.figure(figsize=(10, 6))
    sns.barplot(y=cols, x=values)
    plt.xlabel(metric.replace("_", " ").title())
    plt.title(f"LOFO – {metric} when each feature is removed")
    plt.tight_layout()
    plt.show()
