"""SVM classifier with XGBoost-based feature selection for credit scoring."""

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.visualization.plots import plot_feature_importances


def select_top_features_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 5,
    random_state: int = 42,
) -> list[str]:
    """Use XGBoost feature importances to select the top-*k* features.

    Args:
        X_train: Training features (MinMax-scaled recommended).
        y_train: Training labels.
        top_k: Number of top features to keep (default 5).
        random_state: Random seed for XGBoost.

    Returns:
        List of the *top_k* most important feature names.
    """
    xgb = XGBClassifier(eval_metric="logloss", random_state=random_state)
    xgb.fit(X_train, y_train)

    importances = pd.Series(xgb.feature_importances_, index=X_train.columns)
    top_features = importances.sort_values(ascending=False).head(top_k).index.tolist()
    print(f"Top {top_k} features selected by XGBoost:\n  {top_features}")
    return top_features


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kernel: str = "rbf",
    random_state: int = 42,
) -> SVC:
    """Train an SVM classifier on the provided data.

    Args:
        X_train: Training features (should be scaled to [0, 1]).
        y_train: Training labels.
        kernel: SVM kernel type (default ``"rbf"``).
        random_state: Random seed for reproducibility.

    Returns:
        Fitted :class:`~sklearn.svm.SVC` with ``probability=True``.
    """
    svm = SVC(kernel=kernel, random_state=random_state, probability=True)
    svm.fit(X_train, y_train)
    return svm


def evaluate_svm(
    model: SVC,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a trained SVM and print key metrics.

    Args:
        model: Fitted :class:`~sklearn.svm.SVC`.
        X_test: Test features (same columns and scale as training).
        y_test: True test labels.

    Returns:
        Dictionary with ``roc_auc`` and ``y_pred``.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    return {"roc_auc": roc_auc, "y_pred": y_pred}


def compute_permutation_importance(
    model: SVC,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 0,
) -> None:
    """Compute and plot Permutation Feature Importance for the SVM.

    Only features whose mean importance minus two standard deviations is
    positive are considered reliably important.

    Args:
        model: Fitted :class:`~sklearn.svm.SVC`.
        X_test: Test features.
        y_test: True test labels.
        n_repeats: Number of permutation rounds per feature (default 10).
        random_state: Random seed.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state
    )

    # Print only features with statistically reliable importance
    for i in result.importances_mean.argsort()[::-1]:
        if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
            print(
                f"{X_test.columns[i]:<20} "
                f"{result.importances_mean[i]:.3f} +/- {result.importances_std[i]:.3f}"
            )

    plot_feature_importances(
        result.importances_mean,
        list(X_test.columns),
        title="SVM – Permutation Feature Importance",
    )


def run_xgboost_svm_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    top_k: int = 5,
    random_state: int = 42,
) -> tuple[SVC, list[str], dict]:
    """Full pipeline: scale → XGBoost feature selection → SVM training + evaluation.

    Args:
        X: All features (unscaled).
        y: Target labels.
        test_size: Fraction of data for the test split (default 0.3).
        top_k: Number of features to select via XGBoost (default 5).
        random_state: Random seed.

    Returns:
        Tuple of (fitted SVM, selected feature names, evaluation metrics dict).
    """
    from sklearn.model_selection import train_test_split

    # MinMax scaling is required for SVM to avoid dominance by large-magnitude features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    top_features = select_top_features_xgboost(X_train, y_train, top_k=top_k)
    svm = train_svm(X_train[top_features], y_train, random_state=random_state)
    metrics = evaluate_svm(svm, X_test[top_features], y_test)

    return svm, top_features, metrics
