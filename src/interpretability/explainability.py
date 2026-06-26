"""Model-agnostic and model-specific interpretability tools.

Provides wrappers around SHAP, LIME, Partial Dependence Plots (PDP /ICE),
and Permutation Feature Importance (PFI) that work consistently across the
models in this project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def shap_summary_tree(model, X: pd.DataFrame, max_display: int = 20) -> None:
    """Beeswarm SHAP summary plot for tree-based models.

    Uses :class:`shap.TreeExplainer` which is fast and exact for Random
    Forest, XGBoost, LightGBM, and Decision Tree models.

    Args:
        model: Fitted tree-based model.
        X: Feature matrix (typically the test set).
        max_display: Maximum number of features shown on the plot (default 20).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classifiers sklearn returns a list [class0, class1]
    values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(values_to_plot, X, max_display=max_display)


def shap_summary_linear(model, X: pd.DataFrame) -> None:
    """SHAP summary plot for linear models (e.g. Logistic Regression).

    Uses :class:`shap.Explainer` which selects the appropriate algorithm
    automatically.

    Args:
        model: Fitted linear model.
        X: Feature matrix.
    """
    import shap

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)


def shap_summary_kernel(
    predict_fn,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    nsamples: int = 100,
) -> None:
    """SHAP summary plot for black-box models via KernelExplainer.

    Slower than TreeExplainer; suitable for SVMs, neural networks, and
    stacking ensembles.

    Args:
        predict_fn: Callable that maps a 2-D array to predictions (probabilities
            or raw scores).
        X_background: Representative background dataset used to integrate out
            features (typically a small random sample of the training set).
        X_explain: Data points to explain.
        nsamples: Number of samples for the Kernel SHAP approximation (default 100).
    """
    import shap

    explainer = shap.KernelExplainer(predict_fn, shap.sample(X_background, 100))
    shap_values = explainer.shap_values(X_explain, nsamples=nsamples)
    shap.summary_plot(shap_values, X_explain)


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------


def lime_explain_instance(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    instance_index: int = 0,
    num_features: int = 10,
    class_names: list[str] | None = None,
    predict_fn=None,
) -> None:
    """Explain a single prediction using LIME.

    Args:
        model: Fitted classifier that exposes ``predict_proba``.
        X_train: Training features (used to initialise the LIME explainer).
        X_test: Test features.
        instance_index: Row index in *X_test* to explain (default 0).
        num_features: Number of features to include in the explanation (default 10).
        class_names: Human-readable class labels (default ``["Class 0", "Class 1"]``).
        predict_fn: Custom predict function; falls back to ``model.predict_proba``
            when ``None``.
    """
    import lime.lime_tabular

    if class_names is None:
        class_names = ["Class 0", "Class 1"]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=list(X_train.columns),
        class_names=class_names,
        mode="classification",
    )

    fn = predict_fn if predict_fn is not None else model.predict_proba
    instance = X_test.iloc[instance_index]
    explanation = explainer.explain_instance(instance, fn, num_features=num_features)
    explanation.as_pyplot_figure()
    plt.tight_layout()
    plt.show()

    print("\nTop feature contributions:")
    for feat, weight in explanation.as_list():
        print(f"  {feat}: {weight:.4f}")


# ---------------------------------------------------------------------------
# PDP / ICE
# ---------------------------------------------------------------------------


def plot_pdp_ice(
    model,
    X: pd.DataFrame,
    features: list,
    kind: str = "both",
    subsample: int = 1500,
    grid_resolution: int = 20,
    target: int = 1,
) -> None:
    """Partial Dependence Plot (PDP) and Individual Conditional Expectation (ICE).

    Args:
        model: Fitted scikit-learn–compatible classifier.
        X: Feature matrix to compute PDP/ICE over.
        features: List of feature names or indices to plot.
        kind: ``"average"`` (PDP only), ``"individual"`` (ICE only), or
            ``"both"`` (default).
        subsample: Maximum number of samples for ICE lines (default 1 500).
        grid_resolution: Number of grid points along each feature axis (default 20).
        target: Class index for multi-class models (default 1).
    """
    n_cols = min(3, len(features))
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten()

    for i, feature in enumerate(features):
        PartialDependenceDisplay.from_estimator(
            model,
            X,
            [feature],
            kind=kind,
            subsample=subsample,
            grid_resolution=grid_resolution,
            target=target,
            ax=axes_flat[i],
            random_state=42,
        )
        axes_flat[i].set_title(f"PDP/ICE – {feature}")

    # Hide unused subplot slots
    for j in range(len(features), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Permutation Feature Importance (PFI)
# ---------------------------------------------------------------------------


def plot_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 30,
    random_state: int = 42,
    title: str = "Permutation Feature Importance",
) -> None:
    """Compute and plot Permutation Feature Importance.

    Args:
        model: Fitted scikit-learn–compatible classifier.
        X_test: Test features.
        y_test: True test labels.
        n_repeats: Number of times each feature is permuted (default 30).
        random_state: Random seed.
        title: Plot title.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    sorted_idx = result.importances_mean.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(
        np.array(X_test.columns)[sorted_idx],
        result.importances_mean[sorted_idx],
    )
    plt.xlabel("Mean Decrease in Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.show()
