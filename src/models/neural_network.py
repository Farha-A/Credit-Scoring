"""Two-layer neural network training and evaluation for credit scoring.

Requires TensorFlow / Keras.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score


def build_neural_network(input_dim: int = 20):
    """Construct a two-hidden-layer neural network for binary classification.

    Architecture:
        * Dense(512) → LeakyReLU → Dropout(0.5)
        * Dense(512) → LeakyReLU → Dropout(0.5)
        * Dense(1, sigmoid)

    L2 regularisation (λ=0.2) is applied to both hidden layers to reduce
    overfitting on the large tabular dataset.

    Args:
        input_dim: Number of input features (default 20).

    Returns:
        Compiled :class:`~tensorflow.keras.models.Sequential` model.
    """
    # Keras imports are deferred so the module can be imported without TF
    from tensorflow.keras import Input, regularizers
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.layers import LeakyReLU
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import SGD

    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(512, kernel_regularizer=regularizers.l2(0.2)),
            LeakyReLU(negative_slope=0.01),
            Dropout(0.5),
            Dense(512, kernel_regularizer=regularizers.l2(0.2)),
            LeakyReLU(negative_slope=0.01),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    epochs: int = 20,
    batch_size: int = 64,
    validation_split: float = 0.2,
    verbose: int = 1,
):
    """Build and train the two-layer neural network.

    Args:
        X_train: Training features.
        y_train: Training labels.
        epochs: Number of training epochs (default 20).
        batch_size: Mini-batch size (default 64).
        validation_split: Fraction of training data used for validation (default 0.2).
        verbose: Keras verbosity level – 0 (silent), 1 (progress bar), 2 (one line/epoch).

    Returns:
        Fitted Keras ``Sequential`` model.
    """
    model = build_neural_network(input_dim=X_train.shape[1])
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=verbose,
    )
    return model


def evaluate_neural_network(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Evaluate the trained neural network on the test set.

    Args:
        model: Fitted Keras ``Sequential`` model.
        X_test: Test features.
        y_test: True test labels.
        threshold: Decision threshold for converting probabilities to classes (default 0.5).

    Returns:
        Dictionary with ``accuracy``, ``f1``, and ``roc_auc``.
    """
    y_prob = model.predict(X_test)
    y_pred = (y_prob > threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    return {"accuracy": accuracy, "f1": f1, "roc_auc": roc_auc}


def fit_global_surrogate(
    model,
    X_test: pd.DataFrame,
    y_pred_nn: np.ndarray,
) -> LogisticRegression:
    """Fit a Logistic Regression surrogate to approximate the neural network globally.

    A high R² between the surrogate predictions and the neural network predictions
    indicates the surrogate is a faithful (and interpretable) proxy.

    Args:
        model: Fitted Keras neural network (unused here, kept for API symmetry).
        X_test: Test features.
        y_pred_nn: Binary predictions from the neural network on *X_test*.

    Returns:
        Fitted :class:`~sklearn.linear_model.LogisticRegression` surrogate.
    """
    surrogate = LogisticRegression(random_state=0)
    surrogate.fit(X_test, y_pred_nn.ravel())

    surrogate_pred = surrogate.predict(X_test)
    r2 = r2_score(y_pred_nn.ravel(), surrogate_pred)
    print(f"Global surrogate R² vs NN predictions: {r2:.4f}")

    return surrogate


def lofo_importance(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    epochs: int = 20,
    verbose: int = 0,
) -> dict:
    """Leave-One-Feature-Out importance analysis for the neural network.

    Retrains the network from scratch with each feature removed in turn.
    This is expensive – consider running on a subset of the data or with
    fewer epochs for exploratory analysis.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: True test labels.
        epochs: Epochs per re-train (default 20).
        verbose: Keras verbosity level (default 0 – silent).

    Returns:
        Dictionary mapping feature name (or ``"None"``) to
        ``{"accuracy": …, "f1": …, "roc_auc": …}``.
    """
    results: dict = {}
    for col in X_train.columns:
        nn_tmp = train_neural_network(
            X_train.drop(col, axis=1), y_train, epochs=epochs, verbose=verbose
        )
        y_prob = nn_tmp.predict(X_test.drop(col, axis=1))
        y_pred = (y_prob > 0.5).astype(int)
        results[col] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

    # Baseline with all features
    nn_full = train_neural_network(X_train, y_train, epochs=epochs, verbose=verbose)
    y_prob_full = nn_full.predict(X_test)
    y_pred_full = (y_prob_full > 0.5).astype(int)
    results["None"] = {
        "accuracy": accuracy_score(y_test, y_pred_full),
        "f1": f1_score(y_test, y_pred_full),
        "roc_auc": roc_auc_score(y_test, y_prob_full),
    }
    return results
