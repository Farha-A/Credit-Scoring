"""Credit Scoring Streamlit App.

Layout
------
Sidebar  – model selector (scans model_weights/ for .pkl files). A toggle switches
           between single-model mode and ensemble mode, where you choose how many
           models (and which ones) to combine via soft or hard voting; models are
           listed ordered by accuracy/AUC-ROC taken from their training notebooks.
Tab 1    – Manual Prediction: fill in all 20 loan features, get a prediction.
Tab 2    – Batch Prediction:  upload a CSV, sample N rows, run predictions on them.

Prerequisites
-------------
Place pre-trained model .pkl files (and/or a .keras model) and a scaler.pkl
inside ``model_weights/``. The scaler is applied automatically when present;
raw values are used otherwise. Loading .keras models requires tensorflow.

The stacking ensemble from Nour_Ensemble_Model.ipynb is a 3-file bundle —
``ensemble_xgb.pkl``, ``ensemble_meta_model.pkl``, ``ensemble_tabnet.zip`` —
loaded together as one "Stacking Ensemble" model. Requires pytorch-tabnet.

Launch::

    streamlit run app.py
"""

import os

# PyTorch (pulled in by pytorch-tabnet) and MKL/XGBoost both bundle their own
# OpenMP runtime; on Windows loading both in one process aborts with
# "OMP: Error #15" unless this is set before either is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import glob

import numpy as np
import pandas as pd
import streamlit as st

try:
    from tensorflow import keras as tf_keras
    KERAS_AVAILABLE = True
except ImportError:
    tf_keras = None
    KERAS_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TabNetClassifier = None
    TABNET_AVAILABLE = False

try:
    import h2o
    H2O_AVAILABLE = True
except ImportError:
    h2o = None
    H2O_AVAILABLE = False

# The 3 files the stacking ensemble notebook (Nour_Ensemble_Model.ipynb) saves;
# they must be loaded together, not as independent models.
ENSEMBLE_XGB_FILE    = "ensemble_xgb.pkl"
ENSEMBLE_META_FILE   = "ensemble_meta_model.pkl"
ENSEMBLE_TABNET_FILE = "ensemble_tabnet.zip"
ENSEMBLE_DISPLAY_NAME = "Stacking Ensemble"

# h2o.save_model() writes a single extensionless binary file (Nour_GBM.ipynb);
# it needs a running local H2O (Java) cluster to load and predict.
H2O_GBM_FILE = "gbm_h2o_model"
H2O_GBM_DISPLAY_NAME = "H2O GBM"

# xgboost_svm.pkl (Arwa_XGBoost_SVM.ipynb) is an SVC trained only on the top-5
# features XGBoost selected by importance (printed in that notebook's output),
# not all 20 — it must be sliced down to just these, in this order.
XGBOOST_SVM_FEATURES = [
    "last_fico_range_high",
    "recoveries",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "out_prncp",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHTS_DIR = "model_weights"

# The 20 features the models were trained on (order matters for the scaler)
FEATURES = [
    "last_pymnt_d",
    "total_rec_prncp",
    "last_pymnt_amnt",
    "out_prncp",
    "total_rec_late_fee",
    "last_fico_range_high",
    "installment",
    "loan_amnt",
    "total_rec_int",
    "out_prncp_inv",
    "total_pymnt",
    "funded_amnt_inv",
    "recoveries",
    "debt_settlement_flag",
    "hardship_flag",
    "mo_sin_old_rev_tl_op",
    "revol_util",
    "dti",
    "all_util",
    "annual_inc",
]

# These 3 columns are label-encoded / already-integer flags rather than
# continuous values, so the preprocessing pipeline's StandardScaler was
# fit only on the other 17 columns. Keep this in sync with standardizing()
# in notebooks/Data_Preprocessing.ipynb.
UNSCALED_FEATURES = ["last_pymnt_d", "debt_settlement_flag", "hardship_flag"]
SCALED_FEATURES = [f for f in FEATURES if f not in UNSCALED_FEATURES]

# Human-readable labels for the input form
FEATURE_META = {
    "last_pymnt_d":            ("Last Payment Month (encoded)",          1.0,   120.0,  30.0),
    "total_rec_prncp":         ("Total Principal Received ($)",          0.0, 40000.0, 5000.0),
    "last_pymnt_amnt":         ("Last Payment Amount ($)",               0.0,  5000.0,  300.0),
    "out_prncp":               ("Outstanding Principal ($)",             0.0, 40000.0, 1000.0),
    "total_rec_late_fee":      ("Total Late Fees Received ($)",          0.0,   200.0,    0.0),
    "last_fico_range_high":    ("Last FICO Score (Upper Range)",       580.0,   850.0,  700.0),
    "installment":             ("Monthly Installment ($)",              30.0,  1500.0,  300.0),
    "loan_amnt":               ("Loan Amount ($)",                    1000.0, 40000.0, 10000.0),
    "total_rec_int":           ("Total Interest Received ($)",           0.0, 10000.0, 1000.0),
    "out_prncp_inv":           ("Outstanding Principal – Investor ($)",  0.0, 40000.0, 1000.0),
    "total_pymnt":             ("Total Payments Received ($)",           0.0, 50000.0, 8000.0),
    "funded_amnt_inv":         ("Funded Amount – Investor ($)",        1000.0, 40000.0, 10000.0),
    "recoveries":              ("Post Charge-Off Recovery ($)",          0.0,  5000.0,    0.0),
    "debt_settlement_flag":    ("Debt Settlement (0 = No, 1 = Yes)",    0.0,     1.0,    0.0),
    "hardship_flag":           ("Hardship Flag (0 = No, 1 = Yes)",      0.0,     1.0,    0.0),
    "mo_sin_old_rev_tl_op":    ("Months Since Oldest Revolving Account",6.0,   600.0,  120.0),
    "revol_util":              ("Revolving Utilization (%)",             0.0,   100.0,   50.0),
    "dti":                     ("Debt-to-Income Ratio (%)",             0.0,    45.0,   18.0),
    "all_util":                ("Combined Credit Utilization (%)",       0.0,   100.0,   50.0),
    "annual_inc":              ("Annual Income ($)",                 20000.0, 300000.0, 65000.0),
}

LABEL_MAP = {0: "❌ Bad Credit", 1: "✅ Good Credit"}
COLOR_MAP  = {0: "#FF4B4B",      1: "#21C55D"}

# Accuracy / AUC-ROC of each trained model, as reported in its training notebook.
# Keyed by the display name available_models() derives from the .pkl filename.
# Used only to order/label model pickers — not recomputed at runtime (the app
# has no labeled data to score against). Models with no entry here fall back to
# 0.0 for both metrics and sort to the bottom of any list.
METRICS = {
    "Random Forest":         {"acc": 0.9900, "auc": 0.9993},  # Arwa Random_Forest
    "Decision Tree":         {"acc": 0.9674, "auc": 0.9856},  # Farha_DT
    "Xgboost Svm":           {"acc": 0.9500, "auc": 0.9858},  # Arwa_XGBoost_SVM (SVM)
    "Logistic Regression":   {"acc": 0.9649, "auc": 0.9900},  # Arwa Logistic_Regression (AUC read off ROC plot, approximate)
    "Pltr Model":            {"acc": 0.9253, "auc": 0.9459},  # Farha_PLTR
    "Two Layer Nn":          {"acc": 0.9415, "auc": 0.9846},  # Farha_2L_NN
    "Stacking Ensemble":     {"acc": 0.9904, "auc": 0.9903},  # Nour_Ensemble_Model (TabNet+XGB -> RF)
    "Gbdt Rahma":            {"acc": 0.9894, "auc": 0.9991},  # Rahma_GBDT
    "Lightgbm Model":        {"acc": 0.9903, "auc": 0.9902},  # Nour_LightGBM
    "Svm Rahma":             {"acc": 0.9896, "auc": 0.9973},  # Rahma_SVM (RBF, best)
    "Xgboost Rahma":         {"acc": 0.9900, "auc": 0.9993},  # Rahma_XGBoost
    "H2O GBM":               {"acc": 0.9835, "auc": 0.9965},  # Nour_GBM
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_scaler():
    """Load scaler.pkl from model_weights/ if it exists, else return None."""
    path = os.path.join(WEIGHTS_DIR, "scaler.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


class KerasClassifierWrapper:
    """Adapts a binary sigmoid-output tf.keras model to the sklearn-style
    predict()/predict_proba() interface the rest of the app expects."""

    def __init__(self, keras_model):
        self._model = keras_model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_good = np.asarray(self._model.predict(X, verbose=0)).reshape(-1)
        return np.column_stack([1 - p_good, p_good])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class FeatureSubsetWrapper:
    """Wraps a model trained on a subset of FEATURES (in a specific column
    order) so it can be called with the full FEATURES-ordered array like
    every other model in the app."""

    def __init__(self, model, feature_subset: list[str]):
        self._model = model
        self._idx = [FEATURES.index(f) for f in feature_subset]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X[:, self._idx])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X[:, self._idx])


def _apply_pltr_rules(X_df: pd.DataFrame, rules: list) -> pd.DataFrame:
    """Reproduces apply_rules() from Farha_PLTR.ipynb: turns raw feature
    values into one binary indicator column per decision-tree-derived rule,
    where each rule is a list of (feature_name, op, threshold) conditions."""
    X_rules = pd.DataFrame(index=X_df.index)
    for i, rule in enumerate(rules):
        # Column name must match apply_rules() exactly — the fitted Pipeline's
        # StandardScaler validates against these exact training-time names.
        rule_str = " AND ".join(f"{feat} {op} {val:.2f}" for feat, op, val in rule)
        cond = pd.Series(True, index=X_df.index)
        for feat, op, val in rule:
            cond &= (X_df[feat] <= val) if op == "<=" else (X_df[feat] > val)
        X_rules[f"rule_{i}: {rule_str}"] = cond.astype(int)
    return X_rules


class PLTRWrapper:
    """Wraps the {"model": Pipeline, "rules": [...]} dict saved by
    Farha_PLTR.ipynb: raw features must be converted to rule-indicator
    columns via _apply_pltr_rules() before the linear Pipeline can predict."""

    def __init__(self, bundle: dict):
        self._model = bundle["model"]
        self._rules = bundle["rules"]

    def _rule_features(self, X: np.ndarray) -> pd.DataFrame:
        return _apply_pltr_rules(pd.DataFrame(X, columns=FEATURES), self._rules)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._rule_features(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._rule_features(X))


class StackingEnsembleWrapper:
    """Reproduces predict_with_stack() from Nour_Ensemble_Model.ipynb: TabNet's
    and XGBoost's predictions are appended as 2 extra columns to the raw
    features, and a RandomForest meta-model predicts on that combined input."""

    def __init__(self, tabnet_model, xgb_model, meta_model):
        self._tabnet = tabnet_model
        self._xgb = xgb_model
        self._meta = meta_model

    def _stacked_features(self, X: np.ndarray) -> np.ndarray:
        tabnet_preds = self._tabnet.predict(X)
        xgb_preds = self._xgb.predict(X)
        return np.column_stack([X, tabnet_preds, xgb_preds])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._meta.predict(self._stacked_features(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._meta.predict_proba(self._stacked_features(X))


class H2OGBMWrapper:
    """Wraps an H2OGradientBoostingEstimator (Nour_GBM.ipynb): raw features
    must be converted to an H2OFrame before predict(), and H2O's own p0/p1
    probability columns map directly onto predict_proba()'s [bad, good]."""

    def __init__(self, h2o_model):
        self._model = h2o_model

    def _predict_frame(self, X: np.ndarray):
        df = pd.DataFrame(X, columns=FEATURES)
        return self._model.predict(h2o.H2OFrame(df)).as_data_frame()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_frame(X)["predict"].astype(int).values

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._predict_frame(X)[["p0", "p1"]].values


@st.cache_resource
def _ensure_h2o_started() -> None:
    """Start (or connect to) the local H2O Java cluster exactly once per
    Streamlit process — h2o.init() is slow and must not be repeated."""
    h2o.init()


@st.cache_resource
def load_model(model_path):
    """Load a model from *model_path*: a pickled sklearn-style model, a
    .keras model, the extensionless H2O GBM binary, or (if *model_path* is
    the 3-path ensemble bundle tuple) the TabNet+XGBoost+RandomForest
    stacking ensemble — each wrapped to expose the same
    predict()/predict_proba() API."""
    if isinstance(model_path, tuple):
        xgb_path, tabnet_path, meta_path = model_path
        tabnet_model = TabNetClassifier()
        tabnet_model.load_model(tabnet_path)
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
        with open(meta_path, "rb") as f:
            meta_model = pickle.load(f)
        return StackingEnsembleWrapper(tabnet_model, xgb_model, meta_model)
    if model_path.endswith(".keras"):
        return KerasClassifierWrapper(tf_keras.models.load_model(model_path))
    if os.path.basename(model_path) == H2O_GBM_FILE:
        _ensure_h2o_started()
        return H2OGBMWrapper(h2o.load_model(model_path))
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if os.path.basename(model_path) == "xgboost_svm.pkl":
        return FeatureSubsetWrapper(model, XGBOOST_SVM_FEATURES)
    if os.path.basename(model_path) == "pltr_model.pkl":
        return PLTRWrapper(model)
    return model


def available_models() -> dict[str, str | tuple[str, str, str]]:
    """Return {display_name: file_path} for every .pkl or (if tensorflow is
    installed) .keras file in model_weights/ (except scaler.pkl), plus one
    "Stacking Ensemble" entry (mapped to a (xgb_path, tabnet_path, meta_path)
    tuple) if all 3 of its files are present and pytorch-tabnet is installed."""
    paths = glob.glob(os.path.join(WEIGHTS_DIR, "*.pkl"))
    if KERAS_AVAILABLE:
        paths += glob.glob(os.path.join(WEIGHTS_DIR, "*.keras"))
    # The ensemble's XGBoost/meta-model .pkl files are components of the
    # bundle below, not standalone-usable models (the meta-model alone
    # expects 22 columns, not the app's 20 raw features) — don't list them.
    ensemble_component_names = {
        os.path.splitext(ENSEMBLE_XGB_FILE)[0],
        os.path.splitext(ENSEMBLE_META_FILE)[0],
    }
    models: dict[str, str | tuple[str, str, str]] = {}
    for p in sorted(paths):
        name = os.path.splitext(os.path.basename(p))[0]
        if name == "scaler" or name in ensemble_component_names:
            continue
        # Convert snake_case filename → Title Case display name
        display = name.replace("_", " ").title()
        models[display] = p

    xgb_path    = os.path.join(WEIGHTS_DIR, ENSEMBLE_XGB_FILE)
    meta_path   = os.path.join(WEIGHTS_DIR, ENSEMBLE_META_FILE)
    tabnet_path = os.path.join(WEIGHTS_DIR, ENSEMBLE_TABNET_FILE)
    if TABNET_AVAILABLE and all(os.path.exists(p) for p in (xgb_path, meta_path, tabnet_path)):
        models[ENSEMBLE_DISPLAY_NAME] = (xgb_path, tabnet_path, meta_path)

    h2o_gbm_path = os.path.join(WEIGHTS_DIR, H2O_GBM_FILE)
    if H2O_AVAILABLE and os.path.exists(h2o_gbm_path):
        models[H2O_GBM_DISPLAY_NAME] = h2o_gbm_path

    return models


def ordered_models(models_dict: dict[str, str], sort_by: str = "auc") -> list[tuple[str, str, float, float]]:
    """Return (display, path, acc, auc) tuples sorted best-first by *sort_by*
    ("auc" or "acc"). Models without a METRICS entry default to 0.0 and sort last."""
    rows = []
    for display, path in models_dict.items():
        m = METRICS.get(display, {})
        rows.append((display, path, m.get("acc", 0.0), m.get("auc", 0.0)))
    key_idx = 3 if sort_by == "auc" else 2
    rows.sort(key=lambda r: r[key_idx], reverse=True)
    return rows


def preprocess(df: pd.DataFrame, scaler) -> np.ndarray:
    """Apply scaler (to the subset of columns it was fit on) if available,
    otherwise return raw values as array."""
    out = df[FEATURES].astype(float).copy()
    if scaler is not None:
        out[SCALED_FEATURES] = scaler.transform(out[SCALED_FEATURES])
    return out[FEATURES].values


def predict(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (class_labels, probabilities).  Probabilities may be None."""
    labels = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return labels, probs


def predict_ensemble(
    loaded_models: list[tuple[str, object]],
    X: np.ndarray,
    method: str = "soft",
) -> tuple[np.ndarray, np.ndarray | None, list[list[tuple[str, int, np.ndarray | None]]]]:
    """Combine predictions from multiple models into one consensus.

    Args:
        loaded_models: list of (display_name, model) pairs.
        X: preprocessed feature array (n_rows, n_features).
        method: "soft" averages predict_proba across models (falls back to
            hard votes for any model without predict_proba); "hard" takes a
            plain majority vote of each model's predicted label.

    Returns:
        labels: consensus class per row.
        probs: consensus probability array per row (soft method only; None for hard).
        per_model: for each row, a list of (display_name, label, prob_row) for
            every model — used to render the individual-model breakdown table.
    """
    n_rows = X.shape[0]
    per_model_preds = []  # list of (display, labels_array, probs_array_or_None)
    for display, model in loaded_models:
        labels_i, probs_i = predict(model, X)
        per_model_preds.append((display, labels_i, probs_i))

    per_model: list[list[tuple[str, int, np.ndarray | None]]] = [[] for _ in range(n_rows)]
    for display, labels_i, probs_i in per_model_preds:
        for row in range(n_rows):
            prob_row = probs_i[row] if probs_i is not None else None
            per_model[row].append((display, int(labels_i[row]), prob_row))

    if method == "soft":
        prob_stack = [probs_i for _, _, probs_i in per_model_preds if probs_i is not None]
        if prob_stack:
            probs = np.mean(prob_stack, axis=0)
            labels = np.argmax(probs, axis=1)
            return labels, probs, per_model
        # No model exposed predict_proba — fall back to hard voting below.

    # Hard (majority) voting on labels.
    label_stack = np.array([labels_i for _, labels_i, _ in per_model_preds])  # (n_models, n_rows)
    labels = np.array([
        np.bincount(label_stack[:, row]).argmax() for row in range(n_rows)
    ])
    return labels, None, per_model


def render_single_result(label: int, prob: np.ndarray | None) -> None:
    """Render the prediction card for the manual-input tab."""
    colour = COLOR_MAP[label]
    st.markdown(
        f"<div style='padding:1.2rem;border-radius:10px;background:{colour}22;"
        f"border:2px solid {colour};text-align:center;'>"
        f"<h2 style='color:{colour};margin:0'>{LABEL_MAP[label]}</h2></div>",
        unsafe_allow_html=True,
    )

    if prob is not None:
        st.write("")
        col_bad, col_good = st.columns(2)
        col_bad.metric("Probability – Bad Credit",  f"{prob[0]*100:.1f}%")
        col_good.metric("Probability – Good Credit", f"{prob[1]*100:.1f}%")

        # Simple bar chart for confidence
        bar_df = pd.DataFrame(
            {"Class": ["Bad Credit", "Good Credit"], "Probability": [prob[0], prob[1]]}
        )
        st.bar_chart(bar_df.set_index("Class"))


def render_ensemble_result(
    label: int,
    prob: np.ndarray | None,
    per_model_row: list[tuple[str, int, np.ndarray | None]],
) -> None:
    """Render the consensus card, per-model breakdown table, and agreement line."""
    render_single_result(label, prob)

    n_agree = sum(1 for _, m_label, _ in per_model_row if m_label == label)
    st.caption(f"🤝 Agreement: {n_agree}/{len(per_model_row)} models agree with the consensus.")

    breakdown = pd.DataFrame(
        {
            "Model": [name for name, _, _ in per_model_row],
            "Prediction": [LABEL_MAP[m_label] for _, m_label, _ in per_model_row],
            "P(Good Credit)": [
                f"{m_prob[1] * 100:.1f}%" if m_prob is not None else "n/a"
                for _, _, m_prob in per_model_row
            ],
        }
    )
    st.dataframe(breakdown, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Scoring",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Scoring System")
st.caption("Select a model from the sidebar, then use the tabs below.")

# ---------------------------------------------------------------------------
# Sidebar – model selector
# ---------------------------------------------------------------------------

models_dict = available_models()

if not KERAS_AVAILABLE and glob.glob(os.path.join(WEIGHTS_DIR, "*.keras")):
    st.sidebar.warning(
        "A `.keras` model was found in `model_weights/` but TensorFlow isn't "
        "installed, so it's hidden from the list. Run `pip install tensorflow` "
        "and reload to enable it."
    )

_ensemble_paths = {
    ENSEMBLE_XGB_FILE:    os.path.join(WEIGHTS_DIR, ENSEMBLE_XGB_FILE),
    ENSEMBLE_META_FILE:   os.path.join(WEIGHTS_DIR, ENSEMBLE_META_FILE),
    ENSEMBLE_TABNET_FILE: os.path.join(WEIGHTS_DIR, ENSEMBLE_TABNET_FILE),
}
_ensemble_files_present = [name for name, p in _ensemble_paths.items() if os.path.exists(p)]
if _ensemble_files_present and not (TABNET_AVAILABLE and len(_ensemble_files_present) == 3):
    _reasons = []
    if not TABNET_AVAILABLE:
        _reasons.append("`pytorch-tabnet` isn't installed")
    _missing = [name for name in _ensemble_paths if name not in _ensemble_files_present]
    if _missing:
        _reasons.append(f"missing file(s): {', '.join(_missing)}")
    st.sidebar.warning(
        f"Stacking ensemble files found in `model_weights/` but it can't be "
        f"loaded — {'; '.join(_reasons)}."
    )

if not H2O_AVAILABLE and os.path.exists(os.path.join(WEIGHTS_DIR, H2O_GBM_FILE)):
    st.sidebar.warning(
        f"`{H2O_GBM_FILE}` was found in `model_weights/` but the `h2o` "
        "package isn't installed, so it's hidden from the list. Run "
        "`pip install h2o` and reload to enable it."
    )

ensemble_mode = False
chosen_models: list[tuple[str, object]] = []
voting_method = "soft"

with st.sidebar:
    st.header("⚙️ Model Selection")

    if not models_dict:
        st.error(
            f"No model .pkl files found in `{WEIGHTS_DIR}/`.\n\n"
            "Place your trained model files there and reload the app."
        )
        st.stop()

    scaler = load_scaler()

    ensemble_mode = st.toggle("🧩 Use multiple models (ensemble)")

    if not ensemble_mode:
        # ── Single-model mode ────────────────────────────────────────────
        ranked = ordered_models(models_dict, sort_by="auc")
        options = [display for display, *_ in ranked]
        labels_by_display = {
            display: f"{display} — acc {acc:.2%} / auc {auc:.2%}"
            for display, _, acc, auc in ranked
        }

        chosen_display = st.selectbox(
            "Choose a model",
            options,
            format_func=lambda d: labels_by_display[d],
        )
        chosen_path = models_dict[chosen_display]
        model = load_model(chosen_path)
        chosen_models = [(chosen_display, model)]

        st.success(f"Loaded: **{chosen_display}**")

        st.markdown("---")
        st.markdown("**Available models (best AUC-ROC first)**")
        for display, _, acc, auc in ranked:
            icon = "🔵" if display == chosen_display else "⚪"
            st.markdown(f"{icon} {display} — acc {acc:.2%} / auc {auc:.2%}")

    else:
        # ── Ensemble mode ────────────────────────────────────────────────
        sort_by_label = st.radio("Order models by", ["AUC-ROC", "Accuracy"], horizontal=True)
        sort_by = "auc" if sort_by_label == "AUC-ROC" else "acc"
        voting_label = st.radio("Voting method", ["Soft (avg. probability)", "Hard (majority vote)"])
        voting_method = "soft" if voting_label.startswith("Soft") else "hard"

        ranked = ordered_models(models_dict, sort_by=sort_by)
        options = [display for display, *_ in ranked]
        labels_by_display = {
            display: f"{display} — acc {acc:.2%} / auc {auc:.2%}"
            for display, _, acc, auc in ranked
        }

        max_n = len(options)
        default_n = min(3, max_n)
        n_models = st.slider("Number of models (N)", min_value=1 if max_n < 2 else 2, max_value=max_n, value=default_n)

        default_selection = options[:n_models]
        selected_displays = st.multiselect(
            "Models to include",
            options,
            default=default_selection,
            format_func=lambda d: labels_by_display[d],
        )

        if not selected_displays:
            st.warning("Select at least one model to get a prediction.")
            st.stop()

        for display in selected_displays:
            chosen_models.append((display, load_model(models_dict[display])))

        st.success(f"Ensemble of **{len(chosen_models)}** model(s), {voting_label.split(' ')[0].lower()} voting.")

        st.markdown("---")
        st.markdown(f"**Available models (best {sort_by_label} first)**")
        for display, _, acc, auc in ranked:
            icon = "🔵" if display in selected_displays else "⚪"
            st.markdown(f"{icon} {display} — acc {acc:.2%} / auc {auc:.2%}")

    if scaler:
        st.info("StandardScaler found — inputs will be scaled automatically.")
    else:
        st.warning("No `scaler.pkl` found — raw values will be passed to the model.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_manual, tab_batch = st.tabs(["🔢 Manual Prediction", "📄 Batch Prediction (CSV)"])


# ── Tab 1: Manual Prediction ─────────────────────────────────────────────────

with tab_manual:
    st.subheader("Enter Loan Details")
    st.markdown("Fill in the fields below, then click **Predict**.")

    # Group features into logical sections for readability
    GROUPS = {
        "Loan Details": [
            "loan_amnt", "funded_amnt_inv", "installment",
        ],
        "Income & Debt Ratios": [
            "annual_inc", "dti", "revol_util", "all_util",
        ],
        "Credit Profile": [
            "last_fico_range_high", "mo_sin_old_rev_tl_op",
        ],
        "Payment History": [
            "last_pymnt_d", "last_pymnt_amnt", "total_pymnt",
            "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
        ],
        "Outstanding Balance": [
            "out_prncp", "out_prncp_inv",
        ],
        "Risk Flags & Recovery": [
            "recoveries", "debt_settlement_flag", "hardship_flag",
        ],
    }

    input_values: dict[str, float] = {}

    for group_name, group_features in GROUPS.items():
        with st.expander(group_name, expanded=True):
            cols = st.columns(min(3, len(group_features)))
            for idx, feat in enumerate(group_features):
                label, min_v, max_v, default = FEATURE_META[feat]
                col = cols[idx % len(cols)]

                # Binary flags → selectbox; everything else → number_input
                if feat in ("debt_settlement_flag", "hardship_flag"):
                    choice = col.selectbox(label, options=[0, 1], index=int(default), key=feat)
                    input_values[feat] = float(choice)
                else:
                    input_values[feat] = col.number_input(
                        label,
                        min_value=float(min_v),
                        max_value=float(max_v),
                        value=float(default),
                        step=(1.0 if max_v <= 120 else 100.0),
                        key=feat,
                    )

    st.write("")
    if st.button("🔮 Predict", use_container_width=True, type="primary"):
        row_df = pd.DataFrame([input_values])
        X = preprocess(row_df, scaler)

        st.markdown("---")
        st.subheader("Prediction Result")

        if not ensemble_mode:
            labels, probs = predict(chosen_models[0][1], X)
            prob_row = probs[0] if probs is not None else None
            render_single_result(int(labels[0]), prob_row)
        else:
            labels, probs, per_model = predict_ensemble(chosen_models, X, method=voting_method)
            prob_row = probs[0] if probs is not None else None
            render_ensemble_result(int(labels[0]), prob_row, per_model[0])


# ── Tab 2: Batch Prediction ───────────────────────────────────────────────────

with tab_batch:
    st.subheader("Batch Prediction from CSV")
    st.markdown(
        "Upload a CSV whose columns match the 20 model features. "
        "The app will randomly sample rows and show predictions."
    )

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        try:
            full_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
            st.stop()

        # Validate that the required feature columns are present
        missing_cols = [c for c in FEATURES if c not in full_df.columns]
        if missing_cols:
            st.error(
                f"The uploaded CSV is missing the following required columns:\n\n"
                + ", ".join(f"`{c}`" for c in missing_cols)
            )
            st.stop()

        st.success(f"CSV loaded — {len(full_df):,} rows, {len(full_df.columns)} columns.")

        max_sample = min(len(full_df), 100)
        n_sample = st.slider(
            "Number of rows to sample",
            min_value=1,
            max_value=max_sample,
            value=min(20, max_sample),
        )

        if st.button("🎲 Sample & Predict", use_container_width=True, type="primary"):
            sample_df = full_df.sample(n=n_sample, random_state=None).reset_index(drop=True)

            X_batch = preprocess(sample_df, scaler)

            if not ensemble_mode:
                labels, probs = predict(chosen_models[0][1], X_batch)
                per_model_batch = None
            else:
                labels, probs, per_model_batch = predict_ensemble(chosen_models, X_batch, method=voting_method)

            # Build results table
            results = sample_df[FEATURES].copy()
            results["Prediction"] = [LABEL_MAP[int(l)] for l in labels]
            if probs is not None:
                results["P(Bad Credit)"]  = (probs[:, 0] * 100).round(1).astype(str) + "%"
                results["P(Good Credit)"] = (probs[:, 1] * 100).round(1).astype(str) + "%"

            if ensemble_mode:
                # Agreement column: how many of the chosen models agree with the consensus.
                results["Agreement"] = [
                    f"{sum(1 for _, m_label, _ in per_model_batch[row] if m_label == int(labels[row]))}/{len(chosen_models)}"
                    for row in range(len(labels))
                ]

            st.markdown("---")
            st.subheader(f"Results for {n_sample} sampled rows")

            # Summary metrics
            n_good = int((labels == 1).sum())
            n_bad  = int((labels == 0).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows",  n_sample)
            c2.metric("✅ Good Credit", n_good)
            c3.metric("❌ Bad Credit",  n_bad)

            # Prediction distribution bar chart
            dist_df = pd.DataFrame(
                {"Count": [n_bad, n_good]},
                index=["Bad Credit", "Good Credit"],
            )
            st.bar_chart(dist_df)

            # Full results table
            st.dataframe(results, use_container_width=True)

            # Download button
            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("👆 Upload a CSV file to get started.")
        st.markdown("**Expected columns (20 features):**")
        st.code("\n".join(FEATURES))
