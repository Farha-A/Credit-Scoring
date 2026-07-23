# Credit Scoring

Binary classification task on LendingClub loan data: predict whether a borrower has **good** or **bad** credit.

---

## Project Structure

```
Credit-Scoring/
├── app.py                        # Streamlit web application
├── model_weights/                # Drop pre-trained .pkl files here
│   └── (scaler.pkl, model.pkl, ...)
├── src/
│   ├── data/
│   │   ├── downloader.py         # Google Drive download & .gz decompression
│   │   └── preprocessing.py      # Full 10-step preprocessing pipeline
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── decision_tree.py
│   │   ├── neural_network.py     # 2-layer Keras NN
│   │   ├── svm.py                # XGBoost feature selection → SVM pipeline
│   │   └── ensemble.py           # TabNet + XGBoost + RF stacking ensemble
│   ├── visualization/
│   │   └── plots.py              # Shared plotting helpers
│   └── interpretability/
│       └── explainability.py     # SHAP, LIME, PDP/ICE, PFI wrappers
└── notebooks/
    ├── EDA.ipynb
    ├── Data_Preprocessing.ipynb
    ├── Arwa Logistic_Regression.ipynb
    ├── Arwa Random_Forest.ipynb
    ├── Arwa_XGBoost_SVM.ipynb
    ├── Farha_2L_NN.ipynb
    ├── Farha_DT.ipynb
    ├── Nour_Ensemble_Model.ipynb
    └── ...
```

---

## Models

| Model | File |
|-------|------|
| Logistic Regression | `src/models/logistic_regression.py` |
| Random Forest | `src/models/random_forest.py` |
| Decision Tree | `src/models/decision_tree.py` |
| Neural Network (Keras) | `src/models/neural_network.py` |
| XGBoost → SVM Pipeline | `src/models/svm.py` |
| TabNet + XGBoost + RF Stacking | `src/models/ensemble.py` |

All models are trained on **20 features** selected via XGBoost feature importance. Inputs are preprocessed with `StandardScaler`; categorical columns are label-encoded.

---

## Interpretability

Located in `src/interpretability/explainability.py`:

- **SHAP** — TreeExplainer (tree models), linear Explainer (LR), KernelExplainer (black-box)
- **LIME** — per-instance tabular explanations
- **PDP / ICE** — partial dependence and individual conditional expectation plots
- **PFI** — permutation feature importance

---

## Streamlit App

The app loads pre-trained models from `model_weights/` — it never trains models itself.

### Setup

1. Place trained `.pkl` files in `model_weights/` (e.g. `random_forest.pkl`, `logistic_regression.pkl`).
2. Optionally place `scaler.pkl` there; inputs will be scaled automatically when found.
3. Install dependencies and run:

```bash
pip install streamlit pandas numpy scikit-learn
streamlit run app.py
```

### Features

**Tab 1 — Manual Prediction**
Fill in the 20 loan features (grouped into Loan Details, Income & Debt Ratios, Credit Profile, Payment History, Outstanding Balance, Risk Flags & Recovery) and click **Predict** to see the credit label and probability breakdown.

**Tab 2 — Batch Prediction (CSV)**
Upload a CSV with the 20 feature columns, choose how many rows to randomly sample via a slider, and click **Sample & Predict**. Results include a summary (total / good / bad), distribution chart, full results table, and a download button.

---

## Preprocessing Pipeline

Defined in `src/data/preprocessing.py` — `run_preprocessing_pipeline(csv_file)`:

1. Load CSV
2. Drop columns with >20% nulls (excluding clinically important ones)
3. Fix delinquency nulls
4. Drop remaining rows with nulls
5. Drop useless identifier columns
6. Binary-encode target (`loan_status`)
7. Under-sample majority class for balance
8. Z-score outlier removal (threshold = 3.0)
9. StandardScaler on numeric columns
10. LabelEncoder on categorical columns

---

## Dataset

LendingClub loan dataset. Download via `src/data/downloader.py` (requires a Google Drive folder ID) or place the raw `.gz` file manually and decompress with `decompress_gz()`.
