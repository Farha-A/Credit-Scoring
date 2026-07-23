"""Data cleaning and preprocessing pipeline for the LendingClub credit-scoring dataset.

Typical usage::

    from src.data.preprocessing import run_preprocessing_pipeline
    df = run_preprocessing_pipeline("accepted.csv")
    df.to_csv("df_cleaned.csv", index=False)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Columns that are high-null but kept because they carry signal
_IMPORTANT_HIGH_NULL_COLS = {"mths_since_last_delinq", "all_util"}

# Columns determined to be uninformative for prediction (IDs, URLs, free text, etc.)
_USELESS_COLS = [
    "acceptD",
    "application_type",
    "creditPullD",
    "desc",
    "emp_title",
    "expD",
    "id",
    "listD",
    "mthsSinceMostRecentInq",
    "reviewStatusD",
    "title",
    "url",
    "zip_code",
    "sec_app_inq_last_6mths",
]

# The 20 features selected by XGBoost feature importance in EDA (plus target)
SELECTED_FEATURES = [
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
    "loan_status",
]


def load_dataframe(file_name: str) -> pd.DataFrame:
    """Load the raw dataset CSV into a DataFrame.

    Args:
        file_name: Path to the CSV file.

    Returns:
        Raw DataFrame with all original columns.
    """
    print(f"Loading data from {file_name}")
    return pd.read_csv(file_name, low_memory=False)


def drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.20) -> None:
    """Drop columns whose null ratio exceeds *threshold*, skipping important ones.

    Modifies *df* in-place.

    Args:
        df: The dataset DataFrame.
        threshold: Fraction of rows that may be null before a column is dropped
            (default 0.20 → 20 %).
    """
    print("Dropping high-null columns")
    n_rows = df.shape[0]
    cols_to_drop = [
        col
        for col in df.columns
        if col not in _IMPORTANT_HIGH_NULL_COLS
        and df[col].isna().sum() >= threshold * n_rows
    ]
    df.drop(columns=cols_to_drop, inplace=True)


def fix_delinq_nulls(df: pd.DataFrame) -> None:
    """Set ``mths_since_last_delinq`` to 0 where ``delinq_amnt`` is 0.

    When there is no delinquency amount the corresponding months-since column
    should be 0, not NaN.  Modifies *df* in-place.

    Args:
        df: The dataset DataFrame.
    """
    print("Fixing mths_since_last_delinq nulls using delinq_amnt")
    if "mths_since_last_delinq" not in df.columns or "delinq_amnt" not in df.columns:
        return

    mask = (df["delinq_amnt"] == 0) & df["mths_since_last_delinq"].isna()
    df.loc[mask, "mths_since_last_delinq"] = 0


def drop_rows_with_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every row that still contains at least one null value.

    Because the dataset has >2 million rows, dropping the relatively small
    proportion of remaining null rows is safe.

    Args:
        df: The dataset DataFrame.

    Returns:
        DataFrame with no null values.
    """
    print("Dropping rows with remaining null values")
    return df.dropna().reset_index(drop=True)


def drop_useless_columns(df: pd.DataFrame) -> None:
    """Remove columns that carry no predictive value (IDs, URLs, free text).

    Modifies *df* in-place.  Silently skips columns that are already absent.

    Args:
        df: The dataset DataFrame.
    """
    print("Dropping useless columns")
    cols_present = [c for c in _USELESS_COLS if c in df.columns]
    df.drop(columns=cols_present, inplace=True)


def encode_target(df: pd.DataFrame) -> None:
    """Binary-encode the ``loan_status`` target column in-place.

    Good credit (1): Current, Fully Paid, In Grace Period, Late (16-30 days).
    Bad credit (0):  Charged Off, Late (31-120 days), Default.

    Args:
        df: The dataset DataFrame.
    """
    print("Encoding loan_status target")
    good = {"Current", "Fully Paid", "In Grace Period", "Late (16-30 days)"}
    bad = {"Charged Off", "Late (31-120 days)", "Default"}

    df["loan_status"] = df["loan_status"].apply(
        lambda s: "1" if s in good else ("0" if s in bad else s)
    )


def balance_classes(df: pd.DataFrame, random_state: int = 737) -> pd.DataFrame:
    """Under-sample the majority class to match the minority class count.

    Randomly selects rows labelled "1" so that both classes have the same
    number of samples, then shuffles the result.

    Args:
        df: DataFrame that already has a binary ``loan_status`` column.
        random_state: Seed for reproducibility.

    Returns:
        Balanced and shuffled DataFrame.
    """
    print("Balancing classes via under-sampling")
    n_bad = (df["loan_status"] == "0").sum()
    subset_good = df[df["loan_status"] == "1"].sample(n=n_bad, random_state=random_state)
    subset_bad = df[df["loan_status"] == "0"]

    balanced = pd.concat([subset_good, subset_bad]).sample(frac=1, random_state=random_state)
    balanced.reset_index(inplace=True)
    return balanced


def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Remove rows whose z-score on any numeric column exceeds *z_threshold*.

    Args:
        df: The dataset DataFrame.
        z_threshold: Maximum allowed absolute z-score (default 3.0).

    Returns:
        DataFrame with outlier rows removed.
    """
    print("Removing outliers via z-score")
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != "loan_status"
    ]
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    outlier_mask = (z_scores > z_threshold).any(axis=1)
    return df[~outlier_mask].reset_index(drop=True)


def standardize_numeric(df: pd.DataFrame) -> None:
    """Standardize every numeric column (except ``loan_status``) to zero mean / unit variance.

    Modifies *df* in-place using scikit-learn's :class:`~sklearn.preprocessing.StandardScaler`.

    Args:
        df: The dataset DataFrame.
    """
    print("Standardizing numeric columns")
    scaler = StandardScaler()
    for col in df.columns:
        if df[col].dtype != object and col != "loan_status":
            values = df[col].values.reshape(-1, 1)
            df[col] = scaler.fit_transform(values)


def label_encode_categoricals(df: pd.DataFrame) -> None:
    """Label-encode all object-dtype columns.

    Modifies *df* in-place.

    Args:
        df: The dataset DataFrame.
    """
    print("Label-encoding categorical columns")
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = le.fit_transform(df[col])


def run_preprocessing_pipeline(csv_file: str) -> pd.DataFrame:
    """Execute the full preprocessing pipeline and return a model-ready DataFrame.

    Steps:
        1. Load raw CSV
        2. Drop high-null columns (keep important ones)
        3. Fix delinquency null values
        4. Drop remaining null rows
        5. Drop useless columns
        6. Binary-encode the target
        7. Balance classes
        8. Remove outliers (z-score)
        9. Standardize numeric features
        10. Label-encode categorical features
        11. Retain only the 20 selected features + target

    Args:
        csv_file: Path to the raw accepted-loans CSV.

    Returns:
        Cleaned and feature-selected DataFrame ready for model training.
    """
    df = load_dataframe(csv_file)
    drop_high_null_columns(df)
    fix_delinq_nulls(df)
    df = drop_rows_with_nulls(df)
    drop_useless_columns(df)
    encode_target(df)
    df = balance_classes(df)
    df = remove_outliers(df)
    standardize_numeric(df)
    label_encode_categoricals(df)

    # Remove the helper index column added by balance_classes → reset_index
    if "index" in df.columns:
        df.drop(columns=["index"], inplace=True)

    # Keep only the features chosen during EDA feature selection
    available = [c for c in SELECTED_FEATURES if c in df.columns]
    return df[available]
