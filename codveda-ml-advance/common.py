"""Shared data loading / preprocessing for the churn project."""
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TARGET = "Churn"
SEED = 42


def load_raw():
    train = pd.read_csv(DATA / "churn-bigml-80.csv")
    test = pd.read_csv(DATA / "churn-bigml-20.csv")
    return train, test


def _features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Binary yes/no plans -> 0/1
    for col in ["International plan", "Voice mail plan"]:
        df[col] = (df[col] == "Yes").astype(int)
    # Area code is categorical, not numeric
    df["Area code"] = df["Area code"].astype(str)
    # Charges are a deterministic function of minutes (perfectly collinear) -> drop
    df = df.drop(columns=[c for c in df.columns if c.endswith("charge")])
    return df


def get_data(scale: bool = False):
    """Return X_train, X_test, y_train, y_test (aligned one-hot columns).

    The official split is used: the 80% file for training, the 20% file for testing.
    If scale=True, numeric columns are standardised using train statistics only.
    """
    train, test = load_raw()
    y_train = train[TARGET].astype(int).values
    y_test = test[TARGET].astype(int).values

    X_train = _features(train.drop(columns=[TARGET]))
    X_test = _features(test.drop(columns=[TARGET]))

    X_train = pd.get_dummies(X_train, columns=["State", "Area code"], dtype=int)
    X_test = pd.get_dummies(X_test, columns=["State", "Area code"], dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    if scale:
        num_cols = [c for c in X_train.columns
                    if not (c.startswith("State_") or c.startswith("Area code_"))
                    and c not in ("International plan", "Voice mail plan")]
        scaler = StandardScaler().fit(X_train[num_cols])
        X_train[num_cols] = scaler.transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test
