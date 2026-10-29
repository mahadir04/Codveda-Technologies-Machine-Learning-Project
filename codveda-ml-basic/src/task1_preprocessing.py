"""Level 1 - Task 1: Data Preprocessing for Machine Learning.

Dataset: Telecom churn (churn-bigml-80 + churn-bigml-20 combined, 3333 rows).
Steps: missing values -> encode categoricals -> split -> standardize -> save.
Scaler is fit on the TRAIN split only to avoid data leakage.
"""
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load ---------------------------------------------------------------
df = pd.concat(
    [pd.read_csv(DATA / "churn-bigml-80.csv"), pd.read_csv(DATA / "churn-bigml-20.csv")],
    ignore_index=True,
)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

# 2. Missing data ---------------------------------------------------------
missing = df.isna().sum()
print(f"Missing values total: {missing.sum()}")
df = df.drop_duplicates()
num_cols = df.select_dtypes("number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.drop("Churn").tolist()
# Imputers are kept in the pipeline so it also works on dirty data
# (median for numeric, most frequent for categorical). No-op on this dataset.
orig_dtypes = df[num_cols].dtypes.to_dict()  # imputer returns floats; restore ints
df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
df = df.astype(orig_dtypes)
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

# 3. Encode categoricals --------------------------------------------------
y = df["Churn"].astype(int)
X = df.drop(columns="Churn")
for c in ["International plan", "Voice mail plan"]:  # binary -> label encoding
    X[c] = (X[c] == "Yes").astype(int)
X["Area code"] = X["Area code"].astype(str)  # an ID, not a quantity
X = pd.get_dummies(X, columns=["State", "Area code"], dtype=int)  # one-hot
print(f"After encoding: {X.shape[1]} features")

# 4. Split (stratified: churn is ~14.5% of rows) --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Standardize numeric features (fit on train only) ---------------------
scale_cols = [c for c in num_cols if c != "Area code"]
scaler = StandardScaler().fit(X_train[scale_cols])
X_train, X_test = X_train.copy(), X_test.copy()
X_train[scale_cols] = scaler.transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

# 6. Save -----------------------------------------------------------------
X_train.to_csv(OUT / "X_train.csv", index=False)
X_test.to_csv(OUT / "X_test.csv", index=False)
y_train.to_csv(OUT / "y_train.csv", index=False)
y_test.to_csv(OUT / "y_test.csv", index=False)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Churn rate  train={y_train.mean():.3f}  test={y_test.mean():.3f}")
print("Train scaled-column means (~0):", X_train[scale_cols].mean().abs().max().round(6))
print("Train scaled-column stds  (~1):", X_train[scale_cols].std().round(3).unique())
print(f"Saved to {OUT}")
