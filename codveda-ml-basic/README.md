# Codveda ML Internship — Level 1 (Basic)

Two tasks, Python + pandas + scikit-learn.

| Task | Script | Dataset |
|---|---|---|
| Task 1: Data Preprocessing for ML | `src/task1_preprocessing.py` | Telecom churn (80 + 20 files combined, 3333 rows) |
| Task 3: K-Nearest Neighbors Classifier | `src/task3_knn.py` | Iris (3-class) and preprocessed churn (binary) |

## Run

```bash
pip install -r requirements.txt
python src/task1_preprocessing.py   # writes data/processed/*.csv
python src/task3_knn.py             # writes outputs/*
```

Run Task 1 first: Task 3 reads its output for the churn experiment.

## Task 1 — Preprocessing

- **Missing data:** checked; none found in the churn data. Median (numeric) / most-frequent (categorical) imputers and duplicate dropping are in the pipeline so it handles dirty data too.
- **Encoding:** `International plan`, `Voice mail plan` → 0/1 (label); `State`, `Area code` → one-hot (area code is an ID, so it is not scaled or treated as a number). 19 columns → 71 features.
- **Split:** 80/20, stratified on `Churn` (14.5% in both train and test), `random_state=42`.
- **Scaling:** `StandardScaler` fit on the train split only, then applied to test (no leakage). Train mean ≈ 0, std = 1.

## Task 3 — KNN

K ∈ {1, 3, 5, 7, 9, 11, 15, 21, 31}. Best K is chosen by 5-fold CV on the training set; the test set is only used for reporting.

**Iris** (30 test samples, features standardized in a pipeline) — best K = 5, test accuracy 0.933, macro F1 0.93. Test accuracy sits between 0.867 and 0.967 across K; with only 30 test rows, one sample is 3.3 points, so differences between K = 1…21 are noise. Large K (31) clearly underfits. Versicolor/virginica are the only confused classes.

**Churn** (667 test samples) — best K = 5, test accuracy 0.891, but this is misleading:

- Always predicting "no churn" scores 0.855, so KNN adds little accuracy.
- Churn **recall is only 0.26** (25 of 97 churners found; precision 0.96). Recall trends down as K grows (0.36 at K=1 → 0.12 at K=31, not strictly monotonic) because the majority class dominates the neighbourhood.
- Takeaway: accuracy alone hides the failure; for churn, recall/F1 on the positive class matter. Class weighting, resampling, or a different model would be the next step (out of scope for Level 1).

**Known limitation:** the churn features are scaled once on the full training set in Task 1, so the 5-fold CV used to pick K sees slightly leaked scaling statistics (tiny effect; the test set is unaffected). Iris avoids this by scaling inside a Pipeline.

## Outputs (`outputs/`)

- `iris_k_comparison.csv`, `churn_k_comparison.csv` — metrics for every K
- `*_accuracy_vs_k.png`, `*_confusion_matrix.png`
- `*_classification_report.txt`, `knn_run_log.txt`

## Notes

- Sentiment, stock and house datasets were provided but are not needed for these two tasks (they have no missing values / no suitable labels for KNN as-is), so they are not used.
