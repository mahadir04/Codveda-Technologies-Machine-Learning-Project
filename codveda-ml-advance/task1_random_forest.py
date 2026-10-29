"""Level 3 - Task 1: Random Forest classifier for customer churn."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, accuracy_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score)

from common import OUT, SEED, get_data

X_train, X_test, y_train, y_test = get_data(scale=False)  
print(f"Train {X_train.shape}, test {X_test.shape}, churn rate train={y_train.mean():.3f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ---- Baseline -------------------------------------------------------------
base = RandomForestClassifier(random_state=SEED, n_jobs=-1)
base_cv = cross_val_score(base, X_train, y_train, cv=cv, scoring="f1")
print(f"Baseline CV F1: {base_cv.mean():.4f} +/- {base_cv.std():.4f}")

# ---- Hyperparameter tuning (optimise F1: churn is imbalanced) --------------
grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.3],
    "class_weight": [None, "balanced_subsample"],
}
gs = GridSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=-1), grid,
                  cv=cv, scoring="f1", n_jobs=-1, refit=True)
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
print(f"Best CV F1: {gs.best_score_:.4f}")
best = gs.best_estimator_

# Effect of number of trees / depth (for the report)
res = pd.DataFrame(gs.cv_results_)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
res.groupby("param_n_estimators")["mean_test_score"].max().plot(marker="o", ax=axes[0])
axes[0].set(title="Best CV F1 vs. n_estimators", xlabel="n_estimators", ylabel="CV F1")
res["depth"] = res["param_max_depth"].astype(str)
res.groupby("depth")["mean_test_score"].max().reindex(["10", "20", "None"]).plot(marker="o", ax=axes[1])
axes[1].set(title="Best CV F1 vs. max_depth", xlabel="max_depth", ylabel="CV F1")
plt.tight_layout(); plt.savefig(OUT / "rf_tuning.png", dpi=150); plt.close()

# ---- Cross-validated metrics on the training set ---------------------------
cv_metrics = {m: cross_val_score(best, X_train, y_train, cv=cv, scoring=m)
              for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
cv_summary = {m: (v.mean(), v.std()) for m, v in cv_metrics.items()}
for m, (mu, sd) in cv_summary.items():
    print(f"CV {m:9s}: {mu:.4f} +/- {sd:.4f}")

# ---- Held-out test evaluation ----------------------------------------------
proba = best.predict_proba(X_test)[:, 1]
pred_default = (proba >= 0.5).astype(int)
print("\nTest set (threshold 0.5)")
print(classification_report(y_test, pred_default, target_names=["Stayed", "Churned"]))

# Threshold tuned on training OOF-free CV would be ideal; report a sweep for the analyst
thresholds = np.linspace(0.1, 0.9, 33)
f1s = [f1_score(y_test, (proba >= t).astype(int)) for t in thresholds]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thresholds, f1s); ax.axvline(0.5, ls="--", c="grey")
ax.set(title="Test F1 vs. decision threshold", xlabel="threshold", ylabel="F1")
plt.tight_layout(); plt.savefig(OUT / "rf_threshold_sweep.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, pred_default,
                                        display_labels=["Stayed", "Churned"],
                                        cmap="Blues", ax=ax)
ax.set_title("Random Forest - confusion matrix (test)")
plt.tight_layout(); plt.savefig(OUT / "rf_confusion_matrix.png", dpi=150); plt.close()

# ---- Feature importance -----------------------------------------------------
imp = pd.Series(best.feature_importances_, index=X_train.columns).sort_values(ascending=False)
top = imp.head(15)[::-1]
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(top.index, top.values, color="#3b6ea5")
ax.set(title="Random Forest - top 15 features (impurity importance)", xlabel="importance")
plt.tight_layout(); plt.savefig(OUT / "rf_feature_importance.png", dpi=150); plt.close()
imp.head(15).to_csv(OUT / "rf_feature_importance.csv", header=["importance"])
print("\nTop 10 features:\n", imp.head(10).round(4))

# Permutation importance on the test set (less biased than impurity importance)
from sklearn.inspection import permutation_importance
pi = permutation_importance(best, X_test, y_test, scoring="f1", n_repeats=10,
                            random_state=SEED, n_jobs=-1)
pi_s = pd.Series(pi.importances_mean, index=X_train.columns).sort_values(ascending=False)
print("\nTop 5 permutation importances (test F1 drop):\n", pi_s.head(5).round(4))

results = {
    "best_params": gs.best_params_,
    "baseline_cv_f1": float(base_cv.mean()),
    "tuned_cv": {m: {"mean": float(a), "std": float(b)} for m, (a, b) in cv_summary.items()},
    "test": {
        "accuracy": accuracy_score(y_test, pred_default),
        "precision": precision_score(y_test, pred_default),
        "recall": recall_score(y_test, pred_default),
        "f1": f1_score(y_test, pred_default),
        "roc_auc": roc_auc_score(y_test, proba),
    },
    "top_features": imp.head(10).round(4).to_dict(),
    "top_permutation": pi_s.head(5).round(4).to_dict(),
}
(OUT / "rf_results.json").write_text(json.dumps(results, indent=2, default=float))
