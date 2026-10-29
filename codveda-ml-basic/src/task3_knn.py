"""Level 1 - Task 3: K-Nearest Neighbors classifier.

Two labeled datasets:
  * Iris  (3-class, balanced)           - raw CSV, scaled in a Pipeline
  * Churn (binary, 14.5% positives)     - output of task1_preprocessing.py
For each: train KNN for several K, compare accuracy / precision / recall / F1
on the held-out test set, plot confusion matrices. Best K is picked by 5-fold
cross-validation on TRAIN only (the test set is never used for selection).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, precision_recall_fscore_support)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31]


def run(name, X_train, X_test, y_train, y_test, labels, make_model, pos_label=None):
    rows = []
    for k in K_VALUES:
        model = make_model(k)
        cv = cross_val_score(model, X_train, y_train, cv=5).mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        avg = "binary" if pos_label is not None else "macro"
        p, r, f, _ = precision_recall_fscore_support(
            y_test, pred, average=avg, pos_label=pos_label if pos_label is not None else 1,
            zero_division=0)
        rows.append(dict(K=k, cv_accuracy=cv, test_accuracy=accuracy_score(y_test, pred),
                         precision=p, recall=r, f1=f))
    res = pd.DataFrame(rows).round(4)
    res.to_csv(OUT / f"{name}_k_comparison.csv", index=False)
    best_k = int(res.loc[res.cv_accuracy.idxmax(), "K"])
    print(f"\n===== {name.upper()} =====")
    print(res.to_string(index=False))
    print(f"Best K by 5-fold CV on train: {best_k}")

    # accuracy vs K plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res.K, res.cv_accuracy, "o-", label="CV accuracy (train)")
    ax.plot(res.K, res.test_accuracy, "s--", label="Test accuracy")
    ax.axvline(best_k, color="grey", ls=":", label=f"best K={best_k}")
    ax.set(xlabel="K", ylabel="Accuracy", title=f"{name}: accuracy vs K")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / f"{name}_accuracy_vs_k.png", dpi=150)
    plt.close(fig)

    # final model + confusion matrix
    model = make_model(best_k).fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\nTest report (K={best_k}):\n", classification_report(
        y_test, pred, target_names=labels, zero_division=0))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=labels,
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{name}: confusion matrix (K={best_k})")
    fig.tight_layout(); fig.savefig(OUT / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    (OUT / f"{name}_classification_report.txt").write_text(
        classification_report(y_test, pred, target_names=labels, zero_division=0))


# ---- Iris -----------------------------------------------------------------
iris = pd.read_csv(DATA / "iris.csv")
Xi, yi = iris.drop(columns="species"), iris["species"]
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.2, random_state=42, stratify=yi)
run("iris", Xi_tr, Xi_te, yi_tr, yi_te, sorted(yi.unique()),
    lambda k: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)))

# ---- Churn (preprocessed by Task 1) ----------------------------------------
P = DATA / "processed"
Xc_tr, Xc_te = pd.read_csv(P / "X_train.csv"), pd.read_csv(P / "X_test.csv")
yc_tr, yc_te = pd.read_csv(P / "y_train.csv").squeeze(), pd.read_csv(P / "y_test.csv").squeeze()
print(f"\nChurn majority-class baseline accuracy: {1 - yc_te.mean():.4f}")
run("churn", Xc_tr, Xc_te, yc_tr, yc_te, ["No churn", "Churn"],
    lambda k: KNeighborsClassifier(n_neighbors=k), pos_label=1)
