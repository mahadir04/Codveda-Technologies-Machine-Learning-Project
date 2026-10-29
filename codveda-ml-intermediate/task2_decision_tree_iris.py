"""Level 2 / Task 2: Decision Trees for Classification (Iris)

- Train a decision tree
- Visualize the tree
- Prune it (cost-complexity pruning, alpha chosen by cross-validation)
- Evaluate with accuracy and F1
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c4", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 12,
})

# ---------------------------------------------------------------- load
df = pd.read_csv(ROOT / "data" / "iris.csv")
print(f"Loaded iris: {df.shape}, missing values: {int(df.isna().sum().sum())}")
print(df["species"].value_counts().to_string(), "\n")

X = df.drop(columns="species")
y = df["species"]
feature_names = list(X.columns)
class_names = sorted(y.unique())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED)

# ---------------------------------------------------- 1. unpruned tree
full = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)

def evaluate(model, name):
    pred_tr, pred_te = model.predict(X_train), model.predict(X_test)
    res = {
        "model": name,
        "depth": model.get_depth(),
        "leaves": model.get_n_leaves(),
        "train_acc": accuracy_score(y_train, pred_tr),
        "test_acc": accuracy_score(y_test, pred_te),
        "test_f1_macro": f1_score(y_test, pred_te, average="macro"),
        "test_f1_weighted": f1_score(y_test, pred_te, average="weighted"),
    }
    return res

# ------------------------------------------------------- 2. pruning
# Cost-complexity pruning: candidate alphas from the training data,
# best alpha chosen by 5-fold CV (not by the test set).
path = full.cost_complexity_pruning_path(X_train, y_train)
alphas = np.unique(np.round(path.ccp_alphas, 6))
alphas = alphas[:-1]  # last alpha prunes the tree down to the root
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=SEED),
    {"ccp_alpha": alphas}, cv=5, scoring="accuracy").fit(X_train, y_train)

# Among alphas within one standard error of the best, prefer the largest
# (simplest tree) -- the "1-SE rule".
r = pd.DataFrame(grid.cv_results_)
best = r.loc[r["mean_test_score"].idxmax()]
thresh = best["mean_test_score"] - best["std_test_score"] / np.sqrt(5)
chosen_alpha = r.loc[r["mean_test_score"] >= thresh, "param_ccp_alpha"].max()
pruned = DecisionTreeClassifier(random_state=SEED, ccp_alpha=chosen_alpha
                                ).fit(X_train, y_train)
print(f"Best CV alpha: {best['param_ccp_alpha']:.4f} | "
      f"1-SE chosen alpha: {chosen_alpha:.4f}\n")

results = pd.DataFrame([evaluate(full, "Unpruned"),
                        evaluate(pruned, f"Pruned (ccp_alpha={chosen_alpha:.3f})")])
print(results.round(4).to_string(index=False), "\n")
results.round(4).to_csv(OUT / "iris_tree_metrics.csv", index=False)

print("Classification report (pruned tree, test set):")
report = classification_report(y_test, pruned.predict(X_test), digits=3)
print(report)
(OUT / "iris_classification_report.txt").write_text(report)

# -------------------------------------------------- 3. visualizations
fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                         gridspec_kw={"width_ratios": [1.6, 1]})
plot_tree(full, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, impurity=False, ax=axes[0], fontsize=8)
axes[0].set_title(f"Unpruned tree (depth {full.get_depth()}, "
                  f"{full.get_n_leaves()} leaves)")
plot_tree(pruned, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, impurity=False, ax=axes[1], fontsize=10)
axes[1].set_title(f"Pruned tree (depth {pruned.get_depth()}, "
                  f"{pruned.get_n_leaves()} leaves)")
fig.tight_layout()
fig.savefig(OUT / "iris_tree_structure.png", dpi=150)
plt.close(fig)

# pruning curve (CV accuracy vs alpha) + leaf count
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(r["param_ccp_alpha"].astype(float), r["mean_test_score"],
            yerr=r["std_test_score"], color=BLUE, marker="o", ms=5, lw=2,
            capsize=3, label="5-fold CV accuracy (±1 SD)")
ax.axvline(chosen_alpha, color=ORANGE, ls="--", lw=2,
           label=f"chosen alpha = {chosen_alpha:.3f}")
ax.set_xlabel("ccp_alpha (higher = more pruning)")
ax.set_ylabel("Accuracy")
ax.set_title("Pruning: cross-validated accuracy vs alpha")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "iris_pruning_curve.png", dpi=150)
plt.close(fig)

# confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, m, t in zip(axes, [full, pruned], ["Unpruned", "Pruned"]):
    ConfusionMatrixDisplay.from_estimator(m, X_test, y_test, ax=ax,
                                          cmap="Blues", colorbar=False)
    ax.set_title(f"{t} - test accuracy "
                 f"{accuracy_score(y_test, m.predict(X_test)):.3f}")
    ax.grid(False)
fig.tight_layout()
fig.savefig(OUT / "iris_confusion_matrices.png", dpi=150)
plt.close(fig)

# feature importance
imp = pd.Series(pruned.feature_importances_, index=feature_names).sort_values()
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.barh(imp.index, imp.values, color=BLUE, height=0.55)
ax.set_xlabel("Importance (pruned tree)")
ax.set_title("Which features drive the splits?")
fig.tight_layout()
fig.savefig(OUT / "iris_feature_importance.png", dpi=150)
plt.close(fig)
print("Feature importance:\n", imp.sort_values(ascending=False).round(3).to_string())
