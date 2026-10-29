"""Level 2 / Task 3: K-Means Clustering (customer segmentation, telecom churn data)

- Load + preprocess (drop redundant/identifier columns, scale)
- Elbow method (+ silhouette as a cross-check) to choose k
- Fit K-Means, visualize clusters in 2D (PCA and two raw features)
- Interpret clusters (profile table; churn is NOT used for clustering,
  only to interpret the segments afterwards)

Usage: python task3_kmeans_customer_segmentation.py [K]
       (K omitted -> chosen automatically from the elbow/silhouette curves)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42

# categorical palette, fixed order (blue, orange, aqua, magenta)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c4", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 12,
})

# ------------------------------------------------------------ load
# The two churn files are an 80/20 split of the same dataset. Clustering is
# unsupervised, so there is no train/test need: pool them (3,333 customers).
df = pd.concat([pd.read_csv(ROOT / "data" / "churn-bigml-80.csv"),
                pd.read_csv(ROOT / "data" / "churn-bigml-20.csv")],
               ignore_index=True)
print(f"Loaded: {df.shape}, missing values: {int(df.isna().sum().sum())}, "
      f"duplicates: {int(df.duplicated().sum())}")

# ------------------------------------------------------ preprocess
# Drop: State / Area code (identifiers, not behaviour), *charge columns
# (perfectly linear in minutes -> would double-weight usage), Churn (kept
# aside for interpretation only).
df["International plan"] = (df["International plan"] == "Yes").astype(int)
df["Voice mail plan"] = (df["Voice mail plan"] == "Yes").astype(int)
features = [
    "Account length", "International plan", "Voice mail plan",
    "Number vmail messages",
    "Total day minutes", "Total day calls",
    "Total eve minutes", "Total eve calls",
    "Total night minutes", "Total night calls",
    "Total intl minutes", "Total intl calls",
    "Customer service calls",
]
X = df[features]
X_scaled = StandardScaler().fit_transform(X)

# ---------------------------------------------------- elbow method
ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_, sample_size=2000,
                                random_state=SEED))
curve = pd.DataFrame({"k": list(ks), "inertia": inertia, "silhouette": sil})
print(curve.round(3).to_string(index=False))
curve.round(4).to_csv(OUT / "kmeans_elbow_table.csv", index=False)

# Elbow = point of maximum distance from the line joining first and last point
x = np.array(list(ks), float)
y = np.array(inertia)
p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
v, w = p2 - p1, np.c_[x, y] - p1
d = np.abs(v[0] * w[:, 1] - v[1] * w[:, 0]) / np.linalg.norm(v)
k_elbow = int(x[np.argmax(d)])
# The inertia curve here is smooth (no sharp elbow), so the chord heuristic is
# only a soft hint. Silhouette drops sharply after k=3, so it breaks the tie.
k_sil = int(list(ks)[int(np.argmax(sil))])
K = int(sys.argv[1]) if len(sys.argv) > 1 else k_sil
print(f"\nElbow (max distance to chord) at k={k_elbow}; "
      f"best silhouette at k={k_sil}; using K={K}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(ks), inertia, color=PALETTE[0], marker="o", lw=2)
axes[0].axvline(K, color=PALETTE[1], ls="--", lw=2, label=f"chosen k = {K}")
axes[0].set_xlabel("Number of clusters k")
axes[0].set_ylabel("Inertia (within-cluster SSE)")
axes[0].set_title("Elbow method")
axes[0].legend(frameon=False)
axes[1].plot(list(ks), sil, color=PALETTE[2], marker="o", lw=2)
axes[1].axvline(K, color=PALETTE[1], ls="--", lw=2)
axes[1].set_xlabel("Number of clusters k")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette cross-check")
fig.tight_layout()
fig.savefig(OUT / "kmeans_elbow.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------- fit
km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(X_scaled)
df["cluster"] = km.labels_
print(f"Silhouette (k={K}): "
      f"{silhouette_score(X_scaled, km.labels_, sample_size=2000, random_state=SEED):.3f}")

# --------------------------------------------------- visualize 2D
pca = PCA(n_components=2, random_state=SEED)
Z = pca.fit_transform(X_scaled)
centers_z = pca.transform(km.cluster_centers_)
ev = pca.explained_variance_ratio_

# size labels
sizes = df["cluster"].value_counts().sort_index()
labels = {c: f"Cluster {c} (n={sizes[c]})" for c in sizes.index}

fig, ax = plt.subplots(figsize=(8, 6))
for c in range(K):
    m = df["cluster"] == c
    ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.6, color=PALETTE[c],
               edgecolor="none", label=labels[c])
    ax.scatter(*centers_z[c], s=170, color=PALETTE[c], edgecolor="white",
               linewidth=2, zorder=5)
    ax.annotate(str(c), centers_z[c], ha="center", va="center", color="white",
                fontsize=8, fontweight="bold", zorder=6)
ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)")
ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
ax.set_title(f"K-Means segments (k={K}) projected to 2D with PCA")
ax.legend(frameon=False, loc="lower center", markerscale=1.5)
fig.tight_layout()
fig.savefig(OUT / "kmeans_clusters_pca.png", dpi=150)
plt.close(fig)

# raw-feature scatter (interpretable axes)
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x="Total day minutes", y="Customer service calls",
                hue="cluster", palette=PALETTE[:K], s=18, alpha=0.6,
                edgecolor="none", ax=ax)
ax.set_title("Segments on two raw features")
h, _ = ax.get_legend_handles_labels()
ax.legend(h, [labels[c] for c in range(K)], frameon=False)
fig.tight_layout()
fig.savefig(OUT / "kmeans_clusters_raw.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------ interpret
profile_cols = features + ["Churn"]
df["Churn"] = df["Churn"].astype(int)
profile = df.groupby("cluster")[profile_cols].mean()
profile.insert(0, "size", sizes)
profile.insert(1, "share_%", (sizes / len(df) * 100).round(1))
profile["Churn"] = profile["Churn"] * 100
profile = profile.rename(columns={"Churn": "churn_rate_%"})
pd.set_option("display.width", 250, "display.max_columns", 30)
print("\nCluster profiles (feature means):")
print(profile.round(2).T.to_string())
profile.round(3).to_csv(OUT / "kmeans_cluster_profiles.csv")

# heatmap of z-scored cluster means: shows what makes each segment distinctive
z = (profile[features] - X.mean()) / X.std()
fig, ax = plt.subplots(figsize=(9, 0.5 * len(features) + 1.5))
sns.heatmap(z.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-2.5,
            vmax=2.5, linewidths=2, linecolor="white", cbar_kws={"label":
            "SD from overall mean"}, ax=ax)
ax.set_xlabel("Cluster")
ax.set_title("What makes each segment different")
fig.tight_layout()
fig.savefig(OUT / "kmeans_cluster_heatmap.png", dpi=150)
plt.close(fig)

df.to_csv(OUT / "churn_with_clusters.csv", index=False)
