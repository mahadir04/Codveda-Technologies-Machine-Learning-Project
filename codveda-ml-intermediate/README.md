# Codveda Internship - Level 2 (Intermediate) Project

Two tasks, both completed: **Decision Trees (Iris)** and **K-Means Clustering (customer segmentation, telecom churn data)**.

```
data/        iris.csv, churn-bigml-80.csv, churn-bigml-20.csv
task2_decision_tree_iris.py
task3_kmeans_customer_segmentation.py
outputs/     plots, metric tables, run logs
```

Run: `pip install pandas scikit-learn matplotlib seaborn`, then `python task2_decision_tree_iris.py` and `python task3_kmeans_customer_segmentation.py [K]`. Seeds are fixed (42), so results reproduce.

---

## Task 2 - Decision Trees for Classification

**Data:** Iris, 150 rows, 3 balanced classes, no missing values. Stratified 70/30 split (105 train / 45 test).

**Method**
1. Trained an unpruned `DecisionTreeClassifier`.
2. Pruned with cost-complexity pruning. Candidate `ccp_alpha` values came from the training data, were scored by 5-fold CV (never the test set), and the 1-SE rule picked the simplest tree within one standard error of the best.
3. Evaluated both trees on the held-out test set.

| Model | Depth | Leaves | Train acc | Test acc | Test F1 (macro) |
|---|---|---|---|---|---|
| Unpruned | 5 | 8 | 1.000 | 0.933 | 0.933 |
| Pruned (alpha = 0.0185) | 3 | 4 | 0.981 | 0.933 | 0.933 |

**Findings**
- Pruning cut the tree from 8 leaves to 4 with **no loss in test accuracy or F1**. The unpruned tree reached 100% train accuracy by carving out single-sample leaves (e.g. splits on `sepal_width` for 1-2 flowers), which is overfitting. The pruned tree keeps only the splits that generalize.
- Setosa is perfectly separated by one split (`petal_length <= 2.45`). All test errors are versicolor/virginica confusions (3 of 45 samples).
- Only `petal_length` (0.56) and `petal_width` (0.44) are used by the pruned tree; sepal features add nothing.
- Caveat: with a 45-sample test set, one sample is ~2.2 points of accuracy, so the unpruned/pruned tie should not be over-read. The benefit of pruning shown here is simplicity and robustness, not a higher score.

Outputs: `iris_tree_structure.png` (unpruned vs pruned), `iris_pruning_curve.png`, `iris_confusion_matrices.png`, `iris_feature_importance.png`, `iris_tree_metrics.csv`, `iris_classification_report.txt`.

---

## Task 3 - K-Means Clustering

**Data:** `churn-bigml-80.csv` + `churn-bigml-20.csv` pooled (they are an 80/20 split of one dataset; clustering is unsupervised so no split is needed) = 3,333 customers, no missing values or duplicates.

**Preprocessing**
- Dropped `State` and `Area code` (identifiers, not behavior) and all `*charge` columns (exactly linear in minutes, so they would double-weight usage).
- Encoded the two plan columns as 0/1.
- Standardized all 13 features with `StandardScaler`.
- `Churn` was **not** used for clustering; it is used only afterwards to interpret segments.

**Choosing k**

| k | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| Silhouette | 0.180 | **0.186** | 0.098 | 0.091 | 0.091 |

The inertia curve is smooth with no sharp elbow (a chord-distance heuristic points weakly at k=5), so I used silhouette as the tie-breaker. It drops by almost half after k=3, so **k = 3**. See `kmeans_elbow.png`.

**Segments (k = 3)**

| Cluster | Size | Defining trait | Churn rate |
|---|---|---|---|
| 0 | 323 (9.7%) | **All have an international plan** (+3.0 SD) | **42.4%** |
| 1 | 2,180 (65.4%) | No international plan, **no voicemail plan** | 13.9% |
| 2 | 830 (24.9%) | **All have a voicemail plan**, ~29 voicemail msgs on average | **5.3%** |

Overall churn in the pooled data is about 14.5%.

**Interpretation**
- Cluster 0, international-plan holders, churns at roughly 3x the overall rate. This is the clearest business signal: a high-risk segment worth investigating (pricing, call quality, or competitor offers on international calls).
- Cluster 2, voicemail-plan customers, is the stickiest at 5.3% churn.
- Cluster 1 is the large baseline segment with no add-on plans.

**Important caveat: the segments are driven by plan flags, not usage.** In the heatmap (`kmeans_cluster_heatmap.png`) the only features that separate the clusters are the plan columns and voicemail message count. Usage features (minutes, calls, service calls) are nearly identical across clusters (differences under 0.15 SD). I checked this by clustering on usage features alone: silhouette stays at about 0.07 for every k from 2 to 6, meaning there is **no natural cluster structure in the usage behavior of this dataset**. The k=3 silhouette of 0.186 is largely produced by the binary plan features. So these are meaningful plan-based segments, but should not be read as distinct usage personas. The 2D PCA plot shows the same thing: two clean blobs split on voicemail plan, and the international-plan customers sit inside them rather than forming their own region.

Outputs: `kmeans_elbow.png`, `kmeans_clusters_pca.png`, `kmeans_clusters_raw.png`, `kmeans_cluster_heatmap.png`, `kmeans_cluster_profiles.csv`, `kmeans_elbow_table.csv`, `churn_with_clusters.csv` (data with cluster labels).
