# Customer Churn Prediction — Level 3 (Advanced) Tasks

Covers **Level 3 Task 1 (Random Forest)** and **Level 3 Task 3 (Neural Network with TensorFlow/Keras)**
on the BigML telecom churn dataset (`churn-bigml-80.csv` for training, `churn-bigml-20.csv` for testing).

## Structure

```
common.py                 shared loading + preprocessing
task1_random_forest.py    Level 3 Task 1
task3_neural_network.py   Level 3 Task 3
data/                     churn-bigml-80.csv (2,666 rows), churn-bigml-20.csv (667 rows)
outputs/                  plots, metrics JSON, saved Keras model
requirements.txt
```

Run: `pip install -r requirements.txt && python task1_random_forest.py && python task3_neural_network.py`

## Preprocessing (`common.py`)

- `International plan`, `Voice mail plan` → 0/1.
- `State` and `Area code` → one-hot (area code treated as categorical, not numeric).
- Dropped the four `Total * charge` columns: they are exact linear functions of the minutes columns.
- Official 80/20 files are used as the train/test split; the test set is never touched during tuning.
- Standardisation (train statistics only) is applied for the neural network; not needed for the forest.
- Target is imbalanced (14.6% churn), so F1 / recall are reported, not only accuracy.

## Task 1 — Random Forest

- 5-fold stratified CV grid search over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `class_weight`, optimising F1.
- Best: `n_estimators=100, max_depth=None, min_samples_leaf=4, max_features=0.3, class_weight=balanced_subsample`.

| Metric | 5-fold CV (train) | Test |
|---|---|---|
| Accuracy | 0.948 ± 0.008 | 0.949 |
| Precision (churn) | 0.847 ± 0.054 | 0.821 |
| Recall (churn) | 0.791 ± 0.018 | 0.821 |
| F1 (churn) | 0.817 ± 0.023 | 0.821 |
| ROC-AUC | 0.909 ± 0.010 | 0.919 |

Default (untuned) forest had CV F1 of 0.664, so tuning, mainly `class_weight` and `min_samples_leaf`, was a large gain.

**Feature importance** (impurity): Total day minutes (0.26), Customer service calls (0.17), International plan (0.11),
Total eve minutes (0.08), Total intl minutes (0.07). Permutation importance on the test set gives the same
top group (day minutes, customer service calls, eve minutes, international plan, voice mail plan).
Heavy daytime usage, repeated support calls and having an international plan are the main churn signals.

Plots: `rf_tuning.png`, `rf_confusion_matrix.png`, `rf_feature_importance.png`, `rf_threshold_sweep.png`.

## Task 3 — Neural network (Keras)

Architecture: `Input(67) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, sigmoid)`
(6,465 parameters), L2 regularisation, Adam, binary cross-entropy, class weights for imbalance,
early stopping on validation loss (stopped at epoch 46) and LR reduction on plateau.

| Metric | Test |
|---|---|
| Accuracy | 0.891 |
| Precision (churn) | 0.587 |
| Recall (churn) | 0.779 |
| F1 (churn) | 0.670 |
| ROC-AUC | 0.895 |

Plots: `nn_training_curves.png` (train/validation loss and accuracy), `nn_confusion_matrix.png`.
Model saved to `outputs/churn_mlp.keras`.

## Comparison and takeaways

The Random Forest clearly beats the MLP on this small tabular dataset (F1 0.82 vs 0.67, AUC 0.92 vs 0.89).
The NN's class weighting pushes it to flag more customers (recall 0.78, but precision only 0.59), and the loss curves show
validation loss flattening while training loss keeps falling, i.e. mild overfitting with ~2.3k training rows.
For tabular data of this size, tree ensembles are the stronger default; the NN would need more data or tuning
(wider search, threshold calibration) to compete.

## Caveats

- The task text suggests MNIST for the neural network; this project uses the churn dataset for consistency with Task 1 instead.
- Single train/test split; results on 95 churners in the test set carry noticeable variance.
- Decision thresholds were left at 0.5; the threshold sweep plot is provided for exploring precision/recall trade-offs.
