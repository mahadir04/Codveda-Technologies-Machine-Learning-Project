"""Level 3 - Task 3: Feed-forward neural network (TensorFlow/Keras) for churn."""
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from common import OUT, SEED, get_data

tf.keras.utils.set_random_seed(SEED)

X_train, X_test, y_train, y_test = get_data(scale=True)  # NNs need scaled inputs
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train.values.astype("float32"), y_train, test_size=0.15,
    stratify=y_train, random_state=SEED)
X_te = X_test.values.astype("float32")
print(f"train {X_tr.shape}, val {X_val.shape}, test {X_te.shape}")

# Churn is imbalanced (~14%): weight the minority class during backprop
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
class_weight = {0: float(cw[0]), 1: float(cw[1])}
print("class weights:", class_weight)

# ---- Architecture: input -> 64 -> 32 -> 1 (sigmoid) -------------------------
model = keras.Sequential([
    layers.Input(shape=(X_tr.shape[1],)),
    layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
], name="churn_mlp")
model.summary()

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                  restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
]
hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=200,
                 batch_size=32, class_weight=class_weight,
                 callbacks=callbacks, verbose=0)
print(f"Stopped after {len(hist.history['loss'])} epochs")

# ---- Training curves --------------------------------------------------------
h = hist.history
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(h["loss"], label="train"); axes[0].plot(h["val_loss"], label="validation")
axes[0].set(title="Loss (binary cross-entropy)", xlabel="epoch"); axes[0].legend()
axes[1].plot(h["accuracy"], label="train"); axes[1].plot(h["val_accuracy"], label="validation")
axes[1].set(title="Accuracy", xlabel="epoch"); axes[1].legend()
plt.tight_layout(); plt.savefig(OUT / "nn_training_curves.png", dpi=150); plt.close()

# ---- Evaluation -------------------------------------------------------------
proba = model.predict(X_te, verbose=0).ravel()
pred = (proba >= 0.5).astype(int)
print("\nTest set (threshold 0.5)")
print(classification_report(y_test, pred, target_names=["Stayed", "Churned"]))

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stayed", "Churned"],
                                        cmap="Oranges", ax=ax)
ax.set_title("Neural network - confusion matrix (test)")
plt.tight_layout(); plt.savefig(OUT / "nn_confusion_matrix.png", dpi=150); plt.close()

results = {
    "epochs_run": len(h["loss"]),
    "params": int(model.count_params()),
    "test": {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    },
}
(OUT / "nn_results.json").write_text(json.dumps(results, indent=2, default=float))
model.save(OUT / "churn_mlp.keras")
