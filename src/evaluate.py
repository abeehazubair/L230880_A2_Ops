"""Evaluation helpers for VisionGuard (Task 2.5), used by train.py.
The decision threshold is chosen on VALIDATION data; the test set is only scored with it.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
CLASS_NAMES = ["Real", "AI"]
BETA = 0.5   # F0.5 weights precision twice as much as recall: calling a REAL image "AI" is the worse error


def predict(model, ds):
    """Returns (true labels, predicted P(AI)) in dataset order."""
    y_true, y_prob = [], []
    for x, y in ds:
        y_prob.append(model(x, training=False).numpy().ravel())
        y_true.append(y.numpy())
    return np.concatenate(y_true).astype(int), np.concatenate(y_prob)


def choose_threshold(y_true, y_prob, beta=BETA):
    """Try thresholds 0.05..0.95 on validation data, keep the one with the best F-beta."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def compute_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "f05": fbeta_score(y_true, y_pred, beta=BETA, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def plot_confusion(m, title, path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], [f"predicted {c}" for c in CLASS_NAMES])
    ax.set_yticks([0, 1], [f"true {c}" for c in CLASS_NAMES])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc(y_true, y_prob, thr, title, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    pred = y_prob >= thr
    op_fpr = (pred & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
    op_tpr = (pred & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1, label="random guess")
    ax.scatter([op_fpr], [op_tpr], s=70, color=ORANGE, zorder=3, label=f"chosen threshold = {thr:.2f}")
    ax.set_xlabel("False positive rate (real called AI)")
    ax.set_ylabel("True positive rate (AI caught)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
