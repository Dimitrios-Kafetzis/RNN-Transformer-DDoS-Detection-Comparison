#!/usr/bin/env python3
"""
evaluation/profile_models.py

Automatically discovers all <model>_y_pred.npy / <model>_y_true.npy
files in evaluation/predictions/, then for each one prints a
classification report and writes confusion-matrix + per-class F1
charts under plots/model_profiles/.
"""

import os
import glob
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
import matplotlib.pyplot as plt

# Folders
PRED_DIR = os.path.join(os.path.dirname(__file__), "predictions")
PLOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "plots", "model_profiles")
)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def discover_models():
    """Look for all *_y_pred.npy files and infer the model names."""
    pred_paths = glob.glob(os.path.join(PRED_DIR, "*_y_pred.npy"))
    model_names = [os.path.basename(p)[:-len("_y_pred.npy")] for p in pred_paths]
    return sorted(model_names)

def load_arrays(model_name):
    y_pred_path = os.path.join(PRED_DIR, f"{model_name}_y_pred.npy")
    y_true_path = os.path.join(PRED_DIR, f"{model_name}_y_true.npy")
    if not os.path.isfile(y_pred_path) or not os.path.isfile(y_true_path):
        raise FileNotFoundError(f"Missing y_pred or y_true for {model_name}")
    y_pred = np.load(y_pred_path)
    y_true = np.load(y_true_path)
    return y_true, y_pred

def profile_model(model_name):
    print(f"\n=== Profiling {model_name} ===")
    y_true, y_pred = load_arrays(model_name)

    labels = np.unique(y_true)
    labels_str = [str(l) for l in labels]

    # 1. Classification report
    print(classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=labels_str,
        zero_division=0
    ))

    # 2. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_str)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    ax.set_title(f"Confusion Matrix — {model_name}")
    ensure_dir(PLOT_DIR)
    cm_file = os.path.join(PLOT_DIR, f"{model_name}_confusion_matrix.png")
    fig.savefig(cm_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {cm_file}")

    # 3. Per-class F1
    f1_scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(labels)), f1_scores)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels_str, rotation=90)
    ax.set_ylabel("F1 score")
    ax.set_title(f"Per-class F1 — {model_name}")
    f1_file = os.path.join(PLOT_DIR, f"{model_name}_per_class_f1.png")
    fig.savefig(f1_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {f1_file}")

def main():
    models = discover_models()
    if not models:
        print(f"No prediction files found in {PRED_DIR}")
        return
    for m in models:
        try:
            profile_model(m)
        except Exception as e:
            print(f"[!] Skipping {m}: {e}")

if __name__ == "__main__":
    main()
