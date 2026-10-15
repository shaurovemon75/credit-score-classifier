"""Figures saved to reports/figures (also reused by the notebook)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import config as C

sns.set_theme(style="whitegrid")


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=140)
    plt.close(fig)


def plot_model_comparison(df):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    d = df.sort_values("macro_f1")
    ax.barh(d.index, d["macro_f1"], color="#2a6f97")
    for i, v in enumerate(d["macro_f1"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 (customer-grouped hold-out)")
    ax.set_title("Model comparison")
    _save(fig, "model_comparison.png")


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    pct = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(pct, annot=cm, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=C.LABELS, yticklabels=C.LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (hold-out)")
    _save(fig, "confusion_matrix.png")


def plot_importance(pipeline, feature_names, top=20):
    clf = pipeline.named_steps["clf"]
    prep = pipeline.named_steps["prep"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    imp = clf.booster_.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(np.array(names)[order], imp[order] / imp.sum(), color="#386641")
    ax.set_title(f"Top {top} features (LightGBM gain share)")
    _save(fig, "feature_importance.png")
