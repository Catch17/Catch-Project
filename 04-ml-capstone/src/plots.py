"""
plots.py – save confusion-matrix heatmap and decision-tree visualisation.

Uses the 'Agg' non-interactive backend so no display is required.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree


def save_confusion_matrix(
    cm: np.ndarray,
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Save a heatmap of *cm* to *output_path* (PNG)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(4, cm.shape[0]), max(3, cm.shape[0])))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"[plots] Confusion matrix saved to '{output_path}'")


def save_tree_plot(
    model: DecisionTreeClassifier,
    output_path: str,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
    max_depth: int = 4,
) -> None:
    """Save a decision-tree visualisation to *output_path* (PNG)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        ax=ax,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        max_depth=max_depth,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"[plots] Tree plot saved to '{output_path}'")
