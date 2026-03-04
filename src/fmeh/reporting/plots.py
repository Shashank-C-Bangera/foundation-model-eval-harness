from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_metric_bars(agg_df: pd.DataFrame, metric: str, output_path: str | Path) -> None:
    if agg_df.empty or metric not in agg_df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = agg_df.copy()
    plot_df["series"] = (
        plot_df["task"].astype(str)
        + " | "
        + plot_df["model_id"].astype(str)
        + " | "
        + plot_df["prompt_version"].astype(str)
    )
    plot_df = plot_df.sort_values(metric, ascending=False)

    x = list(range(len(plot_df)))
    ax.bar(x, plot_df[metric])
    ax.set_title(f"{metric} by task/model/prompt")
    ax.set_ylabel(metric)
    ax.set_xticks(x, plot_df["series"], rotation=60, ha="right")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    target_labels: list[str], pred_labels: list[str], output_path: str | Path
) -> None:
    if not target_labels:
        return
    labels = ["yes", "no", "maybe"]
    cm = confusion_matrix(target_labels, pred_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, colorbar=False)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
