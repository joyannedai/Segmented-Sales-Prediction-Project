import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir

logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def plot_model_comparison(results_df: pd.DataFrame, baseline_wape: float, output_dir: str, title_suffix: str = ""):
    if results_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    models = results_df["model"].tolist()
    wapes = results_df["wape"].tolist()
    colors = ["#2ecc71" if w < baseline_wape else "#e74c3c" for w in wapes]

    bars = ax.bar(models, wapes, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, wapes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    if not np.isnan(baseline_wape):
        ax.axhline(y=baseline_wape, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.8, label=f"Baseline ({baseline_wape:.1f}%)")

    ax.set_ylabel("WAPE (%)", fontsize=12)
    ax.set_title(f"Model Comparison {title_suffix}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    ax.legend()
    plt.tight_layout()

    path = os.path.join(ensure_dir(output_dir), f"model_comparison{title_suffix.replace(' ', '_')}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {path}")


def plot_feature_importance(model, feature_names: list, group: str, model_name: str, output_dir: str):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).flatten()
    else:
        return

    n_features = min(len(importance), len(feature_names))
    imp_series = pd.Series(importance[:n_features], index=feature_names[:n_features]).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    top10 = imp_series.head(10)[::-1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top10)))
    ax.barh(range(len(top10)), top10.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10.index, fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"{group.upper()} - {model_name} Feature Importance", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    path = os.path.join(ensure_dir(output_dir), f"feature_importance_{group}_{model_name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {path}")
