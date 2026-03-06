"""
report_cross_results.py
────────────────────────
Visualise & summarise cross-dataset results for *all* balance modes.

Input: cross_dataset_validation_results.csv with columns like:
  - checkpoint
  - model_backbone
  - trained_on
  - training_fold
  - balance_mode ∈ {"weighted_loader", "weighted_loss", "none"}
  - validated_on
  - accuracy, f1_score, precision, recall, specificity

Outputs:
  • Console:
      (A) mean ± std per (balance_mode × TRAIN→EVAL)
      (B) mean ± std per (balance_mode × MODEL × TRAIN→EVAL)
  • Plots (per balance_mode):
      - Heat-map: mean MAIN_METRIC per Model × TRAIN→EVAL
      - Box-plot: MAIN_METRIC spread per backbone
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
CSV = Path("workspace/metrics/all_cross_results/cross_dataset_validation_results.csv")
assert CSV.exists(), f"CSV not found: {CSV}"

# Metric used in heatmaps / boxplots
MAIN_METRIC = "f1_score"
SUMMARY_COLS = ["accuracy", "f1_score", "precision", "recall", "specificity"]


# ----------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------
df = pd.read_csv(CSV)

# Derived column: transfer direction
df["train→val"] = df["trained_on"] + "→" + df["validated_on"]

required_cols = {
    "trained_on",
    "validated_on",
    "model_backbone",
    "training_fold",
    "balance_mode",
    *SUMMARY_COLS,
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

balance_modes = sorted(df["balance_mode"].unique())
print("Found balance modes:", balance_modes)


def print_mean_std_table(stats: pd.DataFrame, title: str) -> None:
    """
    stats is expected to have a MultiIndex on columns: (metric, agg)
    where agg ∈ {'mean','std'}.

    This helper flattens it to 'metric_mean' / 'metric_std' for nicer printing.
    """
    flat = stats.copy()
    flat.columns = [f"{m}_{a}" for (m, a) in flat.columns]
    print(f"\n{title}")
    print(flat.to_string())


# ----------------------------------------------------------------
# 0️⃣  Console summaries (by balance_mode), with mean & std
# ----------------------------------------------------------------
for mode in balance_modes:
    df_mode = df[df["balance_mode"] == mode].copy()
    print("\n" + "=" * 80)
    print(f"BALANCE MODE: {mode}")
    print("=" * 80)

    # (A) mean ± std over folds (and models) for each Train→Eval pair
    pair_stats = (
        df_mode.groupby("train→val")[SUMMARY_COLS]
        .agg(["mean", "std"])
        .round(3)
        .sort_index()
    )
    print_mean_std_table(
        pair_stats,
        "--- A) Mean ± std over rows for each TRAIN→EVAL pair ---",
    )

    # (B) mean ± std for each (MODEL × TRAIN→EVAL)
    model_pair_stats = (
        df_mode.groupby(["model_backbone", "train→val"])[SUMMARY_COLS]
        .agg(["mean", "std"])
        .round(3)
    )
    print_mean_std_table(
        model_pair_stats,
        "--- B) Mean ± std for each MODEL × TRAIN→EVAL pair ---",
    )

# Also show all models found
print("\nUnique model_backbone values:")
print(sorted(df["model_backbone"].unique()))


# ----------------------------------------------------------------
# 1️⃣  HEAT-MAPS (one per balance_mode, using mean MAIN_METRIC)
# ----------------------------------------------------------------
for mode in balance_modes:
    df_mode = df[df["balance_mode"] == mode].copy()
    if df_mode.empty:
        continue

    heat = df_mode.pivot_table(
        index="model_backbone",
        columns="train→val",
        values=MAIN_METRIC,
        aggfunc="mean",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(heat, aspect="auto")

    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)

    ax.set_title(f"Mean {MAIN_METRIC} per transfer direction\n(balance_mode = {mode})")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(MAIN_METRIC)

    fig.tight_layout()


# ----------------------------------------------------------------
# 2️⃣  BOX-PLOTS per backbone (one per balance_mode)
# ----------------------------------------------------------------
for mode in balance_modes:
    df_mode = df[df["balance_mode"] == mode].copy()
    if df_mode.empty:
        continue

    # Order backbones by mean MAIN_METRIC under this balance mode
    order = (
        df_mode.groupby("model_backbone")[MAIN_METRIC]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    box_data = [df_mode.loc[df_mode["model_backbone"] == m, MAIN_METRIC] for m in order]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(box_data, labels=order, showfliers=False, vert=False)
    ax.set_xlabel(MAIN_METRIC)
    ax.set_title(f"Distribution of {MAIN_METRIC} per backbone\n(balance_mode = {mode})")

    fig.tight_layout()

plt.show()
