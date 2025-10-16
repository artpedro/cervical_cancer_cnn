"""
report_cross_results.py
────────────────────────
Visualise & summarise cross-dataset results.
  • Heat-map   : mean accuracy per transfer direction
  • Box-plots  : accuracy spread per backbone
  • Console    : two tables
        1. mean over folds for each Train→Eval pair
        2. mean over folds for each (Model × Train→Eval) pair
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path("cross_dataset_validation_results.csv")
assert CSV.exists(), f"CSV not found: {CSV}"

df = pd.read_csv(CSV)
df["train→val"] = df["trained_on"] + "→" + df["validated_on"]

# ────────────────────────────────────────────────────────────────
# 0️⃣  Console summaries
# ────────────────────────────────────────────────────────────────
summary_cols = ["accuracy", "f1_score", "precision", "recall", "specificity"]

# (A) mean over the 5 folds for each transfer direction
pair_stats = (
    df.groupby("train→val")[summary_cols]
      .mean()
      .round(3)
      .sort_index()
)

# (B) mean for each (backbone × transfer direction) – 5 values per cell
model_pair_stats = (
    df.groupby(["model_backbone", "train→val"])[summary_cols]
      .mean()
      .round(3)
)

print("\n=== AVERAGE OVER 5 FOLDS FOR EACH TRAIN→EVAL PAIR ===")
print(pair_stats.to_string())

print("\n=== AVERAGE OVER 5 FOLDS FOR EACH MODEL × TRAIN→EVAL PAIR ===")
print(model_pair_stats.to_string())
print(df.model_backbone.unique())

# ────────────────────────────────────────────────────────────────
# 1️⃣  HEAT-MAP (mean accuracy, with per-fold dots)
# ────────────────────────────────────────────────────────────────
heat = (
    df.pivot_table(
        index="model_backbone",
        columns="train→val",
        values="f1_score",
        aggfunc="mean",
    )
    .sort_index()
)

fig1, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(heat, aspect="auto")
ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
ax.set_yticks(range(len(heat.index)),   heat.index)
ax.set_title("Mean f1-score per transfer direction (5 folds averaged)")
plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")


fig1.tight_layout()

# ────────────────────────────────────────────────────────────────
# 2️⃣  BOX-PLOTS per backbone (10 runs each: 5 folds × 2 targets)
# ────────────────────────────────────────────────────────────────
order = (
    df.groupby("model_backbone")["f1_score"]
      .mean()
      .sort_values(ascending=False)
      .index
)
box_data = [df.loc[df["model_backbone"] == m, "f1_score"] for m in order]

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.boxplot(box_data, labels=order, showfliers=False, vert=False)
ax2.set_xlabel("F1-Score")
ax2.set_title("Transfer F1-Score distribution per backbone (10 runs)")
fig2.tight_layout()

plt.show()
