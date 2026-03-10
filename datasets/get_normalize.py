#!/usr/bin/env python
"""
Populate datasets/normalization_stats.json with per-fold mean/std for each
dataset and each mixed-dataset combination. Stats for fold_k are computed
only on the training portion of that fold (train_dev rows with fold != k),
so cross-validation does not leak validation data into normalization.

Run from project root: python -m datasets.get_normalize
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from datasets.datasets import (
    NORM_STATS_PATH,
    scan_apacc,
    scan_herlev,
    scan_sipakmed,
    merge_train_dev_with_folds,
    compute_mean_std_for_df,
)

# ─────────── configuration (match train_models / train_models_mixed) ───────────
SEED = 42
NUM_FOLDS = 5
TEST_SIZE = 0.2

ROOTS = {
    "apacc": Path("./datasets/data/apacc"),
    "herlev": Path("./datasets/data/smear2005"),
    "sipakmed": Path("./datasets/data/sipakmed"),
}

COMBINATIONS = [
    ("apacc_sipakmed", ["apacc", "sipakmed"]),
    ("herlev_sipakmed", ["herlev", "sipakmed"]),
    ("herlev_apacc", ["herlev", "apacc"]),
]
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    stats_path = Path(NORM_STATS_PATH)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if stats_path.exists():
        with stats_path.open("r") as f:
            stats = json.load(f)
    else:
        stats = {}

    scanners = {
        "apacc": lambda: scan_apacc(ROOTS["apacc"], num_folds=NUM_FOLDS, seed=SEED),
        "herlev": lambda: scan_herlev(
            ROOTS["herlev"], num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE
        ),
        "sipakmed": lambda: scan_sipakmed(
            ROOTS["sipakmed"], num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE
        ),
    }

    # ─── Single datasets: per-fold stats (training portion only) ───
    for name, scan in scanners.items():
        print(f"\n>> {name}")
        df = scan()
        train_dev = df[df["split"] == "train_dev"]
        if name not in stats:
            stats[name] = {}
        for fold in range(NUM_FOLDS):
            train_only = train_dev[train_dev["fold"] != fold]
            if train_only.empty:
                raise RuntimeError(f"{name} fold {fold}: no training rows")
            mean, std = compute_mean_std_for_df(
                train_only.reset_index(drop=True),
                show_progress=(fold == 0),
            )
            stats[name][f"fold_{fold}"] = {"mean": mean, "std": std}
            if fold == 0:
                print(f"   fold_0 mean: {mean}")
                print(f"   fold_0 std:  {std}")

        # Full train_dev stats (for full-dataset training and cross-dataset evaluation)
        full_train_dev = train_dev.reset_index(drop=True)
        mean_full, std_full = compute_mean_std_for_df(full_train_dev, show_progress=False)
        stats[name]["full"] = {"mean": mean_full, "std": std_full}
        print(f"   full (train_dev) mean: {mean_full}")
        print(f"   full (train_dev) std:  {std_full}")

    # ─── Mixed datasets: per-fold stats (training portion only) ───
    for combined_name, train_names in COMBINATIONS:
        print(f"\n>> {combined_name}")
        dfs_to_merge = []
        for name in train_names:
            dfs_to_merge.append(scanners[name]())
        use_group = "sipakmed" in train_names
        merged_df = merge_train_dev_with_folds(
            dfs_to_merge,
            num_folds=NUM_FOLDS,
            seed=SEED,
            group_column="cluster_id" if use_group else None,
            name_prefixes=train_names,
        )
        if combined_name not in stats:
            stats[combined_name] = {}
        for fold in range(NUM_FOLDS):
            train_only = merged_df[merged_df["fold"] != fold]
            if train_only.empty:
                raise RuntimeError(f"{combined_name} fold {fold}: no training rows")
            mean, std = compute_mean_std_for_df(
                train_only.reset_index(drop=True),
                show_progress=(fold == 0),
            )
            stats[combined_name][f"fold_{fold}"] = {"mean": mean, "std": std}
            if fold == 0:
                print(f"   fold_0 mean: {mean}")
                print(f"   fold_0 std:  {std}")

        # Full merged train_dev stats (for full-dataset training and cross-dataset evaluation)
        mean_full, std_full = compute_mean_std_for_df(merged_df.reset_index(drop=True), show_progress=False)
        stats[combined_name]["full"] = {"mean": mean_full, "std": std_full}
        print(f"   full (train_dev) mean: {mean_full}")
        print(f"   full (train_dev) std:  {std_full}")

    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nNormalization stats written to {stats_path.resolve()}")


if __name__ == "__main__":
    main()
