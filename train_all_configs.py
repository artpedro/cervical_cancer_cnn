"""
Train all configs: single-dataset (apacc, herlev, sipakmed) and mixed-dataset
(apacc_sipakmed, herlev_sipakmed, herlev_apacc). Same models and balance modes
for both. Single log file and single training_time_results.csv for the full run.

Usage:
  uv run python train_all_configs.py              # run everything
  uv run python train_all_configs.py --single     # single datasets only
  uv run python train_all_configs.py --mixed       # mixed datasets only
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from datasets.datasets import (
    NORM_STATS_PATH,
    merge_train_dev_with_folds,
    scan_apacc,
    scan_herlev,
    scan_sipakmed,
)

from train_models import (
    DEVICE,
    BATCH_SIZE,
    METRICS_DIR,
    NUM_FOLDS,
    RUNS_DIR,
    SEED,
    _setup_run_dir,
    train_dataset,
)

# Same model set as in train_models.py / train_models_mixed.py main()
MODELS = {
    **{f"EfficientNet-B{i}": f"efficientnet_b{i}" for i in range(7)},
    "SqueezeNet 1.1": "tv_squeezenet1_1",
    "MobileNet V2 1.0x": "mobilenetv2_100",
    "MobileNet V4 small": "mobilenetv4_conv_small.e2400_r224_in1k",
    "ShuffleNet V2 1.0x": "tv_shufflenet_v2_x1_0",
    "GhostNet V3": "ghostnetv3_100.in1k",
}

BALANCE_MODES = ["weighted_loader", "weighted_loss", "none"]

# Single datasets: (name, root, scanner)
SINGLE_CONFIGS = [
    ("apacc", Path("./datasets/data/apacc"), scan_apacc),
    ("herlev", Path("./datasets/data/smear2005"), scan_herlev),
    ("sipakmed", Path("./datasets/data/sipakmed"), scan_sipakmed),
]

# Mixed: (combined_name, [train_name_a, train_name_b])
MIXED_CONFIGS = [
    ("apacc_sipakmed", ["apacc", "sipakmed"]),
    ("herlev_sipakmed", ["herlev", "sipakmed"]),
    ("herlev_apacc", ["herlev", "apacc"]),
]

ROOTS = {
    "apacc": Path("./datasets/data/apacc"),
    "herlev": Path("./datasets/data/smear2005"),
    "sipakmed": Path("./datasets/data/sipakmed"),
}
SCANNERS = {
    "apacc": scan_apacc,
    "herlev": scan_herlev,
    "sipakmed": scan_sipakmed,
}


def _tee_log(log_path: Path):
    """Redirect stdout/stderr to both console and log file."""
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")

    class _Tee:
        def __init__(self, *streams, isatty_stream=None):
            self.streams = streams
            self._isatty_stream = isatty_stream

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

        def isatty(self):
            if self._isatty_stream is None:
                return False
            return bool(getattr(self._isatty_stream, "isatty", lambda: False)())

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_stdout, log_f, isatty_stream=old_stdout)
    sys.stderr = _Tee(old_stderr, log_f, isatty_stream=old_stderr)
    return log_f, old_stdout, old_stderr


def run_single(results_csv: Path, progress_cb=None) -> None:
    """Train all single-dataset configs (apacc, herlev, sipakmed)."""
    for name, root, scanner in SINGLE_CONFIGS:
        print(f"\nScanning dataset: {name} at {root}")
        if name == "apacc":
            df = scanner(root=root, num_folds=NUM_FOLDS, seed=SEED)
        else:
            df = scanner(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=0.2)

        for balance_mode in BALANCE_MODES:
            print("\n" + "-" * 70)
            print(f"Starting training on {name} with balance_mode = '{balance_mode}'")
            print("-" * 70 + "\n")

            run_dir = _setup_run_dir(METRICS_DIR, dataset_name=name, balance_mode=balance_mode)
            train_dataset(
                name=name,
                df=df,
                run_dir=run_dir,
                models=MODELS,
                balance_mode=balance_mode,
                num_folds=NUM_FOLDS,
                batch_size=BATCH_SIZE,
                device=DEVICE,
                stats_path=NORM_STATS_PATH,
                use_amp=True,
                results_csv=results_csv,
                progress_cb=progress_cb,
            )


def run_mixed(results_csv: Path, progress_cb=None) -> None:
    """Train all mixed-dataset configs (apacc_sipakmed, herlev_sipakmed, herlev_apacc)."""
    for combined_name, train_names in MIXED_CONFIGS:
        print(f"\nCombination: train on {' + '.join(train_names)}")

        dfs_to_merge = []
        for name in train_names:
            root = ROOTS[name]
            scanner = SCANNERS[name]
            print(f"  Scanning {name} at {root}")
            if name == "apacc":
                df_part = scanner(root=root, num_folds=NUM_FOLDS, seed=SEED)
            else:
                df_part = scanner(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=0.2)
            dfs_to_merge.append(df_part)

        use_group = "sipakmed" in train_names
        merged_df = merge_train_dev_with_folds(
            dfs_to_merge,
            num_folds=NUM_FOLDS,
            seed=SEED,
            group_column="cluster_id" if use_group else None,
            name_prefixes=train_names,
        )
        print(f"  Merged train_dev size: {len(merged_df)}")

        for balance_mode in BALANCE_MODES:
            print("\n" + "-" * 70)
            print(f"Starting training on {combined_name} with balance_mode = '{balance_mode}'")
            print("-" * 70 + "\n")

            run_dir = _setup_run_dir(METRICS_DIR, dataset_name=combined_name, balance_mode=balance_mode)
            train_dataset(
                name=combined_name,
                df=merged_df,
                run_dir=run_dir,
                models=MODELS,
                balance_mode=balance_mode,
                num_folds=NUM_FOLDS,
                batch_size=BATCH_SIZE,
                device=DEVICE,
                stats_path=NORM_STATS_PATH,
                use_amp=True,
                results_csv=results_csv,
                progress_cb=progress_cb,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all configs (single + mixed datasets).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--single",
        action="store_true",
        help="Run only single-dataset training (apacc, herlev, sipakmed)",
    )
    group.add_argument(
        "--mixed",
        action="store_true",
        help="Run only mixed-dataset training (apacc_sipakmed, herlev_sipakmed, herlev_apacc)",
    )
    args = parser.parse_args()

    run_single_only = args.single
    run_mixed_only = args.mixed
    if not run_single_only and not run_mixed_only:
        run_single_only = run_mixed_only = True

    print("\n" + "=" * 70)
    print("TRAIN ALL CONFIGS (single + mixed datasets)")
    print("=" * 70)
    print("Modes: weighted_loader | weighted_loss | none")
    print("=" * 70 + "\n")

    run_start_dt = datetime.datetime.now()
    run_start_str = run_start_dt.strftime("%Y-%m-%d %H:%M:%S")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    log_path = RUNS_DIR / f"terminal_all_{run_start_dt.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    results_csv = METRICS_DIR / "training_time_results.csv"

    n_single = len(SINGLE_CONFIGS) * len(BALANCE_MODES) * len(MODELS) * NUM_FOLDS if run_single_only else 0
    n_mixed = len(MIXED_CONFIGS) * len(BALANCE_MODES) * len(MODELS) * NUM_FOLDS if run_mixed_only else 0
    total_configs = n_single + n_mixed
    done_configs = [0]  # use list so progress_cb can mutate

    def progress_cb(**info):
        done_configs[0] += 1
        print(f"[PROGRESS] {done_configs[0]}/{total_configs} configs | start={run_start_str}")

    log_f, old_stdout, old_stderr = _tee_log(log_path)
    try:
        print(f"[START] {run_start_str}")
        print(f"[LOG]   {log_path}")
        print(f"[CSV]   {results_csv}")

        if run_single_only:
            run_single(results_csv, progress_cb=progress_cb)
            if run_mixed_only:
                print("\n" + "=" * 70)
                print("SINGLE-DATASET TRAINING COMPLETE — starting mixed-dataset")
                print("=" * 70 + "\n")

        if run_mixed_only:
            run_mixed(results_csv, progress_cb=progress_cb)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE (all requested configs)")
        print("=" * 70 + "\n")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_f.close()


if __name__ == "__main__":
    main()
