# misc
from typing import Optional, Tuple
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

# Data management
import os, random, csv, warnings, datetime, time
import pandas as pd
import numpy as np

# Datasets and loaders
from datasets.datasets import (
    scan_sipakmed,
    scan_herlev,
    scan_apacc,
    get_loaders,
    get_loaders_weighted,
    APACC_TRAIN_TF,
    APACC_VAL_TF,
    SIPAKMED_TRAIN_TF,
    SIPAKMED_VAL_TF,
    HERLEV_TRAIN_TF,
    HERLEV_VAL_TF,
)


# Models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.amp import autocast, GradScaler
from model_utils import load_any

import gc

# ----------------------------
# Configuration / constants
# ----------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Faster settings (NOT fully deterministic, but still seeded)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path(os.getenv("DATA_DIR", ".\\workspace\\data"))
METRICS_DIR = Path(os.getenv("METRICS_DIR", ".\\workspace\\metrics"))
RUNS_DIR = Path(os.getenv("RUNS_DIR", ".\\workspace\\runs"))

BATCH_SIZE = 32
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
NUM_FOLDS = 5
EPOCHS = 25
LR = 5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-3
SCHEDULER_MILESTONES = [10, 20]
SCHEDULER_GAMMA = 0.1

# global default – can be overridden by env var or CLI later
BALANCE_MODE = os.getenv("BALANCE_MODE", "weighted_loader")
# allowed: "weighted_loader", "weighted_loss", "none"

EPS = 1e-9

# Default model list (can be overridden per dataset)
BASE_TODO = {
    "SqueezeNet 1.1": "tv_squeezenet1_1",
    "MobileNet V2 1.0x": "mobilenetv2_100",
    "MobileNet V4 small": "mobilenetv4_conv_small.e2400_r224_in1k",
    "ShuffleNet V2 1.0x": "tv_shufflenet_v2_x1_0",
    "GhostNet V3": "ghostnetv3_100.in1k",
    **{f"EfficientNet-B{i}": f"efficientnet_b{i}" for i in range(4)},
}

# ----------------------------
# Metrics for model evaluation
# ----------------------------


def compute_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency weights for binary classes.
    Uses df['binary_idx'] and assumes exactly two classes (0 and 1).
    """
    freq = df["binary_idx"].value_counts(normalize=True).sort_index()
    if freq.size != 2:
        raise ValueError("Expected two classes, got " + str(freq.to_dict()))
    return torch.tensor(
        [1.0 / freq[0], 1.0 / freq[1]], dtype=torch.float32, device=device
    )


# ----------------------------
# Epoch runner
# ----------------------------
def _run_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    epoch_num: int,
    split_name: str,
    optimiser: Optional[torch.optim.Optimizer] = None,
    *,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
    device: torch.device = DEVICE,
) -> dict[str, float]:
    training = optimiser is not None
    model.train(training)

    amp_enabled = bool(use_amp and torch.cuda.is_available() and device.type == "cuda")

    total_loss = 0.0
    n_samples = 0

    tp = torch.zeros((), device=device, dtype=torch.int64)
    tn = torch.zeros((), device=device, dtype=torch.int64)
    fp = torch.zeros((), device=device, dtype=torch.int64)
    fn = torch.zeros((), device=device, dtype=torch.int64)

    desc = f"Epoch {epoch_num:02d} ({split_name.capitalize()})"
    progress_bar = tqdm(dataloader, desc=desc, leave=True)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            bs = labels.size(0)
            n_samples += bs

            if training:
                optimiser.zero_grad(set_to_none=True)

            # NEW API: torch.amp.autocast("cuda", ...)
            with autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                if amp_enabled:
                    if scaler is None:
                        raise ValueError("AMP enabled but scaler is None.")
                    scaler.scale(loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    loss.backward()
                    optimiser.step()

            total_loss += float(loss.detach().item()) * bs

            pred = logits.detach().argmax(dim=1)

            # Safer across torch versions (no sum(dtype=...) dependency)
            tp += ((pred == 1) & (labels == 1)).sum().to(torch.int64)
            tn += ((pred == 0) & (labels == 0)).sum().to(torch.int64)
            fp += ((pred == 1) & (labels == 0)).sum().to(torch.int64)
            fn += ((pred == 0) & (labels == 1)).sum().to(torch.int64)

    tp_f = tp.to(torch.float32)
    tn_f = tn.to(torch.float32)
    fp_f = fp.to(torch.float32)
    fn_f = fn.to(torch.float32)

    acc = (tp_f + tn_f) / (tp_f + tn_f + fp_f + fn_f + EPS)
    prec = tp_f / (tp_f + fp_f + EPS)
    rec = tp_f / (tp_f + fn_f + EPS)
    spec = tn_f / (tn_f + fp_f + EPS)
    f1 = (2.0 * prec * rec) / (prec + rec + EPS)
    ppv = prec
    npv = tn_f / (tn_f + fn_f + EPS)

    return {
        "loss": total_loss / max(1, n_samples),
        "acc": float(acc.item()),
        "prec": float(prec.item()),
        "rec": float(rec.item()),
        "spec": float(spec.item()),
        "f1": float(f1.item()),
        "ppv": float(ppv.item()),
        "npv": float(npv.item()),
    }



# ----------------------------
# Utility to create run folder
# ----------------------------


def _setup_run_dir(metrics_dir: Path, dataset_name: str, balance_mode: str) -> Path:
    """
    Create a run directory like:
      metrics_dir / <dataset_name> / <balance_mode> / <timestamp>/
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = metrics_dir / dataset_name / balance_mode / ts
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


# ----------------------------
# Flexible training function
# ----------------------------
def train_dataset(
    name: str,
    df: pd.DataFrame,
    run_dir: Path,
    *,
    models: dict[str, str],
    balance_mode: str = "weighted_loader",
    num_folds: int = NUM_FOLDS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    epochs: int = EPOCHS,
    lr: float = LR,
    momentum: float = MOMENTUM,
    weight_decay: float = WEIGHT_DECAY,
    scheduler_milestones: list[int] | None = None,
    scheduler_gamma: float = SCHEDULER_GAMMA,
    device: torch.device = DEVICE,
    train_tf=None,
    val_tf=None,
    use_amp: bool = True,
    results_csv: Path | None = None,
    progress_cb=None,  # callable(**info)
) -> None:
    """
    NEW:
      - AMP support (use_amp=True) for speed on CUDA
      - Metrics computed accurately on GPU via TP/TN/FP/FN accumulation (no sklearn)
      - Per-config timing saved to results_csv after each model+fold finishes
      - progress_cb called after each config
    """
    assert balance_mode in {"weighted_loader", "weighted_loss", "none"}

    print(f"\n{'=' * 70}")
    print(f"DATASET: {name.upper()}  |  balance_mode = {balance_mode}")
    print(f"Run directory: {run_dir}")
    print(f"{'=' * 70}")

    if "split" in df.columns:
        train_df = df[df["split"].isin(["train", "train_dev"])].reset_index(drop=True)
        print(
            f"Using rows with split in ('train','train_dev') for training (rows: {len(train_df)})"
        )
    else:
        train_df = df.reset_index(drop=True)
        print(f"No 'split' column found; using all {len(train_df)} rows for training.")

    class_dist = train_df["binary_idx"].value_counts().sort_index()
    total = len(train_df)
    print(f"\nClass Distribution (training subset):")
    print(
        f"  Class 0 (Normal):   {class_dist.get(0, 0):>6} samples ({100 * class_dist.get(0, 0) / total:5.1f}%)"
    )
    if 1 in class_dist.index:
        print(
            f"  Class 1 (Abnormal): {class_dist.get(1, 0):>6} samples ({100 * class_dist.get(1, 0) / total:5.1f}%)"
        )
        if class_dist.get(0, 0) > 0 and class_dist.get(1, 0) > 0:
            imbalance_ratio = max(class_dist[0], class_dist[1]) / min(
                class_dist[0], class_dist[1]
            )
            print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")

    if balance_mode == "weighted_loader":
        print("\n→ Using WeightedRandomSampler, criterion = CrossEntropyLoss (no class weights).")
    elif balance_mode == "weighted_loss":
        print("\n→ Using standard DataLoader, criterion = CrossEntropyLoss(class_weights) per fold.")
    else:
        print("\n→ Using standard DataLoader, criterion = CrossEntropyLoss (no balancing).")

    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(" GPU:", torch.cuda.get_device_name())
    amp_flag = bool(use_amp and torch.cuda.is_available() and device.type == "cuda")
    print(f"AMP enabled: {amp_flag}")

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    epoch_file = run_dir / f"epoch_logs_{name}_{balance_mode}.csv"
    summary_file = run_dir / f"summary_{name}_{balance_mode}.csv"

    epoch_cols = [
        "dataset", "model", "origin", "epoch", "split",
        "loss", "acc", "prec", "rec", "spec", "f1", "ppv", "npv",
        "lr", "seconds",
    ]
    pd.DataFrame(columns=epoch_cols).to_csv(epoch_file, index=False)

    summary_cols = [
        "dataset", "model",
        "best_epoch", "best_acc", "best_prec", "best_rec", "best_spec", "best_f1", "best_ppv", "best_npv",
        "fold",
    ]
    pd.DataFrame(columns=summary_cols).to_csv(summary_file, index=False)

    if scheduler_milestones is None:
        scheduler_milestones = SCHEDULER_MILESTONES

    # ---- results CSV append helper ----
    def _append_results(row: dict) -> None:
        if results_csv is None:
            return
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = results_csv.exists() and results_csv.stat().st_size > 0
        pd.DataFrame([row]).to_csv(results_csv, mode="a", header=not file_exists, index=False)

    for friendly_name, backbone_id in models.items():
        for fold in range(num_folds):
            # per-config timer (model+fold)
            config_start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            config_t0 = time.time()

            print(f"\n> {name} | {friendly_name} | fold {fold}")

            # --------- loaders + criterion ---------
            pin = torch.cuda.is_available()
            if balance_mode == "weighted_loader":
                train_loader, val_loader = get_loaders_weighted(
                    df=train_df,
                    fold=fold,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin,
                    train_tf=train_tf,
                    val_tf=val_tf,
                )
                criterion = nn.CrossEntropyLoss()
            else:
                train_loader, val_loader = get_loaders(
                    df=train_df,
                    fold=fold,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin,
                    train_tf=train_tf,
                    val_tf=val_tf,
                )
                if balance_mode == "weighted_loss":
                    train_mask = train_df["fold"] != fold
                    train_fold_df = train_df.loc[train_mask]
                    class_weights = compute_class_weights(train_fold_df, device=device)
                    print(f"  Fold {fold}: class weights = {class_weights.tolist()}")
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                else:
                    criterion = nn.CrossEntropyLoss()

            # --------- model/optim/scheduler/scaler ---------
            model, _, origin = load_any(backbone_id, num_classes=2, pretrained=True)
            model.to(device)

            optimiser = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            scheduler = MultiStepLR(optimiser, milestones=scheduler_milestones, gamma=scheduler_gamma)

            scaler = GradScaler("cuda", enabled=amp_flag)

            best_val = {k: 0.0 for k in ["acc", "prec", "rec", "spec", "f1", "ppv", "npv"]}
            best_val["epoch"] = 0

            # --------- epoch loop ---------
            for epoch in range(1, epochs + 1):
                t0 = time.time()

                train_m = _run_epoch(
                    train_loader, model, criterion, epoch, "train", optimiser,
                    scaler=scaler, use_amp=use_amp, device=device
                )
                val_m = _run_epoch(
                    val_loader, model, criterion, epoch, "val", optimiser=None,
                    scaler=None, use_amp=use_amp, device=device
                )

                scheduler.step()
                duration = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]

                for split_name, m in [("train", train_m), ("val", val_m)]:
                    row_data = OrderedDict(
                        dataset=name,
                        model=friendly_name,
                        origin=origin,
                        epoch=epoch,
                        split=split_name,
                        loss=m["loss"],
                        acc=m["acc"],
                        prec=m["prec"],
                        rec=m["rec"],
                        spec=m["spec"],
                        f1=m["f1"],
                        ppv=m["ppv"],
                        npv=m["npv"],
                        lr=lr_now,
                        seconds=duration,
                    )
                    pd.DataFrame([row_data]).to_csv(epoch_file, mode="a", header=False, index=False)

                if val_m["acc"] > best_val["acc"]:
                    best_val.update(val_m)
                    best_val["epoch"] = epoch
                    ckpt_name = f"{name}_{backbone_id}_{balance_mode}_best_fold{fold}.pt"
                    torch.save(model.state_dict(), run_dir / "checkpoints" / ckpt_name)

                if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                    print(
                        f"Ep{epoch:02d} loss {val_m['loss']:.4f} acc {val_m['acc']:.3f} "
                        f"f1 {val_m['f1']:.3f} (duration {duration:.1f}s)"
                    )

            # fold summary row
            summary_row = [
                name,
                friendly_name,
                best_val["epoch"],
                best_val["acc"],
                best_val["prec"],
                best_val["rec"],
                best_val["spec"],
                best_val["f1"],
                best_val["ppv"],
                best_val["npv"],
                fold,
            ]
            pd.DataFrame([summary_row], columns=summary_cols).to_csv(
                summary_file, mode="a", header=False, index=False
            )

            # write per-config timing row
            config_seconds = time.time() - config_t0
            config_end_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else ""

            _append_results(
                {
                    "run_dir": str(run_dir),
                    "dataset": name,
                    "balance_mode": balance_mode,
                    "model": friendly_name,
                    "backbone_id": backbone_id,
                    "origin": origin,
                    "fold": fold,
                    "config_start": config_start_dt,
                    "config_end": config_end_dt,
                    "config_seconds": config_seconds,
                    "best_epoch": best_val["epoch"],
                    "best_acc": best_val["acc"],
                    "best_f1": best_val["f1"],
                    "device": str(device),
                    "gpu_name": gpu_name,
                    "amp": amp_flag,
                }
            )

            if callable(progress_cb):
                progress_cb(
                    dataset=name,
                    balance_mode=balance_mode,
                    model=friendly_name,
                    backbone_id=backbone_id,
                    fold=fold,
                    config_seconds=config_seconds,
                    run_dir=str(run_dir),
                )

            del model, optimiser, scheduler, train_loader, val_loader, scaler
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nFinished training on dataset {name}. Logs at: {run_dir}")

# ----------------------------
# Main script
# ----------------------------
def main():
    import sys

    print("\n" + "=" * 70)
    print("FLEXIBLE TRAINING SCRIPT")
    print("=" * 70)
    print("Modes:")
    print("  - weighted_loader : WeightedRandomSampler + unweighted loss")
    print("  - weighted_loss   : standard loaders + weighted loss")
    print("  - none            : standard loaders + unweighted loss")
    print("=" * 70 + "\n")

    run_start_dt = datetime.datetime.now()
    run_start_str = run_start_dt.strftime("%Y-%m-%d %H:%M:%S")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    log_path = RUNS_DIR / f"terminal_{run_start_dt.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    results_csv = METRICS_DIR / "training_time_results.csv"

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

    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_stdout, log_f, isatty_stream=old_stdout)
    sys.stderr = _Tee(old_stderr, log_f, isatty_stream=old_stderr)

    try:
        print(f"[START] {run_start_str}")
        print(f"[LOG]   {log_path}")
        print(f"[CSV]   {results_csv}")

        names = ["apacc", "herlev", "sipakmed"]
        roots = [
            Path("./datasets/data/apacc"),
            Path("./datasets/data/smear2005"),
            Path("./datasets/data/sipakmed"),
        ]
        scanners = [scan_apacc, scan_herlev, scan_sipakmed]

        tf_map = {
            "apacc": (APACC_TRAIN_TF, APACC_VAL_TF),
            "herlev": (HERLEV_TRAIN_TF, HERLEV_VAL_TF),
            "sipakmed": (SIPAKMED_TRAIN_TF, SIPAKMED_VAL_TF),
        }

        balance_modes = ["weighted_loader", "weighted_loss", "none"]

        models = {
            **{f"EfficientNet-B{i}": f"efficientnet_b{i}" for i in range(7)},
            "SqueezeNet 1.1": "tv_squeezenet1_1",
            "MobileNet V2 1.0x": "mobilenetv2_100",
            "MobileNet V4 small": "mobilenetv4_conv_small.e2400_r224_in1k",
            "ShuffleNet V2 1.0x": "tv_shufflenet_v2_x1_0",
            "GhostNet V3": "ghostnetv3_100.in1k",
        }

        total_configs = len(names) * len(balance_modes) * len(models) * NUM_FOLDS
        done_configs = 0

        def progress_cb(**info):
            nonlocal done_configs
            done_configs += 1
            print(f"[PROGRESS] Runned {done_configs}/{total_configs} config models | start={run_start_str}")

        for root, scanner, name in zip(roots, scanners, names):
            train_tf, val_tf = tf_map[name]

            print(f"\nScanning dataset: {name} at {root}")
            df = scanner(root=root, num_folds=NUM_FOLDS, seed=SEED)

            for balance_mode in balance_modes:
                print("\n" + "-" * 70)
                print(f"Starting training on {name} with balance_mode = '{balance_mode}'")
                print("-" * 70 + "\n")

                run_dir = _setup_run_dir(METRICS_DIR, dataset_name=name, balance_mode=balance_mode)

                train_dataset(
                    name=name,
                    df=df,
                    run_dir=run_dir,
                    models=models,
                    balance_mode=balance_mode,
                    num_folds=NUM_FOLDS,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    epochs=EPOCHS,
                    lr=LR,
                    momentum=MOMENTUM,
                    weight_decay=WEIGHT_DECAY,
                    scheduler_milestones=SCHEDULER_MILESTONES,
                    scheduler_gamma=SCHEDULER_GAMMA,
                    device=DEVICE,
                    train_tf=train_tf,
                    val_tf=val_tf,
                    use_amp=True,
                    results_csv=results_csv,
                    progress_cb=progress_cb,
                )

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE (all datasets, all balance modes)")
        print("=" * 70 + "\n")

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_f.close()


if __name__ == "__main__":
    main()
