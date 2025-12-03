# misc
from typing import Optional, Tuple
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

# Data management
import os, random, csv, warnings, datetime, time
import pandas as pd
import numpy as np

# Datasets
from datasets.datasets import (
    scan_sipakmed,
    scan_herlev,
    scan_apacc,
    get_loaders,
    get_loaders_weighted,
)

# Models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from model_utils import load_any

# Metrics
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
)

import gc

# ----------------------------
# Configuration / constants
# ----------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path(os.getenv("DATA_DIR", ".\\workspace\\data"))
METRICS_DIR = Path(os.getenv("METRICS_DIR", ".\\workspace\\metrics_flex"))
RUNS_DIR = Path(os.getenv("RUNS_DIR", ".\\workspace\\runs_flex"))

BATCH_SIZE = 32
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
NUM_FOLDS = 5
EPOCHS = 25
LR = 5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-3
SCHEDULER_MILESTONES = [10, 20]
SCHEDULER_GAMMA = 0.1

# change here if you want a global default
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


def _confusion_parts(y_true, y_pred) -> Tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn, fp, fn, tp


def metrics_binary(y_true, y_pred) -> dict[str, float]:
    """Return all requested binary-classification metrics in one dict."""
    tn, fp, fn, tp = _confusion_parts(y_true, y_pred)

    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),  # PPV
        "rec": recall_score(y_true, y_pred, zero_division=0),  # Sensitivity
        "spec": tn / (tn + fp + EPS),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "ppv": tp / (tp + fp + EPS),  # same as precision
        "npv": tn / (tn + fn + EPS),
    }


def compute_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency weights for binary classes.
    Uses df['binary_idx'] and assumes exactly two classes (0 and 1).
    """
    freq = df["binary_idx"].value_counts(normalize=True).sort_index()
    if freq.size != 2:
        raise ValueError("Expected two classes, got " + str(freq.to_dict()))
    return torch.tensor([1.0 / freq[0], 1.0 / freq[1]], dtype=torch.float32, device=device)


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
) -> dict[str, float]:
    """
    Execute a single epoch with a tqdm progress bar.
    - If optimiser is provided -> training mode, gradients ON.
    - If optimiser is None     -> evaluation mode, gradients OFF.
    """
    training = optimiser is not None
    model.train(training)

    total_loss, preds, trues = 0.0, [], []

    # Create a description for the progress bar
    desc = f"Epoch {epoch_num:02d} ({split_name.capitalize()})"
    progress_bar = tqdm(dataloader, desc=desc, leave=True)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss += loss.item() * labels.size(0)
            preds.append(logits.detach().argmax(1).cpu())
            trues.append(labels.cpu())

    y_pred = torch.cat(preds)
    y_true = torch.cat(trues)
    n = len(dataloader.dataset)

    metrics = metrics_binary(y_true, y_pred)
    metrics["loss"] = total_loss / n
    return metrics


# ----------------------------
# Utility to create run folder
# ----------------------------


def _setup_run_dir(metrics_dir: Path, suffix: str = "") -> Path:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{ts}_{suffix}" if suffix else ts
    run_dir = metrics_dir / name
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
) -> None:
    """
    Train all models in `models` on a single dataset using one of three modes:

    balance_mode ∈ {
        "weighted_loader" : WeightedRandomSampler + unweighted CrossEntropyLoss
        "weighted_loss"   : standard loaders + class-weighted CrossEntropyLoss
        "none"            : standard loaders + unweighted CrossEntropyLoss
    }

    `df` must contain:
      - 'path'
      - 'binary_idx'
      - 'fold'

    Optionally:
      - 'split' ∈ {"train","train_dev","test"}; only train/train_dev are used for training.
    """
    assert balance_mode in {"weighted_loader", "weighted_loss", "none"}

    print(f"\n{'=' * 70}")
    print(f"DATASET: {name.upper()}  |  balance_mode = {balance_mode}")
    print(f"{'=' * 70}")

    # -------------------------------
    # Restrict to training portion if split exists
    # -------------------------------
    if "split" in df.columns:
        train_df = df[df["split"].isin(["train", "train_dev"])].reset_index(drop=True)
        print(f"Using rows with split in ('train','train_dev') for training (rows: {len(train_df)})")
    else:
        train_df = df.reset_index(drop=True)
        print(f"No 'split' column found; using all {len(train_df)} rows for training.")

    # -------------------------------
    # Show class distribution in train_df
    # -------------------------------
    class_dist = train_df["binary_idx"].value_counts().sort_index()
    total = len(train_df)
    print(f"\nClass Distribution (training subset):")
    print(f"  Class 0 (Normal):   {class_dist.get(0, 0):>6} samples ({100 * class_dist.get(0, 0) / total:5.1f}%)")
    if 1 in class_dist.index:
        print(f"  Class 1 (Abnormal): {class_dist.get(1, 0):>6} samples ({100 * class_dist.get(1, 0) / total:5.1f}%)")
        if class_dist.get(0, 0) > 0 and class_dist.get(1, 0) > 0:
            imbalance_ratio = max(class_dist[0], class_dist[1]) / min(class_dist[0], class_dist[1])
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

    # -------------------------------
    # Init CSV logs (epoch + summary)
    # -------------------------------
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    epoch_cols = [
        "dataset",
        "model",
        "origin",
        "epoch",
        "split",
        "loss",
        "acc",
        "prec",
        "rec",
        "spec",
        "f1",
        "ppv",
        "npv",
        "lr",
        "seconds",
    ]
    pd.DataFrame(columns=epoch_cols).to_csv(run_dir / "epoch_logs.csv", index=False)

    summary_cols = [
        "dataset",
        "model",
        "best_epoch",
        "best_acc",
        "best_prec",
        "best_rec",
        "best_spec",
        "best_f1",
        "best_ppv",
        "best_npv",
        "fold",
    ]
    pd.DataFrame(columns=summary_cols).to_csv(run_dir / "summary.csv", index=False)

    if scheduler_milestones is None:
        scheduler_milestones = SCHEDULER_MILESTONES

    # -------------------------------
    # Loop over models and folds
    # -------------------------------
    for friendly_name, backbone_id in models.items():
        for fold in range(num_folds):
            print(f"\n> {name} | {friendly_name} | fold {fold}")

            # --------- 1) loaders + criterion based on balance_mode ---------
            if balance_mode == "weighted_loader":
                # WeightedRandomSampler + unweighted CE
                train_loader, val_loader = get_loaders_weighted(
                    df=train_df,
                    fold=fold,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
                criterion = nn.CrossEntropyLoss()

            else:
                # Standard loaders
                train_loader, val_loader = get_loaders(
                    df=train_df,
                    fold=fold,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                )

                if balance_mode == "weighted_loss":
                    # compute class weights on TRAIN part of this fold only
                    train_mask = train_df["fold"] != fold
                    train_fold_df = train_df.loc[train_mask]
                    class_weights = compute_class_weights(train_fold_df, device=device)
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    print(f"  Fold {fold}: class weights = {class_weights.tolist()}")
                else:
                    criterion = nn.CrossEntropyLoss()

            # --------- 2) model, optimiser, scheduler ---------
            model, _, origin = load_any(backbone_id, num_classes=2, pretrained=True)
            model.to(device)

            optimiser = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            scheduler = MultiStepLR(
                optimiser, milestones=scheduler_milestones, gamma=scheduler_gamma
            )

            best_val = {k: 0.0 for k in ["acc", "prec", "rec", "spec", "f1", "ppv", "npv"]}
            best_val["epoch"] = 0

            # --------- 3) epoch loop ---------
            for epoch in range(1, epochs + 1):
                t0 = time.time()

                train_m = _run_epoch(train_loader, model, criterion, epoch, "train", optimiser)
                val_m = _run_epoch(val_loader, model, criterion, epoch, "val")

                scheduler.step()
                duration = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]

                # per-epoch logging
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
                    pd.DataFrame([row_data]).to_csv(
                        run_dir / "epoch_logs.csv",
                        mode="a",
                        header=False,
                        index=False,
                    )

                # track best val acc
                if val_m["acc"] > best_val["acc"]:
                    best_val.update(val_m)
                    best_val["epoch"] = epoch
                    torch.save(
                        model.state_dict(),
                        run_dir / "checkpoints" / f"{name}_{backbone_id}_best_{fold}.pt",
                    )

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
                run_dir / "summary.csv",
                mode="a",
                header=False,
                index=False,
            )

            # hygiene
            del model, optimiser, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nFinished training on dataset {name}. Logs at: {run_dir}")


# ----------------------------
# Main script
# ----------------------------


def main():
    print("\n" + "=" * 70)
    print("FLEXIBLE TRAINING SCRIPT")
    print("=" * 70)
    print("Modes:")
    print("  - weighted_loader : WeightedRandomSampler + unweighted loss")
    print("  - weighted_loss   : standard loaders + weighted loss")
    print("  - none            : standard loaders + unweighted loss")
    print("=" * 70)
    print(f"BALANCE_MODE = {BALANCE_MODE}")
    print("=" * 70 + "\n")

    names = ["apacc", "herlev", "sipakmed"]
    roots = [
        Path("./datasets/data/apacc"),
        Path("./datasets/data/smear2005"),
        Path("./datasets/data/sipakmed"),
    ]
    scanners = [scan_apacc, scan_herlev, scan_sipakmed]

    for root, scanner, name in zip(roots, scanners, names):
        # dataset-specific model list if needed
        if name == "apacc":
            models = {f"EfficientNet-B{i}": f"efficientnet_b{i}" for i in range(4)}
        else:
            models = {
                "SqueezeNet 1.1": "tv_squeezenet1_1",
                "MobileNet V2 1.0x": "mobilenetv2_100",
                "MobileNet V4 small": "mobilenetv4_conv_small.e2400_r224_in1k",
                "ShuffleNet V2 1.0x": "tv_shufflenet_v2_x1_0",
                # "GhostNet V3": "ghostnetv3_100.in1k",
                **{f"EfficientNet-B{i}": f"efficientnet_b{i}" for i in range(4)},
            }

        print(f"\nScanning dataset: {name} at {root}")
        df = scanner(root=root, num_folds=NUM_FOLDS, seed=SEED)

        run_dir = _setup_run_dir(METRICS_DIR, suffix=BALANCE_MODE)

        train_dataset(
            name=name,
            df=df,
            run_dir=run_dir,
            models=models,
            balance_mode=BALANCE_MODE,
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
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
