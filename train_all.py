# misc
from typing import Optional, Tuple, OrderedDict

# Data management
import os, random, csv, warnings, datetime, time
import pandas as pd
import numpy as np

from sipakmed import (
    scan_sipakmed,
    compute_class_weights,
    get_sipakmed_loaders,
)

from herlev import (
    scan_herlev,
    get_herlev_loaders
)

from apacc import (
    scan_apacc,
    get_apacc_loaders
)

from model_utils import load_any

# Models
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# Metrics
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
)
import gc
from pathlib import Path

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

EPS = 1e-9

TODO = {
    "SqueezeNet 1.1": "tv_squeezenet1_1",
    "MobileNet V2 1.0x": "mobilenetv2_100",
    "MobileNet V4 small": "mobilenetv4_conv_small.e2400_r224_in1k",
    "ShuffleNet V2 1.0x": "tv_shufflenet_v2_x1_0",
    "GhostNet V3": "hf_hub:timm/ghostnetv3_100.in1k",
    **{f"EfficientNet-B{i}":f"efficientnet_b{i}" for i in range(4)},}

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
        "prec": precision_score(y_true, y_pred),  # PPV
        "rec": recall_score(y_true, y_pred),  # Sensitivity
        "spec": tn / (tn + fp + EPS),
        "f1": f1_score(y_true, y_pred),
        "ppv": tp / (tp + fp + EPS),  # same as precision
        "npv": tn / (tn + fn + EPS),
    }


def compute_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency weights for binary classes.
    """
    freq = df["binary_idx"].value_counts(normalize=True).sort_index()
    if freq.size != 2:
        raise ValueError("Expected two classes, got " + str(freq.to_dict()))
    return torch.tensor([1 / freq[0], 1 / freq[1]], dtype=torch.float32, device=device)


# ----------------------------
# Epoch runner
# ----------------------------


def _run_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimiser: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Execute a single epoch.
    - If optimiser is provided -> training mode, gradients ON.
    - If optimiser is None     -> evaluation mode, gradients OFF.

    Returns
    -------
    (avg_loss, accuracy, f1, recall, specificity)
    """
    training = optimiser is not None
    model.train(training)

    total_loss, preds, trues = 0.0, [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            model = model.to(DEVICE)
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
# Utility to create run folder & logger
# ----------------------------


def _setup_run_dir(metrics_dir: Path) -> Path:
    run_dir = metrics_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _init_csv_logger(run_dir: Path):
    path = run_dir / "epoch_logs.csv"
    f = path.open("w", newline="")
    w = csv.writer(f)
    w.writerow(
        [
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
    )
    return f, w


# ----------------------------
# Main script
# ----------------------------


def main():
    # ---------------- roots / scanners / loaders ----------------
    names = ["apacc","herlev","sipakmed"]
    roots = [".\\datasets\\apacc", ".\\datasets\\herlev", ".\\datasets\\sipakmed"]
    scanners = [scan_apacc, scan_herlev, scan_sipakmed]
    get_loaders = [get_apacc_loaders, get_herlev_loaders, get_sipakmed_loaders]

    for root, scanner, get_loader, name in zip(roots, scanners, get_loaders,names):
        df = scanner(root, NUM_FOLDS, SEED)

        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(df, DEVICE))

        print(f"Using device: {DEVICE}")
        if torch.cuda.is_available():
            print(" GPU:", torch.cuda.get_device_name())

        run_dir = _setup_run_dir(METRICS_DIR)
        log_file, _ = _init_csv_logger(run_dir)
        history_rows = []
        summary_rows = []

        for friendly_name, backbone_id in TODO.items():
            for fold in range(NUM_FOLDS):
                print(f"\n> {name} | {friendly_name} | fold {fold}")
                train_loader, val_loader = get_loader(
                    df, fold, BATCH_SIZE, NUM_WORKERS, torch.cuda.is_available()
                )

                model, _, origin = load_any(backbone_id, num_classes=2, pretrained=True)
                model.to(DEVICE)

                optimiser = torch.optim.SGD(
                    model.parameters(),
                    lr=LR,
                    momentum=MOMENTUM,
                    weight_decay=WEIGHT_DECAY,
                )
                scheduler = MultiStepLR(
                    optimiser, milestones=SCHEDULER_MILESTONES, gamma=SCHEDULER_GAMMA
                )

                best_val = {
                    k: 0.0 for k in ["name","acc", "prec", "rec", "spec", "f1", "ppv", "npv"]
                }
                best_val["epoch"] = 0

                for epoch in range(1, EPOCHS + 1):
                    t0 = time.time()
                    train_m = _run_epoch(train_loader, model, criterion, optimiser)
                    val_m = _run_epoch(val_loader, model, criterion)
                    scheduler.step()
                    duration = time.time() - t0
                    lr_now = scheduler.get_last_lr()[0]

                    for split_name, m in [("train", train_m), ("val", val_m)]:
                        history_rows.append(
                            OrderedDict(
                                dataset=name
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
                        )

                    if val_m["acc"] > best_val["acc"]:
                        best_val.update(val_m)
                        best_val["epoch"] = epoch
                        torch.save(
                            model.state_dict(),
                            run_dir / "checkpoints" / f"{name}_{backbone_id}_best_{fold}.pt",
                        )

                    if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
                        print(
                            f"Ep{epoch:02d} loss {val_m['loss']:.4f} acc {val_m['acc']:.3f} f1 {val_m['f1']:.3f} (duration {duration:.1f}s)"
                        )

                summary_rows.append(
                    [
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
                )

                # hygiene: free GPU memory between folds
                del model, optimiser, scheduler, train_loader, val_loader
                torch.cuda.empty_cache()
                gc.collect()

        # ----------------------------------
        # Persist logs and summary to disk
        # ----------------------------------
        pd.DataFrame(history_rows).to_csv(
            run_dir / "epoch_logs.csv",
            index=False,
            columns=[
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
            ],
        )

        pd.DataFrame(
            summary_rows,
            columns=[
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
            ],
        ).to_csv(run_dir / "summary.csv", index=False)

        log_file.close()


if __name__ == "__main__":
    main()
