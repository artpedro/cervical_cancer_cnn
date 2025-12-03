from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from datasets import (
    scan_sipakmed,
    scan_herlev,
    scan_apacc,
    HERLEV_TRAIN_TF,
    HERLEV_VAL_TF,
    SIPAKMED_TRAIN_TF,
    SIPAKMED_VAL_TF,
    APACC_TRAIN_TF,
    APACC_VAL_TF,
    get_loaders_weighted,
)


def without_normalize(tf: T.Compose) -> T.Compose:
    """Return a copy of a Compose transform with the final Normalize removed."""
    ops = list(tf.transforms)
    if isinstance(ops[-1], T.Normalize):
        ops = ops[:-1]  # drop it
    return T.Compose(ops)


# --------------------------------------------------------------------------
#  1.  Textual class-balance report
# --------------------------------------------------------------------------
def report_class_balance(df: pd.DataFrame, num_folds: int) -> None:
    """
    Print overall and per-fold class balance (counts and percentages)
    for binary_idx / binary_label.
    """
    # Consistent class order
    classes = sorted(df["binary_idx"].unique())
    # If you want human-readable names:
    label_names = (
        df[["binary_idx", "binary_label"]]
        .drop_duplicates()
        .set_index("binary_idx")["binary_label"]
        .to_dict()
    )

    def format_counts(series: pd.Series) -> str:
        """Return 'cls_name: count (xx.x%)' joined by ' | '."""
        total = int(series.sum())
        if total == 0:
            return "no samples"
        parts = []
        for cls in classes:
            count = int(series.get(cls, 0))
            pct = 100.0 * count / total if total > 0 else 0.0
            cls_name = label_names.get(cls, str(cls))
            parts.append(f"{cls_name}={count:4d} ({pct:5.1f}%)")
        return " | ".join(parts)

    print("\n=== Class balance =================================================")

    # ---- TOTAL ----
    total_counts = df["binary_idx"].value_counts().reindex(classes, fill_value=0)
    print(f"TOTAL (N={len(df)}):")
    print("  " + format_counts(total_counts))

    # ---- PER FOLD ----
    for f in range(num_folds):
        train_counts = (
            df.query("fold != @f")["binary_idx"]
            .value_counts()
            .reindex(classes, fill_value=0)
        )
        val_counts = (
            df.query("fold == @f")["binary_idx"]
            .value_counts()
            .reindex(classes, fill_value=0)
        )

        n_train = int(train_counts.sum())
        n_val = int(val_counts.sum())

        print(f"\nFold {f}:")
        print(f"  train (N={n_train:4d}): {format_counts(train_counts)}")
        print(f"  val   (N={n_val:4d}): {format_counts(val_counts)}")
    print("==================================================================")


def inspect_batch_class_ratios(
    loader: DataLoader,
    dataset_name: str,
    split: Literal["train", "val"],
    n_batches: int = 5,
) -> None:
    """
    Quickly print class ratios for the first `n_batches` of a loader.
    This is cheap and gives a good sense of whether sampling is working.
    """
    total_counts: Counter[int] = Counter()
    total_samples = 0

    print(
        f"\n=== {dataset_name.upper()} [{split}] batch-wise class ratios "
        f"(first {n_batches} batches) ==="
    )

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_batches:
            break

        # Assumes batch is (images, labels, ...)
        images, labels = batch[0], batch[1]

        # Move labels to CPU numpy for counting
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = labels  # in case they are already numpy/ list

        batch_counts = Counter(labels_np)
        batch_size = len(labels_np)

        total_counts.update(batch_counts)
        total_samples += batch_size

        # Print per-batch ratio
        print(f"Batch {batch_idx:02d}: ", end="")
        for cls in sorted(batch_counts.keys()):
            cnt = batch_counts[cls]
            ratio = cnt / batch_size
            print(f"cls {cls}: {cnt:3d} ({ratio:6.2%})  ", end="")
        print()

    # Aggregated over inspected batches
    print(f"\nAggregated over {min(n_batches, len(loader))} batches:")
    for cls in sorted(total_counts.keys()):
        cnt = total_counts[cls]
        ratio = cnt / total_samples
        print(f"  cls {cls}: {cnt:4d} ({ratio:6.2%})")
    print()


def show_one_batch_grid(
    loader: DataLoader,
    dataset_name: str,
    split: str,
    max_images: int = 16,
) -> None:
    """
    Show a small grid of images and their labels from the first batch
    of the given loader.
    """
    batch = next(iter(loader))
    images, labels = batch[0], batch[1]

    # Convert to CPU
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    n = min(max_images, images.shape[0])
    rows = int(math.sqrt(n))
    cols = math.ceil(n / rows)

    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i in range(n):
        img = images[i]
        # assume (C, H, W), unnormalize very roughly if needed
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = img.numpy()

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_np.squeeze(), cmap="gray" if img_np.ndim == 2 else None)
        plt.axis("off")
        plt.title(f"y={labels[i]}")
    plt.suptitle(f"{dataset_name.upper()} [{split}] – first batch samples")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
#  2.  Main
# --------------------------------------------------------------------------
def main() -> None:
    NUM_FOLDS, SEED = 5, 42

    names = ["sipakmed", "herlev", "apacc"]

    roots = [
        Path("./datasets/data/sipakmed"),
        Path("./datasets/data/smear2005"),
        Path("./datasets/data/apacc"),
    ]

    scanners = [scan_sipakmed, scan_herlev, scan_apacc]

    all_tfs = [
        (HERLEV_TRAIN_TF, HERLEV_VAL_TF),
        (SIPAKMED_TRAIN_TF, SIPAKMED_VAL_TF),
        (APACC_TRAIN_TF, APACC_VAL_TF),
    ]

    for name, root, scanner, tfs in zip(names, roots, scanners, all_tfs):
        print(f"\n==================== {name.upper()} ====================")

        df = scanner(root, num_folds=NUM_FOLDS, seed=SEED)

        # 1) class balance
        report_class_balance(df, NUM_FOLDS)
        train_tf, val_tf = tfs

        train_tf_no_norm = without_normalize(train_tf)  # uses your Resize/Crop/Jitter…
        val_tf_no_norm = without_normalize(val_tf)

        train_loader, val_loader = get_loaders_weighted(
            df,
            fold=0,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            train_tf=train_tf_no_norm,
            val_tf=val_tf_no_norm,
        )

        # 2) QUICK NUMERIC CHECK: batch-wise class ratios
        inspect_batch_class_ratios(
            train_loader, dataset_name=name, split="train", n_batches=5
        )
        inspect_batch_class_ratios(
            val_loader, dataset_name=name, split="val", n_batches=5
        )

        # 3) OPTIONAL VISUAL CHECK: show one batch grid
        show_one_batch_grid(
            train_loader, dataset_name=name, split="train", max_images=16
        )
        show_one_batch_grid(
            val_loader, dataset_name=name, split="val", max_images=16
        )


if __name__ == "__main__":
    main()