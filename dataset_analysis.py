from __future__ import annotations
import hashlib, random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm

from apacc import (
    scan_apacc,
    ApaccDataset,
    train_tf,
    val_tf,
)


# --------------------------------------------------------------------------
#  1.  Textual class-balance report
# --------------------------------------------------------------------------
def report_class_balance(df: pd.DataFrame, num_folds: int) -> None:
    """Print overall and per-fold counts."""
    def pretty(series: pd.Series) -> str:
        return " / ".join(f"{cls}:{n}" for cls, n in series.items())

    print("\n=== Class balance ------------------------------------------------")
    total = df.binary_idx.value_counts().sort_index()
    print(f"TOTAL      : {pretty(total)}")

    test = df.query("split == 'test'").binary_idx.value_counts().sort_index()
    print(f"TEST split : {pretty(test)}")

    trainval = df.query("split == 'train'")
    for f in range(num_folds):
        train = trainval.query("fold != @f").binary_idx.value_counts().sort_index()
        val   = trainval.query("fold == @f").binary_idx.value_counts().sort_index()
        print(f"Fold {f}  train: {pretty(train):15s}  |  val: {pretty(val)}")


# --------------------------------------------------------------------------
#  2.  Transform visualisation helper
# --------------------------------------------------------------------------
def show_transforms(df: pd.DataFrame, n: int = 4, seed: int = 0) -> None:
    """Randomly pick n images and visualise original vs. train_tf vs. val_tf."""
    rng = random.Random(seed)
    sample_paths = rng.sample(df.path.tolist(), n)

    ncols = 3
    fig, axes = plt.subplots(n, ncols, figsize=(4.5 * ncols, 4.5 * n))
    titles = ["original", "train_tf", "val_tf"]

    for r, path in enumerate(sample_paths):
        img_pil = Image.open(path).convert("RGB")
        imgs = [
            img_pil,
            train_tf(img_pil).permute(1, 2, 0).numpy(),  # CHW → HWC
            val_tf(img_pil).permute(1, 2, 0).numpy(),
        ]
        for c, im in enumerate(imgs):
            ax = axes[r, c] if n > 1 else axes[c]
            ax.imshow(np.clip(im, 0, 1))
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=14)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
#  3.  Image-size distribution
# --------------------------------------------------------------------------
def plot_size_distribution(df: pd.DataFrame, max_images: Optional[int] = None) -> None:
    """Histogram width, height, area."""
    paths = df.path.tolist()
    if max_images:
        paths = random.sample(paths, max_images)

    widths, heights = [], []
    for p in tqdm(paths, desc="Reading sizes"):
        with Image.open(p) as im:
            w, h = im.size
            widths.append(w)
            heights.append(h)
    widths, heights = np.array(widths), np.array(heights)
    areas = widths * heights

    # 3 histograms
    fig, ax = plt.subplots(3, 1, figsize=(7, 9))
    ax[0].hist(widths, bins=30);   ax[0].set_title("Width distribution")
    ax[1].hist(heights, bins=30);  ax[1].set_title("Height distribution")
    ax[2].hist(areas, bins=30);    ax[2].set_title("Area (W×H) distribution")
    plt.tight_layout(); plt.show()

    print(f"Median WxH: {np.median(widths)} × {np.median(heights)}")
    print(f"Min WxH   : {widths.min()} × {heights.min()}")
    print(f"Max WxH   : {widths.max()} × {heights.max()}")


# --------------------------------------------------------------------------
#  4.  Duplicate-file check (hash based)
# --------------------------------------------------------------------------
def check_duplicates(df: pd.DataFrame, max_bytes: int = 2**20) -> None:
    """
    Spot potential duplicates by hashing the *first* `max_bytes` of each file.
    Cheap but usually enough for image sets.
    """
    print("\n=== Duplicate file check ----------------------------------------")
    hashes = {}
    dups = []
    for path in tqdm(df.path, desc="Hashing"):
        h = hashlib.sha1(open(path, "rb").read(max_bytes)).hexdigest()
        if h in hashes:
            dups.append((hashes[h], path))
        else:
            hashes[h] = path
    if not dups:
        print("No duplicates detected (within sampled bytes).")
        return
    print(f"⚠️  Found {len(dups)} potential duplicates (SHA-1 on first {max_bytes//1024} KB).")
    for a, b in dups[:10]:
        print("   ", a, "<---->", b)
    if len(dups) > 10:
        print("   …")


# --------------------------------------------------------------------------
#  5.  File-type distribution
# --------------------------------------------------------------------------
def filetype_breakdown(df: pd.DataFrame) -> None:
    counts = df.path.apply(lambda p: Path(p).suffix.lower()).value_counts()
    print("\n=== File-type distribution --------------------------------------")
    for ext, n in counts.items():
        print(f"{ext or '[no ext]':>8s} : {n}")


# --------------------------------------------------------------------------
#  6.  Main
# --------------------------------------------------------------------------
def main() -> None:
    
    ROOT = Path("../datasets/apacc_small")   # ←—— change me
    NUM_FOLDS, SEED = 5, 42

    df = scan_apacc(ROOT, num_folds=NUM_FOLDS, seed=SEED)

    # 6-A  class balance
    report_class_balance(df, NUM_FOLDS)

    # 6-B  transform visualisation
    show_transforms(df.query("split == 'train'"), n=5, seed=SEED)

    # 6-C  image-size histogram
    plot_size_distribution(df, max_images=None)  # set to e.g. 2000 for speed

    # 6-D  misc sanity checks
    filetype_breakdown(df)
    check_duplicates(df, max_bytes=2**18)   # 256 KB


if __name__ == "__main__":
    main()