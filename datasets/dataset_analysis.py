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

from cervical_cancer_cnn.datasets.apacc import (
    scan_apacc,
    ApaccDataset,
    train_tf,
    val_tf,
)

def without_normalize(tf: T.Compose) -> T.Compose:
    """Return a copy of a Compose transform with the final Normalize removed."""
    ops = list(tf.transforms)
    if isinstance(ops[-1], T.Normalize):
        ops = ops[:-1]                      # drop it
    return T.Compose(ops)

train_tf_no_norm = without_normalize(train_tf)   # uses your Resize/Crop/Jitter…
val_tf_no_norm   = without_normalize(val_tf)

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

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def mean_std_simple(loader, device="cuda"):
    n_px = 0
    channel_sum = torch.zeros(3, device=device)
    channel_sq  = torch.zeros(3, device=device)

    for batch, _ in loader:
        batch = batch.to(device, non_blocking=True)  # B,C,H,W in 0-1
        b, c, h, w = batch.shape
        n_px += b * h * w
        channel_sum += batch.sum(dim=[0,2,3])
        channel_sq  += (batch ** 2).sum(dim=[0,2,3])

    mean = channel_sum / n_px
    std  = (channel_sq / n_px - mean ** 2).sqrt()
    return mean.cpu(), std.cpu()

def compute_mean_std(
    df_train: pd.DataFrame,
    tf_no_norm: T.Compose,
    batch_size: int = 64,
    num_workers: int = 4,
    max_samples: int | None = None,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One-pass channel-wise mean / std. Uses Welford’s algorithm for numerical stability.
    """
    ds = ApaccDataset(df_train, tf_no_norm)      # or Smear2005BinaryDataset, …
    if max_samples:
        ds = torch.utils.data.Subset(ds, range(max_samples))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    n_pixels = 0
    mean = torch.zeros(3, device=device)
    M2   = torch.zeros(3, device=device)   # sum of squared deviations

    for batch, _ in tqdm(loader, desc="scanning"):
        batch = batch.to(device, non_blocking=True)   # shape B×C×H×W, 0-1
        pixels = batch.numel() // 3                   # B*H*W
        n_pixels += pixels
        batch = batch.view(3, -1)                     # C × (B*H*W)

        # incremental mean / var
        delta = batch.mean(1) - mean
        mean += delta * (pixels / n_pixels)
        M2   += (batch.var(1, unbiased=False) * pixels +
                 (delta ** 2) * (n_pixels - pixels) * pixels / n_pixels)

    var = M2 / n_pixels
    std = torch.sqrt(var)

    n_px = 0
    channel_sum = torch.zeros(3, device=device)
    channel_sq  = torch.zeros(3, device=device)

    for batch, _ in loader:
        batch = batch.to(device, non_blocking=True)  # B,C,H,W in 0-1
        b, c, h, w = batch.shape
        n_px += b * h * w
        channel_sum += batch.sum(dim=[0,2,3])
        channel_sq  += (batch ** 2).sum(dim=[0,2,3])

    mean_simple = channel_sum / n_px
    std_simple  = (channel_sq / n_px - mean ** 2).sqrt()
    print('simple mean', mean_simple, 'simple std', std_simple)
    
    return mean.cpu(), std.cpu()

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
        print(path)
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
    
    ROOT = Path("./datasets/apacc")  
    NUM_FOLDS, SEED = 5, 42

    df = scan_apacc(ROOT, num_folds=NUM_FOLDS, seed=SEED)

    # 6-A  class balance
    report_class_balance(df, NUM_FOLDS)

    # 6-B  transform visualisation
    show_transforms(df.query("split == 'train'"), n=5, seed=SEED)

    train_df = df.query("split == 'train'")
    mean, std = compute_mean_std(train_df, train_tf_no_norm, batch_size=128)
    print("channel-wise mean :", mean.tolist())
    print("channel-wise std  :", std.tolist())
    # 6-C  image-size histogram
    #plot_size_distribution(df, max_images=None)  # set to e.g. 2000 for speed

    # 6-D  misc sanity checks
    #filetype_breakdown(df)
    #check_duplicates(df, max_bytes=2**18)   # 256 KB


if __name__ == "__main__":
    main()