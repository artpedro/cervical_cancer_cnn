import os
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T


# ============================================================
# LABELING
# ============================================================

NORMAL = {
    "Superficial-Intermediate",
    "Parabasal",  # Sipakmed
    "normal_columnar",
    "normal_intermediate",
    "normal_superficiel",  # Herlev
}

ABNORMAL = {
    "Koilocytotic",
    "Dyskeratotic",
    "Metaplastic",  # Sipakmed
    "light_dysplastic",
    "moderate_dysplastic",
    "severe_dysplastic",
    "carcinoma_in_situ",  # Herlev
}

# ============================================================
# NORMALIZATION CONFIG
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Path to JSON file where per-dataset normalization stats are stored
NORM_STATS_PATH = Path("./datasets/normalization_stats.json")

# ============================================================
# TRANSFORM BUILDERS
# ============================================================

def make_tf(normalizing_matriz: Optional[List[List[float]]] = None) -> tuple[T.Compose, T.Compose]:
    """
    Create (train_tf, eval_tf) transforms.

    Parameters
    ----------
    normalizing_matriz : list[list[float]] or None
        [[meanR, meanG, meanB], [stdR, stdG, stdB]].
        If None, falls back to IMAGENET_MEAN/STD.

    Returns
    -------
    train_tf : torchvision.transforms.Compose
    eval_tf  : torchvision.transforms.Compose
    """
    if normalizing_matriz is None:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = normalizing_matriz

    train_tf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.RandomRotation(
                degrees=180,
                interpolation=T.InterpolationMode.BILINEAR,
                fill=(255, 255, 255),
            ),  # white background
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.2, 0.2, 0.2, 0.0),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    eval_tf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    return train_tf, eval_tf


def compute_mean_std_for_df(
    df: pd.DataFrame,
    resize: int = 256,
    crop_size: int = 224,
    max_samples: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[List[float], List[float]]:
    """
    Compute per-channel mean and std for a subset of images defined by df.
    Assumes df has a 'path' column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'path' column with image file paths.
    resize : int
        Resize shorter side to this before center crop.
    crop_size : int
        Center crop size.
    max_samples : int or None
        If set, only use the first `max_samples` images (for speed).
    show_progress : bool
        Whether to display a tqdm progress bar while iterating images.

    Returns
    -------
    mean : list of 3 floats
    std  : list of 3 floats
    """
    tf = T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(crop_size),
            T.ToTensor(),  # [0,1], shape (C,H,W)
        ]
    )

    n_channels = 3
    channel_sum = torch.zeros(n_channels, dtype=torch.float64)
    channel_sq_sum = torch.zeros(n_channels, dtype=torch.float64)
    n_pixels = 0

    paths = df["path"].tolist()
    if max_samples is not None:
        paths = paths[:max_samples]

    if show_progress:
        iterator = tqdm(paths, desc="Computing mean/std", unit="img")
    else:
        iterator = paths

    for p in iterator:
        img = Image.open(p).convert("RGB")
        x = tf(img)  # (C,H,W)

        c, h, w = x.shape
        assert c == n_channels
        x_flat = x.view(c, -1)

        n_pixels += h * w
        channel_sum += x_flat.sum(dim=1).double()
        channel_sq_sum += (x_flat ** 2).sum(dim=1).double()

    if n_pixels == 0:
        raise RuntimeError("No pixels accumulated; check your DataFrame / paths.")

    mean = channel_sum / n_pixels
    var = (channel_sq_sum / n_pixels) - mean ** 2
    std = torch.sqrt(var.clamp_min(1e-12))

    return mean.tolist(), std.tolist()



def get_or_compute_norm_stats(
    df: pd.DataFrame,
    dataset_name: str,
    stats_path: Path,
    *,
    split: str = "train_dev",
    max_samples_per_split: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    """
    Load mean/std for (dataset_name, split) from JSON file, or compute and save if missing.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset DataFrame, must have columns: 'path' and 'split'.
    dataset_name : str
        Name key in the JSON (e.g. "herlev", "sipakmed", "apacc").
    stats_path : Path
        Path to JSON stats file (will be created/updated).
    split : str
        Which split to base normalization on (typically "train_dev").
    max_samples_per_split : int or None
        If not None, limit number of images used to approximate mean/std.

    Returns
    -------
    mean : list[float]
    std  : list[float]
    """
    stats_path = Path(stats_path)

    # Load existing stats if file exists
    if stats_path.exists():
        with stats_path.open("r") as f:
            stats = json.load(f)
    else:
        stats = {}

    # If stats already exist, just return them
    if (
        dataset_name in stats
        and split in stats[dataset_name]
        and "mean" in stats[dataset_name][split]
        and "std" in stats[dataset_name][split]
    ):
        mean = stats[dataset_name][split]["mean"]
        std = stats[dataset_name][split]["std"]
        return mean, std

    # Otherwise, compute from df[split == split]
    split_df = df[df["split"] == split]
    if split_df.empty:
        raise ValueError(f"No samples found for dataset={dataset_name}, split={split}")

    mean, std = compute_mean_std_for_df(
        split_df,
        max_samples=max_samples_per_split,
    )

    # Update in-memory stats and write back to JSON
    if dataset_name not in stats:
        stats[dataset_name] = {}
    stats[dataset_name][split] = {"mean": mean, "std": std}

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    return mean, std


def make_tf_from_stats(
    df: pd.DataFrame,
    dataset_name: str,
    stats_path: Path,
    *,
    split_for_norm: str = "train_dev",
    max_samples_per_split: Optional[int] = None,
) -> tuple[T.Compose, T.Compose]:
    """
    Build (train_tf, eval_tf) using normalization stats loaded from JSON or
    computed on-the-fly from df.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame with 'path' and 'split' columns.
    dataset_name : str
        Name to use in the JSON stats file.
    stats_path : Path
        Path to JSON stats file.
    split_for_norm : str
        Which split to use when computing normalization (typically 'train_dev').
        Note: train & eval transforms will *share* this normalization,
        which is what you want for zero-shot cross-dataset evaluation.
    max_samples_per_split : int or None
        Limit number of images used to compute stats (for speed). None = all.

    Returns
    -------
    train_tf : torchvision.transforms.Compose
    eval_tf  : torchvision.transforms.Compose
    """
    mean, std = get_or_compute_norm_stats(
        df,
        dataset_name=dataset_name,
        stats_path=stats_path,
        split=split_for_norm,
        max_samples_per_split=max_samples_per_split,
    )
    normalizing_matriz = [mean, std]
    return make_tf(normalizing_matriz=normalizing_matriz)


# ============================================================
# DATASET CLASS
# ============================================================

class PapDataset(Dataset):
    """
    Returns (image_tensor, binary_idx) for pap smear datasets (healthy vs. unhealthy).
    """

    def __init__(self, df: pd.DataFrame, tf: T.Compose):
        self.df = df.reset_index(drop=True)
        self.tf = tf

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = Image.open(self.df.path[idx]).convert("RGB")
        return self.tf(img), int(self.df.binary_idx[idx])


# ============================================================
# SCANNERS (WITH HOLDOUT + FOLDS)
# ============================================================

def scan_herlev(
    root: Path,
    *,
    num_folds: int,
    seed: int,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """
    Walk root/<class>/**/*.BMP, ignoring .bmp,
    assign binary labels via NORMAL / ABNORMAL, then do:

      1) stratified holdout split: split ∈ {"train_dev", "test"}
      2) StratifiedKFold folds only inside train_dev.

    Returns DataFrame with columns:
      - path (str)
      - label_full (str)
      - binary_label ("normal" or "abnormal")
      - binary_idx   (0 or 1)
      - split        ("train_dev" or "test")
      - fold         (0..num_folds-1 for train_dev, -1 for test)
    """
    records = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = cls_dir.name  # e.g. "light_dysplastic"
        if label not in NORMAL and label not in ABNORMAL:
            raise ValueError(
                f"Unknown class folder '{label}' not in NORMAL nor ABNORMAL"
            )
        # glob only uppercase .BMP
        for img_path in cls_dir.glob("**/*.BMP"):
            records.append((str(img_path), label))

    if not records:
        raise RuntimeError(f"No .BMP images found under {root}")

    df = pd.DataFrame(records, columns=["path", "label_full"])

    # map to "normal"/"abnormal"
    df["binary_label"] = df["label_full"].apply(
        lambda x: "normal" if x in NORMAL else "abnormal"
    )
    # 0 = normal, 1 = abnormal
    df["binary_idx"] = df["binary_label"].map({"normal": 0, "abnormal": 1})

    # 1) holdout test split
    df["split"] = "train_dev"
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=seed,
        stratify=df["binary_idx"],
    )
    df.loc[test_idx, "split"] = "test"

    # 2) folds only inside train_dev
    df["fold"] = -1
    train_dev_mask = df["split"] == "train_dev"
    train_dev_idx = df.index[train_dev_mask].to_numpy()
    train_dev_labels = df.loc[train_dev_mask, "binary_idx"].to_numpy()

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold_idx, (_, val_rel_idx) in enumerate(skf.split(train_dev_idx, train_dev_labels)):
        val_idx = train_dev_idx[val_rel_idx]
        df.loc[val_idx, "fold"] = fold_idx

    return df

def scan_sipakmed(
    root: Path,
    *,
    num_folds: int,
    seed: int,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """
    Scan SIPaKMeD and build a DataFrame with cluster-level splits.

    SIPaKMeD filenames follow the pattern 'XXX_YY.bmp', where:
      - XXX = cluster id (resets per class)
      - YY  = cell id within the cluster

    We define a *global* cluster identifier as: f"{label_full}_{XXX}",
    so that cluster ids are unique even though the numeric part resets
    per class.

    Output columns:
      - path         : image path
      - label_full   : original SIPaKMeD class label (e.g. "Superficial-Intermediate")
      - binary_label : "normal" or "abnormal"
      - binary_idx   : 0 (normal) or 1 (abnormal)
      - cluster_id   : string like "Superficial-Intermediate_001"
      - split        : "train_dev" or "test" (cluster-level)
      - fold         : 0..num_folds-1 for train_dev, -1 for test
    """
    records = []

    # ---- 1) Walk the dataset and collect (path, label_full, cluster_id) ----
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        # e.g. "im_Superficial-Intermediate" → "Superficial-Intermediate"
        label = cls_dir.name.replace("im_", "")

        for img_path in cls_dir.glob("**/CROPPED/*.bmp"):
            stem = img_path.stem  # e.g. "001_01"
            parts = stem.split("_")
            if len(parts) < 2:
                raise ValueError(
                    f"Unexpected filename format '{img_path.name}', "
                    "expected something like 'XXX_YY.bmp'."
                )
            simple_cluster_id = parts[0]  # "001"
            # Make cluster_id unique by including the class label
            cluster_id = f"{label}_{simple_cluster_id}"

            records.append((str(img_path), label, cluster_id))

    if not records:
        raise RuntimeError(f"No CROPPED .bmp images found under {root}")

    df = pd.DataFrame(records, columns=["path", "label_full", "cluster_id"])

    # ---- 2) Map to binary labels ----
    df["binary_label"] = df["label_full"].apply(
        lambda x: "normal" if x in NORMAL else "abnormal"
    )
    df["binary_idx"] = df["binary_label"].map({"normal": 0, "abnormal": 1})

    if df["binary_idx"].isna().any():
        bad = df[df["binary_idx"].isna()].label_full.unique().tolist()
        raise ValueError(f"Some labels could not be mapped to NORMAL/ABNORMAL: {bad}")

    # ---- 3) Sanity check: clusters are pure (single class) ----
    cluster_purity = df.groupby("cluster_id")["binary_idx"].nunique()
    if not (cluster_purity == 1).all():
        bad_clusters = cluster_purity[cluster_purity != 1].index.tolist()
        raise ValueError(
            f"Found clusters with mixed binary labels (this should not happen): {bad_clusters}"
        )

    # Build a cluster-level label vector
    cluster_labels = df.groupby("cluster_id")["binary_idx"].first()
    cluster_ids = cluster_labels.index.to_numpy()
    cluster_y = cluster_labels.to_numpy()

    # ---- 4) Cluster-level holdout test split ----
    df["split"] = "train_dev"

    train_clusters, test_clusters = train_test_split(
        cluster_ids,
        test_size=test_size,
        random_state=seed,
        stratify=cluster_y,
    )
    test_clusters = set(test_clusters)

    df.loc[df["cluster_id"].isin(test_clusters), "split"] = "test"

    # ---- 5) Cluster-level K-fold inside train_dev using StratifiedGroupKFold ----
    df["fold"] = -1

    train_dev_mask = df["split"] == "train_dev"
    df_train_dev = df[train_dev_mask]

    # y: per-sample binary label; groups: per-sample cluster_id
    y = df_train_dev["binary_idx"].to_numpy()
    groups = df_train_dev["cluster_id"].to_numpy()

    sgkf = StratifiedGroupKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=seed,
    )

    for fold_idx, (_, val_indices) in enumerate(sgkf.split(X=df_train_dev, y=y, groups=groups)):
        real_idx = df_train_dev.iloc[val_indices].index
        df.loc[real_idx, "fold"] = fold_idx

    return df


def scan_apacc(
    root: Path,
    *,
    num_folds: int,
    seed: int,
) -> pd.DataFrame:
    """
    Walk root/train and root/test, build DataFrame with:
      path · label_full · binary_label · binary_idx · split · fold

    split ∈ {"train_dev","test"}; fold ∈ [0..num_folds-1] inside train_dev only.
    """
    records = []
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.exists():
            raise RuntimeError(f"Missing directory: {split_dir}")
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = cls_dir.name.lower()  # "healthy" or "unhealthy"
            for img_path in cls_dir.glob("**/*"):
                if img_path.is_file():
                    records.append((str(img_path), label, split))

    df = pd.DataFrame(records, columns=["path", "label_full", "split"])

    # Map to binary label/index
    df["binary_label"] = df["label_full"].map(
        {"healthy": "normal", "unhealthy": "abnormal"}
    )
    df["binary_idx"] = df["label_full"].map({"healthy": 0, "unhealthy": 1})
    if df.binary_idx.isna().any():
        bad = df[df.binary_idx.isna()].label_full.unique().tolist()
        raise ValueError(f"Unknown classes: {bad}")

    # Rename "train" → "train_dev" to match others
    df["split"] = df["split"].replace({"train": "train_dev"})

    # Placeholder for folds
    df["fold"] = -1

    # Stratified K-fold *inside* the train_dev portion
    train_dev_mask = df["split"] == "train_dev"
    train_dev_idx = df.index[train_dev_mask].to_numpy()
    train_dev_labels = df.loc[train_dev_mask, "binary_idx"].to_numpy()

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold, (_, val_rel_idx) in enumerate(skf.split(train_dev_idx, train_dev_labels)):
        real_idx = train_dev_idx[val_rel_idx]
        df.loc[real_idx, "fold"] = fold

    return df


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-sample weights for WeightedRandomSampler.

    Each sample gets a weight inversely proportional to its class frequency.
    This ensures balanced sampling during training.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'binary_idx' column

    Returns
    -------
    np.ndarray
        Array of weights, one per sample
    """
    # Count how many samples in each class
    class_counts = df["binary_idx"].value_counts().sort_index()

    # Compute weight for each class (inverse frequency)
    class_weights = 1.0 / class_counts.values

    # Assign weight to each sample based on its class
    sample_weights = np.array([class_weights[label] for label in df["binary_idx"]])

    return sample_weights


# ============================================================
# CREATE TFS FOR EACH DATASET
# ============================================================


sipakmed_root = Path("./datasets/data/sipakmed")
herlev_root = Path("./datasets/data/smear2005")
apacc_root = Path("./datasets/data/apacc")

sipakmed_df = scan_sipakmed(sipakmed_root, num_folds=5, seed=42, test_size=0.2)
herlev_df = scan_herlev(herlev_root, num_folds=5, seed=42, test_size=0.2)
apacc_df = scan_apacc(apacc_root, num_folds=5, seed=42)

# Build transforms from stats file (or compute & save if missing)
SIPAKMED_TRAIN_TF, SIPAKMED_VAL_TF = make_tf_from_stats(
    sipakmed_df,
    dataset_name="sipakmed",
    stats_path=NORM_STATS_PATH,
    split_for_norm="train_dev",
)

HERLEV_TRAIN_TF, HERLEV_VAL_TF = make_tf_from_stats(
    herlev_df,
    dataset_name="herlev",
    stats_path=NORM_STATS_PATH,
    split_for_norm="train_dev",
)

APACC_TRAIN_TF, APACC_VAL_TF = make_tf_from_stats(
    apacc_df,
    dataset_name="apacc",
    stats_path=NORM_STATS_PATH,
    split_for_norm="train_dev",
)

# ============================================================
# LOADERS (ONLY TRAIN_DEV SPLIT)
# ============================================================

def get_loaders_weighted(
    df: pd.DataFrame,
    fold: int,
    batch_size: int,
    pin_memory: bool,
    num_workers: int,
    train_tf: T.Compose,
    val_tf: T.Compose,
) -> tuple[DataLoader, DataLoader]:
    """
    Split df by fold and return (train_loader, val_loader),
    restricted to the 'train_dev' split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with fold assignments and 'split' column.
    fold : int
        Which fold to use for validation (0 to k-1).
    batch_size : int
        Batch size.
    num_workers : int
        Number of data loading workers.
    pin_memory : bool
        Whether to pin memory for faster GPU transfer.

    Returns
    -------
    train_loader : DataLoader
        Training data loader with WeightedRandomSampler.
    val_loader : DataLoader
        Validation data loader (standard sampling).
    """
    # Only work on train_dev portion
    train_dev_df = df[df["split"] == "train_dev"].reset_index(drop=True)

    train_df = train_dev_df[train_dev_df["fold"] != fold].reset_index(drop=True)
    val_df = train_dev_df[train_dev_df["fold"] == fold].reset_index(drop=True)

    # Compute sample weights for training set only
    train_weights = compute_sample_weights(train_df)

    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        PapDataset(train_df, train_tf),
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle=True
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        PapDataset(val_df, val_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def get_loaders(
    df: pd.DataFrame,
    fold: int,
    *,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_tf: T.Compose,
    val_tf: T.Compose,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) for the chosen `fold`,
    restricted to the 'train_dev' split.
    """
    train_dev_df = df[df["split"] == "train_dev"].reset_index(drop=True)

    train_df = train_dev_df[train_dev_df["fold"] != fold].reset_index(drop=True)
    val_df = train_dev_df[train_dev_df["fold"] == fold].reset_index(drop=True)

    train_loader = DataLoader(
        PapDataset(train_df, train_tf),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        PapDataset(val_df, val_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader