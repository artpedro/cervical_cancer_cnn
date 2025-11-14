from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image

# Data transforms
train_tf = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomRotation(
            degrees=180,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=(255, 255, 255),
        ),  # white
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.0),
        T.ToTensor(),
        T.Normalize([0.7467415090755142, 0.6917098472446087, 0.7182460675482442],
                    [0.2690863277058493, 0.27271077679411904, 0.2635063080105664]),
    ]
)

val_tf = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.7467415090755142, 0.6917098472446087, 0.7182460675482442],
                    [0.2690863277058493, 0.27271077679411904, 0.2635063080105664]),
    ]
)


# Dataset class
class ApaccDataset(Dataset):
    """
    Returns (image_tensor, binary_idx) for apacc (healthy vs. unhealthy).
    """

    def __init__(self, df: pd.DataFrame, tf):
        self.df, self.tf = df.reset_index(drop=True), tf

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = Image.open(self.df.path[idx]).convert("RGB")
        return self.tf(img), int(self.df.binary_idx[idx])


# Scanning and fold assignment
def scan_apacc(root: Path, *, num_folds: int, seed: int) -> pd.DataFrame:
    """
    Walk root/train and root/test, build DataFrame with:
      path · label_full · binary_idx · split · fold
    `split` ∈ {"trainval", "test"}; `fold` ∈ [0..num_folds-1] inside trainval only.
    """
    records = []
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.exists():
            raise RuntimeError(f"Missing directory: {split_dir}")
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = cls_dir.name.lower()  # "healthy" or "unhealthy"
            for img_path in cls_dir.glob("**/*"):
                records.append((str(img_path), label, split))

    df = pd.DataFrame(records, columns=["path", "label_full", "split"])

    # Map to binary index: healthy=0, unhealthy=1
    df["binary_idx"] = df["label_full"].map({"healthy": 0, "unhealthy": 1})
    if df.binary_idx.isna().any():
        bad = df[df.binary_idx.isna()].label_full.unique().tolist()
        raise ValueError(f"Unknown classes: {bad}")

    # Prepare a placeholder for folds
    df["fold"] = -1

    # Stratified K-fold *inside* the trainval portion
    trainval_mask = df.split == "train"
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    trainval_idx = df.index[trainval_mask].to_numpy()
    trainval_labels = df.loc[trainval_mask, "binary_idx"].to_numpy()
    for fold, (_, val_inds) in enumerate(skf.split(trainval_idx, trainval_labels)):
        # val_inds are indices into trainval_idx array
        real_idx = trainval_idx[val_inds]
        df.loc[real_idx, "fold"] = fold

    return df


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
    
    Example
    -------
    If dataset has 400 healthy (class 0) and 600 unhealthy (class 1):
    - Each healthy sample gets weight: 1/400 = 0.0025
    - Each unhealthy sample gets weight: 1/600 = 0.00167
    
    Result: Healthy samples are 1.5x more likely to be selected per batch
    """
    # Count how many samples in each class
    class_counts = df["binary_idx"].value_counts().sort_index()
    
    # Compute weight for each class (inverse frequency)
    class_weights = 1.0 / class_counts.values
    
    # Assign weight to each sample based on its class
    sample_weights = np.array([class_weights[label] for label in df["binary_idx"]])
    
    return sample_weights


# Get loaders for APACC dataset with WeightedRandomSampler
def get_apacc_loaders_weighted(
    df: pd.DataFrame,
    fold: int,
    *,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) for the chosen `fold`.
    
    KEY DIFFERENCE FROM ORIGINAL:
    ==========================
    Training loader uses WeightedRandomSampler to oversample minority class.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with fold assignments
    fold : int
        Which fold to use for validation (0 to k-1)
    batch_size : int
        Batch size
    num_workers : int
        Number of data loading workers
    pin_memory : bool
        Whether to pin memory for faster GPU transfer
    
    Returns
    -------
    train_loader : DataLoader
        Training data loader WITH WeightedRandomSampler
    val_loader : DataLoader
        Validation data loader (standard sampling)
    """
    # For APACC, filter only training split (fold != -1)
    train_df = df[(df.fold != fold) & (df.fold != -1)].reset_index(drop=True)
    val_df = df[df.fold == fold].reset_index(drop=True)
    
    # Compute sample weights for training set only
    train_weights = compute_sample_weights(train_df)
    
    # Create WeightedRandomSampler
    # replacement=True allows same sample to appear multiple times in one epoch
    # This is essential for balancing when minority class is much smaller
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),  # Draw this many samples per epoch
        replacement=True  # Allow resampling
    )
    
    # Training loader with WeightedRandomSampler
    # NOTE: shuffle=False when using a sampler (sampler controls order)
    train_loader = DataLoader(
        ApaccDataset(train_df, train_tf),
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle=True
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    # Validation loader remains unchanged (no resampling needed for evaluation)
    val_loader = DataLoader(
        ApaccDataset(val_df, val_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
