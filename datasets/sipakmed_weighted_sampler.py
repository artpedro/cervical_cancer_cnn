import os
from pathlib import Path
import shutil

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image
import numpy as np

import kagglehub

# Class labeling
NORMAL = {"Superficial-Intermediate", "Parabasal"}
ABNORMAL = {"Koilocytotic", "Dyskeratotic", "Metaplastic"}

# Data transforms
train_tf = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomRotation(
            degrees=180,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=(255, 255, 255)),  # white
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.0),
        T.ToTensor(),
        T.Normalize([0.5008344085228984, 0.48421222645755946, 0.5820369775544935],
                    [0.20541780104657734, 0.19316191070964503, 0.21721911661703203]),
    ]
)

val_tf = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.5008344085228984, 0.48421222645755946, 0.5820369775544935],
                    [0.20541780104657734, 0.19316191070964503, 0.21721911661703203]),
    ]
)


# Dataset class
class SipakmedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.df, self.tf = df.reset_index(drop=True), tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.path[idx]).convert("RGB")
        return self.tf(img), int(self.df.binary_idx[idx])


# Scanning and fold assignment
def scan_sipakmed(root: Path, num_folds: int, seed: int) -> pd.DataFrame:
    """
    Walk root/**/CROPPED/*.bmp, build DataFrame with:
      - path, label_full, binary_label, binary_idx, fold
    """
    records = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = cls_dir.name.replace("im_", "")
        for img in cls_dir.glob("**/CROPPED/*.bmp"):
            records.append((str(img), label))

    df = pd.DataFrame(records, columns=["path", "label_full"])
    df["binary_label"] = df["label_full"].apply(
        lambda x: "normal" if x in NORMAL else "abnormal"
    )
    df["binary_idx"] = df["binary_label"].map({"normal": 0, "abnormal": 1})

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for i, (_, val_idx) in enumerate(skf.split(df, df["binary_idx"])):
        df.loc[val_idx, "fold"] = i
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
    If dataset has 800 normal (class 0) and 200 abnormal (class 1):
    - Each normal sample gets weight: 1/800 = 0.00125
    - Each abnormal sample gets weight: 1/200 = 0.005
    
    Result: Abnormal samples are 4x more likely to be selected per batch
    """
    # Count how many samples in each class
    class_counts = df["binary_idx"].value_counts().sort_index()
    
    # Compute weight for each class (inverse frequency)
    class_weights = 1.0 / class_counts.values
    
    # Assign weight to each sample based on its class
    sample_weights = np.array([class_weights[label] for label in df["binary_idx"]])
    
    return sample_weights


# Get loaders for SIPaKMeD dataset with WeightedRandomSampler
def get_sipakmed_loaders_weighted(
    df: pd.DataFrame, 
    fold: int, 
    batch_size: int, 
    num_workers: int, 
    pin_memory: bool
) -> tuple[DataLoader, DataLoader]:
    """
    Split df by fold and return (train_loader, val_loader).
    
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
    train_df = df[df.fold != fold].reset_index(drop=True)
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
        SipakmedDataset(train_df, train_tf),
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle=True
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    # Validation loader remains unchanged (no resampling needed for evaluation)
    val_loader = DataLoader(
        SipakmedDataset(val_df, val_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def download_sipakmed(data_dir: Path) -> Path:
    """
    Download SIPaKMeD from Kaggle and unpack into data_dir/sipakmed.
    Returns the root path.
    """
    # Download dataset from Kaggle
    data_dir.mkdir(parents=True, exist_ok=True)
    final_path = data_dir / "sipakmed"
    if final_path.exists():
        print("Dataset already exists at:", final_path)
        return final_path
    else:
        print("Downloading dataset...")
        slug = "prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed"
        raw_path = kagglehub.dataset_download(
            slug
        )  # downloads + unzips under ~/.cache/kagglehub/…
        print("Downloaded to :", raw_path)

        # Move/rename
        if final_path.exists():
            shutil.rmtree(final_path)
        shutil.copytree(raw_path, final_path)
        print("Dataset ready at:", final_path)

        assert final_path.exists(), f"Missing folder: {final_path}"
        return final_path
