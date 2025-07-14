from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image

# Class labeling
NORMAL = {
    "normal_columnar",
    "normal_intermediate",
    "normal_superficiel",
}

ABNORMAL = {
    "light_dysplastic",
    "moderate_dysplastic",
    "severe_dysplastic",
    "carcinoma_in_situ",
}

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
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_tf = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Dataset class
class HerlevDataset(Dataset):
    """
    Dataset that returns (image_tensor, binary_idx) for herlev.
    """

    def __init__(self, df: pd.DataFrame, tf):
        self.df, self.tf = df.reset_index(drop=True), tf

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = Image.open(self.df.path[idx]).convert("RGB")
        return self.tf(img), int(self.df.binary_idx[idx])


# Scanning and fold assignment
def scan_herlev(root: Path, *, num_folds: int, seed: int) -> pd.DataFrame:
    """
    Walk root/<class>/**/*.BMP, ignoring .bmp,
    assign binary labels via NORMAL / ABNORMAL, then do StratifiedKFold.
    Returns DataFrame with columns:
      - path (str)
      - label_full (str)
      - binary_label ("normal" or "abnormal")
      - binary_idx   (0 or 1)
      - fold         (0..num_folds-1)
    """
    records = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = cls_dir.name  # e.g. "light_dysplastic"
        # ensure this label is known
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

    # Stratified K-fold on the binary_idx
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["binary_idx"])):
        df.loc[val_idx, "fold"] = fold_idx

    return df


# Get loaders for Herlev dataset
def get_herlev_loaders(
    df: pd.DataFrame,
    fold: int,
    *,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) for the chosen `fold`.
    """
    train_df = df[df.fold != fold]
    val_df = df[df.fold == fold]

    train_loader = DataLoader(
        HerlevDataset(train_df, train_tf),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        HerlevDataset(val_df, val_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
