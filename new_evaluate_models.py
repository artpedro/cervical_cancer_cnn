# cross_dataset_evaluate.py
# ==============================================================
# Evaluate every *.pt checkpoint under MODELS_DIR on all datasets
# using the new datasets/datasets.py utilities
# ==============================================================

import os
import gc
import re
from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
)
from tqdm import tqdm

from model_utils import load_any

# New unified dataset utilities
from datasets.datasets import (
    PapDataset,
    sipakmed_df,
    herlev_df,
    apacc_df,
    SIPAKMED_VAL_TF,
    HERLEV_VAL_TF,
    APACC_VAL_TF,
)

# ==============================================================
# CONFIGURATION
# ==============================================================

MODELS_DIR = Path(r"./workspace/metrics")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
SEED = 42
EPS = 1e-9

# Checkpoint pattern:
#   <dataset>_<backbone>_<balance_mode>_best_fold<k>.pt
# Examples:
#   apacc_efficientnet_b0_none_best_fold0.pt
#   herlev_tv_squeezenet1_1_weighted_loader_best_fold2.pt
CKPT_RE = re.compile(
    r"^(?P<dataset>[^_]+)_"  # apacc / herlev / sipakmed
    r"(?P<backbone>.+)_"  # efficientnet_b0, ghostnetv3_100.in1k, ...
    r"(?P<balance_mode>weighted_loader|weighted_loss|none)_"  # balance mode
    r"best_fold(?P<fold>\d+)$"  # fold index
)

# ==============================================================
# METRIC HELPERS
# ==============================================================


def _confusion_parts(y_true, y_pred) -> Tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn, fp, fn, tp


def metrics_binary(y_true, y_pred) -> Dict[str, float]:
    tn, fp, fn, tp = _confusion_parts(y_true, y_pred)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "spec": tn / (tn + fp + EPS),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "ppv": tp / (tp + fp + EPS),
        "npv": tn / (tn + fn + EPS),
    }


def evaluate_model(loader: DataLoader, model: nn.Module) -> Dict[str, float]:
    model.eval()
    model.to(DEVICE)
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            preds.append(logits.argmax(1).cpu())
            trues.append(labels.cpu())
    return metrics_binary(torch.cat(trues), torch.cat(preds))


# ==============================================================
# DATASET MAP (USING NEW UTILS)
# ==============================================================

# We reuse the already-built DFs and val transforms from datasets/datasets.py
# Note: PapDataset is generic and works for all three.
dataset_map = {
    "sipakmed": {
        "df": sipakmed_df,
        "dataset": PapDataset,
        "val_tf": SIPAKMED_VAL_TF,
    },
    "herlev": {
        "df": herlev_df,
        "dataset": PapDataset,
        "val_tf": HERLEV_VAL_TF,
    },
    "apacc": {
        "df": apacc_df,
        "dataset": PapDataset,
        "val_tf": APACC_VAL_TF,
    },
}


def build_loader(
    df: pd.DataFrame,
    dataset_cls,
    transform,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin: bool = True,
) -> DataLoader:
    """
    Build a DataLoader over a PapDataset using a given transform.
    """
    ds = dataset_cls(df.reset_index(drop=True), transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )


# ==============================================================
# MAIN
# ==============================================================


def main():
    if not MODELS_DIR.exists():
        print(f"❌ MODELS_DIR not found: {MODELS_DIR}")
        return

    model_files: List[Path] = list(MODELS_DIR.rglob("*.pt"))
    if not model_files:
        print(f"❌ No .pt files found under {MODELS_DIR}")
        return

    print(f"Found {len(model_files)} checkpoints.")
    print(f"Using device: {DEVICE}")

    # Where we store cross-dataset results
    output_dir = MODELS_DIR / "all_cross_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "cross_dataset_validation_results.csv"

    # ----------------------------------------------------------
    # Resume support: skip checkpoints already evaluated
    # ----------------------------------------------------------
    evaluated_names = set()
    if out_csv.exists():
        curr = pd.read_csv(out_csv)
        evaluated_names = {Path(p).name for p in curr["checkpoint"].tolist()}
        print(f"Resuming: {len(evaluated_names)} checkpoints already evaluated.")

    new_model_files: List[Path] = []
    for ckpt in model_files:
        if ckpt.name in evaluated_names:
            print(f"✅  Already evaluated: {ckpt.relative_to(MODELS_DIR)}")
            continue
        new_model_files.append(ckpt)

    model_files = new_model_files
    if not model_files:
        print("Nothing new to evaluate.")
        return

    results: List[dict] = []

    # ----------------------------------------------------------
    # Iterate over each target dataset
    # ----------------------------------------------------------
    for target_name, tgt in dataset_map.items():
        print(f"\n=== Evaluating on {target_name.upper()} dataset ===")

        df_target = tgt["df"]

        # Use test split when available (this is how your scanners are defined)
        if "split" in df_target.columns:
            test_df = df_target[df_target["split"] == "test"].copy()
            if test_df.empty:
                print("  [WARN] split=='test' is empty, falling back to fold==0")
                test_df = df_target[df_target["fold"] == 0].copy()
        else:
            test_df = df_target[df_target["fold"] == 0].copy()

        if test_df.empty:
            print(f"  [WARN] No test data found for {target_name}, skipping.")
            continue

        # Loop over every checkpoint, evaluating on this target dataset
        for ckpt_path in tqdm(model_files, desc=f"Validating on {target_name}"):
            try:
                m = CKPT_RE.match(ckpt_path.stem)
                if not m:
                    print(f"⚠️  Skipping '{ckpt_path}': name does not match pattern.")
                    continue

                training_dataset = m.group("dataset")
                backbone_id = m.group("backbone")
                balance_mode = m.group("balance_mode")
                training_fold = int(m.group("fold"))

                if training_dataset not in dataset_map:
                    print(
                        f"⚠️  Unknown training dataset '{training_dataset}' in {ckpt_path.name}"
                    )
                    continue

                # normalisation to apply = training dataset's val_tf
                eval_tf = dataset_map[training_dataset]["val_tf"]

                # build loader on *target* images but with *training* normalization
                test_loader = build_loader(
                    test_df,
                    tgt["dataset"],
                    eval_tf,
                )

                # load model + weights
                model, _, _ = load_any(backbone_id, num_classes=2)
                state = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(state, strict=True)

                # evaluate
                metrics = evaluate_model(test_loader, model)

                results.append(
                    {
                        "checkpoint": str(ckpt_path.relative_to(MODELS_DIR)),
                        "model_backbone": backbone_id,
                        "trained_on": training_dataset,
                        "training_fold": training_fold,
                        "balance_mode": balance_mode,
                        "validated_on": target_name,
                        "accuracy": metrics["acc"],
                        "f1_score": metrics["f1"],
                        "precision": metrics["prec"],
                        "recall": metrics["rec"],
                        "specificity": metrics["spec"],
                    }
                )

                # tidy GPU / RAM
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"⚠️  Skipping '{ckpt_path}': {e}")

    # ----------------------------------------------------------
    # Save combined results
    # ----------------------------------------------------------
    if not results:
        print("\nNo new cross-dataset evaluations completed.")
        return

    results_df = pd.DataFrame(results)

    # If we are resuming, append to existing file
    if out_csv.exists():
        old_df = pd.read_csv(out_csv)
        results_df = pd.concat([old_df, results_df], ignore_index=True)

    print("\n=== Cross-Dataset Validation Results (latest batch) ===")
    print(results_df.tail().round(4).to_string(index=False))

    results_df.to_csv(out_csv, index=False)
    print(f"\n✅  Results saved/updated at: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
