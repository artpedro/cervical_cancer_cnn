# cross_dataset_evaluate.py
# ==============================================================
# Evaluate every *.pt checkpoint under MODELS_DIR on all datasets
# ==============================================================

# -------- Standard libs ----------
import os
import gc
from pathlib import Path
from typing import Tuple, Dict, List

# -------- Third-party ----------
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_score, confusion_matrix,
)
from tqdm import tqdm

# -------- Project-specific ----------
from model_utils import load_any                               # backbone loader

# SiPaKMeD
from sipakmed import (
    scan_sipakmed, SipakmedDataset, val_tf as sipak_val_tf,
)
# Herlev
from herlev import (
    scan_herlev, HerlevDataset, val_tf as herlev_val_tf,
)
# APaCC
from apacc import (
    scan_apacc, ApaccDataset, val_tf as apacc_val_tf,
)

# ==============================================================
# CONFIGURATION
# ==============================================================
MODELS_DIR     = Path(
    "C:\\Users\\Workstation-Lab\\Documents\\Arquivos do Artur Pedro\\labcity\\cervical_cancer_cnn\\workspace\\metrics"
)                               # folder that holds sub-folders with checkpoints
DEVICE          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE      = 32
NUM_WORKERS     = int(os.getenv("NUM_WORKERS", 2))
SEED            = 42
EPS             = 1e-9

# ==============================================================
# METRIC HELPERS
# ==============================================================
def _confusion_parts(y_true, y_pred) -> Tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn, fp, fn, tp


def metrics_binary(y_true, y_pred) -> Dict[str, float]:
    tn, fp, fn, tp = _confusion_parts(y_true, y_pred)
    return {
        "acc":  accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec":  recall_score(y_true, y_pred,  zero_division=0),
        "spec": tn / (tn + fp + EPS),
        "f1":   f1_score(y_true, y_pred,     zero_division=0),
        "ppv":  tp / (tp + fp + EPS),
        "npv":  tn / (tn + fn + EPS),
    }


def evaluate_model(loader: DataLoader, model: nn.Module) -> Dict[str, float]:
    model.eval(); model.to(DEVICE)
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            preds.append(logits.argmax(1).cpu())
            trues.append(labels.cpu())
    return metrics_binary(torch.cat(trues), torch.cat(preds))

# ==============================================================
# DATASET MAP with per-dataset normalisation
# ==============================================================
dataset_map = {
    "sipakmed": {
        "root":     Path("./datasets/sipakmed"),
        "scanner":  scan_sipakmed,
        "dataset":  SipakmedDataset,
        "val_tf":   sipak_val_tf,
    },
    "herlev": {
        "root":     Path("./datasets/herlev"),
        "scanner":  scan_herlev,
        "dataset":  HerlevDataset,
        "val_tf":   herlev_val_tf,
    },
    "apacc": {
        "root":     Path("./datasets/apacc"),
        "scanner":  scan_apacc,
        "dataset":  ApaccDataset,
        "val_tf":   apacc_val_tf,
    },
}

# --------------------------------------------------------------
# Build a loader with an arbitrary normalisation
# --------------------------------------------------------------
def build_loader(
    df: pd.DataFrame,
    dataset_cls,
    transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin=True,
) -> DataLoader:
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

    results: List[dict] = []
    
    if os.path.exists(MODELS_DIR / "all_cross_results" / "cross_dataset_validation_results.csv"):
        curr_csv = pd.read_csv(MODELS_DIR / "all_cross_results" / "cross_dataset_validation_results.csv")
        evaluated_paths = [path.split("\\")[-1] for path in curr_csv["checkpoint"].tolist()]
        print(evaluated_paths)

    models = []
    for model in model_files:
        if os.path.exists(MODELS_DIR / "all_cross_results" / "cross_dataset_validation_results.csv"):  
            if str(model).split("\\")[-1] in evaluated_paths:
                print(f"✅  Already evaluated: {model.relative_to(MODELS_DIR)}")
                continue
            else:
                models.append(model)
    model_files = models
    # ----------------------------------------------------------
    # iterate over each target dataset
    # ----------------------------------------------------------
    for target_name, tgt in dataset_map.items():
        print(f"\n=== Evaluating on {target_name.upper()} dataset ===")

        # Build dataframe once (all folds); we will slice fold-0 below
        df_target = tgt["scanner"](root=tgt["root"], num_folds=5, seed=SEED)
        
        # loop over every checkpoint
        for ckpt_path in tqdm(model_files, desc=f"Validating on {target_name}"):
            try:
                
                print(ckpt_path)
                # Expect names like  sipakmed_efficientnet_b0_best_0.pt
                parts             = ckpt_path.stem.split("_")
                training_dataset  = parts[0]
        
                training_fold     = int(parts[-1])
                backbone_id       = "_".join(parts[1:-2])   # between dataset tag and 'best'

                # normalisation to apply = training dataset's val_tf
                eval_tf           = dataset_map[training_dataset]["val_tf"]

                # build loader on *target* images but with *eval_tf*
                test_df           = df_target[df_target.fold == 0]     # fold-0 split
                test_loader       = build_loader(
                    test_df,
                    tgt["dataset"],
                    eval_tf,
                )

                # load model + weights
                model, _, _ = load_any(backbone_id, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=True)

                # evaluate
                metrics = evaluate_model(test_loader, model)

                results.append({
                    "checkpoint":    str(ckpt_path.relative_to(MODELS_DIR)),
                    "model_backbone": backbone_id,
                    "trained_on":     training_dataset,
                    "training_fold":  training_fold,
                    "validated_on":   target_name,
                    "accuracy":       metrics["acc"],
                    "f1_score":       metrics["f1"],
                    "precision":      metrics["prec"],
                    "recall":         metrics["rec"],
                    "specificity":    metrics["spec"],
                })

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
        print("\nNo cross-dataset evaluations completed.")
        return

    results_df = pd.DataFrame(results)
    print("\n=== Cross-Dataset Validation Results ===")
    print(results_df.round(4).to_string(index=False))

    output_dir = MODELS_DIR / "all_cross_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "cross_dataset_validation_results_extra.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅  Results saved to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
