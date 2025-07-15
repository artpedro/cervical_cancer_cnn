# misc
import os
import gc
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple

# --- User's custom modules (ensure they are in the same directory or python path) ---
from model_utils import load_any
from sipakmed import scan_sipakmed, get_sipakmed_loaders
from herlev import scan_herlev, get_herlev_loaders
from apacc import scan_apacc, get_apacc_loaders

# Metrics and model imports from the previous script
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
)

# =================================================================================
# SCRIPT CONFIGURATION
# =================================================================================

# --- 1. SET THE PATH TO YOUR TRAINED MODELS ---
# This should be the 'checkpoints' directory created by the training script.
MODELS_DIR = Path("C:\\Users\\Workstation-Lab\\Documents\\Arquivos do Artur Pedro\\labcity\\cervical_cancer_cnn\\evaluate")

# --- 2. OTHER CONFIG ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
SEED = 42
EPS = 1e-9

# =================================================================================
# HELPER FUNCTIONS
# =================================================================================

def _confusion_parts(y_true, y_pred) -> Tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn, fp, fn, tp


def metrics_binary(y_true, y_pred) -> dict[str, float]:
    """Return all requested binary-classification metrics in one dict."""
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


def evaluate_model(dataloader: DataLoader, model: nn.Module) -> dict[str, float]:
    """Runs a single evaluation pass and returns metrics."""
    model.eval()
    model.to(DEVICE)
    preds, trues = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            preds.append(logits.detach().argmax(1).cpu())
            trues.append(labels.cpu())

    y_pred = torch.cat(preds)
    y_true = torch.cat(trues)
    return metrics_binary(y_true, y_pred)


# =================================================================================
# MAIN VALIDATION LOGIC
# =================================================================================

def main():
    """
    Main function to validate models from one dataset on all other datasets.
    """
    if not MODELS_DIR.exists():
        print(f"Error: Models directory not found at '{MODELS_DIR}'")
        return

    # --- Mapping to select the correct data loader ---
    dataset_map = {
        "sipakmed": {"scanner": scan_sipakmed, "loader": get_sipakmed_loaders, "path": Path(".\\datasets\\sipakmed")},
        "herlev": {"scanner": scan_herlev, "loader": get_herlev_loaders, "path": Path(".\\datasets\\herlev")},
        "apacc": {"scanner": scan_apacc, "loader": get_apacc_loaders, "path": Path(".\\datasets\\apacc")},
    }

    # --- Find all model files ---
    model_files = list(MODELS_DIR.glob("*.pt"))
    if not model_files:
        print(f"Error: No '.pt' model files found in '{MODELS_DIR}'.")
        return

    print(f"Found {len(model_files)} models to evaluate across all datasets.")
    print(f"Using device: {DEVICE}")
    
    results = []

    # --- Loop through each dataset as a validation target ---
    for target_name, target_info in dataset_map.items():
        print(f"\n--- Preparing to validate on dataset: {target_name.upper()} ---")

        # Prepare the target dataset's test loader (using validation set of fold 0)
        df_target = target_info["scanner"](root=target_info["path"], num_folds=5, seed=SEED)
        _, test_loader = target_info["loader"](
            df=df_target,
            fold=0,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

        # --- Loop through models and evaluate on the current target dataset ---
        progress_desc = f"Validating on {target_name}"
        for model_path in tqdm(model_files, desc=progress_desc):
            try:
                # Parse filename: e.g., "sipakmed_efficientnet_b0_best_0.pt"
                parts = model_path.stem.split('_')
                training_dataset = parts[0]
                
                # *** Skip if the model was trained on the current target dataset ***
                if training_dataset == target_name:
                    continue

                training_fold = int(parts[-1])
                backbone_id = "_".join(parts[0:-2])

                # Load model architecture and weights
                model, _, _ = load_any(backbone_id, num_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))

                # Evaluate
                metrics = evaluate_model(test_loader, model)

                # Store results
                results.append({
                    "model_backbone": backbone_id,
                    "trained_on": training_dataset,
                    "training_fold": training_fold,
                    "validated_on": target_name,
                    "accuracy": metrics["acc"],
                    "f1_score": metrics["f1"],
                    "precision": metrics["prec"],
                    "recall": metrics["rec"],
                    "specificity": metrics["spec"],
                })

                # Clean up memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"\nWarning: Could not process file '{model_path.name}': {e}")

    # --- Display and save final results ---
    if not results:
        print("\nNo cross-dataset validations could be performed.")
        return
        
    results_df = pd.DataFrame(results)
    
    print("\n\n--- Cross-Dataset Validation Results ---")
    print(results_df.round(4).to_string())

    # Save to CSV in the parent directory of the checkpoints folder
    output_path = MODELS_DIR.parent / "cross_dataset_validation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")


if __name__ == "__main__":
    main()