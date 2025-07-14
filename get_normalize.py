#!/usr/bin/env python
"""
compute_dataset_stats.py – reproducible μ/σ for every dataset (Windows-safe)

• Works with helper modules that expose:
      scan_<name>(root, num_folds, seed)  -> DataFrame
      <Name>Dataset                       -> __init__(df, transform)
      val_tf (or eval_tf)                 -> deterministic eval transform
• Writes one YAML file: dataset_stats.yaml
"""
from __future__ import annotations
import importlib, random, yaml
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

# ─────────── configuration ────────────
SEED = 42
DATASETS = {
    "apacc": {"root": r".\datasets\apacc", "module": "apacc",    "num_folds": 5},
    "herlev"  : {"root": r".\datasets\herlev",   "module": "herlev","num_folds": 5},
    "sipakmed"   : {"root": r".\datasets\sipakmed",    "module": "sipakmed", "num_folds": 5},
    # add more …
}
OUT_YAML = Path("dataset_stats.yaml")
NUM_WORKERS = 4            # set 0 to avoid multiprocessing entirely
# ──────────────────────────────────────


def without_normalize(tf: T.Compose) -> T.Compose:
    ops = list(tf.transforms)
    if ops and isinstance(ops[-1], T.Normalize):
        ops = ops[:-1]
    return T.Compose(ops)


def calc_mean_std(loader, device="cuda"):
    n_pix  = 0
    c_sum  = torch.zeros(3, dtype=torch.float64, device=device)
    c_sq   = torch.zeros(3, dtype=torch.float64, device=device)

    with torch.no_grad():
        for batch, _ in tqdm(loader, leave=False):
            batch = batch.to(device, non_blocking=True)       # B×C×H×W
            b, c, h, w = batch.shape
            n_pix  += b * h * w
            c_sum  += batch.sum(dim=[0, 2, 3])
            c_sq   += (batch ** 2).sum(dim=[0, 2, 3])

    mean = c_sum / n_pix                    # still on GPU
    std  = torch.sqrt(c_sq / n_pix - mean ** 2)

    return mean.cpu().tolist(), std.cpu().tolist()

def main() -> None:
    # reproducible RNG
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats: dict[str, dict] = {}

    for nick, cfg in DATASETS.items():
        print(f"\n▶  {nick}")
        mod = importlib.import_module(cfg["module"])

        # scan function
        scan_fn = next(getattr(mod, n) for n in dir(mod) if n.startswith("scan"))
        df = scan_fn(Path(cfg["root"]), num_folds=cfg["num_folds"], seed=SEED)

        # dataset class (module-stem + 'Dataset')
        stem = cfg["module"].split(".")[-1]
        dataset_cls = next(v for k, v in vars(mod).items()
                           if k.lower().startswith(stem) and k.endswith("Dataset"))

        # deterministic transform (eval tf w/o Normalize)
        if hasattr(mod, "val_tf"):
            stat_tf = without_normalize(mod.val_tf)
        elif hasattr(mod, "eval_tf"):
            stat_tf = without_normalize(mod.eval_tf)
        else:  # fallback
            stat_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

        # choose rows
        df_train = df[df["split"] == "train"] if "split" in df.columns else df

        loader = DataLoader(
            dataset_cls(df_train.reset_index(drop=True), stat_tf),
            batch_size=128,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        mean, std = calc_mean_std(loader, device=device)
        stats[nick] = {"mean": mean, "std": std}
        print("   mean :", mean)
        print("   std  :", std)

    with OUT_YAML.open("w") as f:
        yaml.safe_dump(stats, f, sort_keys=False)
    print(f"\n📄  Statistics written to {OUT_YAML.resolve()}")


# Windows (spawn) requires the guard so that worker processes can import this file
if __name__ == "__main__":
    main()
