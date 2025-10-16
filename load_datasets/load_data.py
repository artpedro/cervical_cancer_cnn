import pathlib
import shutil
import kagglehub
import os
import shutil
import zipfile
import urllib.request as urllib_request
import itertools, json, math, time, copy, random, warnings, textwrap
from pathlib import Path
from collections import defaultdict
import pandas as pd

## download from kaggle
print("✅ KaggleHub ready — proceeding to download…")

DATA_DIR = pathlib.Path(".\\data")  # <- central location for everything
DATA_DIR.mkdir(parents=True, exist_ok=True)

slug = "prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed"
raw_path = kagglehub.dataset_download(
    slug
)  # downloads + unzips under ~/.cache/kagglehub/…
print("Downloaded to :", raw_path)

# Move/rename to a predictable place in /content/data
final_path = DATA_DIR / "sipakmed"
if final_path.exists():
    shutil.rmtree(final_path)
shutil.copytree(raw_path, final_path)
print("🗂  Dataset ready at:", final_path)

DATA_ROOT = Path(".\\data\\sipakmed")  # after KaggleHub copy
assert DATA_ROOT.exists(), f"Missing folder: {DATA_ROOT}"

# ── Crawl: one class = one *top-level* im_* folder ─────────────────────
records = []  # (path, label) rows
cls_counts = defaultdict(int)
for class_dir in sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()]):
    label = class_dir.name.replace("im_", "")  # neat label
    # bmp / png / jpg inside any “…/CROPPED/” subdir
    imgs = list(class_dir.glob("**\\CROPPED\\*.bmp"))
    cls_counts[label] = len(imgs)
    records += [(str(p), label) for p in imgs]

df_counts = pd.Series(cls_counts, name="#images").sort_index().to_frame()
print(df_counts)
print(f"Total images found: {len(records)}")
