# Cervical Cancer Classification via Convolutional Neural Networks

## Abstract

This repository implements a comprehensive experimental framework for evaluating lightweight convolutional neural network (CNN) architectures on cervical cytology image classification tasks. The system encompasses three publicly available datasets (SIPaKMeD, Herlev, and APaCC), employs stratified k-fold cross-validation, and conducts cross-dataset transfer learning experiments to assess model generalizability across distinct data distributions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Core Components](#core-components)
4. [Experimental Pipeline](#experimental-pipeline)
5. [Module Interactions](#module-interactions)
6. [Usage Instructions](#usage-instructions)
7. [Dependencies](#dependencies)

---

## Project Overview

### Objective

The primary objective of this research infrastructure is to evaluate the performance and generalization capabilities of efficient CNN architectures for binary classification of cervical cell images (normal vs. abnormal). The framework facilitates:

1. **Intra-dataset validation**: K-fold cross-validation within individual datasets
2. **Cross-dataset transfer learning**: Evaluation of models trained on one dataset and validated on others
3. **Architecture comparison**: Systematic comparison of mobile-optimized networks (e.g., EfficientNet, MobileNet, SqueezeNet)

### Dataset Taxonomy

The experimental framework incorporates three cervical cytology datasets:

- **SIPaKMeD**: Five-class dataset (Superficial-Intermediate, Parabasal, Koilocytotic, Dyskeratotic, Metaplastic) collapsed into binary labels
- **Herlev**: Seven-class dataset spanning normal columnar/intermediate/superficial cells and varying dysplasia grades
- **APaCC**: Binary dataset (healthy vs. unhealthy) with explicit train/test partitioning

---

## Repository Structure

```
cervical_cancer_cnn/
├── datasets/                      # Dataset-specific modules
│   ├── __init__.py               # Package initialization
│   ├── sipakmed.py               # SIPaKMeD dataset interface
│   ├── herlev.py                 # Herlev dataset interface
│   ├── apacc.py                  # APaCC dataset interface
│   ├── dataset_analysis.py       # Exploratory data analysis utilities
│   ├── get_normalize.py          # Normalization statistics computation
│   └── dataset_stats.yaml        # Precomputed channel-wise μ/σ values
│
├── load_datasets/                 # Legacy data loading utilities
│   ├── __init__.py
│   └── load_data.py              # Initial SIPaKMeD download script
│
├── model_utils.py                 # Backbone loading and adaptation
├── train_sipakmed.py              # Single-dataset training (SIPaKMeD)
├── train_all.py                   # Multi-dataset training orchestrator
├── evaluate_models.py             # Cross-dataset evaluation pipeline
├── visualize_perfomrance.py       # Results visualization utilities
├── cross_dataset_metrics.ipynb    # Interactive analysis notebook
├── requirements.txt               # Python dependencies
└── __init__.py                    # Root package marker
```

---

## Core Components

### 1. Dataset Modules (`datasets/`)

Each dataset module (`sipakmed.py`, `herlev.py`, `apacc.py`) implements a standardized interface comprising:

#### **Dataset Scanning Function** (`scan_<dataset>`)
- **Input**: Root directory path, number of folds, random seed
- **Output**: `pandas.DataFrame` with columns:
  - `path`: Absolute file path to image
  - `label_full`: Original multi-class label
  - `binary_label`: String representation ("normal" / "abnormal")
  - `binary_idx`: Integer encoding (0=normal, 1=abnormal)
  - `fold`: Stratified fold assignment (0 to k-1)
- **Functionality**: Recursively traverses directory structure, applies dataset-specific labeling conventions, and performs stratified k-fold splitting to ensure balanced class distribution across folds

#### **PyTorch Dataset Class** (`<Dataset>Dataset`)
- **Inheritance**: `torch.utils.data.Dataset`
- **Constructor Parameters**: DataFrame, transformation pipeline
- **`__getitem__`**: Returns (transformed_image_tensor, binary_label_index) tuple
- **Design Rationale**: Decouples data indexing from transformation logic, enabling flexible augmentation strategies

#### **Data Transformations** (`train_tf`, `val_tf`)
- **Training Transformations** (`train_tf`):
  - Geometric: Resize(256), CenterCrop(224), RandomRotation(±180°), RandomHorizontalFlip(p=0.5)
  - Photometric: ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
  - Normalization: Dataset-specific channel-wise standardization
  
- **Validation Transformations** (`val_tf`):
  - Deterministic resize-crop pipeline
  - Identical normalization to training set

**Critical Design Decision**: Each dataset employs distinct normalization statistics (μ, σ per RGB channel) computed from its training partition. This preserves dataset-specific color distributions during intra-dataset validation while introducing a controlled confound during cross-dataset transfer experiments.

#### **DataLoader Factory** (`get_<dataset>_loaders`)
- **Parameters**: DataFrame, fold index, batch size, worker count, memory pinning flag
- **Returns**: (train_loader, val_loader) tuple
- **Partitioning Logic**: 
  - Training: All samples where `fold ≠ current_fold`
  - Validation: All samples where `fold == current_fold`

---

### 2. Normalization Infrastructure

#### **`dataset_stats.yaml`**
Persistent storage of precomputed channel-wise statistics for reproducibility:

```yaml
sipakmed:
  mean: [0.5008, 0.4842, 0.5820]
  std:  [0.2054, 0.1932, 0.2172]
herlev:
  mean: [0.3264, 0.2441, 0.6026]
  std:  [0.3175, 0.2581, 0.2517]
apacc:
  mean: [0.7467, 0.6917, 0.7182]
  std:  [0.2691, 0.2727, 0.2635]
```

#### **`get_normalize.py`**
Automated computation script employing single-pass Welford's algorithm for numerical stability:

1. Dynamically imports dataset modules via `importlib`
2. Instantiates dataset with normalization-free transforms
3. Accumulates channel-wise sum and squared-sum across entire training set
4. Computes μ = Σ(pixels) / N, σ² = Σ(pixels²) / N - μ²
5. Persists results to YAML for deterministic reproduction

---

### 3. Model Infrastructure (`model_utils.py`)

#### **`load_any(name, num_classes, pretrained)`**
Unified model instantiation interface supporting multiple sources:

- **timm library**: Primary source for modern architectures (EfficientNet, MobileNetV4, GhostNet)
- **torchvision**: Fallback for legacy models (SqueezeNet, MobileNetV2, ShuffleNet)
- **torch.hub**: Secondary fallback for specific architectures

**Returns**: (model, input_features_dim, origin_string) tuple

#### **`_adapt_head(model, num_classes)`**
Classification head replacement logic:

1. **Fast path**: Invokes `model.reset_classifier(num_classes)` if available (timm models)
2. **Manual adaptation**: Iterates through common attribute names (`head`, `classifier`, `fc`, `_fc`)
   - For `nn.Linear`: Replaces with Linear(in_features, num_classes)
   - For `nn.Sequential`: Traverses backwards to find terminal Linear/Conv2d layer
3. **Returns**: Input feature dimensionality for logging purposes

**Design Rationale**: Enables seamless integration of pretrained ImageNet weights while adapting output dimensionality to binary classification task.

---

### 4. Training Scripts

#### **`train_sipakmed.py`**
Single-dataset training script demonstrating baseline methodology:

- **Configuration Constants**:
  - `SEED=42`: Ensures reproducibility across NumPy, PyTorch, and CUDA operations
  - `BATCH_SIZE=32`, `EPOCHS=25`, `NUM_FOLDS=5`
  - `LR=5e-4`, `MOMENTUM=0.9`, `WEIGHT_DECAY=5e-3`
  - Scheduler: MultiStepLR with milestones [10, 20], γ=0.1

- **Optimization Strategy**:
  - Loss Function: CrossEntropyLoss with inverse-frequency class weights
  - Optimizer: Stochastic Gradient Descent with momentum
  - Learning Rate Schedule: Step decay (10× reduction at epochs 10 and 20)

- **Training Loop** (`_run_epoch`):
  ```python
  for epoch in [1..EPOCHS]:
      train_metrics = _run_epoch(train_loader, model, criterion, optimizer)
      val_metrics   = _run_epoch(val_loader, model, criterion, None)
      scheduler.step()
      if val_accuracy > best_val_accuracy:
          save_checkpoint(model, f"{backbone_id}_best_{fold}.pt")
  ```

- **Metrics Computation** (`metrics_binary`):
  - Accuracy, Precision, Recall (Sensitivity)
  - Specificity, F1-score
  - PPV (Positive Predictive Value), NPV (Negative Predictive Value)
  - Derived from confusion matrix: (TN, FP, FN, TP)

- **Output Artifacts**:
  - `epoch_logs.csv`: Per-epoch metrics for all models/folds
  - `summary.csv`: Best validation metrics per fold
  - `checkpoints/*.pt`: State dictionaries for best-performing models

#### **`train_all.py`**
Multi-dataset orchestration extension:

**Key Enhancement**: Outer loop over datasets enables systematic evaluation:

```python
datasets = ["apacc", "herlev", "sipakmed"]
for dataset_name, root, scanner, loader_fn in zip(names, roots, scanners, get_loaders):
    df = scanner(root, num_folds=NUM_FOLDS, seed=SEED)
    for model_name, backbone_id in TODO.items():
        for fold in range(NUM_FOLDS):
            # Training loop identical to train_sipakmed.py
```

**Architecture Coverage**:
```python
TODO = {
    "SqueezeNet 1.1": "tv_squeezenet1_1",
    "MobileNet V2": "mobilenetv2_100",
    "MobileNet V4 small": "mobilenetv4_conv_small.e2400_r224_in1k",
    "ShuffleNet V2": "tv_shufflenet_v2_x1_0",
    "GhostNet V3": "ghostnetv3_100.in1k",
    "EfficientNet-B0": "efficientnet_b0",
    "EfficientNet-B1": "efficientnet_b1",
    "EfficientNet-B2": "efficientnet_b2",
    "EfficientNet-B3": "efficientnet_b3",
}
```

---

### 5. Evaluation Pipeline (`evaluate_models.py`)

**Objective**: Quantify cross-dataset generalization by evaluating all checkpoints on all datasets.

#### **Methodology**

1. **Checkpoint Discovery**: Recursively scans `MODELS_DIR` for `*.pt` files
2. **Filename Parsing**: Extracts metadata via convention `{dataset}_{backbone}_{best}_{fold}.pt`
3. **Critical Design Decision**: Applies **training dataset's normalization** to target dataset images
   - Example: Model trained on SIPaKMeD (μ=[0.50, 0.48, 0.58]) evaluates Herlev images using SIPaKMeD's μ/σ
   - Rationale: Simulates deployment scenario where target data distribution is unknown

4. **Evaluation Protocol**:
   ```python
   for target_dataset in ["sipakmed", "herlev", "apacc"]:
       for checkpoint in all_checkpoints:
           training_dataset = parse_checkpoint_name(checkpoint)
           transform = dataset_map[training_dataset]["val_tf"]  # ← key insight
           test_loader = build_loader(target_data, transform)
           metrics = evaluate_model(test_loader, model)
   ```

5. **Output**: `cross_dataset_validation_results.csv` with schema:
   - `checkpoint`: Relative path to weights file
   - `model_backbone`: Architecture identifier
   - `trained_on`: Source dataset
   - `training_fold`: Fold index (0-4)
   - `validated_on`: Target dataset
   - `accuracy`, `f1_score`, `precision`, `recall`, `specificity`: Performance metrics

#### **Incremental Evaluation Support**
- Checks for existing results CSV to avoid redundant computation
- Filters already-evaluated checkpoints via path matching
- Appends new results to `cross_dataset_validation_results_extra.csv`

---

### 6. Visualization and Analysis

#### **`visualize_perfomrance.py`**
Command-line visualization script generating:

1. **Heatmap**: Mean F1-score per transfer direction (train→val pairs)
2. **Boxplots**: Distribution of F1-scores across folds for each backbone

**Statistical Summaries**:
- Table 1: Mean metrics over 5 folds per transfer direction
- Table 2: Mean metrics per (model × transfer direction) combination

#### **`cross_dataset_metrics.ipynb`**
Interactive Jupyter notebook for exploratory analysis:

- **Cell 2**: Aggregate statistics per transfer direction
- **Cell 3-4**: Per-model performance tables
- **Cell 5**: Dataset-level ranking (which datasets generalize best?)
- **Cell 6**: Model-level ranking (which architectures are most robust?)
- **Cell 8**: Friedman test for statistical significance across models
- **Cell 9**: Multi-metric ranking system (average rank across 5 metrics)
- **Cell 10**: Pareto frontier analysis (models not dominated on all metrics)
- **Cell 11**: Radar plot of top-3 balanced performers

**Key Findings Encoded**:
- Pareto-optimal models: EfficientNet-B2, EfficientNet-B0, EfficientNet-B3
- Best generalizer (mean rank): EfficientNet-B2
- Friedman test p-values < 0.05 confirm statistically significant differences

---

### 7. Auxiliary Modules

#### **`dataset_analysis.py`**
Exploratory data analysis utilities:

- **`report_class_balance`**: Per-fold class distribution verification
- **`compute_mean_std`**: Welford's algorithm implementation for normalization statistics
- **`show_transforms`**: Visual comparison of original vs. augmented images
- **`plot_size_distribution`**: Histogram analysis of image dimensions
- **`check_duplicates`**: SHA-1 hash-based duplicate detection
- **`filetype_breakdown`**: File extension enumeration

#### **`load_datasets/load_data.py`**
Legacy download script for SIPaKMeD via KaggleHub API (superseded by `sipakmed.download_sipakmed()`).

---

## Experimental Pipeline

### Phase 1: Data Preparation

```bash
# Automatic download (SIPaKMeD only; others require manual placement)
python -c "from datasets.sipakmed import download_sipakmed; download_sipakmed(Path('./datasets'))"

# Compute normalization statistics
cd datasets
python get_normalize.py
```

**Output**: `dataset_stats.yaml` containing channel-wise μ/σ for all datasets.

---

### Phase 2: Intra-Dataset Training

```bash
# Option A: Single dataset (SIPaKMeD example)
python train_sipakmed.py

# Option B: All datasets
python train_all.py
```

**Environment Variables** (optional):
- `DATA_DIR`: Dataset root (default: `./workspace/data`)
- `METRICS_DIR`: Output directory (default: `./workspace/metrics`)
- `NUM_WORKERS`: DataLoader workers (default: 2)

**Output Structure**:
```
workspace/metrics/YYYY-MM-DD_HH-MM-SS/
├── epoch_logs.csv          # Per-epoch training history
├── summary.csv             # Best validation metrics per fold
└── checkpoints/
    ├── sipakmed_efficientnet_b0_best_0.pt
    ├── sipakmed_efficientnet_b0_best_1.pt
    └── ...
```

**Training Outputs**:
- **Total Checkpoints**: 3 datasets × 8 models × 5 folds = **120 model files**
- **CSV Logs**: Detailed metrics for ~6,000 training epochs (3 × 8 × 5 × 25)

---

### Phase 3: Cross-Dataset Evaluation

```bash
python evaluate_models.py
```

**Execution Flow**:
1. Discovers all `*.pt` files in `MODELS_DIR`
2. For each (checkpoint, target_dataset) pair:
   - Loads model with pretrained backbone structure
   - Restores trained weights from checkpoint
   - Applies source dataset's normalization to target images
   - Evaluates on target's fold-0 validation set
3. Aggregates results to `cross_dataset_validation_results.csv`

**Output**: 120 checkpoints × 3 target datasets = **360 evaluation records**

---

### Phase 4: Results Analysis

```bash
# Generate static visualizations
python visualize_perfomrance.py

# Interactive exploration
jupyter notebook cross_dataset_metrics.ipynb
```

**Analysis Dimensions**:
1. **Intra-dataset performance**: `trained_on == validated_on`
2. **Transfer learning efficacy**: Performance drop when `trained_on ≠ validated_on`
3. **Architecture comparison**: Which models maintain performance across domains?
4. **Dataset characteristics**: Which datasets transfer best to others?

---

## Module Interactions

### Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Phase                           │
└─────────────────────────────────────────────────────────────┘
                              │
      ┌───────────────────────┼───────────────────────┐
      ▼                       ▼                       ▼
┌──────────┐          ┌──────────┐            ┌──────────┐
│ sipakmed │          │  herlev  │            │  apacc   │
│   .py    │          │   .py    │            │   .py    │
└─────┬────┘          └─────┬────┘            └─────┬────┘
      │                     │                       │
      │  scan_*()           │  scan_*()             │  scan_*()
      │  *Dataset           │  *Dataset             │  *Dataset
      │  get_*_loaders()    │  get_*_loaders()      │  get_*_loaders()
      │                     │                       │
      └─────────────────────┼───────────────────────┘
                            ▼
                  ┌─────────────────┐
                  │  train_all.py   │
                  │                 │
                  │  ┌──────────┐   │
                  │  │ _run_epoch│  │
                  │  └──────────┘   │
                  └────────┬────────┘
                           │ imports
                           ▼
                  ┌─────────────────┐
                  │ model_utils.py  │
                  │                 │
                  │  load_any()     │
                  │  _adapt_head()  │
                  └─────────────────┘
                           │
                           │ creates models, saves checkpoints
                           ▼
            ┌──────────────────────────────┐
            │  workspace/metrics/          │
            │    YYYY-MM-DD_HH-MM-SS/      │
            │      checkpoints/*.pt        │
            └──────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Phase                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │ evaluate_models.py   │
                  │                      │
                  │  discovers *.pt      │
                  │  parses filenames    │
                  │  builds loaders      │
                  │  evaluates models    │
                  └──────────┬───────────┘
                             │ imports
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
    ┌──────────┐       ┌──────────┐      ┌──────────┐
    │ sipakmed │       │  herlev  │      │  apacc   │
    │ val_tf   │       │  val_tf  │      │  val_tf  │
    └──────────┘       └──────────┘      └──────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
              ┌────────────────────────────────┐
              │ cross_dataset_validation_      │
              │         results.csv            │
              └────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌─────────────────────┐    ┌─────────────────────────┐
    │ visualize_          │    │ cross_dataset_metrics   │
    │  perfomrance.py     │    │        .ipynb           │
    │                     │    │                         │
    │  • heatmaps         │    │  • statistical tests    │
    │  • boxplots         │    │  • Pareto analysis      │
    │  • summary tables   │    │  • radar plots          │
    └─────────────────────┘    └─────────────────────────┘
```

### Data Flow Summary

1. **Dataset Modules** provide standardized interfaces (scan, Dataset class, loaders)
2. **Training Scripts** consume dataset interfaces + model_utils to generate checkpoints
3. **Evaluation Script** loads checkpoints and dataset interfaces to produce aggregated CSV
4. **Visualization Tools** consume aggregated CSV to generate insights

**Critical Coupling Point**: `evaluate_models.py` relies on **filename convention** to infer training dataset, enabling correct normalization selection. Breaking this convention (`{dataset}_{backbone}_best_{fold}.pt`) will cause evaluation failures.

---

## Usage Instructions

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Acquisition

1. **SIPaKMeD**: Automatic via `datasets.sipakmed.download_sipakmed()`
2. **Herlev**: Manual download, place in `./datasets/herlev/`
3. **APaCC**: Manual download, place in `./datasets/apacc/`

Expected directory structures:
```
./datasets/sipakmed/im_<class>/...
./datasets/herlev/<class>/*.BMP
./datasets/apacc/train/<class>/, test/<class>/
```

### Execution Sequence

```bash
# 1. Verify dataset statistics (optional)
cd datasets
python get_normalize.py
cd ..

# 2. Train models on all datasets
python train_all.py
# Expected runtime: ~48-72 hours on RTX 3090 (120 models × 25 epochs)

# 3. Cross-dataset evaluation
python evaluate_models.py
# Expected runtime: ~4-6 hours (360 evaluations)

# 4. Generate reports
python visualize_perfomrance.py
jupyter notebook cross_dataset_metrics.ipynb
```

### Configuration Customization

Modify constants in training scripts:
```python
BATCH_SIZE = 32        # Increase for larger GPUs
NUM_FOLDS = 5          # Reduce for faster experimentation
EPOCHS = 25            # Adjust based on convergence
LR = 5e-4              # Learning rate
```

---

## Dependencies

**Core Framework**:
- `torch==2.7.1+cu128`: Deep learning framework
- `torchvision==0.22.1`: Image transformations and models
- `timm==1.0.16`: Modern architecture implementations

**Data Management**:
- `pandas==2.3.0`: Tabular data manipulation
- `numpy==2.3.0`: Numerical operations
- `scikit-learn==1.7.0`: Stratified splitting, metrics
- `pillow==11.2.1`: Image I/O

**Visualization**:
- `matplotlib==3.10.3`: Plotting utilities

**Utilities**:
- `kagglehub==0.3.12`: Dataset downloading
- `PyYAML==6.0.2`: Configuration persistence
- `tqdm==4.67.1`: Progress monitoring

See `requirements.txt` for complete dependency list with pinned versions.

---

## Reproducibility Considerations

1. **Random Seed Control**: All stochastic operations seeded with `SEED=42`
   - NumPy: `np.random.seed(42)`
   - PyTorch: `torch.manual_seed(42)`
   - CUDA: `torch.backends.cudnn.deterministic=True`

2. **Stratified Splitting**: `StratifiedKFold` ensures balanced class distribution across folds

3. **Normalization Persistence**: `dataset_stats.yaml` decouples statistics computation from training

4. **Checkpoint Naming Convention**: Encodes metadata for unambiguous evaluation

5. **Version Pinning**: `requirements.txt` specifies exact package versions

**Known Variability Sources**:
- CUDA non-determinism in some operations (e.g., atomicAdd)
- DataLoader multiprocessing on different hardware
- Pretrained weight initialization from timm/torchvision

---

## Citation

If using this framework, please cite:

```bibtex
@software{cervical_cancer_cnn,
  author = {[Author Names]},
  title = {Cervical Cancer Classification via Convolutional Neural Networks},
  year = {2025},
  url = {https://github.com/[repository]},
}
```

---

## License

[Specify License]

---

## Contact

For questions or collaboration inquiries, contact: [Contact Information]
