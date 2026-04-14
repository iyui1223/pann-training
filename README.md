# PANN Training Pipeline

Training and evaluation code for the **Process-Additive Neural Network (PANN)** —
a physics-partitioned architecture that learns additive bias corrections from
OpenIFS single-column model (SCM) tendency outputs.

Each physical process (convection, cloud microphysics, radiation SW/LW, vertical
diffusion, non-orographic GWD) is handled by an independent branch whose outputs
are summed.  This additive decomposition enables Shapley-based importance attribution
per branch and per vertical level.

## Quick start

```bash
# 1. Clone this repo
git clone <repo-url> pann-training
cd pann-training

# 2. Install dependencies
pip install -r requirements.txt      # or use the Poetry pyproject.toml from the parent project

# 3. Download the training dataset (see below) and place it at:
#      data/training_dataset_partitioned.nc

# 4. Run the full pipeline (Slurm cluster)
bash scripts/main.sh

# 5. Or run steps individually (interactive / non-Slurm)
python src/train.py    --config config/config.yaml --data data/training_dataset_partitioned.nc --save_dir output/models
python src/evaluate.py --model_dir output/models   --data data/training_dataset_partitioned.nc --fig_dir output/figs
```

## Training dataset

The pipeline expects a single NetCDF file: **`training_dataset_partitioned.nc`**

This file is produced by the OpenIFS SCM data pipeline (M03 step) and contains
~10k single-column 1-hour runs with per-process tendency decompositions.

**The dataset is not version-controlled in this repo** due to its size (~200 MB).
Download it from one of:

| Source | Link |
|--------|------|
| Google Drive | *TODO: paste link here* |
| Hugging Face | *TODO: paste link here* |

After downloading, place it at `data/training_dataset_partitioned.nc`.

### Dataset summary

| Dimension | Size | Description |
|-----------|------|-------------|
| `sample`  | ~9,874 | Individual 1-hour SCM runs |
| `level`   | 91 | Full pressure levels (top-of-atmosphere to surface) |

Key variables per sample:

- **Inputs**: Per-process tendencies for T, Q, QL, QI (convection, cloud, radiation, vdif, nogw)
- **Target**: Bias = SCM total tendency minus actual tendency (or synthetic Gaussian target for validation)
- **Metadata**: Pressure levels, initial state fields

See `config/config.yaml` → `target.mode` to switch between:
- `scm_minus_reference` — real SCM bias (8 branches, T+Q output)
- `simple_gaussian_q` — synthetic Q-only Gaussian target (2 branches; for architecture validation)

## Repository layout

```
pann-training/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml          # architecture + training hyperparameters
├── src/
│   ├── train.py             # k-fold training (supports level-blocking)
│   ├── evaluate.py          # metrics, profile plots, Shapley attribution
│   ├── extract_hidden.py    # hidden-layer activation extraction
│   ├── model.py             # PANN model definition (flat / hourglass / conv1d)
│   └── dataset.py           # data loading, normalisation, target modes
├── scripts/
│   ├── env_setting.sh       # paths, Slurm config (edit for your system)
│   ├── main.sh              # pipeline orchestrator (auto-detects blocking)
│   ├── step1_train.sh       # Slurm job: training
│   ├── step2_evaluate.sh    # Slurm job: evaluation + figures
│   └── step3_extract_hidden.sh  # Slurm job: hidden extraction
├── data/                    # put training_dataset_partitioned.nc here
├── output/
│   ├── models/              # trained model checkpoints
│   ├── figs/                # evaluation figures
│   └── hidden_activations/  # extracted activations
└── logs/                    # Slurm stdout/stderr
```

## Configuration

All hyperparameters live in `config/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `architecture` | `conv1d` | Branch type: `flat`, `hourglass`, or `conv1d` |
| `hidden_dim` | 16 | Channel count / hidden width |
| `bottleneck_dim` | 10 | Vertical-mode bottleneck channels (conv1d/hourglass) |
| `block_size` | 4 | Levels per block (level-blocking mode) |
| `block_overlap` | 2 | Shared levels between adjacent blocks |
| `learning_rate` | 1e-3 | Adam learning rate |
| `epochs` | 39 | Max training epochs |
| `patience` | 20 | Early stopping patience |
| `n_folds` | 4 | k-fold cross-validation folds |
| `level_weight` | `pressure` | Loss weighting: `pressure` or `uniform` |

## Running without Slurm

For local / interactive use, skip `main.sh` and call the Python scripts directly:

```bash
# Train
python src/train.py \
    --config config/config.yaml \
    --data data/training_dataset_partitioned.nc \
    --save_dir output/models \
    --device cpu

# Evaluate
python src/evaluate.py \
    --model_dir output/models \
    --data data/training_dataset_partitioned.nc \
    --fig_dir output/figs \
    --device cpu

# Extract hidden activations (optional)
python src/extract_hidden.py \
    --model_dir output/models \
    --data data/training_dataset_partitioned.nc \
    --out_dir output/hidden_activations \
    --device cpu
```

## Dependencies

Core requirements: Python 3.11+, PyTorch, NumPy, xarray, matplotlib, PyYAML.

See `requirements.txt` for pinned versions.

## License

MIT

## Citation

If you use this code, please cite:

> Ichikawa, Y. (2026). Process-Additive Neural Network for physics-partitioned
> SCM bias correction. *In preparation.*
