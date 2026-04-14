#!/usr/bin/env bash
#
# Environment settings for PANN training pipeline (standalone repo).
#
# Usage: source scripts/env_setting.sh   (from repo root)
#

set -euo pipefail

# ── Project root (the directory containing this repo) ────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Python virtual environment ───────────────────────────────────────────────
# Option A: Poetry-managed venv (uncomment & adjust the glob)
# PYTHON_ENV="$HOME/.cache/pypoetry/virtualenvs/pann-*/bin/activate"
# Option B: Conda / plain venv (uncomment & set path)
# PYTHON_ENV="/path/to/your/venv/bin/activate"
PYTHON_ENV="${PYTHON_ENV:-}"   # leave empty to use system Python

# ── Source code ──────────────────────────────────────────────────────────────
SOURCE_DIR="${ROOT}/src"

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG_YAML="${ROOT}/config/config.yaml"

# ── Input data ───────────────────────────────────────────────────────────────
# Download the training NetCDF and place it here (see README for links):
TRAINING_NC="${ROOT}/data/training_dataset_partitioned.nc"

# ── Output directories ───────────────────────────────────────────────────────
MODELS_DIR="${ROOT}/output/models"
HIDDEN_DIR="${ROOT}/output/hidden_activations"
FIGS_DIR="${ROOT}/output/figs"
LOG_DIR="${ROOT}/logs"

mkdir -p "${MODELS_DIR}" "${HIDDEN_DIR}" "${FIGS_DIR}" "${LOG_DIR}"

# ── Slurm defaults (edit for your allocation) ────────────────────────────────
SLURM_ACCOUNT="${SLURM_ACCOUNT:-CRANMER-SL3-CPU}"
SLURM_PARTITION="${SLURM_PARTITION:-icelake}"
SLURM_TIME_TRAIN="${SLURM_TIME_TRAIN:-00:30:00}"
SLURM_TIME_EVAL="${SLURM_TIME_EVAL:-00:30:00}"

export ROOT SOURCE_DIR CONFIG_YAML
export TRAINING_NC
export MODELS_DIR HIDDEN_DIR FIGS_DIR LOG_DIR
export PYTHON_ENV
export SLURM_ACCOUNT SLURM_PARTITION SLURM_TIME_TRAIN SLURM_TIME_EVAL

echo "PANN training environment loaded:"
echo "  Root:          ${ROOT}"
echo "  Config:        ${CONFIG_YAML}"
echo "  Training data: ${TRAINING_NC}"
echo "  Models:        ${MODELS_DIR}"
echo "  Figures:       ${FIGS_DIR}"
