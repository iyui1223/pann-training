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
# Activate script for the venv holding torch/xarray/netCDF4/matplotlib.
# A glob is allowed (first match wins), e.g. for a Poetry-managed venv:
#   PYTHON_ENV="$HOME/.cache/pypoetry/virtualenvs/pann-*/bin/activate"
# Set PYTHON_ENV="" to fall back to whatever `python` is already on PATH.
PYTHON_ENV="${PYTHON_ENV:-$HOME/venvs/pann-training/bin/activate}"

# Batch nodes are headless; pin a non-interactive matplotlib backend so
# figure generation cannot fail on a missing or stale DISPLAY.
export MPLBACKEND="${MPLBACKEND:-Agg}"

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
SLURM_TIME_TRAIN="${SLURM_TIME_TRAIN:-02:00:00}"

# Max block-training tasks running at once.  Each task imports ~930 MB across
# ~28k files from the shared filesystem, so a low cap keeps startup near its
# ~3 min floor instead of degrading badly, and limits wasted credit if the
# first wave fails.  Raise for throughput, lower for caution.
SLURM_ARRAY_THROTTLE="${SLURM_ARRAY_THROTTLE:-4}"
SLURM_TIME_EVAL="${SLURM_TIME_EVAL:-00:30:00}"

export ROOT SOURCE_DIR CONFIG_YAML
export TRAINING_NC
export MODELS_DIR HIDDEN_DIR FIGS_DIR LOG_DIR
export PYTHON_ENV
export SLURM_ACCOUNT SLURM_PARTITION SLURM_TIME_TRAIN SLURM_TIME_EVAL
export SLURM_ARRAY_THROTTLE

echo "PANN training environment loaded:"
echo "  Root:          ${ROOT}"
echo "  Config:        ${CONFIG_YAML}"
echo "  Training data: ${TRAINING_NC}"
echo "  Models:        ${MODELS_DIR}"
echo "  Figures:       ${FIGS_DIR}"
echo "  Python env:    ${PYTHON_ENV:-<system python>}"
