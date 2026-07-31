#!/bin/bash
#SBATCH --job-name=pann_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

# Slurm runs a spooled copy of this script from /var/spool/slurm/..., so
# BASH_SOURCE cannot locate the repo.  SLURM_SUBMIT_DIR is the directory
# sbatch was invoked from, which main.sh guarantees is the repo root.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/scripts/env_setting.sh"

if [[ -n "${PYTHON_ENV}" ]]; then
    for f in ${PYTHON_ENV}; do source "$f"; break; done
fi

cd "${ROOT}"

echo "=== Step 1: Train PANN ==="
echo "Config:  ${CONFIG_YAML}"
echo "Data:    ${TRAINING_NC}"
echo "Output:  ${MODELS_DIR}"

# Block index comes from the array task id, or an explicit BLOCK_IDX export
# when a single block is submitted by hand.
BLOCK_IDX="${BLOCK_IDX:-${SLURM_ARRAY_TASK_ID:-}}"

BLOCK_ARG=""
if [[ -n "${BLOCK_IDX}" ]]; then
    echo "Block:   ${BLOCK_IDX}"
    BLOCK_ARG="--block_idx ${BLOCK_IDX}"
fi
echo ""

# -u keeps stdout unbuffered so progress appears in the log as it happens
# rather than only when the process exits.
python -u "${SOURCE_DIR}/train.py" \
    --config "${CONFIG_YAML}" \
    --data   "${TRAINING_NC}" \
    --save_dir "${MODELS_DIR}" \
    --device cpu \
    ${BLOCK_ARG}

echo ""
echo "[$(date)] Step 1 complete."
