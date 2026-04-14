#!/bin/bash
#SBATCH --job-name=pann_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_setting.sh"

if [[ -n "${PYTHON_ENV}" ]]; then
    for f in ${PYTHON_ENV}; do source "$f"; break; done
fi

cd "${ROOT}"

echo "=== Step 1: Train PANN ==="
echo "Config:  ${CONFIG_YAML}"
echo "Data:    ${TRAINING_NC}"
echo "Output:  ${MODELS_DIR}"

BLOCK_ARG=""
if [[ -n "${BLOCK_IDX:-}" ]]; then
    echo "Block:   ${BLOCK_IDX}"
    BLOCK_ARG="--block_idx ${BLOCK_IDX}"
fi
echo ""

python "${SOURCE_DIR}/train.py" \
    --config "${CONFIG_YAML}" \
    --data   "${TRAINING_NC}" \
    --save_dir "${MODELS_DIR}" \
    --device cpu \
    ${BLOCK_ARG}

echo ""
echo "[$(date)] Step 1 complete."
