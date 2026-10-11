#!/bin/bash
#SBATCH --job-name=pann_zab
#SBATCH --output=logs/zero_anchored_bias_%A_%a.out
#SBATCH --error=logs/zero_anchored_bias_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#
# Zero-anchored branch-bias experiment: one array task per bias variant.
# Submit from the repo root:
#   sbatch -A "${SLURM_ACCOUNT:-CRANMER-SL3-CPU}" -p icelake experiments/zero_anchored_bias_slurm.sh
# Then combine all variants into one table + figure:
#   python experiments/zero_anchored_bias.py --plot_only

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/scripts/env_setting.sh"

if [[ -n "${PYTHON_ENV}" ]]; then
    for f in ${PYTHON_ENV}; do source "$f"; break; done
fi

cd "${ROOT}"

VARIANTS=(free gated anchored)
VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT_DIR="${ROOT}/output/experiments/zero_anchored_bias"

echo "=== Zero-anchored bias experiment: ${VARIANT} ==="
python -u experiments/zero_anchored_bias.py \
    --config   "${CONFIG_YAML}" \
    --data     "${TRAINING_NC}" \
    --out_dir  "${OUT_DIR}" \
    --variants "${VARIANT}" \
    --seeds 0 1 2 \
    --epochs 150 \
    --patience 20

echo "[$(date)] ${VARIANT} complete."
