#!/bin/bash
#SBATCH --job-name=pann_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=00:30:00
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

echo "=== Step 2: Evaluate PANN ==="

python -u "${SOURCE_DIR}/evaluate.py" \
    --model_dir "${MODELS_DIR}" \
    --data      "${TRAINING_NC}" \
    --fig_dir   "${FIGS_DIR}" \
    --device    cpu

echo ""
echo "[$(date)] Step 2 complete.  Figures in ${FIGS_DIR}"
