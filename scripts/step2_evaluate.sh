#!/bin/bash
#SBATCH --job-name=pann_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=00:30:00
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

echo "=== Step 2: Evaluate PANN ==="

python "${SOURCE_DIR}/evaluate.py" \
    --model_dir "${MODELS_DIR}" \
    --data      "${TRAINING_NC}" \
    --fig_dir   "${FIGS_DIR}" \
    --device    cpu

echo ""
echo "[$(date)] Step 2 complete.  Figures in ${FIGS_DIR}"
