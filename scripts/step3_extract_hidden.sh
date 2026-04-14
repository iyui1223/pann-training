#!/bin/bash
#SBATCH --job-name=pann_hidden
#SBATCH --output=logs/hidden_%j.out
#SBATCH --error=logs/hidden_%j.err
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

echo "=== Step 3: Extract hidden-layer activations ==="

python "${SOURCE_DIR}/extract_hidden.py" \
    --model_dir "${MODELS_DIR}" \
    --data      "${TRAINING_NC}" \
    --out_dir   "${HIDDEN_DIR}" \
    --device    cpu

echo ""
echo "[$(date)] Step 3 complete.  Hidden activations in ${HIDDEN_DIR}"
