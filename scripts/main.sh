#!/usr/bin/env bash
#
# PANN pipeline orchestrator.
#
# If block_size is set in config.yaml, submits parallel per-block training
# jobs followed by a single evaluation.  Otherwise, standard single-model
# training.
#
# Usage (from repo root):
#   bash scripts/main.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_setting.sh"

cd "${ROOT}"

if [[ -n "${PYTHON_ENV}" ]]; then
    # shellcheck disable=SC1090
    for f in ${PYTHON_ENV}; do source "$f"; break; done
fi

echo "=============================================="
echo "  PANN Training Pipeline"
echo "=============================================="
echo ""

if [[ ! -f "${TRAINING_NC}" ]]; then
    echo "ERROR: Training data not found at ${TRAINING_NC}"
    echo "Download it first — see README.md for links."
    exit 1
fi

TOTAL_LEVELS=$(ncdump -h "${TRAINING_NC}" | grep 'level = ' | head -1 | sed 's/[^0-9]//g')
N_BLOCKS=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG_YAML}'))
bs = cfg.get('block_size')
bo = cfg.get('block_overlap')
if bs is None:
    print(0)
else:
    lo = cfg.get('level_start') or 0
    hi = cfg.get('n_levels') or ${TOTAL_LEVELS}
    n = min(hi, ${TOTAL_LEVELS}) - lo
    stride = bs - bo
    count = 0
    s = 0
    while s < n:
        e = min(s + bs, n)
        if e - s < bs and count > 0:
            s = max(n - bs, 0)
        count += 1
        if e >= n:
            break
        s += stride
    print(count)
")

if [[ "${N_BLOCKS}" -gt 0 ]]; then
    echo "Level-blocking mode: ${N_BLOCKS} blocks"
    echo ""

    TRAIN_JOBS=""
    for B in $(seq 0 $((N_BLOCKS - 1))); do
        JOB=$(sbatch \
            --account="${SLURM_ACCOUNT}" \
            --partition="${SLURM_PARTITION}" \
            --time="${SLURM_TIME_TRAIN}" \
            --export=ALL,BLOCK_IDX=${B} \
            "${SCRIPT_DIR}/step1_train.sh" \
            | awk '{print $4}')
        echo "  Block ${B} train: job ${JOB}"
        TRAIN_JOBS="${TRAIN_JOBS}:${JOB}"
    done

    JOB_EVAL=$(sbatch \
        --account="${SLURM_ACCOUNT}" \
        --partition="${SLURM_PARTITION}" \
        --time="${SLURM_TIME_EVAL}" \
        --dependency=afterok${TRAIN_JOBS} \
        "${SCRIPT_DIR}/step2_evaluate.sh" \
        | awk '{print $4}')
    echo ""
    echo "  Evaluate: job ${JOB_EVAL}  (after all blocks)"

else
    echo "Standard (non-blocked) mode"
    echo ""

    JOB1=$(sbatch \
        --account="${SLURM_ACCOUNT}" \
        --partition="${SLURM_PARTITION}" \
        --time="${SLURM_TIME_TRAIN}" \
        "${SCRIPT_DIR}/step1_train.sh" \
        | awk '{print $4}')
    echo "Step 1 (train):    job ${JOB1}"

    JOB2=$(sbatch \
        --account="${SLURM_ACCOUNT}" \
        --partition="${SLURM_PARTITION}" \
        --time="${SLURM_TIME_EVAL}" \
        --dependency=afterok:${JOB1} \
        "${SCRIPT_DIR}/step2_evaluate.sh" \
        | awk '{print $4}')
    echo "Step 2 (evaluate): job ${JOB2}  (after ${JOB1})"

    JOB3=$(sbatch \
        --account="${SLURM_ACCOUNT}" \
        --partition="${SLURM_PARTITION}" \
        --time="${SLURM_TIME_EVAL}" \
        --dependency=afterok:${JOB1} \
        "${SCRIPT_DIR}/step3_extract_hidden.sh" \
        | awk '{print $4}')
    echo "Step 3 (hidden):   job ${JOB3}  (after ${JOB1})"
fi

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       ${LOG_DIR}/"
