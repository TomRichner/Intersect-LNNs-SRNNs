#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# launch_batch.sh — Launch all seeds for an experiment
# ─────────────────────────────────────────────────────────────────────
# Usage:
#   ./cloud/launch_batch.sh <run_name> <experiment> <model>
#
# Examples:
#   ./cloud/launch_batch.sh prod har srnn        # launches 5 VMs
#   ./cloud/launch_batch.sh smoke20 smnist srnn  # launches 5 VMs
#
# Reads N_SEEDS from the experiment .env file.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <run_name> <experiment> <model>"
    echo "  run_name:   required label (e.g. 'prod', 'smoke20')"
    echo "  experiment: har, gesture, occupancy, smnist, traffic, power, ozone, person, cheetah"
    echo "  model:      srnn, ltc"
    exit 1
fi

RUN_NAME=$1
EXPERIMENT=$2
MODEL=$3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/experiments/${EXPERIMENT}.env"

N=${N_SEEDS:-5}

echo "=== Batch Launch: ${RUN_NAME} / ${MODEL} / ${EXPERIMENT} (${N} seeds) ==="
echo ""

LAUNCHED=0
SKIPPED=0

for SEED in $(seq 1 ${N}); do
    echo "--- Seed ${SEED}/${N} ---"
    if "${SCRIPT_DIR}/launch_run.sh" "${RUN_NAME}" "${EXPERIMENT}" "${MODEL}" "${SEED}"; then
        LAUNCHED=$((LAUNCHED + 1))
    else
        SKIPPED=$((SKIPPED + 1))
    fi
    echo ""
    # Small delay between launches to avoid API rate limits
    sleep 3
done

echo "=== Batch complete: ${LAUNCHED} launched, ${SKIPPED} skipped ==="
