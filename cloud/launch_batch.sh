#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# launch_batch.sh — Launch all seeds for an experiment
# ─────────────────────────────────────────────────────────────────────
# Usage:
#   ./cloud/launch_batch.sh <experiment> <model>
#
# Examples:
#   ./cloud/launch_batch.sh har srnn        # launches 5 VMs
#   ./cloud/launch_batch.sh smnist srnn     # launches 5 VMs
#
# Reads N_SEEDS from the experiment .env file.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment> <model>"
    echo "  experiment: har, gesture, occupancy, smnist, traffic, power, ozone, person, cheetah"
    echo "  model:      srnn, ltc"
    exit 1
fi

EXPERIMENT=$1
MODEL=$2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/experiments/${EXPERIMENT}.env"

N=${N_SEEDS:-5}

echo "=== Batch Launch: ${MODEL} / ${EXPERIMENT} (${N} seeds) ==="
echo ""

LAUNCHED=0
SKIPPED=0

for SEED in $(seq 1 ${N}); do
    echo "--- Seed ${SEED}/${N} ---"
    if "${SCRIPT_DIR}/launch_run.sh" "${EXPERIMENT}" "${MODEL}" "${SEED}"; then
        LAUNCHED=$((LAUNCHED + 1))
    else
        SKIPPED=$((SKIPPED + 1))
    fi
    echo ""
    # Small delay between launches to avoid API rate limits
    sleep 3
done

echo "=== Batch complete: ${LAUNCHED} launched, ${SKIPPED} skipped ==="
