#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# smoke_test.sh — Launch all experiments for 1 epoch (parallel, seed 1)
# ─────────────────────────────────────────────────────────────────────
# Usage:
#   ./cloud/smoke_test.sh              # 1 epoch (default)
#   ./cloud/smoke_test.sh --epochs 2   # 2 epochs
#
# Launches 9 VMs simultaneously (18 vCPUs, well under 64 quota).
# Each runs 1 epoch with the env's default bs/size, then self-deletes.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS="${2:-1}"

# Parse optional --epochs flag
OVERRIDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --epochs) OVERRIDE="--epochs $2"; shift 2 ;;
        *)        shift ;;
    esac
done
OVERRIDE="${OVERRIDE:---epochs 1}"

EXPERIMENTS=(har gesture occupancy smnist traffic power ozone person cheetah)
MODEL=srnn
SEED=1

echo "═══════════════════════════════════════════════════════════════"
echo "  Smoke Test — All Experiments (seed ${SEED})"
echo "  Override: ${OVERRIDE}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

LAUNCHED=0
FAILED=0

for exp in "${EXPERIMENTS[@]}"; do
    echo "── Launching ${exp}..."
    if "${SCRIPT_DIR}/launch_run.sh" "${exp}" "${MODEL}" "${SEED}" ${OVERRIDE}; then
        LAUNCHED=$((LAUNCHED + 1))
    else
        echo "  ⚠ FAILED to launch ${exp}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
    sleep 2  # small delay to avoid API rate limits
done

echo "═══════════════════════════════════════════════════════════════"
echo "  Smoke test launched: ${LAUNCHED}/${#EXPERIMENTS[@]} experiments"
if [ ${FAILED} -gt 0 ]; then
    echo "  ⚠ Failed: ${FAILED}"
fi
echo ""
echo "  Monitor all: gcloud compute instances list"
echo "  Monitor one: gcloud compute ssh srnn-<exp>-seed1 --command='tail -f /tmp/training.log'"
echo "═══════════════════════════════════════════════════════════════"
