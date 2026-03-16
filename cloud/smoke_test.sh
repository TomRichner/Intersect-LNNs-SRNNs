#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# smoke_test.sh — Launch all experiments for quick validation (seed 1)
# ─────────────────────────────────────────────────────────────────────
# Usage:
#   ./cloud/smoke_test.sh --run smoke20              # 1 epoch (default)
#   ./cloud/smoke_test.sh --run smoke20 --epochs 20  # 20 epochs
#
# Launches 9 VMs simultaneously (18 vCPUs, well under 64 quota).
# Each runs with the env's default bs/size, then self-deletes.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse flags
RUN_NAME=""
OVERRIDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --run)    RUN_NAME="$2"; shift 2 ;;
        --epochs) OVERRIDE="--epochs $2"; shift 2 ;;
        *)        shift ;;
    esac
done
OVERRIDE="${OVERRIDE:---epochs 1}"

if [ -z "${RUN_NAME}" ]; then
    echo "Usage: $0 --run <name> [--epochs N]"
    echo "  --run:    required label (e.g. 'smoke20', 'quick-test')"
    echo "  --epochs: override epoch count (default: 1)"
    exit 1
fi

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
    if "${SCRIPT_DIR}/launch_run.sh" "${RUN_NAME}" "${exp}" "${MODEL}" "${SEED}" ${OVERRIDE}; then
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
echo "  Monitor all: ./cloud/monitor.sh ${RUN_NAME}"
echo "  Monitor one: gcloud compute ssh ${RUN_NAME}-srnn-<exp>-seed1 --tunnel-through-iap --command='tail -f /tmp/training.log'"
echo "═══════════════════════════════════════════════════════════════"
