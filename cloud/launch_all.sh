#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# launch_all.sh — Launch all experiments × all seeds (batched for quota)
# ─────────────────────────────────────────────────────────────────────
# Usage:
#   ./cloud/launch_all.sh <run_name>                    # all experiments, seeds 1-5
#   ./cloud/launch_all.sh <run_name> --seeds 3          # seeds 1-3 only
#   ./cloud/launch_all.sh <run_name> --epochs 10        # override epochs
#   ./cloud/launch_all.sh <run_name> --dry-run          # print what would launch
#
# Quota: 24 VMs max, 64 vCPUs max.
# With 9 experiments × 2 vCPUs each, we can fit 2 seeds per wave
# (18 VMs = 36 vCPUs), leaving headroom for other VMs.
#
# Waves:
#   Wave 1: seeds 1-2 (18 VMs)
#   Wave 2: seeds 3-4 (18 VMs)
#   Wave 3: seed 5    (9 VMs)
# Each wave waits for all previous VMs to self-delete before launching.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
MAX_SEEDS=5
OVERRIDE=""
DRY_RUN=false
MAX_VMS=24
WAVE_SIZE=2  # seeds per wave (2 × 9 = 18 VMs < 24 limit)

# Parse arguments — first positional arg is required run name
if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_name> [--seeds N] [--epochs N] [--bs N] [--dry-run]"
    echo "  run_name: required label (e.g. 'prod', 'smoke20')"
    exit 1
fi

RUN_NAME=$1
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --seeds)   MAX_SEEDS=$2; shift 2 ;;
        --epochs)  OVERRIDE="${OVERRIDE} --epochs $2"; shift 2 ;;
        --bs)      OVERRIDE="${OVERRIDE} --bs $2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *)         echo "Unknown flag: $1"; exit 1 ;;
    esac
done

EXPERIMENTS=(har gesture occupancy smnist traffic power ozone person cheetah)
MODEL=srnn

TOTAL_VMS=$(( ${#EXPERIMENTS[@]} * MAX_SEEDS ))

echo "═══════════════════════════════════════════════════════════════"
echo "  Full Experiment Launch: ${RUN_NAME}"
echo "  Experiments:  ${#EXPERIMENTS[@]}"
echo "  Seeds:        1-${MAX_SEEDS}"
echo "  Total VMs:    ${TOTAL_VMS}"
echo "  Wave size:    ${WAVE_SIZE} seeds ($(( WAVE_SIZE * ${#EXPERIMENTS[@]} )) VMs per wave)"
echo "  VM quota:     ${MAX_VMS}"
echo "  Override:     ${OVERRIDE:-none}"
echo "  Dry run:      ${DRY_RUN}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Helper: wait until running srnn VMs < threshold ──────────────────
wait_for_slots() {
    local max_running=$1
    while true; do
        RUNNING=$(gcloud compute instances list \
            --filter="name~'^${RUN_NAME}-' AND status=RUNNING" \
            --format="value(name)" 2>/dev/null | wc -l | tr -d ' ')
        if [ "${RUNNING}" -lt "${max_running}" ]; then
            return
        fi
        echo "  $(date +%H:%M:%S) — ${RUNNING} VMs running (limit: ${max_running}), waiting..."
        sleep 60
    done
}

# ── Launch in waves ──────────────────────────────────────────────────
WAVE=1
SEED=1
LAUNCHED=0
FAILED=0

while [ ${SEED} -le ${MAX_SEEDS} ]; do
    WAVE_END=$(( SEED + WAVE_SIZE - 1 ))
    if [ ${WAVE_END} -gt ${MAX_SEEDS} ]; then
        WAVE_END=${MAX_SEEDS}
    fi
    WAVE_VMS=$(( (WAVE_END - SEED + 1) * ${#EXPERIMENTS[@]} ))

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Wave ${WAVE}: seeds ${SEED}-${WAVE_END} (${WAVE_VMS} VMs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Wait for enough slots before launching this wave
    if [ "${DRY_RUN}" = false ] && [ ${WAVE} -gt 1 ]; then
        echo ""
        echo "  ⏳ Waiting for Wave $(( WAVE - 1 )) to finish..."
        wait_for_slots ${WAVE_VMS}
        echo ""
    fi

    for s in $(seq ${SEED} ${WAVE_END}); do
        for exp in "${EXPERIMENTS[@]}"; do
            if [ "${DRY_RUN}" = true ]; then
                echo "  [dry-run] launch_run.sh ${RUN_NAME} ${exp} ${MODEL} ${s} ${OVERRIDE}"
            else
                echo "── Launching ${exp} seed ${s}..."
                if "${SCRIPT_DIR}/launch_run.sh" "${RUN_NAME}" "${exp}" "${MODEL}" "${s}" ${OVERRIDE}; then
                    LAUNCHED=$((LAUNCHED + 1))
                else
                    echo "  ⚠ FAILED: ${exp} seed ${s}"
                    FAILED=$((FAILED + 1))
                fi
                sleep 2
            fi
        done
    done

    SEED=$(( WAVE_END + 1 ))
    WAVE=$(( WAVE + 1 ))
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
if [ "${DRY_RUN}" = true ]; then
    echo "  Dry run complete. ${TOTAL_VMS} VMs would be launched."
else
    echo "  All waves launched: ${LAUNCHED} succeeded, ${FAILED} failed"
fi
echo "  Monitor: gcloud compute instances list --filter=\"name~'^${RUN_NAME}-'\""
echo "═══════════════════════════════════════════════════════════════"
