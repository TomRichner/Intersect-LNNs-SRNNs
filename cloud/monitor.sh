#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# monitor.sh — Check experiment status, quota usage, and results
# ─────────────────────────────────────────────────────────────────────
# Usage:
#   ./cloud/monitor.sh              # show everything
#   ./cloud/monitor.sh har srnn     # show just har/srnn results
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

FILTER_EXP="${1:-}"
FILTER_MODEL="${2:-}"

# ── vCPU Quota ─────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  vCPU Quota"
echo "═══════════════════════════════════════════════════════════════"

# Global quota (CPUS_ALL_REGIONS) — this is typically the binding constraint
GLOBAL_INFO=$(gcloud compute project-info describe \
    --format="json(quotas)" 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); cpus=[q for q in d['quotas'] if q['metric']=='CPUS_ALL_REGIONS'][0]; print(f\"{int(cpus['usage'])},{int(cpus['limit'])}\")")

GLOBAL_USAGE=$(echo "${GLOBAL_INFO}" | cut -d',' -f1)
GLOBAL_LIMIT=$(echo "${GLOBAL_INFO}" | cut -d',' -f2)
GLOBAL_AVAIL=$(( GLOBAL_LIMIT - GLOBAL_USAGE ))
MAX_VMS=$(( GLOBAL_AVAIL / 2 ))

# Regional quota (for reference)
REGIONAL_INFO=$(gcloud compute regions describe us-central1 \
    --format="json(quotas)" 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); cpus=[q for q in d['quotas'] if q['metric']=='CPUS'][0]; print(f\"{int(cpus['usage'])},{int(cpus['limit'])}\")")

REGIONAL_USAGE=$(echo "${REGIONAL_INFO}" | cut -d',' -f1)
REGIONAL_LIMIT=$(echo "${REGIONAL_INFO}" | cut -d',' -f2)

echo "  Global:   ${GLOBAL_USAGE} / ${GLOBAL_LIMIT} vCPUs  ← binding limit"
echo "  Regional: ${REGIONAL_USAGE} / ${REGIONAL_LIMIT} vCPUs (us-central1)"
echo "  Can launch ${MAX_VMS} more e2-highmem-2 VMs"
echo ""

# ── Running VMs ────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  Running VMs"
echo "═══════════════════════════════════════════════════════════════"

VM_LIST=$(gcloud compute instances list \
    --filter="status=RUNNING" \
    --format="table(name,machineType.basename(),zone.basename(),status,creationTimestamp.date())" \
    2>/dev/null)

VM_COUNT=$(echo "${VM_LIST}" | tail -n +2 | wc -l | tr -d ' ')

if [ "${VM_COUNT}" -gt 0 ]; then
    echo "${VM_LIST}"
else
    echo "  No running VMs."
fi
echo ""

# ── Results in GCS ─────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  Results in GCS"
echo "═══════════════════════════════════════════════════════════════"

MODELS=("srnn" "ltc")
EXPERIMENTS=(har gesture occupancy smnist traffic power ozone person cheetah)

# Apply filters
if [ -n "${FILTER_MODEL}" ]; then
    MODELS=("${FILTER_MODEL}")
fi
if [ -n "${FILTER_EXP}" ]; then
    EXPERIMENTS=("${FILTER_EXP}")
fi

for model in "${MODELS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        SEEDS_DONE=0
        SEEDS_TOTAL=0
        STATUS_LINE=""

        source "${SCRIPT_DIR}/experiments/${exp}.env" 2>/dev/null || continue
        N=${N_SEEDS:-5}

        for seed in $(seq 1 ${N}); do
            SEEDS_TOTAL=$((SEEDS_TOTAL + 1))
            RESULT="${GCS_BUCKET}/results/${model}/${exp}/seed${seed}/final_metrics.json"

            if gsutil -q stat "${RESULT}" 2>/dev/null; then
                STATUS_LINE="${STATUS_LINE} ✅${seed}"
                SEEDS_DONE=$((SEEDS_DONE + 1))
            elif gsutil -q stat "${GCS_BUCKET}/results/${model}/${exp}/seed${seed}/training_log.txt" 2>/dev/null; then
                STATUS_LINE="${STATUS_LINE} ⏳${seed}"
            else
                STATUS_LINE="${STATUS_LINE} ·${seed}"
            fi
        done

        if [ ${SEEDS_DONE} -eq ${SEEDS_TOTAL} ]; then
            OVERALL="✅"
        elif [ ${SEEDS_DONE} -gt 0 ]; then
            OVERALL="🔶"
        else
            OVERALL="  "
        fi

        printf "  %s %-5s / %-10s  [%d/%d]  %s\n" \
            "${OVERALL}" "${model}" "${exp}" "${SEEDS_DONE}" "${SEEDS_TOTAL}" "${STATUS_LINE}"
    done
done

echo ""
echo "  Legend: ✅=done  ⏳=in progress  ·=not started"
echo ""
