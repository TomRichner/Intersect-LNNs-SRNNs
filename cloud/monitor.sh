#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# monitor.sh — Check experiment status, quota usage, and results
# ─────────────────────────────────────────────────────────────────────
# Usage:
#   ./cloud/monitor.sh <run_name>              # show all results for a run
#   ./cloud/monitor.sh <run_name> har srnn     # show just har/srnn results
#   ./cloud/monitor.sh                         # show all VMs + quota (no results)
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

RUN_NAME="${1:-}"
FILTER_EXP="${2:-}"
FILTER_MODEL="${3:-}"

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
echo "  Can launch ${MAX_VMS} more n4d-highmem-2 VMs"
echo ""

# ── Running VMs ────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  Running VMs"
echo "═══════════════════════════════════════════════════════════════"

if [ -n "${RUN_NAME}" ]; then
    VM_FILTER="name~'^${RUN_NAME}-' AND status=RUNNING"
else
    VM_FILTER="status=RUNNING"
fi

VM_LIST=$(gcloud compute instances list \
    --filter="${VM_FILTER}" \
    --format="table(name,machineType.basename(),zone.basename(),status,creationTimestamp.date())" \
    2>/dev/null)

VM_COUNT=$(echo "${VM_LIST}" | tail -n +2 | wc -l | tr -d ' ')

if [ "${VM_COUNT}" -gt 0 ]; then
    echo "${VM_LIST}"
else
    echo "  No running VMs${RUN_NAME:+ for run '${RUN_NAME}'}."
fi
echo ""

# ── Results in GCS ─────────────────────────────────────────────────
if [ -z "${RUN_NAME}" ]; then
    echo "  (Specify a run name to see GCS results)"
    echo ""
    exit 0
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Results in GCS — run: ${RUN_NAME}"
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
            RESULT="${GCS_BUCKET}/results/${RUN_NAME}/${model}/${exp}/seed${seed}/training_log.txt"

            if gsutil -q stat "${RESULT}" 2>/dev/null; then
                STATUS_LINE="${STATUS_LINE} ✅${seed}"
                SEEDS_DONE=$((SEEDS_DONE + 1))
            else
                # Check if VM is running for this seed
                VM_CHECK="${RUN_NAME}-${model}-${exp}-seed${seed}"
                if gcloud compute instances describe "${VM_CHECK}" --zone="${GCP_ZONE}" &>/dev/null 2>&1; then
                    STATUS_LINE="${STATUS_LINE} ⏳${seed}"
                else
                    STATUS_LINE="${STATUS_LINE} ·${seed}"
                fi
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
echo "  Legend: ✅=done  ⏳=running  ·=not started"
echo ""
