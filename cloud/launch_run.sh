#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# launch_run.sh — Launch a single experiment VM
# ─────────────────────────────────────────────────────────────────────
# Usage:
#   ./cloud/launch_run.sh <experiment> <model> <seed> [--epochs N] [--bs N]
#
# Examples:
#   ./cloud/launch_run.sh har srnn 1              # uses env defaults
#   ./cloud/launch_run.sh har srnn 1 --epochs 1   # smoke test (1 epoch)
#   ./cloud/launch_run.sh smnist srnn 3 --bs 64 --epochs 2
#
# The script:
#   1. Reads cloud/config.env for GCP settings
#   2. Reads cloud/experiments/<experiment>.env for training args
#   3. Creates a VM with the startup script and all config as metadata
#   4. VM boots, trains, uploads results, self-deletes
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Parse arguments ────────────────────────────────────────────────
if [ $# -lt 3 ]; then
    echo "Usage: $0 <experiment> <model> <seed> [--epochs N] [--bs N]"
    echo "  experiment: har, gesture, occupancy, smnist, traffic, power, ozone, person, cheetah"
    echo "  model:      srnn, ltc"
    echo "  seed:       1-5"
    echo "  --epochs N: override epoch count (appended to env ARGS)"
    echo "  --bs N:     override batch size (appended to env ARGS)"
    exit 1
fi

EXPERIMENT=$1
MODEL=$2
SEED=$3
shift 3

# Parse optional overrides
OVERRIDE_ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --epochs) OVERRIDE_ARGS="${OVERRIDE_ARGS} --epochs $2"; shift 2 ;;
        --bs)     OVERRIDE_ARGS="${OVERRIDE_ARGS} --bs $2"; shift 2 ;;
        *)        echo "Unknown override: $1"; exit 1 ;;
    esac
done

# ── Load config ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

EXP_ENV="${SCRIPT_DIR}/experiments/${EXPERIMENT}.env"
if [ ! -f "${EXP_ENV}" ]; then
    echo "ERROR: Experiment config not found: ${EXP_ENV}"
    echo "Available experiments:"
    ls "${SCRIPT_DIR}/experiments/"*.env | xargs -I{} basename {} .env
    exit 1
fi
source "${EXP_ENV}"

# ── Apply CLI overrides (appended last → last-wins in Julia parsers) ─
if [ -n "${OVERRIDE_ARGS}" ]; then
    ARGS="${ARGS}${OVERRIDE_ARGS}"
fi

# ── Determine VM name and machine type ─────────────────────────────
VM_NAME="${MODEL}-${EXPERIMENT_NAME}-seed${SEED}"
VM_MACHINE="${MACHINE_TYPE:-${GCP_MACHINE_TYPE}}"

# ── Check quota before launching ───────────────────────────────────
echo "=== Pre-launch Check ==="
CURRENT_VMS=$(gcloud compute instances list --filter="status=RUNNING" --format="value(name)" 2>/dev/null | wc -l | tr -d ' ')
# Get vCPU count for the chosen machine type
VCPUS_PER_VM=$(echo "${VM_MACHINE}" | grep -oE '[0-9]+$' || echo "2")
CURRENT_VCPUS=$(( CURRENT_VMS * 2 ))  # approximate, most are 2-vCPU
echo "  Current running VMs:  ${CURRENT_VMS}"
echo "  Approx vCPUs in use:  ${CURRENT_VCPUS} / 64"
echo ""

# ── Check if VM already exists ─────────────────────────────────────
if gcloud compute instances describe "${VM_NAME}" --zone="${GCP_ZONE}" &>/dev/null; then
    echo "ERROR: VM '${VM_NAME}' already exists!"
    echo "  To delete: gcloud compute instances delete ${VM_NAME} --zone=${GCP_ZONE} --quiet"
    exit 1
fi

# ── Check if results already exist ─────────────────────────────────
RESULT_PATH="${GCS_BUCKET}/results/${MODEL}/${EXPERIMENT_NAME}/seed${SEED}/final_metrics.json"
if gsutil -q stat "${RESULT_PATH}" 2>/dev/null; then
    echo "WARNING: Results already exist for ${MODEL}/${EXPERIMENT_NAME}/seed${SEED}"
    read -p "  Overwrite? (y/N): " CONFIRM
    if [ "${CONFIRM}" != "y" ]; then
        echo "  Skipping."
        exit 0
    fi
fi

# ── Create the VM ──────────────────────────────────────────────────
echo "=== Launching VM ==="
echo "  Name:     ${VM_NAME}"
echo "  Machine:  ${VM_MACHINE}"
echo "  Zone:     ${GCP_ZONE}"
echo "  Image:    ${GCP_IMAGE_FAMILY}"
echo "  Spot:     ${GCP_USE_SPOT}"
echo "  Script:   ${TRAIN_SCRIPT}"
echo "  Args:     ${ARGS}"
echo "  Seed:     ${SEED}"
echo ""

SPOT_FLAGS=""
if [ "${GCP_USE_SPOT}" = "true" ]; then
    SPOT_FLAGS="--provisioning-model=SPOT --instance-termination-action=STOP"
fi

gcloud compute instances create "${VM_NAME}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --machine-type="${VM_MACHINE}" \
    --image-family="${GCP_IMAGE_FAMILY}" \
    --boot-disk-size=30GB \
    ${SPOT_FLAGS} \
    --metadata="experiment=${EXPERIMENT_NAME},model=${MODEL},seed=${SEED},train-script=${TRAIN_SCRIPT},train-args=${ARGS},gcs-bucket=${GCS_BUCKET}" \
    --metadata-from-file=startup-script="${SCRIPT_DIR}/startup.sh" \
    --scopes=storage-full,compute-rw

echo ""
echo "=== VM '${VM_NAME}' launched ==="
echo "  Monitor:  gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} --command='tail -f /tmp/training.log'"
echo "  Status:   gcloud compute instances describe ${VM_NAME} --zone=${GCP_ZONE} --format='value(status)'"
echo "  Delete:   gcloud compute instances delete ${VM_NAME} --zone=${GCP_ZONE} --quiet"
