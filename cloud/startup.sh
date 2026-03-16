#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# startup.sh — VM startup script for cloud experiment runs
# ─────────────────────────────────────────────────────────────────────
# This script runs automatically on VM boot (as root). It is passed as
# --metadata-from-file=startup-script=cloud/startup.sh when creating
# the VM via launch_run.sh.
#
# It reads experiment config from VM metadata, downloads the dataset,
# runs training, uploads results to GCS, and self-deletes the VM.
#
# NOTE: Julia and git commands run as 'tom' (the user who built the
# VM image) so that pre-compiled packages are found correctly.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# Helper: run a command as the tom user (who owns the Julia depot)
run_as_tom() { sudo -u tom -i bash -c "$*"; }

# ── Read VM metadata (set by launch_run.sh) ─────────────────────────
META_URL="http://metadata.google.internal/computeMetadata/v1/instance"
META_HEADER="Metadata-Flavor: Google"

get_meta() { curl -s "${META_URL}/attributes/$1" -H "${META_HEADER}"; }

EXPERIMENT=$(get_meta experiment)
MODEL=$(get_meta model)
SEED=$(get_meta seed)
TRAIN_SCRIPT=$(get_meta train-script)
TRAIN_ARGS=$(get_meta train-args)
GCS_BUCKET=$(get_meta gcs-bucket)
RUN_NAME=$(get_meta run-name)
VM_NAME=$(curl -s "${META_URL}/name" -H "${META_HEADER}")
VM_ZONE=$(curl -s "${META_URL}/zone" -H "${META_HEADER}" | awk -F/ '{print $NF}')

RESULT_PATH="${GCS_BUCKET}/results/${RUN_NAME}/${MODEL}/${EXPERIMENT}/seed${SEED}"
CHECKPOINT_DIR="/tmp/checkpoints"
LOG_FILE="/tmp/training.log"

# ── Redirect all output to log file ────────────────────────────────
touch "$LOG_FILE" && chmod 666 "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "═══════════════════════════════════════════════════════════════"
echo "  SRNN Cloud Experiment Runner"
echo "═══════════════════════════════════════════════════════════════"
echo "  VM:         ${VM_NAME}"
echo "  Run:        ${RUN_NAME}"
echo "  Experiment: ${EXPERIMENT}"
echo "  Model:      ${MODEL}"
echo "  Seed:       ${SEED}"
echo "  Script:     ${TRAIN_SCRIPT}"
echo "  Args:       ${TRAIN_ARGS}"
echo "  GCS Bucket: ${GCS_BUCKET}"
echo "  Results:    ${RESULT_PATH}"
echo "  Started:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 1: Update code ────────────────────────────────────────────
echo ""
echo "=== Step 1: Pulling latest code ==="
run_as_tom "git config --global --add safe.directory /opt/srnn-repo && cd /opt/srnn-repo && git pull --ff-only" || echo "WARNING: git pull failed, using baked-in version"
echo "  Commit: $(cd /opt/srnn-repo && git rev-parse --short HEAD)"

# ── Step 2: Download dataset from GCS ──────────────────────────────
echo ""
echo "=== Step 2: Downloading dataset '${EXPERIMENT}' ==="
DATASET_DIR="/opt/srnn-repo/JuliaLang/data/${EXPERIMENT}"
mkdir -p "${DATASET_DIR}"
chown -R tom:tom "${DATASET_DIR}"
gsutil -m cp -r "${GCS_BUCKET}/datasets/${EXPERIMENT}/*" "${DATASET_DIR}/" 2>&1
chown -R tom:tom "${DATASET_DIR}"
echo "  Downloaded to ${DATASET_DIR}"
ls -lh "${DATASET_DIR}/"

# ── Step 3: Check for existing checkpoint (Spot VM resume) ─────────
echo ""
echo "=== Step 3: Checking for existing checkpoint ==="
CHECKPOINT_GCS="${GCS_BUCKET}/checkpoints/${RUN_NAME}/${MODEL}-${EXPERIMENT}-seed${SEED}"
RESUME_FLAG=""
mkdir -p "${CHECKPOINT_DIR}"
chmod 777 "${CHECKPOINT_DIR}"

if gsutil -q stat "${CHECKPOINT_GCS}/latest.jld2" 2>/dev/null; then
    echo "  Found existing checkpoint — downloading for resume"
    gsutil cp "${CHECKPOINT_GCS}/latest.jld2" "${CHECKPOINT_DIR}/resume.jld2"
    chmod 666 "${CHECKPOINT_DIR}/resume.jld2"
    RESUME_FLAG="--resume ${CHECKPOINT_DIR}/resume.jld2"
    echo "  Resume flag: ${RESUME_FLAG}"
else
    echo "  No existing checkpoint — starting fresh"
fi

# ── Step 4: Run training ───────────────────────────────────────────
echo ""
echo "=== Step 4: Starting training ==="
JULIA_CMD="cd /opt/srnn-repo && stdbuf -oL julia --project=JuliaLang ${TRAIN_SCRIPT} --model ${MODEL} --seed ${SEED} --save ${CHECKPOINT_DIR} ${TRAIN_ARGS} ${RESUME_FLAG}"
echo "  Command: ${JULIA_CMD}"
echo ""

TRAIN_START=$(date +%s)

run_as_tom "${JULIA_CMD}"

TRAIN_EXIT=$?
TRAIN_END=$(date +%s)
TRAIN_DURATION=$(( TRAIN_END - TRAIN_START ))

echo ""
echo "  Training exit code: ${TRAIN_EXIT}"
echo "  Training duration:  ${TRAIN_DURATION}s ($(( TRAIN_DURATION / 60 ))m $(( TRAIN_DURATION % 60 ))s)"

# ── Step 5: Upload results to GCS ──────────────────────────────────
echo ""
echo "=== Step 5: Uploading results ==="

# Upload the final log
gsutil cp "${LOG_FILE}" "${RESULT_PATH}/training_log.txt"

# Upload best checkpoint
if ls ${CHECKPOINT_DIR}/*best* 1>/dev/null 2>&1; then
    gsutil cp ${CHECKPOINT_DIR}/*best* "${RESULT_PATH}/"
    echo "  Uploaded best checkpoint"
fi

# Upload final metrics JSON
if [ -f "${CHECKPOINT_DIR}/final_metrics.json" ]; then
    gsutil cp "${CHECKPOINT_DIR}/final_metrics.json" "${RESULT_PATH}/"
    echo "  Uploaded final_metrics.json"
fi

# Upload run metadata (args, config)
if [ -f "${CHECKPOINT_DIR}/run_metadata.json" ]; then
    gsutil cp "${CHECKPOINT_DIR}/run_metadata.json" "${RESULT_PATH}/"
    echo "  Uploaded run_metadata.json"
fi

# Also save latest checkpoint for Spot VM resume of other seeds
if ls ${CHECKPOINT_DIR}/*epoch* 1>/dev/null 2>&1; then
    LATEST=$(ls -t ${CHECKPOINT_DIR}/*epoch* | head -1)
    gsutil cp "${LATEST}" "${CHECKPOINT_GCS}/latest.jld2"
    echo "  Uploaded latest checkpoint for resume"
fi

# Upload final log again (now includes upload messages)
gsutil cp "${LOG_FILE}" "${RESULT_PATH}/training_log.txt"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Results at:  ${RESULT_PATH}"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 6: Self-delete the VM ─────────────────────────────────────
echo ""
echo "=== Step 6: Self-deleting VM ==="
# Give a moment for the final log upload
sleep 5
gcloud compute instances delete "${VM_NAME}" --zone="${VM_ZONE}" --quiet
