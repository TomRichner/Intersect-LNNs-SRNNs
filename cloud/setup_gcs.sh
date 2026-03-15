#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# setup_gcs.sh — One-time GCS bucket creation and dataset upload
# ─────────────────────────────────────────────────────────────────────
# Usage: ./cloud/setup_gcs.sh
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - cloud/config.env filled in
#   - Datasets exist at DATASETS_SOURCE_DIR
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# Path to the original Python repo's data directory
DATASETS_SOURCE_DIR="${DATASETS_SOURCE_DIR:-/Users/tom/Desktop/local_code/liquid_time_constant_networks/experiments_with_ltcs/data}"

echo "=== GCS Setup ==="
echo "  Project:  ${GCP_PROJECT}"
echo "  Bucket:   ${GCS_BUCKET}"
echo "  Datasets: ${DATASETS_SOURCE_DIR}"
echo ""

# ── Step 1: Create bucket ───────────────────────────────────────────
echo "Creating bucket ${GCS_BUCKET}..."
if gcloud storage buckets describe "${GCS_BUCKET}" &>/dev/null; then
    echo "  Bucket already exists, skipping."
else
    gcloud storage buckets create "${GCS_BUCKET}" \
        --location=us-central1 \
        --uniform-bucket-level-access
    echo "  Bucket created."
fi

# ── Step 2: Upload datasets ────────────────────────────────────────
DATASETS=(cheetah gesture har occupancy ozone person power traffic)

echo ""
echo "Uploading datasets..."
for ds in "${DATASETS[@]}"; do
    SRC="${DATASETS_SOURCE_DIR}/${ds}"
    DST="${GCS_BUCKET}/datasets/${ds}/"
    if [ -d "$SRC" ]; then
        echo "  Uploading ${ds}..."
        gsutil -m cp -r "${SRC}" "${GCS_BUCKET}/datasets/"
    else
        echo "  WARNING: ${SRC} not found, skipping."
    fi
done

# ── Step 3: Verify ─────────────────────────────────────────────────
echo ""
echo "Verifying uploads..."
gsutil ls "${GCS_BUCKET}/datasets/"
echo ""
gsutil du -sh "${GCS_BUCKET}/datasets/"
echo ""
echo "=== Setup complete ==="
