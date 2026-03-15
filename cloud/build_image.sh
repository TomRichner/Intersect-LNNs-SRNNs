#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# build_image.sh — Create a custom VM image with Julia pre-installed
# ─────────────────────────────────────────────────────────────────────
# Usage: ./cloud/build_image.sh
#
# This creates a reusable VM image with:
#   - Julia (version from config.env)
#   - All Julia packages pre-compiled
#   - git, build-essential
#
# Each experiment VM boots from this image, saving ~10-15 min of setup.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

BUILDER_VM="julia-image-builder"
IMAGE_NAME="srnn-julia-v1"
BOOT_DISK_SIZE="30GB"

echo "=== Building Julia VM Image ==="
echo "  Project:       ${GCP_PROJECT}"
echo "  Zone:          ${GCP_ZONE}"
echo "  Julia version: ${JULIA_VERSION}"
echo "  Image name:    ${IMAGE_NAME}"
echo "  Image family:  ${GCP_IMAGE_FAMILY}"
echo ""

# ── Step 1: Create temporary builder VM ─────────────────────────────
echo "Step 1: Creating builder VM '${BUILDER_VM}'..."
gcloud compute instances create "${BUILDER_VM}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --machine-type=e2-standard-4 \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size="${BOOT_DISK_SIZE}" \
    --scopes=storage-full

echo "  Waiting for VM to be ready..."
sleep 30

# ── Step 2: Install Julia and packages ──────────────────────────────
echo "Step 2: Installing Julia ${JULIA_VERSION} and packages..."

# Determine Julia download URL components
JULIA_MINOR=$(echo "${JULIA_VERSION}" | cut -d. -f1,2)

gcloud compute ssh "${BUILDER_VM}" --zone="${GCP_ZONE}" --command="
set -euo pipefail

echo '--- Installing system packages ---'
sudo apt-get update -qq
sudo apt-get install -y -qq git build-essential curl gzip libssl-dev libcurl4-openssl-dev > /dev/null

echo '--- Installing Julia ${JULIA_VERSION} ---'
curl -sL https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_MINOR}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz | sudo tar -xz -C /opt/
sudo ln -sf /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia
julia --version

echo '--- Cloning repo ---'
sudo git clone https://github.com/TomRichner/Intersect-LNNs-SRNNs.git /opt/srnn-repo
sudo chown -R \$(whoami) /opt/srnn-repo

echo '--- Installing Julia packages ---'
cd /opt/srnn-repo
julia --project=JuliaLang -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
    println(\"All packages installed and precompiled.\")
'

echo '--- Verifying key packages ---'
julia --project=JuliaLang -e '
    using Lux, Zygote, Optimisers, NNlib, JLD2
    println(\"Key packages load OK.\")
'

echo '--- Julia setup complete ---'
"

# ── Step 3: Stop the VM ────────────────────────────────────────────
echo ""
echo "Step 3: Stopping builder VM..."
gcloud compute instances stop "${BUILDER_VM}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}"

echo "  Waiting for VM to fully stop..."
sleep 15

# ── Step 4: Create image from disk ─────────────────────────────────
echo "Step 4: Creating image '${IMAGE_NAME}' from builder disk..."
gcloud compute images create "${IMAGE_NAME}" \
    --project="${GCP_PROJECT}" \
    --source-disk="${BUILDER_VM}" \
    --source-disk-zone="${GCP_ZONE}" \
    --family="${GCP_IMAGE_FAMILY}" \
    --description="Julia ${JULIA_VERSION} + SRNN packages, built $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Step 5: Clean up builder VM ────────────────────────────────────
echo "Step 5: Deleting builder VM..."
gcloud compute instances delete "${BUILDER_VM}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --quiet

# ── Verify ─────────────────────────────────────────────────────────
echo ""
echo "=== Image build complete ==="
gcloud compute images describe "${IMAGE_NAME}" \
    --project="${GCP_PROJECT}" \
    --format="table(name,family,status,diskSizeGb,creationTimestamp)"
