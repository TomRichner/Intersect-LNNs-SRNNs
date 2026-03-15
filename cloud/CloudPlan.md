# Cloud Experiment Plan — SRNN vs LTC (Hasani et al. 2021, Table 3)

## Goal

Reproduce the experiments from Hasani et al. 2021 (Tables 3, 4, 5, 6) using:
- **SRNN** (our model, `srnn.jl`)
- **LTC** (reimplemented in Julia, `ltc1.jl`)

All training runs are executed on Google Cloud ephemeral VMs. Each run gets its own
VM, results are stored centrally in a GCS bucket, and VMs self-delete after completion.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL MAC (orchestrator)                                       │
│                                                                 │
│  launch_run.sh ──┬── gcloud compute instances create vm-1       │
│                  ├── gcloud compute instances create vm-2       │
│                  └── ...                                        │
│                                                                 │
│  monitor.sh ──────── polls VM status + GCS results              │
│  collect_results.sh ── downloads results, computes mean ± std   │
└─────────────────────────────────────────────────────────────────┘
         │                         ▲
         │  (create VMs)           │  (upload results)
         ▼                         │
┌─────────────────────────────────────────────────────────────────┐
│  GOOGLE CLOUD                                                   │
│                                                                 │
│  ┌──────────────────────────────┐   ┌────────────────────────┐  │
│  │  GCS Bucket                  │   │  VM: srnn-har-seed1    │  │
│  │  gs://srnn-experiments/      │   │  ┌──────────────────┐  │  │
│  │                              │   │  │ 1. Clone repo    │  │  │
│  │  ├── datasets/               │◄──│  │ 2. Pull data     │  │  │
│  │  │   ├── har/                │   │  │ 3. Train model   │  │  │
│  │  │   ├── gesture/            │   │  │ 4. Upload results│  │  │
│  │  │   └── ...                 │   │  │ 5. Self-delete   │  │  │
│  │  ├── results/                │   │  └──────────────────┘  │  │
│  │  │   ├── har/srnn/seed1/     │   └────────────────────────┘  │
│  │  │   ├── har/srnn/seed2/     │                               │
│  │  │   └── ...                 │   ┌────────────────────────┐  │
│  │  └── checkpoints/            │   │  VM: srnn-har-seed2    │  │
│  │      └── (for Spot resume)   │   │  (same as above)       │  │
│  └──────────────────────────────┘   └────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Experiments To Run

From Hasani et al. 2021, we need to reproduce the following. All reported as
**mean ± std** over **n = 5** seeds.

### Table 3 — Time Series Prediction

| # | Experiment   | Task Type      | Metric   | Seq Len | Features | Classes/Out | VM Size         |
|---|-------------|----------------|----------|---------|----------|-------------|-----------------|
| 1 | HAR          | Classification | Accuracy | 16      | 561      | 6           | e2-standard-4   |
| 2 | Gesture      | Classification | Accuracy | 32      | varies   | 5           | e2-standard-4   |
| 3 | Occupancy    | Classification | Accuracy | 16      | 5        | 2           | e2-standard-4   |
| 4 | SMnist       | Classification | Accuracy | 784     | 1        | 10          | e2-standard-8   |
| 5 | Traffic      | Regression     | MSE      | 32      | varies   | 1           | e2-standard-4   |
| 6 | Power        | Regression     | MSE      | 32      | varies   | 1           | e2-standard-4   |
| 7 | Ozone        | Classification | F1-score | 32      | 72       | 2           | e2-standard-4   |

### Tables 4, 5 — Person Activity

| # | Experiment   | Task Type      | Metric   | Setting | VM Size         |
|---|-------------|----------------|----------|---------|-----------------|
| 8 | Person (1st) | Classification | Accuracy | Standard| e2-standard-4   |
| 9 | Person (2nd) | Classification | Accuracy | Rubanova| e2-standard-4   |

### Table 6 — Half-Cheetah

| #  | Experiment   | Task Type  | Metric | VM Size         |
|----|-------------|------------|--------|-----------------|
| 10 | Cheetah      | Regression | MSE    | e2-standard-4   |

### Total Runs

- **10 experiments × 5 seeds × 2 models (SRNN + LTC) = 100 runs**
- Phase 1 (SRNN only): 50 runs
- Phase 2 (LTC): 50 runs

---

## Directory Structure

```
Intersect-LNNs-SRNNs/
├── cloud/
│   ├── CloudPlan.md              ← this file
│   ├── config.env.example        ← template (committed)
│   ├── config.env                ← your values (gitignored)
│   ├── setup_gcs.sh              ← one-time: create bucket, upload datasets
│   ├── build_image.sh            ← one-time: create VM image with Julia
│   ├── startup.sh                ← runs inside each VM on boot
│   ├── launch_run.sh             ← launch one VM for one experiment + seed
│   ├── launch_batch.sh           ← launch all runs for an experiment (5 seeds)
│   ├── monitor.sh                ← check status of VMs and results
│   ├── collect_results.sh        ← download results, compute mean ± std
│   └── experiments/              ← per-experiment config
│       ├── har.env
│       ├── gesture.env
│       ├── occupancy.env
│       ├── smnist.env
│       ├── traffic.env
│       ├── power.env
│       ├── ozone.env
│       ├── person.env
│       └── cheetah.env
├── JuliaLang/
│   ├── scripts/
│   │   ├── train_har_srnn.jl     ← already done ✅
│   │   ├── train_gesture_srnn.jl ← TODO
│   │   ├── train_occupancy_srnn.jl ← TODO
│   │   ├── train_smnist_srnn.jl  ← TODO
│   │   ├── train_traffic_srnn.jl ← TODO
│   │   ├── train_power_srnn.jl   ← TODO
│   │   ├── train_ozone_srnn.jl   ← TODO
│   │   ├── train_person_srnn.jl  ← TODO
│   │   └── train_cheetah_srnn.jl ← TODO
│   └── ...
└── ...
```

---

## Implementation Plan

### Phase 0 — Prerequisites (local, one-time)

- [x] `gcloud` CLI installed and authenticated
- [ ] Verify GCP project ID: `gcloud config get-value project`
- [ ] Verify billing is enabled: `gcloud billing accounts list`
- [ ] Enable Compute Engine API: `gcloud services enable compute.googleapis.com`
- [ ] Enable Cloud Storage API: `gcloud services enable storage.googleapis.com`
- [ ] Copy `config.env.example` → `config.env` and fill in project ID

### Phase 1 — Cloud Infrastructure Setup

#### Step 1.1: `setup_gcs.sh` — Create GCS bucket and upload datasets
- Create the GCS bucket
- Upload all datasets from `liquid_time_constant_networks/experiments_with_ltcs/data/`
- Verify uploads with `gsutil ls`

#### Step 1.2: `build_image.sh` — Create custom VM image with Julia pre-installed
This avoids 10+ min of Julia/package installation on every run.

1. Create a temporary VM from a base Debian/Ubuntu image
2. SSH in and install:
   - Julia (version from config.env)
   - OS packages: `build-essential`, `git`
3. Pre-install Julia packages:
   - Clone the repo, `cd JuliaLang`, `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`
4. Create a VM image from this disk: `gcloud compute images create srnn-julia-v1 --source-disk=...`
5. Delete the temporary VM

#### Step 1.3: `startup.sh` — VM startup script
This runs automatically when each VM boots:

```bash
#!/bin/bash
# 1. Set metadata variables (experiment, seed, bucket, etc.)
# 2. Clone repo (or pull latest)
# 3. Download dataset from GCS
# 4. Check for existing checkpoint in GCS (for Spot VM resume)
# 5. Run training:
#      julia --project=JuliaLang JuliaLang/scripts/train_<exp>_srnn.jl \
#            --seed $SEED $ARGS \
#            --save /tmp/checkpoints
# 6. Upload results + best checkpoint to GCS
# 7. Self-delete the VM
```

### Phase 2 — Training Scripts (Julia)

Write training scripts for each experiment, adapting from the Python originals
in `liquid_time_constant_networks/experiments_with_ltcs/`. Each script follows
the same pattern as `train_har_srnn.jl`:

| Script | Status | Key Differences from HAR |
|--------|--------|--------------------------|
| `train_har_srnn.jl` | ✅ Done | — |
| `train_gesture_srnn.jl` | TODO | Different data loader (per-file traces), seq_len=32 |
| `train_occupancy_srnn.jl` | TODO | Binary classification, 5 features, two test sets |
| `train_smnist_srnn.jl` | TODO | Very long sequences (784), pixel-by-pixel input |
| `train_traffic_srnn.jl` | TODO | Regression (MSE loss), seq_len=32 |
| `train_power_srnn.jl` | TODO | Regression (MSE loss), seq_len=32 |
| `train_ozone_srnn.jl` | TODO | Binary classification, F1 metric, 72 features |
| `train_person_srnn.jl` | TODO | Two settings (standard + Rubanova), bs=64 |
| `train_cheetah_srnn.jl` | TODO | Autoregressive regression, MuJoCo rollouts |

### Phase 3 — Launch & Monitor

#### Step 3.1: `launch_run.sh` — Launch a single run
```
./cloud/launch_run.sh <experiment> <model> <seed>
# Example:
./cloud/launch_run.sh har srnn 1
```
- Reads `cloud/experiments/har.env` for args
- Reads `cloud/config.env` for GCP settings
- Creates a VM named `srnn-har-seed1` with the startup script
- VM metadata carries: experiment name, seed, training args, GCS paths

#### Step 3.2: `launch_batch.sh` — Launch all seeds for an experiment
```
./cloud/launch_batch.sh har srnn
# Creates 5 VMs: srnn-har-seed1 through srnn-har-seed5
```

#### Step 3.3: `monitor.sh` — Check run status
```
./cloud/monitor.sh
# Shows: VM name, status (RUNNING/TERMINATED), GCS result status
```

#### Step 3.4: `collect_results.sh` — Aggregate results
```
./cloud/collect_results.sh har srnn
# Downloads results from GCS, computes mean ± std across seeds
```

### Phase 4 — Repeat with LTC model

Once SRNN experiments are validated, repeat with `ltc1.jl` training scripts.

---

## GCS Bucket Layout

```
gs://srnn-experiments/
├── datasets/
│   ├── har/UCI HAR Dataset/         # uploaded once
│   ├── gesture/                     # uploaded once
│   ├── occupancy/                   # uploaded once
│   ├── smnist/                      # uploaded once
│   ├── traffic/                     # uploaded once
│   ├── power/                       # uploaded once
│   ├── ozone/                       # uploaded once
│   ├── person/                      # uploaded once
│   └── cheetah/                     # uploaded once
├── results/
│   ├── srnn/
│   │   ├── har/
│   │   │   ├── seed1/
│   │   │   │   ├── training_log.txt
│   │   │   │   ├── best_checkpoint.jld2
│   │   │   │   └── final_metrics.json
│   │   │   ├── seed2/
│   │   │   └── ...
│   │   ├── gesture/
│   │   └── ...
│   └── ltc/
│       └── (same structure)
└── checkpoints/                     # transient, for Spot VM resume
    └── srnn-har-seed1/
        └── latest.jld2
```

---

## Cost Estimate

| Component | Estimate |
|-----------|----------|
| **e2-standard-4** ($0.134/hr) × 100 runs × ~3 hrs avg | ~$40 |
| **Spot discount** (60-70% off) | ~$13-16 |
| **e2-standard-8** for SMnist ($0.268/hr) × 10 runs × ~5 hrs | ~$13 (or ~$5 Spot) |
| **GCS storage** (<10 GB datasets + results) | ~$0.20/month |
| **Network egress** (results download) | ~$0.10 |
| **VM image storage** (~10 GB) | ~$0.50/month |
| | |
| **Total estimate (with Spot VMs)** | **~$20-25** |
| **Total estimate (without Spot)** | **~$55** |

---

## VM Types Reference

| Machine Type   | vCPUs | RAM (GB) | $/hr (standard) | $/hr (Spot) | Use Case |
|---------------|-------|----------|-----------------|-------------|----------|
| e2-standard-2  | 2     | 8        | $0.067          | ~$0.020     | Tiny datasets |
| e2-standard-4  | 4     | 16       | $0.134          | ~$0.040     | Most experiments |
| e2-standard-8  | 8     | 32       | $0.268          | ~$0.080     | SMnist, large datasets |
| e2-standard-16 | 16    | 64       | $0.536          | ~$0.161     | If 32 GB isn't enough |

> Julia single-threaded BPTT doesn't benefit much from >4 cores. RAM is the
> bottleneck, so choose VM size based on memory needs.

---

## Authentication & Keys

**No API keys, secrets, or service accounts needed.** The `gcloud` CLI uses your
existing Google Cloud authentication:

| Resource | Auth Method |
|----------|-------------|
| `gcloud` CLI (create VMs) | `gcloud auth login` (already done) |
| `gsutil` (GCS bucket) | Same `gcloud` auth, automatic |
| Inside VMs (access GCS) | Default compute service account (automatic) |

The only config value is your **GCP Project ID** (not secret), stored in `config.env`.

### Required APIs (enable once)
```bash
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
```

---

## Execution Order

```
Phase 0: Prerequisites
  └── Verify gcloud, billing, APIs, fill in config.env

Phase 1: Infrastructure  (do once)
  ├── Step 1.1: setup_gcs.sh       → create bucket, upload datasets
  ├── Step 1.2: build_image.sh     → create Julia VM image
  └── Step 1.3: write startup.sh   → VM boot script

Phase 2: Training Scripts  (can overlap with Phase 1)
  ├── Write train_gesture_srnn.jl
  ├── Write train_occupancy_srnn.jl
  ├── Write train_smnist_srnn.jl
  ├── Write train_traffic_srnn.jl
  ├── Write train_power_srnn.jl
  ├── Write train_ozone_srnn.jl
  ├── Write train_person_srnn.jl
  └── Write train_cheetah_srnn.jl

Phase 3: Launch SRNN Experiments
  ├── Test: launch one HAR run locally first
  ├── Test: launch one HAR run on cloud
  ├── Launch all HAR seeds (5 VMs)
  ├── Verify results, iterate if needed
  └── Launch remaining experiments

Phase 4: Launch LTC Experiments
  └── Same process with LTC training scripts

Phase 5: Collect & Analyze
  ├── collect_results.sh → download all
  ├── Compute mean ± std per experiment
  └── Generate comparison table (SRNN vs LTC vs Hasani's Table 3)
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Spot VM preemption mid-training | Checkpointing already built; startup.sh checks for existing checkpoints in GCS |
| Julia compilation slow on fresh VM | Custom VM image with pre-compiled packages |
| Dataset too large for VM RAM | Start with e2-standard-4, upgrade to e2-standard-8 if OOM |
| Training script bug wastes cloud $$ | Test each script locally on 1-2 epochs first |
| GCS bucket permissions | VMs use default service account with storage read/write |
| Runaway costs | Set budget alerts in GCP console; VMs self-delete after training |

---

## Notes

- All training scripts should accept a `--seed` flag for reproducible runs
- Each script should write a `final_metrics.json` with the final test metric for easy collection
- The startup script should capture stdout/stderr to a log file and upload it to GCS
- Consider adding `--save_to_gcs` flag to training scripts for direct GCS checkpoint upload

---

## Next 5 Steps (Concrete)

**GCP Project ID:** `liquidneuralnets` (project number: `1042969478371`)

### Step 1: Configure local environment and enable GCP APIs ✅

Create `config.env` from the template and set the active project:

```bash
# 1a. Set gcloud to use our new project
gcloud config set project liquidneuralnets

# 1b. Verify it's set
gcloud config get-value project
# Expected output: liquidneuralnets

# 1c. Enable required APIs
gcloud services enable compute.googleapis.com storage.googleapis.com

# 1d. Create config.env from the template
cp cloud/config.env.example cloud/config.env
# Then edit cloud/config.env to set:
#   GCP_PROJECT=liquidneuralnets
```

**Verification:** `gcloud services list --enabled` should show both Compute Engine
and Cloud Storage in the list.

- [x] Done (2025-03-15)

### Step 2: Create GCS bucket and upload datasets (`setup_gcs.sh`) ✅

Write and run `cloud/setup_gcs.sh`:

```bash
# 2a. Create the bucket (us-central1 for cheapest compute)
gcloud storage buckets create gs://liquidneuralnets-experiments \
    --location=us-central1 \
    --uniform-bucket-level-access

# 2b. Upload datasets from the existing Python repo
#     Source: liquid_time_constant_networks/experiments_with_ltcs/data/
gsutil -m cp -r <dataset_dirs> gs://liquidneuralnets-experiments/datasets/

# 2c. Verify
gsutil ls gs://liquidneuralnets-experiments/datasets/
```

Total uploaded: **431.3 MiB** across 8 datasets (cheetah, gesture, har, occupancy,
ozone, person, power, traffic). SMnist uses keras download (handled in training script).

**Verification:** All 8 dataset directories present in bucket.

- [x] Done (2025-03-15)

### Step 3: Build a custom Julia VM image (`build_image.sh`)

Write and run `cloud/build_image.sh`. This creates a reusable disk image
with Julia 1.11.4 + all packages pre-compiled so each experiment VM boots
in ~60 seconds instead of ~15 minutes:

```bash
# 3a. Create a temporary VM from a base image
gcloud compute instances create julia-image-builder \
    --zone=us-central1-a \
    --machine-type=e2-standard-4 \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=30GB

# 3b. SSH in and install Julia + packages
gcloud compute ssh julia-image-builder --zone=us-central1-a --command='
    # Install Julia
    wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.4-linux-x86_64.tar.gz
    sudo tar -xzf julia-1.11.4-linux-x86_64.tar.gz -C /opt/
    sudo ln -s /opt/julia-1.11.4/bin/julia /usr/local/bin/julia
    rm julia-1.11.4-linux-x86_64.tar.gz

    # Install git, build deps
    sudo apt-get update && sudo apt-get install -y git build-essential

    # Clone repo and precompile Julia packages
    git clone https://github.com/TomRichner/Intersect-LNNs-SRNNs.git /opt/srnn-repo
    cd /opt/srnn-repo
    julia --project=JuliaLang -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
'

# 3c. Stop the VM (required before creating image)
gcloud compute instances stop julia-image-builder --zone=us-central1-a

# 3d. Create image from the disk
gcloud compute images create srnn-julia-v1 \
    --source-disk=julia-image-builder \
    --source-disk-zone=us-central1-a \
    --family=srnn-julia

# 3e. Delete the temporary VM
gcloud compute instances delete julia-image-builder --zone=us-central1-a --quiet
```

**Verification:** `gcloud compute images list --filter="family=srnn-julia"` shows
`srnn-julia-v1`.

- [ ] Done

### Step 4: Write the VM startup script (`startup.sh`) and launch script (`launch_run.sh`)

Write `cloud/startup.sh` — the script that runs inside each VM on boot:

```bash
#!/bin/bash
set -euo pipefail

# Read VM metadata (set by launch_run.sh)
EXPERIMENT=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/experiment" -H "Metadata-Flavor: Google")
MODEL=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/model" -H "Metadata-Flavor: Google")
SEED=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/seed" -H "Metadata-Flavor: Google")
TRAIN_ARGS=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/train-args" -H "Metadata-Flavor: Google")
GCS_BUCKET=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket" -H "Metadata-Flavor: Google")
VM_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
VM_ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}')

RESULT_PATH="${GCS_BUCKET}/results/${MODEL}/${EXPERIMENT}/seed${SEED}"
LOG_FILE="/tmp/training.log"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Starting ${MODEL}-${EXPERIMENT}-seed${SEED} at $(date) ==="

# 1. Pull latest code
cd /opt/srnn-repo && git pull

# 2. Download dataset
mkdir -p JuliaLang/data/${EXPERIMENT}
gsutil -m cp -r "${GCS_BUCKET}/datasets/${EXPERIMENT}/*" "JuliaLang/data/${EXPERIMENT}/"

# 3. Check for existing checkpoint (Spot VM resume)
CHECKPOINT_PATH="${GCS_BUCKET}/checkpoints/${MODEL}-${EXPERIMENT}-seed${SEED}/latest.jld2"
RESUME_FLAG=""
if gsutil -q stat "$CHECKPOINT_PATH" 2>/dev/null; then
    gsutil cp "$CHECKPOINT_PATH" /tmp/resume_checkpoint.jld2
    RESUME_FLAG="--resume /tmp/resume_checkpoint.jld2"
fi

# 4. Run training
julia --project=JuliaLang "JuliaLang/scripts/train_${EXPERIMENT}_srnn.jl" \
    --seed $SEED \
    --save /tmp/checkpoints \
    $TRAIN_ARGS \
    $RESUME_FLAG

# 5. Upload results
gsutil cp "$LOG_FILE" "${RESULT_PATH}/training_log.txt"
gsutil cp /tmp/checkpoints/*best* "${RESULT_PATH}/" 2>/dev/null || true
gsutil cp /tmp/checkpoints/final_metrics.json "${RESULT_PATH}/" 2>/dev/null || true

echo "=== Completed at $(date) ==="
gsutil cp "$LOG_FILE" "${RESULT_PATH}/training_log.txt"

# 6. Self-delete
gcloud compute instances delete "$VM_NAME" --zone="$VM_ZONE" --quiet
```

Write `cloud/launch_run.sh`:

```bash
#!/bin/bash
# Usage: ./cloud/launch_run.sh <experiment> <model> <seed>
# Example: ./cloud/launch_run.sh har srnn 1
source cloud/config.env
source "cloud/experiments/$1.env"
MODEL=${2:-srnn}
SEED=${3:-1}

VM_NAME="${MODEL}-${EXPERIMENT_NAME}-seed${SEED}"

gcloud compute instances create "$VM_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --machine-type="${MACHINE_TYPE:-$GCP_MACHINE_TYPE}" \
    --image-family="$GCP_IMAGE_FAMILY" \
    ${GCP_USE_SPOT:+--provisioning-model=SPOT --instance-termination-action=STOP} \
    --metadata=experiment=${EXPERIMENT_NAME},model=${MODEL},seed=${SEED},train-args="${ARGS}",gcs-bucket="${GCS_BUCKET}" \
    --metadata-from-file=startup-script=cloud/startup.sh \
    --scopes=storage-full

echo "Launched: $VM_NAME"
```

**Verification:** Run a dry-run test:
`./cloud/launch_run.sh har srnn 1` → check VM appears in
`gcloud compute instances list`.

- [ ] Done

### Step 5: Smoke test with one HAR cloud run

Before launching 50+ runs, validate the entire pipeline end-to-end:

```bash
# 5a. Launch a single HAR run with a small epoch count
#     (temporarily edit har.env to --epochs 3 for testing)
./cloud/launch_run.sh har srnn 1

# 5b. Monitor the VM (watch startup log)
gcloud compute ssh srnn-har-seed1 --zone=us-central1-a --command='tail -f /tmp/training.log'

# 5c. After completion, check results landed in GCS
gsutil ls gs://liquidneuralnets-experiments/results/srnn/har/seed1/

# 5d. Check the VM self-deleted
gcloud compute instances list

# 5e. If everything works, restore har.env to --epochs 200
#     and launch the full batch:
./cloud/launch_batch.sh har srnn
```

**Verification:**
- [ ] VM booted and started training within ~2 min
- [ ] Training log visible
- [ ] Results appeared in GCS after training
- [ ] VM self-deleted after completion
- [ ] Done
