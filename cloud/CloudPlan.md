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
| 1 | HAR          | Classification | Accuracy | 16      | 561      | 6           | n4-highmem-2    |
| 2 | Gesture      | Classification | Accuracy | 32      | 32       | 5           | n4-highmem-2    |
| 3 | Occupancy    | Classification | Accuracy | 16      | 5        | 2           | n4-highmem-2    |
| 4 | SMnist       | Classification | Accuracy | 784     | 1        | 10          | n4-highmem-4    |
| 5 | Traffic      | Regression     | MSE      | 32      | varies   | 1           | n4-highmem-2    |
| 6 | Power        | Regression     | MSE      | 32      | varies   | 1           | n4-highmem-2    |
| 7 | Ozone        | Classification | F1-score | 32      | 72       | 2           | n4-highmem-2    |

### Tables 4, 5 — Person Activity

| # | Experiment   | Task Type      | Metric   | Setting | VM Size         |
|---|-------------|----------------|----------|---------|-----------------|
| 8 | Person (1st) | Classification | Accuracy | Standard| n4-highmem-2    |
| 9 | Person (2nd) | Classification | Accuracy | Rubanova| n4-highmem-2    |

### Table 6 — Half-Cheetah

| #  | Experiment   | Task Type  | Metric | VM Size         |
|----|-------------|------------|--------|-----------------|
| 10 | Cheetah      | Regression | MSE    | n4-highmem-2    |

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
│   │   ├── train_har_srnn.jl       ← ✅ done + cloud tested
│   │   ├── train_gesture_srnn.jl   ← ✅ done, locally verified
│   │   ├── train_occupancy_srnn.jl ← ✅ done, locally verified
│   │   ├── train_smnist_srnn.jl    ← ✅ done, locally verified
│   │   ├── train_traffic_srnn.jl   ← ✅ done, locally verified (first regression task, bs=64)
│   │   ├── train_power_srnn.jl     ← ✅ done, locally verified (regression, MSE/MAE)
│   │   ├── train_ozone_srnn.jl     ← ✅ done, locally verified (F1 metric, weighted CE)
│   │   ├── train_person_srnn.jl    ← ✅ done, locally verified (per-timestep classification, bs=16)
│   │   └── train_cheetah_srnn.jl   ← ✅ done, locally verified (vector autoregression, NPZ.jl)
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
| `train_har_srnn.jl` | ✅ Done | Baseline script with --seed, checkpointing, batched BPTT |
| `train_gesture_srnn.jl` | ✅ Done | 7 CSV files, interleaved windowing, seq_len=32, 5 classes, 3-way split |
| `train_occupancy_srnn.jl` | ✅ Done | CSV.jl data loader, z-score norm, 5 features, 2 classes, two test sets |
| `train_smnist_srnn.jl` | TODO | Very long sequences (784), pixel-by-pixel input |
| `train_traffic_srnn.jl` | TODO | Regression (MSE loss), seq_len=32 |
| `train_power_srnn.jl` | TODO | Regression (MSE loss), seq_len=32 |
| `train_ozone_srnn.jl` | TODO | Binary classification, F1 metric, 72 features |
| `train_person_srnn.jl` | TODO | Two settings (standard + Rubanova), bs=64 |
| `train_cheetah_srnn.jl` | ✅ Done | Autoregressive regression (17→17), MuJoCo rollouts, NPZ.jl for .npy loading |

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

#### Step 3.3: `monitor.sh` — Check run status and quota usage
```
./cloud/monitor.sh
# Output:
#   === vCPU Quota ===
#   USED: 10 / 64  (5 VMs × 2 vCPU)
#   AVAILABLE: 54  (can launch 27 more e2-highmem-2)
#
#   === Running VMs ===
#   NAME              STATUS     MACHINE         ZONE
#   srnn-har-seed1    RUNNING    e2-highmem-2    us-central1-a
#   srnn-har-seed2    RUNNING    e2-highmem-2    us-central1-a
#   ...
#
#   === Results in GCS ===
#   srnn/har/seed1:  ✅ final_metrics.json found
#   srnn/har/seed2:  ⏳ training in progress
#   srnn/har/seed3:  ❌ no results yet
```
Key features:
- Shows vCPU quota usage vs limit (64 max) to avoid launch failures
- Lists all running VMs with status
- Checks GCS bucket for completed results per experiment/seed

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
| **e2-highmem-2** ($0.090/hr) × 90 runs × ~3 hrs avg | ~$24 |
| **e2-highmem-4** for SMnist ($0.181/hr) × 10 runs × ~5 hrs | ~$9 |
| **Spot discount** (60-70% off) | ~$10-12 total |
| **GCS storage** (<10 GB datasets + results) | ~$0.20/month |
| **Network egress** (results download) | ~$0.10 |
| **VM image storage** (~30 GB) | ~$1.50/month |
| | |
| **Total estimate (with Spot VMs)** | **~$12-15** |
| **Total estimate (without Spot)** | **~$35** |

---

## VM Types Reference

| Machine Type    | vCPUs | RAM (GB) | CPU | $/hr (standard) | Use Case |
|----------------|-------|----------|-----|-----------------|----------|
| **n4-highmem-2** | 2   | 16       | Intel Emerald Rapids (5th gen Xeon) | ~$0.12 | **Default — all experiments** |
| n4-highmem-4    | 4    | 32       | Intel Emerald Rapids | ~$0.24 | SMnist (long sequences) |

> **Why n4-highmem-2?** Julia's Zygote BPTT is single-threaded — extra CPUs are
> wasted. N4 has ~2× single-core performance vs E2 (Intel Emerald Rapids vs shared-core).
> RAM usage observed: ~1.8 GB for HAR (32 neurons), so 16 GB is plenty.

> **Why not Spot?** First-epoch JIT compilation takes 15-25 min. Spot preemptions
> during JIT waste the entire compilation. Standard VMs are more reliable.

> **vCPU Quota:** Project has a 64 global vCPU quota (binding), 200 regional (us-central1).
> With n4-highmem-2 (2 vCPU each), we can run **32 concurrent VMs**.

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

### Step 3: Build a custom Julia VM image (`build_image.sh`) ✅

Write and run `cloud/build_image.sh`. This creates a reusable disk image
with Julia 1.12.5 + all packages pre-compiled so each experiment VM boots
in ~60 seconds instead of ~15 minutes.

Ran via `./cloud/build_image.sh` (with manual SSH for install step).
Required `libssl-dev` and `libcurl4-openssl-dev` for Julia's LibGit2 stdlib.

**Result:** Image `srnn-julia-v1` (family: `srnn-julia`), 30 GB disk.
545 packages precompiled in ~17 minutes. Key packages verified (Lux, Zygote,
Optimisers, NNlib, JLD2).

**Verification:** `gcloud compute images describe srnn-julia-v1` shows `STATUS: READY`.

- [x] Done (2025-03-15)

### Step 4: Write the VM startup and launch scripts ✅

Scripts created:

| Script | Purpose |
|--------|---------|
| `cloud/startup.sh` | Runs on VM boot: pulls code, downloads data, checks for resume checkpoint, trains, uploads results to GCS, self-deletes |
| `cloud/launch_run.sh` | Launches a single VM: pre-flight checks (quota, existing VM, existing results), passes all config as VM metadata |
| `cloud/launch_batch.sh` | Launches N_SEEDS VMs for an experiment (calls launch_run.sh in a loop) |
| `cloud/monitor.sh` | Shows vCPU quota usage, running VMs, and per-seed result status from GCS |

**Updates since initial creation:**
- `--seed` flag added to `train_har_srnn.jl` — seeds both model init RNG and global RNG
- `startup.sh` fixed: runs as root but uses `sudo -u tom -i` for Julia/git (package cache compatibility)
- `stdbuf -oL` added to startup.sh for line-buffered Julia output
- `monitor.sh` shows both global (64) and regional (200) vCPU quotas
- Switched from Spot to standard VMs (Spot preemptions during JIT compilation)
- Switched from e2-highmem-2 to **n4-highmem-2** (~2× single-core performance)

- [x] Done (2025-03-15)

### Step 5: Smoke test with one HAR cloud run ⏳

Smoke test with `--epochs 3` on non-preemptible e2-highmem-2 (launched before n4 switch).

**Progress:**
- [x] VM booted and ran startup.sh correctly (root→tom user context fixed)
- [x] git pull, data download (269.5 MiB) worked  
- [x] Julia started training with **no recompilation** (pre-compiled packages found)
- [x] Gradient smoke test passed (initial loss 1.56, expected ~1.79)
- [x] Epoch 0 completed: train acc 41.1%, valid acc 13.8%
- [x] Best checkpoint saved at epoch 1 (valid acc 29.9%)
- [ ] Waiting for epochs 1-2 to complete (stdout buffered — stdbuf fix is in for next run)
- [ ] Results uploaded to GCS
- [ ] VM self-deleted

**Key findings:**
- First-epoch JIT compilation takes ~30 min on e2-highmem-2 (single-threaded)
- RAM usage: ~1.8 GB RSS (well within 16 GB)
- CPU: 98% of one core (single-threaded Zygote BPTT confirmed)
- Spot VM was preempted during first attempt — switched to standard VMs

### Step 6: Write remaining training scripts

| Script | Status | Verified Locally |
|--------|--------|------------------|
| `train_har_srnn.jl` | ✅ | ✅ + cloud tested |
| `train_occupancy_srnn.jl` | ✅ | ✅ (loss 0.68, expected ~0.69) |
| `train_gesture_srnn.jl` | ✅ | ✅ (loss 1.43, expected ~1.61) |
| `train_traffic_srnn.jl` | TODO | Next (first regression task) |
| `train_power_srnn.jl` | TODO | |
| `train_ozone_srnn.jl` | TODO | |
| `train_smnist_srnn.jl` | TODO | |
| `train_person_srnn.jl` | TODO | |
| `train_cheetah_srnn.jl` | ✅ | ✅ (MSE 18.5→4.2 in 2 epochs) |
