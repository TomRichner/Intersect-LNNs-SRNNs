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
