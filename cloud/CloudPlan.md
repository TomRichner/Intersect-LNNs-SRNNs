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
│                                                                 │
│  SSH via IAP:  gcloud compute ssh <vm> --tunnel-through-iap     │
│  Serial logs:  gcloud compute instances get-serial-port-output  │
└─────────────────────────────────────────────────────────────────┘
         │                         ▲
         │ (create VMs, no ext IP) │ (upload results via PGA)
         ▼                         │
┌─────────────────────────────────────────────────────────────────┐
│  GOOGLE CLOUD                                                   │
│                                                                 │
│  ┌─────────────┐  ┌───────────┐  ┌────────────────────────┐     │
│  │ Cloud NAT   │  │ Private   │  │  VM: srnn-har-seed1    │     │
│  │ (srnn-nat)  │  │ Google    │  │  (no external IP)      │     │
│  │ Outbound    │  │ Access    │  │  ┌──────────────────┐  │     │
│  │ internet    │  │ (GCS/API) │  │  │ 1. git pull      │──┼──►NAT│
│  │ (git pull)  │  │           │  │  │ 2. gsutil data   │──┼──►PGA│
│  └─────────────┘  └───────────┘  │  │ 3. Train model   │  │     │
│                                  │  │ 4. Upload results│──┼──►PGA│
│  ┌──────────────────────────────┐│  │ 5. Self-delete   │  │     │
│  │  GCS Bucket                  ││  └──────────────────┘  │     │
│  │  gs://liquidneuralnets-      │└────────────────────────┘     │
│  │       experiments/           │                               │
│  │  ├── datasets/               │  ┌────────────────────────┐   │
│  │  ├── results/srnn/<exp>/     │  │  VM: srnn-har-seed2    │   │
│  │  └── checkpoints/            │  │  (same as above)       │   │
│  └──────────────────────────────┘  └────────────────────────┘   │
│                                                                 │
│  IAP Firewall: allow-iap-ssh (35.235.240.0/20 → tcp:22)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Experiments To Run

From Hasani et al. 2021, we need to reproduce the following. All reported as
**mean ± std** over **n = 5** seeds.

### Table 3 — Time Series Prediction

| # | Experiment   | Task Type      | Metric   | Seq Len | Features | Classes/Out | VM Size         |
|---|-------------|----------------|----------|---------|----------|-------------|-----------------|
| 1 | HAR          | Classification | Accuracy | 16      | 561      | 6           | n4d-highmem-2   |
| 2 | Gesture      | Classification | Accuracy | 32      | 32       | 5           | n4d-highmem-2   |
| 3 | Occupancy    | Classification | Accuracy | 16      | 5        | 2           | n4d-highmem-2   |
| 4 | SMnist       | Classification | Accuracy | 784     | 1        | 10          | n4d-highmem-4   |
| 5 | Traffic      | Regression     | MSE      | 32      | varies   | 1           | n4d-highmem-2   |
| 6 | Power        | Regression     | MSE      | 32      | varies   | 1           | n4d-highmem-2   |
| 7 | Ozone        | Classification | F1-score | 32      | 72       | 2           | n4d-highmem-2   |

### Tables 4, 5 — Person Activity

| # | Experiment   | Task Type      | Metric   | Setting | VM Size         |
|---|-------------|----------------|----------|---------|-----------------|
| 8 | Person (1st) | Classification | Accuracy | Standard| n4d-highmem-4   |
| 9 | Person (2nd) | Classification | Accuracy | Rubanova| n4d-highmem-4   |

### Table 6 — Half-Cheetah

| #  | Experiment   | Task Type  | Metric | VM Size         |
|----|-------------|------------|--------|-----------------|
| 10 | Cheetah      | Regression | MSE    | n4d-highmem-2   |

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
#      julia --project=JuliaLang JuliaLang/scripts/train_<exp>.jl \
#            --model $MODEL --seed $SEED $ARGS \
#            --save /tmp/checkpoints
# 6. Upload results + best checkpoint to GCS
# 7. Self-delete the VM
```

### Phase 2 — Training Scripts (Julia)

Model-agnostic training scripts accept a `--model` flag (`srnn`, `ltc`).
Model dispatch is handled by `JuliaLang/src/model_registry.jl` which provides:
- `build_cell(model, n, n_in, args, rng)` — construct the right cell type
- `initial_state(cell, B)` — zero state (dispatch on cell type)
- `readout(cell, S, ps)` — already dispatched in `srnn.jl` and `ltc1.jl`

Each script is identical for all models — only the `--model` flag changes.

| Script | Replaces | Task Type |
|--------|----------|-----------|
| `train_har.jl` | `train_har_srnn.jl` | Classification (6 classes, last-timestep) |
| `train_gesture.jl` | `train_gesture_srnn.jl` | Classification (5 classes, last-timestep) |
| `train_occupancy.jl` | `train_occupancy_srnn.jl` | Classification (2 classes, last-timestep) |
| `train_smnist.jl` | `train_smnist_srnn.jl` | Classification (10 classes, last-timestep) |
| `train_traffic.jl` | `train_traffic_srnn.jl` | Regression (MSE, per-timestep) |
| `train_power.jl` | `train_power_srnn.jl` | Regression (MSE, per-timestep) |
| `train_ozone.jl` | `train_ozone_srnn.jl` | Classification (2 classes, F1 metric) |
| `train_person.jl` | `train_person_srnn.jl` | Classification (7 classes, per-timestep) |
| `train_cheetah.jl` | `train_cheetah_srnn.jl` | Regression (MSE, autoregressive 17→17) |

Old `train_*_srnn.jl` scripts are preserved for reference.

#### SRNN-specific flags

| Flag | Default | Description |
|------|---------|-------------|
| `--n_a_E` | 3 | SFA timescale count, E neurons (0 = no SFA) |
| `--n_a_I` | 0 | SFA timescale count, I neurons |
| `--n_b_E` | 1 | STD timescale count, E neurons (0 = no STD) |
| `--n_b_I` | 0 | STD timescale count, I neurons |
| `--dales` | off | Dale's law enforcement via softplus (E columns ≥ 0, I columns ≤ 0) |
| `--per_neuron` | off | Per-neuron dynamics params (τ_d, c, a_0, tau endpoints) |

When `--dales` is enabled, W is initialized from `connectivity.jl` (`generate_rmt_matrix`)
and sign constraints are enforced via softplus parameterization in the forward pass.

When `n_a_E=1`, a single `log_tau_a_E` is used instead of the lo/hi range.

### Phase 3 — Launch & Monitor

All commands require a **run name** as the first argument. This scopes VM names,
GCS result paths, and checkpoint paths so different runs never collide.
The `--model` flag is passed automatically by `startup.sh` from VM metadata.

#### Step 3.1: `launch_run.sh` — Launch a single run
```
./cloud/launch_run.sh <run_name> <experiment> <model> <seed> [--epochs N]
# Example:
./cloud/launch_run.sh prod har srnn 1
./cloud/launch_run.sh prod har ltc 1          # same script, different model
./cloud/launch_run.sh smoke20 gesture srnn 1 --epochs 20
```
- Creates a VM named `<run>-<model>-<experiment>-seed<N>` (e.g. `prod-srnn-har-seed1`)
- Results go to: `gs://<bucket>/results/<run>/<model>/<experiment>/seed<N>/`
- VM metadata carries: run name, model, experiment, seed, training args, GCS paths
- `startup.sh` passes `--model $MODEL` to the Julia training script automatically

#### Step 3.2: `launch_batch.sh` — Launch all seeds for an experiment
```
./cloud/launch_batch.sh <run_name> <experiment> <model>
# Example:
./cloud/launch_batch.sh prod har srnn
./cloud/launch_batch.sh prod har ltc          # LTC comparison
# Creates 5 VMs: prod-srnn-har-seed1 through prod-srnn-har-seed5
```

#### Step 3.3: `launch_all.sh` — Launch all experiments in waves
```
./cloud/launch_all.sh <run_name> [--seeds N] [--epochs N] [--dry-run]
# Example:
./cloud/launch_all.sh prod                     # full 5-seed run
./cloud/launch_all.sh prod --dry-run            # preview without launching
```

#### Step 3.4: `smoke_test.sh` — Quick validation (seed 1 only)
```
./cloud/smoke_test.sh --run <name> [--epochs N]
# Example:
./cloud/smoke_test.sh --run smoke20 --epochs 20
```

#### Step 3.5: `monitor.sh` — Check run status and quota usage
```
./cloud/monitor.sh <run_name>              # show VMs + GCS results for a run
./cloud/monitor.sh                         # show all VMs + quota only
```

#### Step 3.6: `collect_results.sh` — Aggregate results
```
./cloud/collect_results.sh <run_name>
# Downloads results from GCS, computes mean ± std across seeds
```

#### Cleanup
```bash
# Delete all results for an exploratory run:
gcloud storage rm -r gs://liquidneuralnets-experiments/results/smoke20/
```

### Phase 4 — Run LTC experiments

No separate LTC scripts needed — use the same model-agnostic scripts with `--model ltc`:
```bash
# Run all LTC experiments (same scripts, just change model name)
./cloud/launch_all.sh prod-ltc --model ltc
# Or individual:
./cloud/launch_run.sh prod har ltc 1
```

---

## GCS Bucket Layout

```
gs://liquidneuralnets-experiments/
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
│   ├── smoke20/                     # run name scopes all results
│   │   └── srnn/
│   │       ├── har/seed1/
│   │       │   ├── training_log.txt
│   │       │   └── srnn_har_best.jld2
│   │       ├── gesture/seed1/
│   │       └── ...
│   ├── prod/                        # production run
│   │   └── srnn/
│   │       ├── har/seed1/ ... seed5/
│   │       ├── gesture/seed1/ ... seed5/
│   │       └── ...
│   └── ltc/
│       └── (same structure)
└── checkpoints/                     # transient, for Spot VM resume
    ├── smoke20/srnn-har-seed1/
    │   └── latest.jld2
    └── prod/srnn-har-seed1/
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

| Machine Type     | vCPUs | RAM (GB) | $/hr (standard) | Use Case |
|-----------------|-------|----------|-----------------|----------|
| **n4d-highmem-2** | 2   | 16       | ~$0.12 | **Default** — HAR, Gesture, Occupancy, Traffic, Ozone, Cheetah |
| **n4d-highmem-4** | 4   | 32       | ~$0.24 | **Large** — SMnist, Power, Person (high RAM usage) |

Observed RAM usage (smoke test):
- HAR: ~1.8 GB, Gesture/Ozone: small, Occupancy: small
- Traffic: 9.6 GB (tight on 16 GB), SMnist: 7.7 GB, Power: 8.8 GB
- Person: >16 GB (OOM on n4d-highmem-2, needs n4d-highmem-4)

> **Why n4d-highmem-2?** Julia's Zygote BPTT is single-threaded — extra CPUs are
> wasted. N4D has ~2× single-core performance vs E2 (Intel Emerald Rapids vs shared-core).

> **Why not Spot?** First-epoch JIT compilation takes 15-25 min. Spot preemptions
> during JIT waste the entire compilation. Standard VMs are more reliable.

> **vCPU Quota:** Project has a 64 global vCPU quota (binding), 200 regional (us-central1).
> With n4d-highmem-2 (2 vCPU each), we can run **32 concurrent VMs**.

---

## Networking — IAP + Cloud NAT (no external IPs)

VMs are created with `--no-address` (no external IPv4). This removes the IPv4
address quota bottleneck (was limited to 8 VMs). Networking is handled by:

| Component | Purpose | Setup Command |
|-----------|---------|---------------|
| **Private Google Access** | VMs reach GCS and Compute API without external IP | `gcloud compute networks subnets update default --region=us-central1 --enable-private-ip-google-access` |
| **IAP Firewall Rule** | Allow SSH tunneling through Google's IAP proxy | `gcloud compute firewall-rules create allow-iap-ssh --rules=tcp:22 --source-ranges=35.235.240.0/20` |
| **Cloud Router** | Required for NAT gateway | `gcloud compute routers create srnn-router --region=us-central1 --network=default` |
| **Cloud NAT** | Outbound internet for git pull (shared, no per-VM IP) | `gcloud compute routers nats create srnn-nat --router=srnn-router --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges` |

**SSH access:** `gcloud compute ssh <vm> --zone=us-central1-a --tunnel-through-iap`
**Serial logs:** `gcloud compute instances get-serial-port-output <vm>` (works without SSH)

**Cost:** Cloud NAT gateway costs ~$0.044/hr (~$1/day). Delete when not running experiments:
```bash
gcloud compute routers nats delete srnn-nat --router=srnn-router --region=us-central1 --quiet
```

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
| `cloud/startup.sh` | Runs on VM boot: pulls code via Cloud NAT, downloads data from GCS via PGA, checks for resume checkpoint, trains, uploads results to GCS, self-deletes |
| `cloud/launch_run.sh` | Launches a single VM with `--no-address` (no external IP): pre-flight checks (quota, existing VM, existing results), passes all config as VM metadata |
| `cloud/launch_batch.sh` | Launches N_SEEDS VMs for an experiment (calls launch_run.sh in a loop) |
| `cloud/launch_all.sh` | Launches all experiments in waves, respecting vCPU quota |
| `cloud/smoke_test.sh` | Quick validation: launches all experiments for a reduced number of epochs |
| `cloud/monitor.sh` | Shows vCPU quota usage, running VMs, and per-seed result status from GCS |
| `cloud/config.env` | GCP project settings, VM types, image family, IAP/NAT documentation |
| `cloud/experiments/*.env` | Per-experiment config: script path, args, machine type, seed count |

**Key design decisions:**
- `startup.sh` runs as root but uses `sudo -u tom -i` for Julia/git (package cache compatibility)
- `stdbuf -oL` for line-buffered Julia output visible in serial console
- VMs created with `--no-address` — uses IAP for SSH, Cloud NAT for git, PGA for GCS
- Standard VMs (not Spot) — JIT compilation takes 15-25 min, preemption wastes it
- Switched from e2-highmem to **n4d-highmem** (~2× single-core performance)

- [x] Done (2025-03-15), updated for IAP/NAT (2025-03-16)

### Step 5: Cloud smoke test (20 epochs, all experiments) ✅

Ran `./cloud/smoke_test.sh --epochs 20` overnight. 8 of 9 experiments launched
(Cheetah blocked by IPv4 quota — fixed by IAP migration).

**Results (single seed, 20 epochs vs Hasani et al. 2021 Table 3):**

| Dataset | Metric | LTC (Hasani) | SRNN (ours) | Verdict |
|---------|--------|-------------|-------------|----------|
| Occupancy | accuracy | 94.63% | **98.71%** | Beats all models |
| HAR | accuracy | 95.67% | 94.55% | Competitive |
| Traffic | MSE | 0.099 | 0.166 (ep 11) | Approaching LSTM |
| Power | MSE | 0.642 | 0.026 (ep 1) | Suspiciously good |
| SMnist | accuracy | 97.57% | 41.23% (ep 2) | Too early |
| Gesture | accuracy | 69.55% | 43.33% | Below all |
| Ozone | F1 | 0.302 | 0.129 | Below all |
| Person | accuracy | 85.48% | OOM (exit 137) | Needs n4d-highmem-4 |
| Cheetah | MSE | 2.308 | Not launched | IP quota limit |

Full results: `JuliaLang/results/cloud_smoke_test_20epoch.md`

**Key findings:**
- Person OOM'd on n4d-highmem-2 (16 GB) — updated to n4d-highmem-4 (32 GB)
- SMnist and Power are very slow (~3.5 and ~4.5 hrs/epoch)
- Occupancy result (98.71%) is a genuine standout

- [x] Done (2025-03-16)

### Step 6: IAP + Cloud NAT migration ✅

Removed external IPv4 address quota bottleneck (was limited to 8 VMs).
See "Networking — IAP + Cloud NAT" section above for details.

- [x] Done (2025-03-16)

### Step 7: Write remaining training scripts ✅

| Script | Status | Verified |
|--------|--------|----------|
| `train_har_srnn.jl` | ✅ | Local + cloud (20 epochs) |
| `train_gesture_srnn.jl` | ✅ | Local + cloud (20 epochs) |
| `train_occupancy_srnn.jl` | ✅ | Local + cloud (20 epochs) |
| `train_smnist_srnn.jl` | ✅ | Local + cloud (2/20 epochs) |
| `train_traffic_srnn.jl` | ✅ | Local + cloud (12/20 epochs) |
| `train_power_srnn.jl` | ✅ | Local + cloud (2/20 epochs) |
| `train_ozone_srnn.jl` | ✅ | Local + cloud (20 epochs) |
| `train_person_srnn.jl` | ✅ | Local (cloud OOM, VM type updated) |
| `train_cheetah_srnn.jl` | ✅ | Local (cloud not yet launched) |

- [x] Done (2025-03-16)

### Step 8: Full production run (5 seeds × 9 experiments)

- [ ] Launch all SRNN experiments with `launch_all.sh`
- [ ] Monitor progress with `monitor.sh` and serial port logs
- [ ] Collect results with `collect_results.sh`
- [ ] Generate comparison table (SRNN vs Hasani Table 3)
