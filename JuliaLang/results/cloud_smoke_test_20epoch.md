# Cloud Smoke Test Results — 20 Epoch Run (Seed 1)

**Date:** 2026-03-15 → 2026-03-16  
**Launched:** ~21:20 PDT (March 15) via `./cloud/smoke_test.sh --epochs 20`  
**Purpose:** Verify all training scripts run correctly on cloud VMs before full 5-seed production run.

---

## SRNN Model Configuration

All experiments used identical SRNN hyperparameters:

| Parameter | Value |
|-----------|-------|
| Hidden size (n) | 32 |
| Excitatory neurons (n_E) | 16 |
| Inhibitory neurons (n_I) | 16 |
| SFA timescales (n_a_E) | 3 |
| STD timescales (n_b_E) | 0 (disabled) |
| ODE solver | semi_implicit (fused) |
| ODE step size (h) | 0.02 (1/50) |
| ODE unfolds per input step | 6 |
| Readout | synaptic |
| Per-neuron dynamics | false (shared scalars) |
| Total parameters | ~19,180 (HAR; varies by input dim) |
| State dimension | 80 (32 neurons × (1 V + 1 c + 3 a_E - 1 extra)) |

### Training Settings

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.005 |
| LR schedule | linear taper (final 20% of epochs) |
| Batch size | 128 |
| Epochs (smoke test) | 20 |
| Seed | 1 |
| Checkpointing | Best (by valid metric) + every 5 epochs |

---

## Cloud Infrastructure

| Setting | Value |
|---------|-------|
| GCP Project | liquidneuralnets |
| Default VM type | n4d-highmem-2 (2 vCPU, 16 GB RAM) |
| Large VM type | n4d-highmem-4 (4 vCPU, 32 GB RAM) |
| VM image | srnn-julia-v1 (Julia 1.12.5, 545 packages precompiled) |
| Zone | us-central1-a |
| Spot VMs | No (standard — JIT too fragile for preemption) |
| Results bucket | gs://liquidneuralnets-experiments/results/ |

---

## Run Status Summary

| # | Experiment | Status | VM Type | Duration | RAM (RSS) |
|---|-----------|--------|---------|----------|-----------|
| 1 | HAR | ✅ Done (20/20) | n4d-highmem-2 | 305 min (5.1 hr) | ~1.8 GB |
| 2 | Gesture | ✅ Done (20/20) | n4d-highmem-2 | 38 min | — |
| 3 | Occupancy | ✅ Done (20/20) | n4d-highmem-2 | 335 min (5.6 hr) | — |
| 4 | Ozone | ✅ Done (20/20) | n4d-highmem-2 | 43 min | — |
| 5 | Traffic | 🔄 Running (12/20) | n4d-highmem-2 | ~50 min/epoch | 9.6 GB |
| 6 | SMnist | 🔄 Running (2/20) | n4d-highmem-4 | ~3.5 hr/epoch | 7.7 GB |
| 7 | Power | 🔄 Running (2/20) | n4d-highmem-4 | ~4.5 hr/epoch | 8.8 GB |
| 8 | Person | 💀 OOM (exit 137) | n4d-highmem-2 | crashed <1 epoch | >16 GB |
| 9 | Cheetah | ❌ Not launched | — | — | — |

**Note:** Only 8 of 9 experiments launched due to an external IPv4 address quota of 8.
Person crashed with OOM on 16 GB VM; needs n4d-highmem-4 (32 GB). Env file updated for next run.

---

## Results vs. Hasani et al. 2021 Table 3

### Table 3 — Time Series Prediction

| Dataset | Metric | LSTM | CT-RNN | Neural ODE | CT-GRU | **LTC** | **SRNN (ours)** | Verdict |
|---------|--------|------|--------|------------|--------|---------|-----------------|---------|
| Gesture | accuracy | 64.57% | 59.01% | 46.97% | 68.31% | **69.55%** | 43.33% | ⚠️ Below all |
| Occupancy | accuracy | 93.18% | 94.54% | 90.15% | 91.44% | 94.63% | **98.71%** | 🟢 Beats all |
| Activity (HAR) | accuracy | 95.85% | 95.73% | **97.26%** | 96.16% | 95.67% | 94.55% | 🟡 Competitive |
| Seq. MNIST | accuracy | **98.41%** | 96.73% | 97.61% | 98.27% | 97.57% | 41.23%* | ⏳ Too early |
| Traffic | MSE | 0.169 | 0.224 | 1.512 | 0.389 | **0.099** | 0.166* | ⏳ Approaching LSTM |
| Power | MSE | 0.628 | 0.742 | 1.254 | **0.586** | 0.642 | 0.026* | ⏳ Suspiciously good |
| Ozone | F1-score | 0.284 | 0.236 | 0.168 | 0.260 | **0.302** | 0.129 | ⚠️ Below all |

*Hasani reports mean ± std over n=5 seeds with full training (50–200 epochs). Our numbers are single seed, 20 epochs.*  
*\* Still running — current best, not final.*

### Table 4 — Person Activity (1st Setting)

| Algorithm | Accuracy |
|-----------|----------|
| LSTM | 83.59% ± 0.40 |
| CT-RNN | 81.54% ± 0.33 |
| CT-GRU | 85.27% ± 0.39 |
| **LTC** | **85.48%** ± 0.40 |
| **SRNN (ours)** | 💀 OOM crash |

### Table 6 — Half-Cheetah Dynamics

| Algorithm | MSE |
|-----------|-----|
| LSTM | 2.500 ± 0.140 |
| CT-RNN | 2.838 ± 0.112 |
| CT-GRU | 3.014 ± 0.134 |
| **LTC** | **2.308** ± 0.015 |
| **SRNN (ours)** | ❌ Not launched |

---

## Epoch-by-Epoch Training Curves

### HAR — Activity Recognition ✅ (Best @ epoch 19: valid 98.77%, test 94.55%)

```
Epoch  Train Loss  Train Acc   Valid Acc    Test Acc
  0     1.43        50.23%      13.78%      15.53%
  1     0.82        84.51%      63.57%      60.22%
  2     0.42        90.81%      92.09%      89.37%
  3     0.25        94.21%      94.95%      90.19%
  4     0.20        95.34%      93.86%      85.83%
  5     0.19        94.90%      96.04%      93.46%
  6     0.20        95.65%      96.04%      93.73%
  7     0.17        96.49%      97.00%      90.19%
  8     0.18        96.80%      96.86%      93.19%
  9     0.17        96.16%      93.86%      88.83%
 10     0.24        91.85%      95.63%      85.83%    ← dip (LR taper region)
 11     0.26        89.49%      90.18%      86.65%
 12     0.20        93.01%      81.31%      85.83%
 13     0.13        97.12%      98.50%      95.10%    ← recovery
 14     0.15        96.23%      98.36%      94.82%
 15     0.13        96.46%      98.36%      93.46%
 16     0.10        98.13%      95.23%      93.73%
 17     0.09        98.21%      98.50%      94.55%
 18     0.09        98.28%      97.68%      94.82%
 19     0.08        98.41%      98.77%      94.55%    ← BEST
```

### Gesture ✅ (Best @ epoch 8: valid 55.00%, test 43.33%)

```
Epoch  Train Loss  Train Acc   Valid Acc    Test Acc
  0     1.79        18.23%      25.00%      17.78%
  1     1.60        30.73%      25.00%      17.78%
  2     1.45        46.35%      40.00%      27.78%
  3     1.44        44.53%      43.33%      40.00%
  4     1.32        50.00%      43.33%      43.33%
  5     1.27        49.22%      46.67%      43.33%
  6     1.25        49.48%      48.33%      42.22%
  7     1.23        53.39%      51.67%      42.22%
  8     1.20        52.08%      55.00%      43.33%    ← BEST
  9     1.15        55.47%      53.33%      46.67%
 ...    (plateaus through epoch 19)
 19     1.02        61.20%      48.33%      45.56%
```

### Occupancy ✅ (Best @ epoch 2: valid 98.89%, test 98.71%)

```
Epoch  Train Loss  Train Acc   Valid Acc    Test Acc
  0     0.45        79.62%      69.33%      75.60%
  1     0.12        96.93%      82.51%      72.82%
  2     0.05        98.74%      98.89%      98.71%    ← BEST
  3     0.05        98.71%      98.77%      94.25%
  4     0.04        98.73%      98.89%      96.00%
  ...   (overfits: valid stable ~98.7%, test degrades)
 19     0.02        99.29%      98.77%      89.48%
```

### Ozone ✅ (Best @ epoch 9: valid F1 0.121, test F1 0.129)

```
Epoch  Train Loss  Valid F1    Prec     Recall    Test F1
  0     0.22        0.0142     6.25%    0.80%     0.0164
  1     0.15        0.0138     5.00%    0.80%     0.0161
  2     0.14        0.1185     6.30%    100.0%    0.1259
  3     0.14        0.1185     6.30%    100.0%    0.1259
  ...   (plateaus — predicts positive class indiscriminately)
  9     0.13        0.1207     6.43%    97.60%    0.1293    ← BEST
 19     0.13        0.1198     6.38%    97.60%    0.1298
```

### Traffic 🔄 (Best so far @ epoch 10: valid MSE 0.166, test MSE 0.168)

```
Epoch  Train Loss   Valid MSE   Valid MAE   Test MSE    Test MAE
  0     1.2130       1.9826      1.1386      2.0123      1.1486
  1     0.7454       0.9832      0.8610      0.9933      0.8662
  2     0.5746       0.3748      0.4840      0.3801      0.4872
  3     0.2653       0.2950      0.4133      0.3042      0.4201
  4     0.1907       0.2115      0.3548      0.2163      0.3585
  5     0.1816       0.1814      0.3196      0.1845      0.3231
  6     0.1782       0.1729      0.3095      0.1768      0.3129
  7     0.2780       0.2032      0.3395      0.2079      0.3422   ← spike
  8     0.2680       0.4344      0.5512      0.4337      0.5517   ← spike
  9     0.1844       0.2196      0.3634      0.2211      0.3651
 10     0.1830       0.1662      0.3061      0.1684      0.3080   ← best
 11     0.1642       0.1896      0.3124      0.1921      0.3160
```

### SMnist 🔄 (2 epochs completed)

```
Epoch  Train Loss  Train Acc   Valid Acc    Test Acc
  0     2.06        25.36%      10.45%      10.10%
  1     1.20        58.24%      42.05%      41.23%
```

### Power 🔄 (2 epochs completed)

```
Epoch  Train Loss  Valid MSE   Valid MAE   Test MSE    Test MAE
  0     0.2045      1.4023      0.7983      1.3874      0.7966
  1     0.0043      0.0263      0.0783      0.0259      0.0785
```

---

## Analysis & Issues

### 🟢 Strong Results

1. **Occupancy (98.71% test accuracy):** Beats all models in Hasani Table 3, including LTC (94.63%).
   Converged extremely fast (best at epoch 2). Shows overfitting after: valid stays ~98.8% but test
   degrades from 98.71% → 89.48% by epoch 19. Early stopping is critical for this task.

2. **HAR (94.55% test accuracy):** Competitive with Hasani's LTC (95.67%). Valid accuracy reached
   98.77% — the gap to test (94.55%) suggests some overfitting. With more seeds and full epochs,
   could match or exceed.

3. **Power (MSE 0.026 at epoch 1):** 22× better than Hasani's best (0.586 CT-GRU). We verified
   both scripts use identical z-score normalization and per-timestep MSE — the metrics should be
   directly comparable. Needs more epochs to confirm this isn't transient.

4. **Traffic (MSE 0.166 at epoch 10):** Already matching LSTM (0.169) and approaching LTC (0.099).
   Still improving with 8 epochs remaining.

### ⚠️ Needs Investigation

5. **Gesture (43.33% test accuracy):** Below all Hasani models including Neural ODE (46.97%).
   Learning curve shows continued improvement through epoch 19 (train acc 61.2%) but valid/test
   plateau. Possible causes: seq_len=32 may be too short for this dataset with SRNN dynamics,
   or 20 epochs insufficient. Note: Hasani trained 200 epochs.

6. **Ozone (F1 0.129):** Below all Hasani models (worst was Neural ODE at 0.168). The model
   achieves ~97.6% recall but only ~6.4% precision — it's predicting nearly everything as positive.
   This is a heavily imbalanced binary classification task. Likely needs: class weighting adjustments,
   threshold tuning, or different training regime.

### 💀 Failed / Not Run

7. **Person (OOM):** Julia process killed by Linux OOM-killer (exit code 137) on n4d-highmem-2
   (16 GB) within 51 minutes, before completing first epoch. RSS exceeded available RAM.
   Fix: Updated `person.env` to use n4d-highmem-4 (32 GB, 4 vCPU).

8. **Cheetah (not launched):** 9th experiment couldn't launch due to external IPv4 address quota
   of 8. Fix: Migrate to IAP tunneling with `--no-address` VMs (planned).

### Timing & Cost

| Experiment | Per-Epoch Time | 20-Epoch Time | Est. 50-Epoch Time |
|-----------|---------------|---------------|-------------------|
| HAR | ~15 min | 305 min | ~12.7 hr |
| Gesture | ~2 min | 38 min | ~1.6 hr |
| Occupancy | ~17 min | 335 min | ~14.0 hr |
| Ozone | ~2 min | 43 min | ~1.8 hr |
| Traffic | ~50 min | — (running) | ~41.7 hr |
| SMnist | ~3.5 hr | — (running) | ~7.3 days |
| Power | ~4.5 hr | — (running) | ~9.4 days |
| Person | — | OOM | — |
| Cheetah | — | not run | — |

**SMnist and Power are extremely slow** — 3.5 and 4.5 hours per epoch respectively.
At 50 epochs, they'd run for 7–9 days each on a single VM. This needs investigation:
likely caused by very long sequences (SMnist: 784 steps) or large datasets (Power: ~2M samples).

---

## Next Steps

1. **Fix IPv4 quota bottleneck:** Switch to IAP tunneling + Cloud NAT (VMs with `--no-address`)
2. **Relaunch Person** on n4d-highmem-4 (env already updated)
3. **Launch Cheetah** (was blocked by IP quota)
4. **Investigate slow experiments:** SMnist and Power epoch times need optimization
5. **Investigate Gesture and Ozone** underperformance
6. **Full production run:** 5 seeds × 9 experiments = 45 VMs (needs IAP for >8 concurrent)
7. **Add `final_metrics.json`** output to training scripts (currently only training_log.txt)
