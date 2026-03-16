# training_utils.jl — shared training utilities for SRNN experiment scripts
#
# Provides:
#   lr_schedule(epoch, total_epochs) — three-phase LR schedule

"""
    lr_schedule(epoch, total_epochs; lr_min=0.0005f0, lr_max=0.01f0,
                warmup_frac=0.1, taper_frac=0.4)

Three-phase learning rate schedule (percentage-based):
  1. Warmup (first 10%):  linear ramp from lr_min to lr_max
  2. Hold   (middle 50%): constant at lr_max
  3. Taper  (last 40%):   linear decay from lr_max to lr_min
"""
function lr_schedule(epoch::Int, total_epochs::Int;
                     lr_min::Float32=0.0005f0, lr_max::Float32=0.01f0,
                     warmup_frac::Float64=0.1, taper_frac::Float64=0.4)
    warmup_end = warmup_frac * total_epochs
    taper_start = (1.0 - taper_frac) * total_epochs

    if epoch < warmup_end
        # Phase 1: linear ramp from lr_min to lr_max
        frac = Float32(epoch / warmup_end)
        return lr_min + (lr_max - lr_min) * frac
    elseif epoch >= taper_start
        # Phase 3: linear taper from lr_max to lr_min
        remaining = Float32((total_epochs - epoch) / (taper_frac * total_epochs))
        return lr_min + (lr_max - lr_min) * clamp(remaining, 0.0f0, 1.0f0)
    else
        # Phase 2: hold at lr_max
        return lr_max
    end
end

