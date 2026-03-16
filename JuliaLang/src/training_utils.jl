# training_utils.jl — shared training utilities for SRNN experiment scripts
#
# Provides:
#   lr_schedule(epoch, total_epochs) — three-phase LR schedule

"""
    lr_schedule(epoch, total_epochs; lr_min=0.0005f0, lr_max=0.01f0,
                warmup_epochs=5, taper_epochs=20)

Three-phase learning rate schedule:
  1. Warmup: linear ramp from lr_min to lr_max over first `warmup_epochs`
  2. Hold:   constant at lr_max
  3. Taper:  linear decay from lr_max to lr_min over last `taper_epochs`
"""
function lr_schedule(epoch::Int, total_epochs::Int;
                     lr_min::Float32=0.0005f0, lr_max::Float32=0.01f0,
                     warmup_epochs::Int=5, taper_epochs::Int=20)
    if epoch < warmup_epochs
        # Phase 1: linear ramp from lr_min to lr_max
        frac = Float32(epoch) / Float32(warmup_epochs)
        return lr_min + (lr_max - lr_min) * frac
    elseif epoch >= total_epochs - taper_epochs
        # Phase 3: linear taper from lr_max to lr_min
        remaining = Float32(total_epochs - epoch) / Float32(taper_epochs)
        return lr_min + (lr_max - lr_min) * clamp(remaining, 0.0f0, 1.0f0)
    else
        # Phase 2: hold at lr_max
        return lr_max
    end
end
