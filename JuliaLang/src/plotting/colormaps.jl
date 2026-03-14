# colormaps.jl — Shared colormaps and palette utilities
#
# Used by plot_srnn.jl and plot_weights.jl.

using Plots: RGB, cgrad

# ═══════════════════════════════════════════════════════════════════════
# E/I PALETTES (ported from SRNNModel2.excitatory_colormap / inhibitory_colormap)
# ═══════════════════════════════════════════════════════════════════════

const E_PALETTE = [
    1.00 0.00 0.00;
    1.00 0.75 0.00;
    0.85 0.20 0.45;
    0.90 0.10 0.60;
    0.90 0.55 0.00;
    0.55 0.27 0.27;
    0.86 0.08 0.24;
    0.60 0.15 0.45;
]

const I_PALETTE = [
    0.00 0.45 0.74;
    0.00 0.75 1.00;
    0.20 0.47 0.62;
    0.00 0.50 0.50;
    0.30 0.75 0.93;
    0.25 0.62 0.75;
    0.00 0.80 0.80;
    0.15 0.55 0.65;
]

"""Cycle through a palette for line `i` (1-based)."""
_palette_color(palette, i) = RGB(palette[mod1(i, size(palette, 1)), :]...)

# ═══════════════════════════════════════════════════════════════════════
# DIVERGING COLORMAPS
# ═══════════════════════════════════════════════════════════════════════

"""
    rwb_colormap(n=256)

Red-White-Blue diverging colormap. Port of `redwhiteblue_colormap.m`.
Blue (negative) → White (zero) → Red (positive).
"""
function rwb_colormap(n=256)
    half = n ÷ 2
    blues = [RGB(t, t, 1.0) for t in range(0, 1, length=half)]
    reds  = [RGB(1.0, t, t) for t in range(1, 0, length=half)]
    return vcat(blues, reds)
end
