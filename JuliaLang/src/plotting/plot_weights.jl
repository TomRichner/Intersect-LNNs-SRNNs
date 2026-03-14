# plot_weights.jl — Weight matrix visualization
#
# Depends on colormaps.jl (must be included first).

using Plots

"""
    plot_weight_matrix(W; title_str="", figsize=(600, 550))

Plot a weight matrix as a heatmap with red-white-blue diverging colormap.
Symmetric color limits centered at zero.
"""
function plot_weight_matrix(W::AbstractMatrix; title_str="W", figsize=(600, 550))
    wmax = maximum(abs.(W))
    fig = heatmap(W; yflip=true, aspect_ratio=:equal,
        color=cgrad(rwb_colormap()), clims=(-wmax, wmax),
        title=title_str, xlabel="pre", ylabel="post",
        size=figsize)
    return fig
end
