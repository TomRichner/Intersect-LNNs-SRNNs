# activations.jl — Differentiable activation functions (port of PiecewiseSigmoid.m)
#
# All functions are Zygote-compatible: no mutation, no branching on array values.
# Uses ifelse() for branchless element-wise conditionals.

"""
    piecewise_sigmoid(x; S_a=0.9, S_c=0.35)

Piecewise linear/quadratic sigmoid bounded in [0, 1].
Port of MATLAB PiecewiseSigmoid.m.

- Linear region centered at `S_c` with slope 1
- Quadratic rounding at corners
- Saturates at 0 and 1

Parameters:
- `S_a`: Linear fraction parameter (0 to 1), controls the width of the linear region
- `S_c`: Center/shift parameter
"""
function piecewise_sigmoid(x; S_a=0.9, S_c=0.35)
    a = S_a / 2.0
    c = S_c
    k = 0.5 / (1.0 - 2.0 * a)

    x1 = c + a - 1.0
    x2 = c - a
    x3 = c + a
    x4 = c + 1.0 - a

    # Branchless piecewise evaluation using ifelse
    # Regions: x < x1 → 0 | x1..x2 → left quad | x2..x3 → linear | x3..x4 → right quad | x4< → 1
    left_quad = k .* (x .- x1).^2
    linear    = (x .- c) .+ 0.5
    right_quad = 1.0 .- k .* (x .- x4).^2

    y = ifelse.(x .< x1, 0.0,
        ifelse.(x .< x2, left_quad,
        ifelse.(x .<= x3, linear,
        ifelse.(x .<= x4, right_quad,
        1.0))))

    return y
end

"""
    piecewise_sigmoid_scalar(x; S_a=0.9, S_c=0.35)

Scalar version of piecewise_sigmoid for use in broadcasting.
"""
function piecewise_sigmoid_scalar(x::Real; S_a=0.9, S_c=0.35)
    a = S_a / 2.0
    c = S_c
    k = 0.5 / (1.0 - 2.0 * a)

    x1 = c + a - 1.0
    x2 = c - a
    x3 = c + a
    x4 = c + 1.0 - a

    if x < x1
        return zero(x)
    elseif x < x2
        return k * (x - x1)^2
    elseif x <= x3
        return (x - c) + 0.5
    elseif x <= x4
        return 1.0 - k * (x - x4)^2
    else
        return one(x)
    end
end

# For convenience: create a closure with fixed parameters
"""
    make_piecewise_sigmoid(; S_a=0.9, S_c=0.35)

Returns a function `f(x)` that applies the piecewise sigmoid element-wise.
Suitable for use as an activation function in Lux layers.
"""
function make_piecewise_sigmoid(; S_a=0.9, S_c=0.35)
    return x -> piecewise_sigmoid(x; S_a=S_a, S_c=S_c)
end
