using Test

using Bijectors
using Bijectors: RationalQuadraticSpline

knots = 10
widths, heights, derivatives = Bijectors.Unconstrained.((randn(knots), randn(knots), randn(knots - 1)))
B = 10 # defines the [-B, B] interval where the map is non-identity

b = RationalQuadraticSpline(widths, heights, derivatives, B)

b(3.0)

using Plots
xs = -B:0.01:B
plot(xs, b(xs), label="b(x)")
vline!(b.widths, label="knots")

using ForwardDiff

function bijector_from_array(θ)
    widths = Bijectors.Unconstrained(θ[1:knots])
    heights = Bijectors.Unconstrained(θ[knots + 1:2 * knots])
    derivatives = Bijectors.Unconstrained(θ[2 * knots + 1:end])

    b = RationalQuadraticSpline(widths, heights, derivatives, B)
end

b = bijector_from_array(randn(3 * knots - 1))

xs = -B:0.01:B
ys = b(xs)
objective(b) = mean(abs2, b(xs) - ys)
f(θ) = (objective ∘ bijector_from_array)(θ)

θ = randn(3 * knots - 1)
f(θ)

θ = randn(3 * knots - 1)
ForwardDiff.gradient(f, θ)

using Optim
res = optimize(f, θ, LBFGS(); autodiff=:forward)
b = bijector_from_array(res.minimizer)

xs = -B:0.01:B
plot(xs, b(xs), label="b(x)", alpha=0.5, width=5)
plot!(xs, ys, label="ys", alpha=1, width=1)
vline!(b.widths, label="knots")

# with a `CouplingLayer`
d = 2

function bijector_from_array(θ)
    i = 0
    widths = Bijectors.Unconstrained(reshape(θ[i + 1:i + d * knots], (knots, d)))
    i += knots * d
    
    heights = Bijectors.Unconstrained(reshape(θ[i + 1:i + d * knots], (knots, d)))
    i += d * knots
    derivatives = Bijectors.Unconstrained(reshape(θ[i + 1:end], (knots - 1, d)))

    b = RationalQuadraticSpline(widths, heights, derivatives, B)
end

θ = randn((3 * knots - 1) * d)
b = bijector_from_array(θ)

b(ones(2))
b(zeros(2))

mask = Bijectors.PartitionMask(d, [1], [2])
cl = CouplingLayer(mask, b)

# TODO: Add some map x₂ ↦ θ(x₂) used to parameterize a RQS

# Improving the `Unconstrained` constructor
import NNlib
w = reshape(randn(knots * d), (knots, d))
w_normed = NNlib.softmax(w)
cumsum(w_normed; dims=1)

### Trying it out
# TODO: Considering whether or not to drop the `vcat` call in the constructor of `RationalQuadraticSpline`,
# in which case we have to do some work to make sure we're not needlessly restricting functionality :/
θ = randn((3 * knots - 1) * d)
i = 0
widths = Bijectors.Unconstrained(reshape(θ[i + 1:i + d * knots], (knots, d)))
i += knots * d
heights = Bijectors.Unconstrained(reshape(θ[i + 1:i + d * knots], (knots, d)))
i += d * knots
derivatives = Bijectors.Unconstrained(reshape(θ[i + 1:end], (knots - 1, d)))

w = 2 * B * (cumsum(NNlib.softmax(values(widths)); dims=1) .- 0.5)
h = 2 * B * (cumsum(NNlib.softmax(values(heights)); dims=1) .- 0.5)
d = NNlib.softplus.(values(derivatives))

size(w), size(h), size(d)

widths, heights, derivatives = w[:, 1], h[:, 1], d[:, 1]
x = -10

K = length(widths)
k = searchsortedfirst(widths, x)

# Observe that by early return if `x ∉ [-B, B]`, we're never going to have `k ≥ K`,
# since `B ∈ weights`.
w = if k == 1
    widths[k + 1] + widths[end] # again, `widths[end] == B` and we want `-B`
else
    widths[k + 1] - widths[k]
end
    

w = (k == 1) ? widths[k + 1] + widths[end] : widths[k + 1] - widths[k]
Δy = (k == 1) ? heights[k + 1] + heights[end] : heights[k + 1] - heights[k]

function rqs_univariate_new(widths, heights, derivatives, x::Real)
    # We're working on [-B, B] and `widths[end]` is `B`
    if (x ≤ -widths[end]) || (x ≥ widths[end])
        return x
    end
    
    K = length(widths)
    # @assert K == length(x) "length(x) should be the same length as the widhts/heights"

    # Find which bin `x` is in; subtract 1 because `searchsortedfirst` returns idx of ≥ not ≤
    k = searchsortedfirst(widths, x) - 1

    # Width
    # If k == 0 then we should put it in the bin `[-B, widths[1]]`
    wₖ = (k == 0) ? widths[end] : widths[k]
    w = widths[k + 1] - wₖ

    # Slope
    hₖ = (k == 0) ? heights[end] : heights[k]
    Δy = heights[k + 1] - hₖ

    s = Δy / w
    ξ = (x - wₖ) / w

    # Eq. (14)
    dₖ = (k == 0) ? 1. : derivatives[k]
    numerator = Δy * (s * ξ^2 + dₖ * ξ * (1 - ξ))
    denominator = s + (derivatives[k + 1] + dₖ - 2s) * ξ * (1 - ξ)
    g = hₖ + numerator / denominator

    return g
end

w = 2 * B * (cumsum(NNlib.softmax(values(widths)); dims=1) .- 0.5)
h = 2 * B * (cumsum(NNlib.softmax(values(heights)); dims=1) .- 0.5)
d = NNlib.softplus.(values(derivatives))

# ON boundary
x = -10.
@test rqs_univariate_new(w, h, d, x) ≈ b(x)
x = 10.
@test rqs_univariate_new(w, h, d, x) ≈ b(x)

# INSIDE
x = -5.
@test rqs_univariate_new(w, h, d, x) ≈ b(x)
x = 5.
@test rqs_univariate_new(w, h, d, x) ≈ b(x)

# OUTSIDE
x = -20.
@test rqs_univariate_new(w, h, d, x) ≈ b(x)
x = 20.
@test rqs_univariate_new(w, h, d, x) ≈ b(x)
