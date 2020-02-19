using Bijectors

struct Residual{G, N} <: Bijector{N}
    g::G
    m::Int # Number of samples for Hutchin estimator 
end

closedform(::Residual) = false

(b::Residual)(x::AbstractVector) = 1 .+ b.g(x)

function logabsdetjac(b::Residual, x::AbstractVector)
    J = Bijectors.jacobian(b.g, x)
    
    # Hutchin estimator
    d = length(x)

    acc = 0.0 # TODO: fix typing

    vs = randn(d, b.m)
    J_k = J
    
    for k = 1:n
        sgn = (k mod 2 == 0) ? (-1.) : 1.
        for v in eachcol(vs)
            acc += (sgn / k) * (v' * (J_k) * v)
        end
    end

    vs' * J * vs

    # Russian roulette estimator
end

d = 2
x = randn(d)

f(x) = (1 / d) .* x

m = 10
vs = randn(d, m)
b = PlanarLayer(2)

using ForwardDiff
Bijectors.jacobian(b::Bijector, x::AbstractVector) = ForwardDiff.jacobian(b, x)
Bijectors.jacobian(b::typeof(f), x::AbstractVector) = ForwardDiff.jacobian(b, x)
J = Bijectors.jacobian(f, x)
vs' * J * vs

n = 100
vs = randn(d, 100)
J_k = J
acc = 0.0 # TODO: fix typing

for k = 1:n
    sgn = (k % 2 == 0) ? (-1.) : 1.
    for v in eachcol(vs)
        acc += (sgn / k) * (v' * (J_k) * v)
    end
end
acc

n = 100
m = 1000
exact = false

vs = randn(d, m)
J_k = J
acc = 0.0 # TODO: fix typing

for k = 1:n
    if !exact
        for v in eachcol(vs)
            acc += ((-1)^(k + 1) / k) * (v' * (J_k) * v) / m
        end
    else
        acc += LinearAlgebra.tr(((-1)^(k + 1) / k) .* J_k)
    end
    J_k *= J
end
acc

using Plots
abstract type TraceEstimator end
struct ExactEstimator <: TraceEstimator end

struct HutchinsonEstimator <: TraceEstimator
    m::Int
end

using LinearAlgebra
function LinearAlgebra.tr(est::HutchinsonEstimator, A::AbstractMatrix)
    @assert size(A, 1) == size(A, 2)
    d = size(A, 1)
    return mean(v' * A * v for v in eachcol(randn(d, est.m)))
end

est = HutchinsonEstimator(1000)
LinearAlgebra.tr(est, J)

LinearAlgebra.tr(J)

res = [
    mean([LinearAlgebra.tr(HutchinsonEstimator(j), J) for i = 1:100])
    for j = 1:10:1000
]

plot(res)
savefig("test.png")

# Russian Roulette series estimator
dist = Geometric(0.01)
n = rand(dist)
cdf(dist, n)


acc = 0.0

# Compute first two terms exactly
J_k = J
acc += LinearAlgebra.tr(J_k)
J_k *= J
acc += - 0.5 * LinearAlgebra.tr(J_k)

start_idx = 3

n = rand(dist)

if n > 0
    for k = start_idx:n
        # P(N ≥ k) = P(N > k - 1) = 1 - P(N ≤ k - 1) = 1 - F(k - 1)
        p = cdf(dist, k - 1)
        J_k *= J
        A = ((-1)^(k + 1) / k) .* J_k
        acc += LinearAlgebra.tr(est, A) / (1 - p)
        # acc += LinearAlgebra.tr(A) / (1 - p)
    end
end

acc

true_acc = sum([LinearAlgebra.tr(((-1)^(k + 1) / k) .* J^k) for k = 1:100])
