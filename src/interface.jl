using Distributions, Bijectors
using ForwardDiff
using Tracker

import Base: inv, ∘

import Random: AbstractRNG
import Distributions: logpdf, rand, rand!, _rand!, _logpdf, params
import StatsBase: entropy

#######################################
# AD stuff "extracted" from Turing.jl #
#######################################

abstract type ADBackend end
struct ForwardDiffAD <: ADBackend end
struct TrackerAD <: ADBackend end

const ADBACKEND = Ref(:forward_diff)
function setadbackend(backend_sym)
    @assert backend_sym == :forward_diff || backend_sym == :reverse_diff
    backend_sym == :forward_diff
    ADBACKEND[] = backend_sym
end

ADBackend() = ADBackend(ADBACKEND[])
ADBackend(T::Symbol) = ADBackend(Val(T))
function ADBackend(::Val{T}) where {T}
    if T === :forward_diff
        return ForwardDiffAD
    else
        return TrackerAD
    end
end

######################
# Bijector interface #
######################

"Abstract type for a `Bijector`."
abstract type Bijector end

"Abstract type for a `Bijector` making use of auto-differentation (AD)."
abstract type ADBijector{AD} <: Bijector end

"""
    inv(b::Bijector)
    Inversed(b::Bijector)

A `Bijector` representing the inverse transform of `b`.
"""
struct Inversed{B <: Bijector} <: Bijector
    orig::B
end

Broadcast.broadcastable(b::Bijector) = Ref(b)

"""
    logabsdetjac(b::Bijector, x)
    logabsdetjac(ib::Inversed{<: Bijector}, y)

Computes the log(abs(det(J(x)))) where J is the jacobian of the transform.
Similarily for the inverse-transform.
"""
logabsdetjac(b::T1, y::T2) where {T<:Bijector,T1<:Inversed{T},T2} = 
    error("`logabsdetjac(b::$T1, y::$T2)` is not implemented.")

"""
    forward(b::Bijector, x)
    forward(ib::Inversed{<: Bijector}, y)

Computes both `transform` and `logabsdetjac` in one forward pass, and
returns a named tuple `(rv=b(x), logabsdetjac=logabsdetjac(b, x))`.

This defaults to the call above, but often one can re-use computation
in the computation of the forward pass and the computation of the
`logabsdetjac`. `forward` allows the user to take advantange of such
efficiencies, if they exist.
"""
forward(b::T1, y::T2) where {T<:Bijector,T1<:Inversed{T},T2} = 
    error("`forward(b::$T1, y::$T2)` is not implemented.")

# default `forward` implementations; should in general implement efficient way
# of computing both `transform` and `logabsdetjac` together.
forward(b::Bijector, x) = (rv=b(x), logabsdetjac=logabsdetjac(b, x))
forward(ib::Inversed{<: Bijector}, y) = (rv=ib(y), logabsdetjac=logabsdetjac(ib, y))

# defaults implementation for inverses
logabsdetjac(ib::Inversed{<: Bijector}, y) = - logabsdetjac(ib.orig, ib(y))

inv(b::Bijector) = Inversed(b)
inv(ib::Inversed{<:Bijector}) = ib.orig

# AD implementations
function jacobian(b::ADBijector{<: ForwardDiffAD}, y::Real)
    return ForwardDiff.derivative(b, y)
end
function jacobian(b::ADBijector{<: ForwardDiffAD}, y::AbstractVector{<: Real})
    return ForwardDiff.jacobian(b, y)
end
function jacobian(b::Inversed{<: ADBijector{<: ForwardDiffAD}}, y::Real)
    return ForwardDiff.derivative(b, y)
end
function jacobian(b::Inversed{<: ADBijector{<: ForwardDiffAD}}, y::AbstractVector{<: Real})
    return ForwardDiff.jacobian(b, y)
end

function jacobian(b::ADBijector{<: TrackerAD}, y::Real)
    return Tracker.gradient(b, y)[1]
end
function jacobian(b::ADBijector{<: TrackerAD}, y::AbstractVector{<: Real})
    # we extract `data` so that we don't returne a `Tracked` type
    return Tracker.data(Tracker.jacobian(b, y))
end
function jacobian(b::Inversed{<: ADBijector{<: TrackerAD}}, y::Real)
    return Tracker.gradient(b, y)[1]
end
function jacobian(b::Inversed{<: ADBijector{<: TrackerAD}}, y::AbstractVector{<: Real})
    return Tracker.data(Tracker.jacobian(b, y))
end

# TODO: allow batch-computation, especially for univariate case?
logabsdetjac(b::ADBijector, x::Real) = log(abs(jacobian(b, x)))
# logabsdetjac(b::ADBijector, x::AbstractVector{<:Real}) = logabsdet(jacobian(b, x))[1]
function logabsdetjac(b::ADBijector, x::AbstractVector{<:Real})
    fact = lu(jacobian(b, x), check=false)
    return issuccess(fact) ? log(abs(det(fact))) : -Inf # TODO: or smallest possible float?
end

"""
    logabsdetjacinv(b::Bijector, x)

Just an alias for `logabsdetjac(inv(b), b(x))`.
"""
logabsdetjacinv(b::Bijector, x) = logabsdetjac(inv(b), b(x))

###############
# Composition #
###############

"""
    ∘(b1::Bijector, b2::Bijector)
    compose(ts::Bijector...)

A `Bijector` representing composition of bijectors.

# Examples
It's important to note that `∘` does what is expected mathematically, which means that the
bijectors are applied to the input right-to-left, e.g. first applying `b2` and then `b1`:
```
(b1 ∘ b2)(x) == b1(b2(x))     # => true
```
But in the `Composed` struct itself, we store the bijectors left-to-right, so that
```
cb1 = b1 ∘ b2                  # => Composed.ts == [b2, b1]
cb2 = compose(b2, b1)
cb1(x) == cb2(x) == b1(b2(x))  # => true
```
"""
struct Composed{A} <: Bijector
    ts::A
end
compose(ts::Bijector...) = Composed(ts)

# The transformation of `Composed` applies functions left-to-right
# but in mathematics we usually go from right-to-left; this reversal ensures that
# when we use the mathematical composition ∘ we get the expected behavior.
# TODO: change behavior of `transform` of `Composed`?
∘(b1::Bijector, b2::Bijector) = compose(b2, b1)

inv(ct::Composed) = Composed(map(inv, reverse(ct.ts)))

# # TODO: should arrays also be using recursive implementation instead?
function (cb::Composed{<: AbstractArray{<: Bijector}})(x)
    res = x
    for b ∈ cb.ts
        res = b(res)
    end

    return res
end

# recursive implementation like this allows type-inference
_transform(x, b1::Bijector, b2::Bijector) = b2(b1(x))
_transform(x, b::Bijector, bs::Bijector...) = _transform(b(x), bs...)
(cb::Composed{<: Tuple})(x) = _transform(x, cb.ts...)

# TODO: implement `forward` recursively
function forward(cb::Composed, x)
    res = (rv=x, logabsdetjac=0)
    for t in cb.ts
        res′ = forward(t, res.rv)
        res = (rv=res′.rv, logabsdetjac=res.logabsdetjac + res′.logabsdetjac)
    end
    return res
end

function _logabsdetjac(x, b1::Bijector, b2::Bijector)
    logabsdetjac(b2, b1(x)) + logabsdetjac(b1, x)
    res = forward(b1, x)
    return logabsdetjac(b2, res.rv) + res.logabsdetjac
end
function _logabsdetjac(x, b1::Bijector, bs::Bijector...)
    res = forward(b1, x)
    return _logabsdetjac(res.rv, bs...) + res.logabsdetjac
end
logabsdetjac(cb::Composed, x) = _logabsdetjac(x, cb.ts...)

###########
# Stacked #
###########
"""
    Stacked(bs)
    Stacked(bs, ranges)
    vcat(bs::Bijector...)

A `Bijector` which stacks bijectors together which can then be applied to a vector
where `bs[i]::Bijector` is applied to `x[ranges[i]]`.

# Examples
```
b1 = Logistic(0.0, 1.0)
b2 = Identity()
b = vcat(b1, b2)
b([0.0, 1.0]) == [b1(0.0), 1.0]  # => true
```
"""
struct Stacked{B, N} <: Bijector where N
    bs::B
    ranges::NTuple{N, UnitRange{Int}}
end
Stacked(bs) = Stacked(bs, NTuple{length(bs), UnitRange{Int}}([i:i for i = 1:length(bs)]))
Stacked(bs, ranges) = Stacked(bs, NTuple{length(bs), UnitRange{Int}}(ranges))

Base.vcat(bs::Bijector...) = Stacked(bs)

inv(sb::Stacked) = Stacked(inv.(sb.bs))

# TODO: Is there a better approach to this?
@generated function _transform(x, rs::NTuple{N, UnitRange{Int}}, bs::Bijector...) where N
    exprs = []
    for i = 1:N
        push!(exprs, :(bs[$i](x[rs[$i]])))
    end

    return :(vcat($(exprs...)))
end
_transform(x, rs::NTuple{1, UnitRange{Int}}, b::Bijector) = b(x)

(sb::Stacked)(x) = _transform(x, sb.ranges, sb.bs...)

# TODO: implement jacobian using matrices with BlockDiagonal.jl
# jacobian(sb::Stacked, x) = BDiagonal([jacobian(sb.bs[i], x[sb.ranges[i]]) for i = 1:length(sb.ranges)])
function logabsdetjac(sb::Stacked, x::AbstractArray{<: Real})
    # We also sum each of the `logabsdetjac()` calls because in the case we're `x`
    # is a vector, since we're using ranges to index we get back a vector.
    # In this case, 1D bijectors will act elementwise and return a vector of equal length.
    # TODO: Don't do this double-sum? Would be nice to be able to batch things, right?
    return sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges[i]])) for i = 1:length(sb.ranges)])
end


##############################
# Example bijector: Identity #
##############################

struct Identity <: Bijector end
(::Identity)(x) = x
(::Inversed{Identity})(y) = y

forward(::Identity, x) = (rv=x, logabsdetjac=zero(x))

logabsdetjac(::Identity, y::T) where T <: Real = zero(T)
logabsdetjac(::Identity, y::AbstractVector{T}) where T <: Real = zero(T)

const IdentityBijector = Identity()

###############################
# Example: Logit and Logistic #
###############################
using StatsFuns: logit, logistic

struct Logit{T<:Real} <: Bijector
    a::T
    b::T
end

(b::Logit)(x) = @. logit((x - b.a) / (b.b - b.a))
(ib::Inversed{Logit{T}})(y) where T <: Real = @. (ib.orig.b - ib.orig.a) * logistic(y) + ib.orig.a

logabsdetjac(b::Logit{<:Real}, x) = @. log((x - b.a) * (b.b - x) / (b.b - b.a))

struct Shift{T <: Real} <: Bijector
    val::T
end

(b::Shift)(x) = @. x + b.val
(b::Inversed{Shift{<: Real}})(y) = @. x - b.val

logabsdetjac(b::Shift{T}, x) where T <: Real = zero(T)


#######################################################
# Constrained to unconstrained distribution bijectors #
#######################################################
"""
    DistributionBijector(d::Distribution)
    DistributionBijector{<: ADBackend, D}(d::Distribution)

This is the default `Bijector` for a distribution. 

It uses `link` and `invlink` to compute the transformations, and `AD` to compute
the `jacobian` and `logabsdetjac`.
"""
struct DistributionBijector{AD, D} <: ADBijector{AD} where D <: Distribution
    dist::D
end
function DistributionBijector(dist::D) where D <: Distribution
    DistributionBijector{ADBackend(), D}(dist)
end

# Simply uses `link` and `invlink` as transforms with AD to get jacobian
(b::DistributionBijector)(x) = link(b.dist, x)
(ib::Inversed{<: DistributionBijector})(y) = invlink(ib.orig.dist, y)

"""
    bijector(d::Distribution)

Returns the constrained-to-unconstrained bijector for distribution `d`.
"""
bijector(d::Distribution) = DistributionBijector(d)

# Transformed distributions
struct TransformedDistribution{D, B, V} <: Distribution{V, Continuous} where {D <: Distribution{V, Continuous}, B <: Bijector}
    dist::D
    transform::B
end
function TransformedDistribution(d::Distribution{V, Continuous}, b::B) where {V <: VariateForm, B <: Bijector}
    return TransformedDistribution{typeof(d), B, V}(d, b)
end


const UnivariateTransformed = TransformedDistribution{<: Distribution, <: Bijector, Univariate}
const MultivariateTransformed = TransformedDistribution{<: Distribution, <: Bijector, Multivariate}
const Transformed = Union{UnivariateTransformed, MultivariateTransformed}

# Can implement these on a case-by-case basis
"""
    transformed(d::Distribution)
    transformed(d::Distribution, b::Bijector)

Couples the distribution `d` with the bijector `b` by returning a `UnivariateTransformed`
or `MultivariateTransformed`, depending on type `D`.

The resulting distribution will sample `x` from `d` and return `b(x)`.
The `logpdf` will be 
"""
transformed(d::Distribution, b::Bijector) = TransformedDistribution(d, b)
transformed(d::Distribution) = transformed(d, bijector(d))

# can specialize further by
bijector(d::Normal) = IdentityBijector
bijector(d::MvNormal) = IdentityBijector
bijector(d::Beta{T}) where T <: Real = Logit(zero(T), one(T))

##############################
# Distributions.jl interface #
##############################

# size
Base.length(td::MultivariateTransformed) = length(td.dist)

# logp
function logpdf(td::UnivariateTransformed, y::Real)
    # logpdf(td.dist, transform(inv(td.transform), y)) .+ logabsdetjac(inv(td.transform), y)
    logpdf_with_trans(td.dist, inv(td.transform)(y), true)
end
function _logpdf(td::MultivariateTransformed, y::AbstractVector{<: Real})
    # logpdf(td.dist, transform(inv(td.transform), y)) .+ logabsdetjac(inv(td.transform), y)
    logpdf_with_trans(td.dist, inv(td.transform)(y), true)
end

function logpdf_with_jac(td::UnivariateTransformed, y::Real)
    z = logabsdetjac(inv(td.transform), y)
    return (logpdf(td.dist, inv(td.transform)(y)) .+ z, z)
end

function logpdf_with_jac(td::MultivariateTransformed, y::AbstractVector{<:Real})
    z = logabsdetjac(inv(td.transform), y)
    return (logpdf(td.dist, inv(td.transform)(y)) .+ z, z)
end

# rand
rand(td::UnivariateTransformed) = td.transform(rand(td.dist))
rand(rng::AbstractRNG, td::UnivariateTransformed) = td.transform(rand(rng, td.dist))

rand(td::MultivariateTransformed) = td.transform(rand(td.dist))
function rand(td::MultivariateTransformed, num_samples::Int)
    res = hcat([td.transform(rand(td.dist)) for i = 1:num_samples]...)
    return res
end

function _rand!(rng::AbstractRNG, td::MultivariateTransformed, x::AbstractVector{<: Real})
    rand!(rng, td.dist, x)
    y = td.transform(x)
    copyto!(x, y)
end

# utility stuff
params(td::Transformed) = params(td.dist)
entropy(td::Transformed) = entropy(td.dist)

# logabsdetjac for distributions
logabsdetjac(d::UnivariateDistribution, x::T) where T <: Real = zero(T)
logabsdetjac(d::MultivariateDistribution, x::AbstractVector{T}) where T <: Real = zero(T)

# for transformed distributions the `y` is going to be the transformed variable
# and so we use the inverse transform to get what we want
# TODO: should this be renamed to `logabsdetinvjac`?
"""
    logabsdetjac(td::UnivariateTransformed, y::Real)
    logabsdetjac(td::MultivariateTransformed, y::AbstractVector{<:Real})

Computes the `logabsdetjac` of the _inverse_ transformation, since `rand(td)` returns
the _transformed_ random variable.
"""
logabsdetjac(td::UnivariateTransformed, y::Real) = logabsdetjac(inv(td.transform), y)
logabsdetjac(td::MultivariateTransformed, y::AbstractVector{<:Real}) = logabsdetjac(inv(td.transform), y)

# updating params of distributions
# TODO: should go somewhere else
"""
    update(d::D, θ)

Takes a distribution `d::Distribution` and parameters `θ` and returns a new distribution of
same UnionAll type but potentially parameterized by a different type. 

# Examples
This is very useful for
cases where one has a distribution parameterized by `Float64` and want to use AD to differentate,
say, the `logpdf` wrt. parameters of the distribution:
```julia-repl
julia> using Distributions, ForwardDiff

julia> # WRONG!
       grad(d::D, θ, x) where D <: Distribution = ForwardDiff.gradient(z -> logpdf(D(z...), x), θ)
grad (generic function with 1 method)

julia> d = Normal()
Normal{Float64}(μ=0.0, σ=1.0)

julia> grad(d, [1.0, 1.0], 0.0)
ERROR: MethodError: no method matching Float64(::ForwardDiff.Dual{ForwardDiff.Tag{getfield(Main, Symbol("##22#23")){Normal{Float64},Float64},Float64},Float64,2})
Closest candidates are:
  Float64(::Real, ::RoundingMode) where T<:AbstractFloat at rounding.jl:185
  Float64(::T<:Number) where T<:Number at boot.jl:725
  Float64(::Int8) at float.jl:60
  ...

julia> # WORKS!
       grad(d::Distribution, θ, x)= ForwardDiff.gradient(z -> logpdf(update(d, z), x), θ)
grad (generic function with 1 method)

julia> grad(d, [1.0, 1.0], 0.0)
2-element Array{Float64,1}:
 -1.0
  0.0
```

"""
@generated function update(d::D, θ) where D <: Distribution
    return :($(nameof(D))(θ...))
end

function update(d::TransformedDistribution{D, B, V}, θ) where {V, D <: Distribution{V, Continuous}, B <: Bijector}
    return TransformedDistribution(update(d.dist, θ), d.transform)
end
