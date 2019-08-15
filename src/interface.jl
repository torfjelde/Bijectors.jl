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
    backend_sym == :forward_diff && CHUNKSIZE[] == 0 && setchunksize(40)
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

abstract type Bijector end
abstract type ADBijector{AD} <: Bijector end

struct Inversed{B <: Bijector} <: Bijector
    orig::B
end

Broadcast.broadcastable(b::Bijector) = Ref(b)

"Computes the log(abs(det(J(x)))) where J is the jacobian of the transform."
logabsdetjac(b::T1, y::T2) where {T<:Bijector,T1<:Inversed{T},T2} = 
    error("`logabsdetjac(b::$T1, y::$T2)` is not implemented.")

"Transforms the input using the bijector."
transform(b::T1, y::T2) where {T<:Bijector,T1<:Inversed{T},T2} =
    error("`transform(b::$T1, y::$T2)` is not implemented.")

"Computes both `transform` and `logabsdetjac` in one forward pass."
forward(b::T1, y::T2) where {T<:Bijector,T1<:Inversed{T},T2} = 
    error("`forward(b::$T1, y::$T2)` is not implemented.")


transform(b::Bijector) = x -> transform(b, x)
(ib::Inversed{<: Bijector})(y) = transform(ib, y)

# default `forward` implementations; should in general implement efficient way
# of computing both `transform` and `logabsdetjac` together.
forward(b::Bijector, x) = (rv=transform(b, x), logabsdetjac=logabsdetjac(b, x))
forward(ib::Inversed{<: Bijector}, y) = (rv=transform(ib, y), logabsdetjac=logabsdetjac(ib, y))

# defaults implementation for inverses
logabsdetjac(ib::Inversed{<: Bijector}, y) = - logabsdetjac(ib.orig, transform(ib, y))

inv(b::Bijector) = Inversed(b)
inv(ib::Inversed{<:Bijector}) = ib.orig

# AD implementations
function jacobian(b::ADBijector{<: ForwardDiffAD}, y::Real)
    return ForwardDiff.derivative(z -> transform(b, z), y)
end
function jacobian(b::ADBijector{<: ForwardDiffAD}, y::AbstractVector{<: Real})
    return ForwardDiff.jacobian(z -> transform(b, z), y)
end
function jacobian(b::Inversed{<: ADBijector{<: ForwardDiffAD}}, y::Real)
    return ForwardDiff.derivative(z -> transform(b, z), y)
end
function jacobian(b::Inversed{<: ADBijector{<: ForwardDiffAD}}, y::AbstractVector{<: Real})
    return ForwardDiff.jacobian(z -> transform(b, z), y)
end

function jacobian(b::ADBijector{<: TrackerAD}, y::Real)
    return Tracker.gradient(z -> transform(b, z), y)[1]
end
function jacobian(b::ADBijector{<: TrackerAD}, y::AbstractVector{<: Real})
    return Tracker.jacobian(z -> transform(b, z), y)
end
function jacobian(b::Inversed{<: ADBijector{<: TrackerAD}}, y::Real)
    return Tracker.gradient(z -> transform(b, z), y)[1]
end
function jacobian(b::Inversed{<: ADBijector{<: TrackerAD}}, y::AbstractVector{<: Real})
    return Tracker.jacobian(z -> transform(b, z), y)
end

# TODO: allow batch-computation, especially for univariate case?
"Computes the absolute determinant of the Jacobian of the inverse-transformation."
logabsdetjac(b::ADBijector, x::Real) = log(abs(jacobian(b, x)))
function logabsdetjac(b::ADBijector, x::AbstractVector{<:Real})
    fact = lu(jacobian(b, x), check=false)
    return issuccess(fact) ? log(abs(det(fact))) : -Inf # TODO: or smallest possible float?
end

###############
# Composition #
###############

struct Composed{A} <: Bijector
    ts::A
end
compose(ts...) = Composed(ts)

# The transformation of `Composed` applies functions left-to-right
# but in mathematics we usually go from right-to-left; this reversal ensures that
# when we use the mathematical composition ∘ we get the expected behavior.
# TODO: change behavior of `transform` of `Composed`?
∘(b1::Bijector, b2::Bijector) = compose(b2, b1)

inv(ct::Composed) = Composed(map(inv, reverse(ct.ts)))

# # TODO: can we implement this recursively, and with aggressive inlining, make this type-stable?
function transform(cb::Composed{<: AbstractArray{<: Bijector}}, x)
    res = x
    for b ∈ cb.ts
        res = transform(b, res)
    end

    return res
end

# recursive implementation like this allows type-inference
_transform(x, b1::Bijector, b2::Bijector) = transform(b2, transform(b1, x))
_transform(x, b::Bijector, bs::Bijector...) = _transform(transform(b, x), bs...)
transform(cb::Composed{<: Tuple}, x) = _transform(x, cb.ts...)
(cb::Composed)(x) = transform(cb, x)

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
    logabsdetjac(b2, transform(b1, x)) + logabsdetjac(b1, x)
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
struct Stacked{B, N} <: Bijector where N
    bs::B
    ranges::NTuple{N, UnitRange{Int}}
end
Stacked(bs) = Stacked(bs, NTuple{length(bs), UnitRange{Int}}([i:i for i = 1:length(bs)]))
Stacked(bs, ranges) = Stacked(bs, NTuple{length(bs), UnitRange{Int}}(ranges))

Base.vcat(bs::Bijector...) = Stacked(bs)

inv(sb::Stacked) = Stacked(inv.(st.bs))

function transform(sb::Stacked{<: AbstractArray{<: Bijector}}, x)
    res = similar(x)
    for (i, r) in enumerate(sb.ranges)
        if length(r) == 1
            res[r] .= sb.bs[i](x[r[1]])
        else
            res[r] .= sb.bs[i](x[r])
        end
    end

    return res
end

# function _transform(x, rs::NTuple{2, UnitRange{Int}}, b1::Bijector, b2::Bijector)
#     idx1 = length(rs[1]) == 1 ? rs[1][1] : rs[1]
#     idx2 = length(rs[2]) == 1 ? rs[2][1] : rs[2]
#     x1 = x[idx1]
#     x2 = x[idx2]
#     return vcat(transform(b1, x1), transform(b2, x2))
# end

# function _transform(x, rs::NTuple{N, UnitRange{Int}}, b1::Bijector, bs::Bijector...) where N
#     idx = length(rs[1]) == 1 ? rs[1][1] : rs[1]
#     return vcat(transform(b1, x[idx]), _transform(x, rs[2:end], bs...))
# end

@generated function _transform(x, rs::NTuple{N, UnitRange{Int}}, bs::Bijector...) where N
    exprs = []
    for i = 1:N
        push!(exprs, :(transform(bs[$i], length(rs[$i]) == 1 ? x[rs[$i][1]] : x[rs[$i]])))
    end

    return :(vcat($(exprs...), ))
end

transform(sb::Stacked{<: Tuple}, x) = _transform(x, sb.ranges, sb.bs...)

# jacobian(sb::Stacked, x) = BDiagonal([jacobian(sb.bs[i], x[sb.ranges[i]]) for i = 1:length(sb.ranges)])
function logabsdetjac(sb::Stacked, x)
    # We also sum each of the `logabsdetjac()` calls because in the case we're `x`
    # is a vector, since we're using ranges to index we get back a vector.
    # In this case, 1D bijectors will act elementwise and return a vector of equal length.
    # TODO: Don't do this? Would be nice to be able to batch things, right?
    return sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges[i]])) for i = 1:length(sb.ranges)])
end


##############################
# Example bijector: Identity #
##############################

struct Identity <: Bijector end
transform(::Identity, x) = x
transform(::Inversed{Identity}, y) = y
(b::Identity)(x) = transform(b, x)

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

transform(b::Logit, x) = @. logit((x - b.a) / (b.b - b.a))
transform(ib::Inversed{Logit{T}}, y) where T <: Real = @. (ib.orig.b - ib.orig.a) * logistic(y) + ib.orig.a
(b::Logit)(x) = transform(b, x)

logabsdetjac(b::Logit{<:Real}, x) = @. log((x - b.a) * (b.b - x) / (b.b - b.a))
# forward(b::Logit, x::Real) = (rv=transform(b, x), logabsdetjac=-logabsdetjac(b, x))

struct Shift{T <: Real} <: Bijector
    val::T
end

transform(b::Shift, x) = @. x + b.val
transform(b::Inversed{Shift{<: Real}}, y) = @. x - b.val
(b::Shift)(x) = transform(b, x)

logabsdetjac(b::Shift{T}, x) where T <: Real = zero(T)


#######################################################
# Constrained to unconstrained distribution bijectors #
#######################################################
struct DistributionBijector{AD, D} <: ADBijector{AD} where D <: Distribution
    dist::D
end
function DistributionBijector(dist::D) where D <: Distribution
    DistributionBijector{ADBackend(), D}(dist)
end

# Simply uses `link` and `invlink` as transforms with AD to get jacobian
transform(b::DistributionBijector, x) = link(b.dist, x)
transform(ib::Inversed{<: DistributionBijector}, y) = invlink(ib.orig.dist, y)
(b::DistributionBijector)(x) = transform(b, x)

"Returns the constrained-to-unconstrained bijector for distribution `d`."
bijector(d::Distribution) = DistributionBijector(d)

# Transformed distributions
struct UnivariateTransformed{D, B} <: Distribution{Univariate, Continuous} where {D <: UnivariateDistribution, B <: Bijector}
    dist::D
    transform::B
end

struct MultivariateTransformed{D, B} <: Distribution{Multivariate, Continuous} where {D <: MultivariateDistribution, B <: Bijector}
    dist::D
    transform::B
end

const Transformed = Union{UnivariateTransformed, MultivariateTransformed}

# Can implement these on a case-by-case basis
transformed(d::UnivariateDistribution, b::Bijector) = UnivariateTransformed(d, b)
transformed(d::MultivariateDistribution, b::Bijector) = MultivariateTransformed(d, b)
transformed(d) = transformed(d, bijector(d))

# can specialize further by
bijector(d::Normal) = IdentityBijector
bijector(d::Beta{T}) where T <: Real = Logit(zero(T), one(T))

##############################
# Distributions.jl interface #
##############################

# size
Base.length(td::MultivariateTransformed) = length(td.dist)

# logp
function logpdf(td::UnivariateTransformed, y::Real)
    # logpdf(td.dist, transform(inv(td.transform), y)) .+ logabsdetjac(inv(td.transform), y)
    logpdf_with_trans(td.dist, transform(inv(td.transform), y), true)
end
function _logpdf(td::MultivariateTransformed, y::AbstractVector{<: Real})
    # logpdf(td.dist, transform(inv(td.transform), y)) .+ logabsdetjac(inv(td.transform), y)
    logpdf_with_trans(td.dist, transform(inv(td.transform), y), true)
end

function logpdf_with_jac(td::UnivariateTransformed, y::Real)
    z = logabsdetjac(inv(td.transform), y)
    return (logpdf(td.dist, transform(inv(td.transform), y)) .+ z, z)
end

function logpdf_with_jac(td::MultivariateTransformed, y::AbstractVector{<:Real})
    z = logabsdetjac(inv(td.transform), y)
    return (logpdf(td.dist, transform(inv(td.transform), y)) .+ z, z)
end

# rand
rand(rng::AbstractRNG, td::UnivariateTransformed) = transform(td.transform, rand(td.dist))
function _rand!(rng::AbstractRNG, td::MultivariateTransformed, x::AbstractVector{<: Real})
    rand!(rng, td.dist, x)
    y = transform(td.transform, x)
    copyto!(x, y)
end

# utility stuff
params(td::Transformed) = params(td.dist)
entropy(td::Transformed) = entropy(td.dist)


logabsdetjac(d::UnivariateDistribution, x::T) where T <: Real = zero(T)
logabsdetjac(d::MultivariateDistribution, x::AbstractVector{T}) where T <: Real = zero(T)

# for transformed distributions the `y` is going to be the transformed variable
# and so we use the inverse transform to get what we want
# TODO: should this be renamed to `logabsdetinvjac`?
logabsdetjac(td::UnivariateTransformed, y::Real) = logabsdetjac(inv(td.transform), y)
logabsdetjac(td::MultivariateTransformed, y::AbstractArray{ <: Real}) = logabsdetjac(inv(td.transform), y)
