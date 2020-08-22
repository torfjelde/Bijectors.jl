#############
# Exp & Log #
#############

struct Exp{N} <: Bijector{N} end
struct Log{N} <: Bijector{N} end

up1(::Exp{N}) where {N} = Exp{N + 1}()
up1(::Log{N}) where {N} = Log{N + 1}()

inv(b::Exp{N}) where {N} = Log{N}()
inv(b::Log{N}) where {N} = Exp{N}()

Exp() = Exp{0}()
Log() = Log{0}()

@deftransform transform(b::Exp{0}, x::Real) = exp(x)
@deftransform transform(b::Log{0}, x::Real) = log(x)

@deftransform transform(b::Exp{1}, x::AbstractArray{<:Real}) = exp.(x)
@deftransform transform(b::Log{1}, x::AbstractArray{<:Real}) = log.(x)

@deftransform transform(b::Exp{2}, x::AbstractMatrix{<:Real}) = exp.(x)
@deftransform transform(b::Log{2}, x::AbstractMatrix{<:Real}) = log.(x)

logabsdetjac(b::Exp{0}, x::Real) = x
logabsdetjac(b::Exp{1}, x::AbstractVector) = sum(x)
logabsdetjac(b::Exp{2}, x::AbstractMatrix) = sum(x)

logabsdetjac(b::Log{0}, x::Real) = -log(x)
logabsdetjac(b::Log{1}, x::AbstractVector) = - sum(log, x)
logabsdetjac(b::Log{2}, x::AbstractMatrix) = - sum(log, x)
