using Distributions
using LinearAlgebra
using Random
using Flux


#########################################################################################################################
# Planar and Radial Flows : Variational Inference with Normalizing Flows, D. Rezende, S. Mohamed(2015) arXiv:1505.05770 #
#########################################################################################################################

mutable struct PlanarLayer <: Bijector
    w
    
    u
    u_hat
    b
end

mutable struct RadialLayer <: Bijector
    α
    β
    z_not
end

function update_u_hat(u, w)
    # to preserve invertibility
    u_hat = u + (m(transpose(w)*u) - transpose(w)*u)[1]*w/(norm(w[:,1],2)^2)
end

function update_u_hat!(flow::PlanarLayer)
    flow.u_hat = flow.u + (m(transpose(flow.w)*flow.u) - transpose(flow.w)*flow.u)[1]*flow.w/(norm(flow.w,2)^2)
end


function PlanarLayer(dims::Int)
    w = param(randn(dims, 1))
    u = param(randn(dims, 1))
    b = param(randn(1))
    u_hat = update_u_hat(u, w)
    return PlanarLayer(w, u, u_hat, b)
end

function RadialLayer(dims::Int)
    α_ = params(randn(1))
    β = params(randn(1))
    z_not = param(randn(dims, 1))
    return RadialLayer(α, β, z_not)
end

m(x) = -1 .+ log.(1 .+ exp.(x)) #for planar flow
dtanh(x) = 1 .- (tanh.(x)).^2 #for planar flow
ψ(z, w, b) = dtanh(transpose(w)*z .+ b).*w #for planar flow
softplus(x) = log.(1 .+ exp.(x)) #for radial flow
h(α, r) = 1 ./ (α .+ r) #for radial flow
dh(α, r) = -dh(α, r).^2 #for radial flow

function transform(flow::PlanarLayer, z)
    return z + flow.u_hat*tanh.(transpose(flow.w)*z .+ flow.b)
end

function transform(flow::RadialLayer, z)
    α = softplus(flow.α_)
    β_hat = -α + softplus(flow.β)
    r = norm.(z - flow.z_not, 1)
    return z + β_hat*h(α, r)*(z - flow.z_not)
end

function forward(flow::T, z) where {T<:PlanarLayer}
    update_u_hat!(flow)
    # compute log_det_jacobian
    transformed = transform(flow, z)
    psi = ψ(transformed, flow.w, flow.b)
    log_det_jacobian = log.(abs.(1.0 .+ transpose(psi)*flow.u_hat))

    return (rv=transformed, logabsdetjacob=Bijector)
end


function forward(flow::T, z) where {T<:RadialLayer}
    # compute log_det_jacobian
    transformed = transform(flow, z)
    α = softplus(flow.α_)
    β_hat = -α + softplus(flow.β)
    r = norm.(z - flow.z_not, 1)
    d = size(flow.z_not)[1]
    log_det_jacobian = log.(((1 + β_hat*h(α, r)).^(d-1)) .* ( 1 +  β_hat*h(α, r) + β_hat*dh(α, r)*r))
    return (rv=transformed, logabsdetjacob=log_det_jacobian)
end