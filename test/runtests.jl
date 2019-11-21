using Bijectors, Random
using Test

Random.seed!(123456)

@testset "Interface" begin
    include("interface.jl")

    include("bijectors/coupling_layer.jl")
end
include("transform.jl")
