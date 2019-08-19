using Test
using Bijectors
using Random
using LinearAlgebra

Random.seed!(123)

# Scalar tests
@testset "Interface" begin
    # Tests with scalar-valued distributions.
    uni_dists = [
        Arcsine(2, 4),
        Beta(2,2),
        BetaPrime(),
        Biweight(),
        Cauchy(),
        Chi(3),
        Chisq(2),
        Cosine(),
        Epanechnikov(),
        Erlang(),
        Exponential(),
        FDist(1, 1),
        Frechet(),
        Gamma(),
        InverseGamma(),
        InverseGaussian(),
        # Kolmogorov(),
        Laplace(),
        Levy(),
        Logistic(),
        LogNormal(1.0, 2.5),
        Normal(0.1, 2.5),
        Pareto(),
        Rayleigh(1.0),
        TDist(2),
        TruncatedNormal(0, 1, -Inf, 2),
    ]
    
    for dist in uni_dists
        @testset "$dist: dist" begin
            td = transformed(dist)

            # single sample
            y = rand(td)
            x = inv(td.transform)(y)
            @test logpdf(td, y) ≈ logpdf_with_trans(dist, x, true)

            # multi-sample
            y = rand(td, 10)
            x = inv(td.transform).(y)
            @test logpdf.(td, y) ≈ logpdf_with_trans.(dist, x, true)
        end

        @testset "$dist: ForwardDiff AD" begin
            x = rand(dist)
            b = DistributionBijector{Bijectors.ADBackend(:forward_diff), typeof(dist)}(dist)
            
            @test abs(det(Bijectors.jacobian(b, x))) > 0
            @test logabsdetjac(b, x) ≠ Inf

            y = b(x)
            b⁻¹ = inv(b)
            @test abs(det(Bijectors.jacobian(b⁻¹, y))) > 0
            @test logabsdetjac(b⁻¹, y) ≠ Inf
        end

        @testset "$dist: Tracker AD" begin
            x = rand(dist)
            b = DistributionBijector{Bijectors.ADBackend(:reverse_diff), typeof(dist)}(dist)
            
            @test abs(det(Bijectors.jacobian(b, x))) > 0
            @test logabsdetjac(b, x) ≠ Inf

            y = b(x)
            b⁻¹ = inv(b)
            @test abs(det(Bijectors.jacobian(b⁻¹, y))) > 0
            @test logabsdetjac(b⁻¹, y) ≠ Inf
        end
    end

    @testset "Composition <: Bijector" begin
        d = Beta()
        td = transformed(d)

        x = rand(d)
        y = td.transform(x)

        b = Bijectors.compose(td.transform, Bijectors.Identity())
        ib = inv(b)

        @test forward(b, x) == forward(td.transform, x)
        @test forward(ib, y) == forward(inv(td.transform), y)

        # inverse works fine for composition
        cb = b ∘ ib
        @test cb(x) ≈ x

        cb2 = cb ∘ cb
        @test cb(x) ≈ x

        # order of composed evaluation
        b1 = DistributionBijector(d)
        b2 = DistributionBijector(Gamma())

        cb = b1 ∘ b2
        @test cb(x) ≈ b1(b2(x))

        # contrived example
        b = bijector(d)
        cb = inv(b) ∘ b
        cb = cb ∘ cb
        @test (cb ∘ cb ∘ cb ∘ cb ∘ cb)(x) ≈ x
    end

    @testset "Stacked <: Bijector" begin
        # `logabsdetjac` without AD
        d = Beta()
        b = bijector(d)
        x = rand(d)
        y = b(x)
        sb = vcat(b, b, inv(b), inv(b))
        @test logabsdetjac(sb, [x, x, y, y]) ≈ 0.0

        # `logabsdetjac` with AD
        b = DistributionBijector(d)
        y = b(x)
        sb1 = vcat(b, b, inv(b), inv(b))             # <= tuple
        sb2 = Stacked([b, b, inv(b), inv(b)])        # <= Array
        @test logabsdetjac(sb1, [x, x, y, y]) ≈ 0.0
        @test logabsdetjac(sb2, [x, x, y, y]) ≈ 0.0

        @testset "Stacked: ADVI with MvNormal" begin
            # MvNormal test
            d = MvNormal(zeros(10), ones(10))
            dists = [
                Beta(),
                Beta(),
                Beta(),
                InverseGamma(),
                InverseGamma(),
                Gamma(),
                Gamma(),
                InverseGamma(),
                Cauchy(),
                Gamma()
            ]
            bs = bijector.(dists)    # constrained-to-unconstrained bijectors for dists
            ibs = inv.(bs)           # invert, so we get unconstrained-to-constrained
            sb = vcat(ibs...)        # => Stacked <: Bijector
            @test sb isa Stacked

            td = transformed(d, sb)  # => MultivariateTransformed <: Distribution{Multivariate, Continuous}
            @test td isa Distribution{Multivariate, Continuous}

            y = rand(td)

            bs = bijector.(tuple(dists...))
            ibs = inv.(bs)
            sb = vcat(ibs...)
            @test sb isa Stacked{<: Tuple}

            y = rand(td)

            # TODO: add AD tests
        end
    end

    @testset "Update" begin
        dists = [Normal(), Gamma()]
        for d in dists
            td = transformed(d)

            θ = (1.0, 1.0)

            td = update(td, θ)

            @test params(td.dist) == θ
            @test params(td) == θ
        end
    end

    @testset "Example: ADVI single" begin
        # Usage in ADVI
        d = Beta()
        b = DistributionBijector(d)    # [0, 1] → ℝ
        ib = inv(b)                    # ℝ → [0, 1]
        td = transformed(Normal(), ib) # x ∼ 𝓝(0, 1) then f(x) ∈ [0, 1]
        x = rand(td)                   # ∈ [0, 1]
        @test 0 ≤ x ≤ 1
    end
end
