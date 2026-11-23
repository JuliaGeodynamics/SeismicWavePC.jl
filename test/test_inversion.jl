# Interfaces to inversion packages: parametrizations, objective and gradient, Optim, LogDensityProblems, ChainRules, ForwardDiff
using LogDensityProblems, ChainRulesCore, ForwardDiff

@testset "inversion interfaces" begin
    m = test_model(:psv)
    solid = m.fields.vs .> 0
    # parametrizations: round trips
    p = ModelParameters(m, :vs, :rho; log=true)
    θ = initial_parameters(p)
    @test length(θ) == length(p) == 2count(solid)
    @test to_model(p, θ).fields.vs ≈ m.fields.vs && to_model(p, θ).fields.vp == m.fields.vp
    basis = [Float64.(solid .& (m.z.val .> -1)), Float64.(solid .& (m.z.val .<= -1))]
    pb = ModelParameters(m, :vs; basis)
    @test initial_parameters(pb) == [0.0, 0.0]
    @test to_model(pb, [0.1, -0.2]).fields.vs ≈ m.fields.vs .+ 0.1 .* basis[1] .- 0.2 .* basis[2]
    @test_throws ErrorException ModelParameters(m, :nofield)
    @test_throws ErrorException ModelParameters(m)

    kw = (; dt=DT, nt=150, T=Float64)
    src = test_source()
    truth = to_model(pb, [0.05, -0.03])
    obs = simulate(truth; kw..., sources=src, stations=test_stations(:psv), verbose=false)
    prior(θ) = (-sum(abs2, θ) / 2, -θ)
    prob = WaveformProblem(pb, obs; sources=src, prior, kw...)
    θ0 = [0.01, 0.02]
    Φ, g = objective_and_gradient(prob, θ0)
    @test Φ ≈ objective(WaveformProblem(pb, obs; sources=src, prior, kw...), θ0)      # without the cache
    fd = [(objective(prob, θ0 .+ 1e-5 .* (1:2 .== i)) - objective(prob, θ0 .- 1e-5 .* (1:2 .== i))) / 2e-5 for i in 1:2]
    @test g ≈ fd rtol = 1e-5
    G = zeros(2)
    @test fg!(prob)(0.0, G, θ0) ≈ Φ && G ≈ g && fg!(prob)(0.0, nothing, θ0) ≈ Φ          # Optim.only_fg!
    @test LogDensityProblems.dimension(prob) == 2
    @test LogDensityProblems.capabilities(typeof(prob)) == LogDensityProblems.LogDensityOrder{1}()
    @test all(LogDensityProblems.logdensity_and_gradient(prob, θ0) .≈ (-Φ, -g))
    Φr, pb_ = ChainRulesCore.rrule(objective, prob, θ0)
    @test Φr ≈ Φ && pb_(2.0)[3] ≈ 2g
    @test ForwardDiff.gradient(θ -> objective(prob, θ), θ0) ≈ g                         # Turing's default AD backend
    # invalid models: Φ = Inf, log density -Inf
    @test objective(prob, [-5.0, 0.0]) == Inf
    @test LogDensityProblems.logdensity(prob, [-5.0, 0.0]) == -Inf
    # cells with log parameters: gradient by the chain rule
    pc = ModelParameters(m, :vs; log=true)
    probc = WaveformProblem(pc, obs; sources=src, kw...)
    θc = initial_parameters(pc)
    gc = objective_and_gradient(probc, θc)[2]
    k = argmax(abs.(gc))
    e = zeros(length(θc)); e[k] = 1e-4
    @test gc[k] ≈ (objective(probc, θc .+ e) - objective(probc, θc .- e)) / 2e-4 rtol = 1e-4
end
