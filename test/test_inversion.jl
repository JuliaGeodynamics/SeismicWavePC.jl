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

@testset "inversion helpers" begin
    # misfits: gradients against finite differences on random traces (two components, three stations)
    noise(sz...) = reshape(sin.((1:prod(sz)) .* 1.618 .+ prod(sz)) .* cos.((1:prod(sz)) .* 0.37), sz)   # deterministic
    nt, dt = 200, 0.01
    t = (0:nt-1) .* dt
    wav(t0, f) = @. exp(-(f * (t - t0))^2) * cos(2π * f * (t - t0))
    obs = stack(stack(wav(0.8 + 0.1c + 0.05n, 4.0 + n) for c in 1:2) for n in 1:3)
    syn = obs .+ 0.3 .* stack(stack(wav(0.85 + 0.1c + 0.05n, 4.0 + n) for c in 1:2) for n in 1:3) .+ 0.01 .* noise(nt, 2, 3)
    misfits = [L2Misfit(normalize=true, window=(0.3, 1.6)), EnvelopeMisfit(weights=[1.0, 2.0, 0.5]), PhaseMisfit(),
               TFPhaseMisfit(width=0.15, band=(2.0, 12.0)), Filtered(L2Misfit(), (2.0, 8.0)),
               Filtered(PhaseMisfit(), (0.0, 10.0); order=2), TravelTimeMisfit(window=(0.3, 1.8))]
    for m in misfits
        J, adj = m(syn, obs, t)
        @test J > 0 && m(obs, obs, t)[1] < 1e-20
        d = noise(size(syn)...)
        h = 1e-4
        fd = (m(syn .+ h .* d, obs, t)[1] - m(syn .- h .* d, obs, t)[1]) / 2h
        @test sum(adj .* d) ≈ fd rtol = 1e-4
    end
    @test L2Misfit(normalize=true)(2obs, obs, t)[1] ≈ 6 / 2                       # 6 traces, each residual = its energy
    late = stack(stack(wav(0.9 + 0.1c + 0.05n, 4.0 + n) for c in 1:2) for n in 1:3)  # 0.1 s late
    @test PhaseMisfit()(3obs, obs, t)[1] < 0.02 * PhaseMisfit()(late, obs, t)[1]     # amplitudes hardly count (waterlevel)
    @test EnvelopeMisfit()(-obs, obs, t)[1] < 1e-20                                 # phase independent
    # band-pass filter: zero phase, symmetric (its own adjoint), removes frequencies outside the band
    x, y = noise(nt, 3), noise(nt + 1, 3)[2:end, :]
    @test sum(bandpass(x, dt, (1.0, 5.0)) .* y) ≈ sum(x .* bandpass(y, dt, (1.0, 5.0)))
    s20 = sin.(2π * 20 .* t) .* exp.(-((t .- 1) ./ 0.3) .^ 2)
    s3 = sin.(2π * 3 .* t) .* exp.(-((t .- 1) ./ 0.3) .^ 2)
    @test maximum(abs, bandpass(s20, dt, (0.0, 5.0))) < 1e-3 && maximum(abs, bandpass(s3, dt, (0.0, 10.0)) .- s3) < 1e-2
    @test argmax(bandpass(wav(1.0, 5.0), dt, (2.0, 8.0))) == argmax(wav(1.0, 5.0))       # no phase shift
    # geographic weights: a cluster of three stations weighs about as much as a single station
    sts = [Station("a", 0.0, 0.0, 0.0), Station("b", 0.1, 0.0, 0.0), Station("c", 0.0, 0.1, 0.0), Station("d", 50.0, 0.0, 0.0)]
    w = geographic_weights(sts, 5.0)
    @test sum(w) ≈ 4
    @test sum(w[1:3]) ≈ w[4] rtol = 0.01

    # smoothed cell parameters: initial model, smooth updates, exact gradient
    m = test_model(:psv)
    p = ModelParameters(m, :vs; log=true, smooth=0.5)
    θ0 = initial_parameters(p)
    @test to_model(p, θ0).fields.vs ≈ m.fields.vs
    e = zeros(length(θ0)); e[length(θ0)÷2] = 0.1
    dv = to_model(p, θ0 .+ e).fields.vs .- m.fields.vs
    @test count(abs.(dv) .> 1e-6) > 10                                              # a spike becomes a smooth blob
    kw = (; dt=DT, nt=150, T=Float64)
    src = test_source()
    sts2 = test_stations(:psv)
    obs1 = simulate(to_model(p, θ0 .+ 5e); kw..., sources=src, stations=sts2, verbose=false)
    prob = WaveformProblem(p, obs1; sources=src, misfit=PhaseMisfit(), kw...)
    Φ, g = objective_and_gradient(prob, θ0)
    k = argmax(abs.(g))
    ek = zeros(length(θ0)); ek[k] = 1e-3
    @test g[k] ≈ (objective(prob, θ0 .+ ek) - objective(prob, θ0 .- ek)) / 2e-3 rtol = 1e-3
    @test_throws ErrorException ModelParameters(m, :vs; smooth=0.5, basis=[ones(size(m))])
    # one misfit per shot (event weights): the weighted sum of the misfits of the shots
    J1 = misfit_gradient(m, obs1; kw..., sources=src)[1]
    Js, gs = misfit_gradient(m, [obs1, obs1]; kw..., sources=[src, src], misfit=[L2Misfit(weights=2.0), L2Misfit()])
    @test Js ≈ 3J1
    @test_throws "one per shot" misfit_gradient(m, [obs1, obs1]; kw..., sources=[src, src], misfit=[L2Misfit()])
end
