# Misfit gradients: adjoint-state method and automatic differentiation (Enzyme extension), against each other and against
# finite differences of whole simulations
using Enzyme, LinearAlgebra

perturbed(m, field, I, d) = CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, NamedTuple{(field,)}(((A = copy(m.fields[field]); A[I] += d; A),))))
observed(m, kw, mode) = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); kw...,
                                 stations=test_stations(mode), verbose=false)
misfit(m, obs, kw) = (r = simulate(m; kw..., stations=obs.stations, verbose=false); sum(abs2, r.v .- obs.v) / 2)

@testset "misfit gradient" begin
    m = test_model(:psv)
    kw = (; dt=DT, nt=150, sources=test_source(), T=Float64)
    obs = observed(m, kw, :psv)
    J, g = misfit_gradient(m, obs; kw...)
    Jad, gad = misfit_gradient(m, obs; kw..., method=:ad, checkpoint_steps=20)
    @test J ≈ misfit(m, obs, kw) ≈ Jad
    @test g isa CartData && size(g) == size(m) && keys(g.fields) == (:rho, :vp, :vs, :Qp, :Qs)
    for f in keys(g.fields)                                         # the adjoint is the exact gradient of the discrete scheme
        @test g.fields[f] ≈ gad.fields[f] rtol = 1e-10
    end
    I = CartesianIndex(110, 1, 75)
    for (field, h) in ((:vs, 1e-4), (:rho, 1e-1), (:Qp, 1e-2))
        fd = (misfit(perturbed(m, field, I, h), obs, kw) - misfit(perturbed(m, field, I, -h), obs, kw)) / 2h
        @test g.fields[field][I] ≈ fd rtol = 1e-4
    end
    @test all(iszero, g.fields.vs[:, 1, end-5:end])                           # air
    # waves in the PML: the adjoint of the PML is exact too (cell next to the bottom absorber, whose cells copy it)
    kwp = (; kw..., nt=500)
    obsp = observed(m, kwp, :psv)
    gp = misfit_gradient(m, obsp; kwp...)[2]
    for (field, h, Ip) in ((:vp, 1e-4, CartesianIndex(94, 1, 21)), (:rho, 1e-1, CartesianIndex(30, 1, 50)))
        fd = (misfit(perturbed(m, field, Ip, h), obsp, kwp) - misfit(perturbed(m, field, Ip, -h), obsp, kwp)) / 2h
        @test gp.fields[field][Ip] ≈ fd rtol = 1e-4
    end
    # Float32, other checkpoint intervals, record_every, Cerjan
    J32, g32 = misfit_gradient(m, obs; kw..., T=Float32, checkpoint_steps=7)
    @test J32 ≈ J rtol = 1e-3
    @test g32.fields.vs ≈ g.fields.vs rtol = 1e-2
    kw2 = (; kw..., record_every=3, absorber=:cerjan)
    obs2 = observed(m, kw2, :psv)
    fd = (misfit(perturbed(m, :vp, I, 1e-4), obs2, kw2) - misfit(perturbed(m, :vp, I, -1e-4), obs2, kw2)) / 2e-4
    @test misfit_gradient(m, obs2; kw2...)[2].fields.vp[I] ≈ fd rtol = 1e-4
    @test_throws ErrorException misfit_gradient(m, obs; kw..., nt=100)        # does not match the observed seismograms
    @test_throws ErrorException misfit_gradient(m, obs; kw..., method=:fd)
end

@testset "misfit gradient SH, 3D" begin
    for (mode, I) in ((:sh, CartesianIndex(110, 1, 75)), (:p3d, CartesianIndex(45, 30, 40)))
        m = test_model(mode)
        kw = (; dt=DT, nt=mode === :sh ? 150 : 100, sources=test_source(), T=Float64, wave=mode === :sh ? :sh : :psv, na=mode === :sh ? 20 : 8)
        obs = observed(m, kw, mode)
        J, g = misfit_gradient(m, obs; kw...)
        @test J ≈ misfit(m, obs, kw)
        for (field, h) in ((:vs, 1e-4), (:rho, 1e-1), (:Qs, 1e-2))
            fd = (misfit(perturbed(m, field, I, h), obs, kw) - misfit(perturbed(m, field, I, -h), obs, kw)) / 2h
            @test g.fields[field][I] ≈ fd rtol = 1e-4
        end
    end
end

@testset "source gradients, misfits, several shots" begin
    m = test_model(:psv)
    src = test_source()
    force = ForceSource(0.5, 0.0, -1.2; fx=1e13, fz=-2e13, t0=0.1, trise=0.3)
    kw = (; dt=DT, nt=150, T=Float64)
    stations = test_stations(:psv)
    obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); kw..., sources=[src, force], stations, verbose=false)
    misfit_of(srcs, mis=L2Misfit()) = (r = simulate(m; kw..., sources=srcs, stations, verbose=false); mis(r.v, obs.v, r.t)[1])
    fd(f, h, mis=L2Misfit()) = (misfit_of(f(h), mis) - misfit_of(f(-h), mis)) / 2h
    J, g, sg = misfit_gradient(m, obs; kw..., sources=[src, force])
    @test length(sg) == 2 && keys(sg[1]) == (:M, :M0, :x, :y, :z, :t0, :trise, :stf, :t) && keys(sg[2])[1] == :F
    mt(; M=src.M, M0=src.M0, t0=src.t0) = MomentTensorSource(src.x, src.y, src.z, M0, M; t0, trise=src.trise)
    @test sg[1].M[1] ≈ fd(h -> [mt(M=(src.M[1] + h / src.M0, src.M[2:6]...)), force], 1e-4 * src.M0) rtol = 1e-5
    @test sg[1].M[5] ≈ fd(h -> [mt(M=(src.M[1:4]..., src.M[5] + h / src.M0, src.M[6])), force], 1e-4 * src.M0) rtol = 1e-5
    @test sg[1].M0 ≈ fd(h -> [mt(M0=src.M0 + h), force], 1e-4 * src.M0) rtol = 1e-5
    @test sg[1].t0 ≈ fd(h -> [mt(t0=src.t0 + h), force], 1e-6) rtol = 1e-4
    @test sg[2].F[3] ≈ fd(h -> [src, ForceSource(force.x, force.y, force.z; fx=force.F[1], fz=force.F[3] + h, t0=force.t0, trise=force.trise)], 1e9) rtol = 1e-5
    @test sg[1].x ≈ fd(h -> [MomentTensorSource(src.x + h, src.y, src.z, src.M0, src.M; t0=src.t0, trise=src.trise), force], 0.05) rtol = 0.05
    # misfits: weights and windows, travel times
    I = CartesianIndex(110, 1, 75)
    for mis in (L2Misfit(weights=[1.0, 2.0, 0.5], window=(0.1, 0.5)), TravelTimeMisfit(window=(0.1, 0.55)))
        Jm, gm = misfit_gradient(m, obs; kw..., sources=[src, force], misfit=mis)
        @test Jm ≈ misfit_of([src, force], mis)
        mm(d) = (r = simulate(perturbed(m, :vs, I, d); kw..., sources=[src, force], stations, verbose=false); mis(r.v, obs.v, r.t)[1])
        @test gm.fields.vs[I] ≈ (mm(1e-4) - mm(-1e-4)) / 2e-4 rtol = 1e-4
    end
    # several shots: sums of the single shots
    obs1 = simulate(m; kw..., sources=force, stations, verbose=false)
    Js, gs, sgs = misfit_gradient(m, [obs, obs1]; kw..., sources=[[src, force], force])
    J1, g1 = misfit_gradient(m, obs1; kw..., sources=force)
    @test Js ≈ J + J1 && gs.fields.vp ≈ g.fields.vp .+ g1.fields.vp && length(sgs) == 2
    @test_throws ErrorException misfit_gradient(m, [obs, obs1]; kw..., sources=[src])
end

@testset "source time functions, covariance, muting" begin
    m = test_model(:psv)
    src = test_source()
    kw = (; dt=DT, nt=150, T=Float64, sources=src, stations=test_stations(:psv), verbose=false)
    # source time functions: any function, sampled values, Gaussians
    r0 = simulate(m; kw..., stf=:kupper)
    @test simulate(m; kw..., stf=(t, t0, tr) -> moment_rate(:kupper, t, t0, tr)).v == r0.v
    ts = range(0, src.trise, length=2001)
    sampled = SampledSTF(ts, [moment_rate(:kupper, t, 0.0, src.trise) for t in ts])
    @test simulate(m; kw..., stf=sampled).v ≈ r0.v rtol = 1e-4
    integral(stf) = sum(moment_rate(stf, t, 0.0, 0.3) for t in range(-1, 3, step=1e-4)) * 1e-4
    @test integral(:gaussian) ≈ 1 rtol = 1e-6
    @test abs(integral(:gaussian_derivative)) < 1e-8
    @test_throws ErrorException SampledSTF([0.0, 1.0], [1.0])
    # the gradient with respect to the source time function, chained to the samples of a SampledSTF
    tq = collect(range(0, 0.4, length=21))
    vals = [moment_rate(:kupper, t, 0.0, 0.3) for t in tq]
    obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); kw..., stf=SampledSTF(tq, vals))
    J, g, sg = misfit_gradient(m, obs; dt=DT, nt=150, T=Float64, sources=src, stf=SampledSTF(tq, vals .* 1.1))
    k = 8
    weight(t) = (τ = t - src.t0; i = searchsortedlast(tq, τ); i == k ? 1 - (τ - tq[k]) / (tq[k+1] - tq[k]) : i == k - 1 ? (τ - tq[k-1]) / (tq[k] - tq[k-1]) : 0.0)
    dJ = sum(sg[1].stf[n] * weight(sg[1].t[n]) for n in eachindex(sg[1].t))
    misfit(v) = (r = simulate(m; kw..., stations=obs.stations, stf=SampledSTF(tq, v)); sum(abs2, r.v .- obs.v) / 2)
    h = 1e-3
    @test dJ ≈ (misfit(vals .* 1.1 .+ h .* (1:21 .== k)) - misfit(vals .* 1.1 .- h .* (1:21 .== k))) / 2h rtol = 1e-5
    # L2 misfit with an inverse covariance matrix: identity = plain L2; derivative by finite differences
    syn, o = randn(30, 2, 3), randn(30, 2, 3)
    t = range(0, 1, length=30)
    @test L2Misfit(invcov=Matrix(1.0I, 30, 30))(syn, o, t)[1] ≈ L2Misfit()(syn, o, t)[1]
    A = randn(30, 30); C = A * A' / 30 + I
    mis = L2Misfit(invcov=C, weights=[1.0, 2.0, 0.5], window=(0.2, 0.8))
    Jc, adj = mis(syn, o, t)
    e = zeros(size(syn)); e[12, 2, 3] = 1e-6
    @test adj[12, 2, 3] ≈ (mis(syn .+ e, o, t)[1] - mis(syn .- e, o, t)[1]) / 2e-6 rtol = 1e-6
    # muting: zero at the source and the stations, unchanged far from them
    J1, g1 = misfit_gradient(m, obs; dt=DT, nt=150, T=Float64, sources=src, stf=SampledSTF(tq, vals .* 1.1))
    _, g2 = misfit_gradient(m, obs; dt=DT, nt=150, T=Float64, sources=src, stf=SampledSTF(tq, vals .* 1.1), mute=(sources=1.0, stations=0.5))
    near(px, pz) = argmin((m.x.val .- px) .^ 2 .+ (m.z.val .- pz) .^ 2)
    @test abs(g2.fields.vs[near(src.x, src.z)]) < 0.05 * abs(g1.fields.vs[near(src.x, src.z)])   # cell ≈ 35 m from the source
    st = obs.stations[1]
    @test abs(g2.fields.vs[near(st.x, st.z)]) < 0.05 * abs(g1.fields.vs[near(st.x, st.z)]) + 1e-30
    far = near(src.x + 2.5, src.z - 1.5)
    @test g2.fields.vs[far] == g1.fields.vs[far]
    # illumination Σₜ|v|² of the source wavefield: largest at the source, not muted, the gradient unchanged; summed over shots
    kwi = (; dt=DT, nt=150, T=Float64, sources=src, stf=SampledSTF(tq, vals .* 1.1), mute=(sources=1.0, stations=0.5))
    _, g3 = misfit_gradient(m, obs; kwi..., illumination=true)
    L = g3.fields.illumination
    @test keys(g3.fields) == (:rho, :vp, :vs, :Qp, :Qs, :illumination) && g3.fields.vs == g2.fields.vs
    @test all(>=(0), L) && Tuple(argmax(L)) == Tuple(near(src.x, src.z)) && L[far] < 1e-3 * maximum(L)
    _, g4 = misfit_gradient(m, [obs, obs]; kwi..., sources=[src, src], illumination=true)
    @test g4.fields.illumination ≈ 2L
    @test_throws ErrorException misfit_gradient(m, obs; kwi..., illumination=true, method=:ad)
end

@testset "stretched grid in depth" begin
    # spacing in depth from 0.6 dx (top) to 1.4 dx (bottom): forward, adjoint and source gradients stay exact
    for mode in (:psv, :sh)
        m = test_model(mode; stretched=true)
        kw = (; dt=DT, nt=150, sources=test_source(), T=Float64, wave=mode)
        obs = observed(m, kw, mode)
        J, g = misfit_gradient(m, obs; kw...)
        Jad, gad = misfit_gradient(m, obs; kw..., method=:ad, checkpoint_steps=20)
        @test J ≈ misfit(m, obs, kw) ≈ Jad
        for f in (:rho, :vs, :Qs)
            @test g.fields[f] ≈ gad.fields[f] rtol = 1e-10
        end
    end
    m = test_model(:psv; stretched=true)
    kw = (; dt=DT, nt=500, sources=test_source(), T=Float64)            # in the PML (whose profile follows the depths)
    obs = observed(m, kw, :psv)
    I = CartesianIndex(94, 1, 21)
    fd = (misfit(perturbed(m, :vp, I, 1e-4), obs, kw) - misfit(perturbed(m, :vp, I, -1e-4), obs, kw)) / 2e-4
    @test misfit_gradient(m, obs; kw...)[2].fields.vp[I] ≈ fd rtol = 1e-5
    # source location between grid points
    src = MomentTensorSource(-0.37, 0.0, -1.03, 1e15, (0.3, -0.5, 0.2, 0.4, 0.7, -0.3); trise=0.3, t0=0.05)
    kws = (; dt=DT, nt=200, T=Float64, interpolation=:sinc)
    obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); kws..., sources=src,
                   stations=test_stations(:psv), verbose=false)
    sg = misfit_gradient(m, obs; kws..., sources=src)[3][1]
    mf(dz) = misfit(m, obs, (; kws..., sources=MomentTensorSource(src.x, src.y, src.z + dz, src.M0, src.M; t0=src.t0, trise=src.trise)))
    @test sg.z ≈ (mf(1e-4) - mf(-1e-4)) / 2e-4 rtol = 1e-5
    # spacing that changes too fast; plane waves need a uniform grid
    z = m.z.val[1, 1, :]
    zj = [z[1:50]; z[50] .+ cumsum(fill(3 * (z[51] - z[50]), 50))]
    Xj, Yj, Zj = xyz_grid(m.x.val[:, 1, 1], 0, zj)
    @test_throws ErrorException simulate(CartData(Xj, Yj, Zj, map(A -> A[:, :, [1:50; 50:99]], m.fields)); dt=DT, nt=2, sources=test_source(), verbose=false)
    @test_throws ErrorException simulate(m; dt=DT, nt=2, sources=PlaneWave(:p; depth=1.0, length=2.0), verbose=false)
end
