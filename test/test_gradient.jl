# Misfit gradients: adjoint-state method and automatic differentiation (Enzyme extension), against each other and against
# finite differences of whole simulations
using Enzyme

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
