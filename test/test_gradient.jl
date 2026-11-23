# Misfit gradients (Enzyme extension) against finite differences of whole simulations
using Enzyme

@testset "misfit gradient (Enzyme)" begin
    m = test_model(:psv)
    kw = (; dt=DT, nt=150, sources=test_source(), T=Float64)
    obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); kw...,
                   stations=test_stations(:psv), verbose=false)
    J, g = misfit_gradient(m, obs; kw..., checkpoint_steps=20)
    misfit(mm) = (r = simulate(mm; kw..., stations=obs.stations, verbose=false); sum(abs2, r.v .- obs.v) / 2)
    @test J ≈ misfit(m)
    @test g isa CartData && size(g) == size(m) && keys(g.fields) == (:rho, :vp, :vs, :Qp, :Qs)
    I = CartesianIndex(110, 1, 75)
    for (field, h) in ((:vs, 1e-4), (:rho, 1e-1), (:Qp, 1e-2))
        pert(d) = CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, NamedTuple{(field,)}(((A = copy(m.fields[field]); A[I] += d; A),))))
        fd = (misfit(pert(h)) - misfit(pert(-h))) / 2h
        @test g.fields[field][I] ≈ fd rtol = 1e-4
    end
    @test all(iszero, g.fields.vs[:, 1, end-5:end])                           # air
    @test_throws ErrorException misfit_gradient(m, obs; kw..., nproc=2)
    @test_throws ErrorException misfit_gradient(m, obs; kw..., nt=100)        # does not match the observed seismograms
end
