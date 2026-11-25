# VTI anisotropy (Thomsen parameters): the isotropic limit equals the isotropic kernels, homogeneous VTI gives the
# horizontal and vertical velocities, Born modelling is the derivative (also of epsilon, delta, gamma) and the transpose
# of the adjoint gradient (dot test, with attenuation, topography and PML).
using LinearAlgebra, Random

with(m, f) = CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, f))
thomsen(m, ε, δ, γ) = with(m, (; epsilon=fill(ε, size(m)), delta=fill(δ, size(m)), gamma=fill(γ, size(m))))

@testset "VTI anisotropy" begin
    # isotropic limit: ε = δ = γ = 0 through the VTI kernels
    for (mode, wave) in ((:psv, :psv), (:sh, :sh), (:p3d, :psv))
        m = test_model(mode; step=true)
        kw = (; dt=DT, nt=200, sources=test_source(), stations=test_stations(mode), wave, T=Float64, verbose=false)
        @test simulate(thomsen(m, 0, 0, 0); kw...).v ≈ simulate(m; kw...).v rtol = 1e-12
    end

    # homogeneous VTI: the ratio of the arrival times along x and z is that of the vertical and horizontal velocities
    x, z = range(-6, 6, length=241), range(-12, 0, length=241)
    X, Y, Z = xyz_grid(x, 0, z)
    ε, γ = 0.2, 0.15
    m = CartData(X, Y, Z, (; rho=fill(2500.0, size(X)), vp=fill(3.0, size(X)), vs=fill(1.7, size(X)),
                           epsilon=fill(ε, size(X)), delta=fill(0.1, size(X)), gamma=fill(γ, size(X))))
    st = [Station("x", 4.0, 0.0, -6.0), Station("z", 0.0, 0.0, -2.0)]
    kw = (; dt=0.004, nt=700, stations=st, T=Float64, verbose=false)
    pick(r, c, n) = (v = r.v[:, c, n]; r.t[findfirst(abs.(v) .> 0.02maximum(abs, v))])
    p = simulate(m; kw..., sources=ForceSource(0.0, 0.0, -6.0; fx=1e12, fz=1e12, trise=0.15))
    @test pick(p, 1, 1) / pick(p, 2, 2) ≈ 1 / sqrt(1 + 2ε) rtol = 0.01
    s = simulate(m; kw..., wave=:sh, sources=ForceSource(0.0, 0.0, -6.0; fy=1e12, trise=0.15))
    @test pick(s, 1, 1) / pick(s, 1, 2) ≈ 1 / sqrt(1 + 2γ) rtol = 0.01
    @test_throws "isotropic" simulate(m; kw..., sources=PlaneWave(:p; depth=8.0, length=1.0))

    # Born modelling (all fields) against central differences, and the dot test with the adjoint gradient
    for (mode, wave, nt) in ((:psv, :psv, 300), (:sh, :sh, 300), (:p3d, :psv, 120))
        m = thomsen(test_model(mode; step=true), 0.15, 0.05, 0.1)
        X, Y, Z = m.x.val, m.y.val, m.z.val
        b(x0, z0, w) = @. exp(-((X - x0)^2 + Y^2 + (Z - z0)^2) / w^2) * (m.fields.vp > 0)
        δm = CartData(X, Y, Z, (; rho=100 .* b(0.3, -1.0, 0.4), vp=0.2 .* b(-0.5, -1.6, 0.5), vs=0.1 .* b(0.8, -0.8, 0.4),
                                epsilon=0.1 .* b(0.0, -1.2, 0.5), delta=0.05 .* b(0.4, -1.4, 0.5), gamma=0.1 .* b(-0.3, -1.0, 0.5),
                                Qp=5 .* b(0.0, -1.2, 0.5), Qs=4 .* b(0.0, -1.2, 0.5)))
        kw = (; dt=DT, nt, sources=test_source(), stations=test_stations(mode), wave, T=Float64, verbose=false)
        d0, δd = born(m, δm; kw...)
        shifted(h) = CartData(X, Y, Z, map(k -> m.fields[k] .+ h .* δm.fields[k], NamedTuple{keys(m.fields)}(keys(m.fields))))
        err(h) = norm(δd.v .- (simulate(shifted(h); kw...).v .- simulate(shifted(-h); kw...).v) ./ 2h) / norm(δd.v)
        @test err(1e-3) < 1e-6
        r = randn(MersenneTwister(1), size(d0.v))
        g = misfit_gradient(m, d0; Base.structdiff(kw, NamedTuple{(:stations, :verbose)})..., misfit=(syn, obs, t) -> (sum(r .* syn), r))[2]
        @test keys(g.fields) == (:rho, :vp, :vs, :epsilon, :delta, :gamma, :Qp, :Qs)
        @test dot(δd.v, r) ≈ sum(dot(g.fields[k], δm.fields[k]) for k in keys(δm.fields)) rtol = 1e-10
    end
    iso = test_model(:psv)
    @test_throws "give the model" born(iso, CartData(iso.x.val, iso.y.val, iso.z.val, (; epsilon=zero(iso.fields.vp))); dt=DT, nt=10,
                                       sources=test_source(), stations=test_stations(:psv))
end
