# Sources and stations between grid points (interpolation = :sinc): accuracy, gradients (incl. exact source locations)
@testset "sources and stations between grid points" begin
    # homogeneous medium: shifting source and stations together by half a cell does not change the seismograms
    m0 = test_model(:psv)
    f = m0.fields
    hom = CartData(m0.x.val, m0.y.val, m0.z.val, merge(f, (; rho=ifelse.(f.vs .> 0, 2500.0, f.rho), vp=ifelse.(f.vs .> 0, 4.0, f.vp),
                                                          vs=ifelse.(f.vs .> 0, 2.3, f.vs))))
    x0, z0 = m0.x.val[40, 1, 1], m0.z.val[1, 1, 50]
    src(d) = MomentTensorSource(x0 + d, 0.0, z0, 1e15, (0.3, -0.5, 0.2, 0.4, 0.7, -0.3); trise=0.3)
    sts(d) = [Station("A", x0 + 0.8 + d, 0.0, z0 - 0.4), Station("B", x0 + 1.3 + d, 0.0, z0 + 0.5)]
    run(d) = simulate(hom; dt=DT, nt=300, sources=src(d), stations=sts(d), interpolation=:sinc, verbose=false)
    r1, r2 = run(0.0), run(0.025)
    @test sqrt(sum(abs2, r1.v .- r2.v) / sum(abs2, r1.v)) < 2e-3
    @test sqrt(sum(abs2, r1.stress .- r2.stress) / sum(abs2, r1.stress)) < 2e-3
    @test_throws ErrorException simulate(hom; dt=DT, nt=10, sources=src(0.0), stations=[Station("air", 0.0, 0.0, 0.4)],
                                         interpolation=:sinc, verbose=false)
    @test_throws ErrorException simulate(hom; dt=DT, nt=10, sources=src(0.0), interpolation=:linear, verbose=false)
    # explosion and station in water: no shear stress points, the shear stress is zero
    water = CartData(m0.x.val, m0.y.val, m0.z.val, merge(f, (; vs=zero(f.vs))))
    rw = simulate(water; dt=DT, nt=100, sources=MomentTensorSource(x0, 0.0, z0, 1e15, (1, 0, 1, 0, 0, 0); trise=0.3),
                  stations=[Station("W", x0 + 0.5, 0.0, z0)], interpolation=:sinc, verbose=false)
    @test maximum(abs, rw.v) > 0 && all(iszero, rw.stress[:, 3, :])
    # station on a single cell above the surface (SH): no Sxy point at its level, and the levels below have weight 0
    msh = test_model(:sh)
    fb = map(copy, msh.fields)
    i = 120
    k = findlast(>(0), fb.vs[i, 1, :]) + 1
    foreach(q -> (q[i, 1, k] = q[i, 1, k-1]), (fb.rho, fb.vp, fb.vs, fb.Qp, fb.Qs))
    rb = simulate(CartData(msh.x.val, msh.y.val, msh.z.val, fb); dt=DT, nt=100, sources=test_source(), wave=:sh,
                  stations=[Station("bump", msh.x.val[i, 1, 1] + 0.01, 0.0, 0.0; surface=true)], interpolation=:sinc, verbose=false)
    @test maximum(abs, rb.v) > 0 && maximum(abs, rb.stress[:, 1, :]) > 0 && all(iszero, rb.stress[:, 2, :])

    # gradients: model, source location (exact with :sinc), cells below a station on the free surface
    m = test_model(:psv)
    mt = MomentTensorSource(-0.37, 0.0, -1.03, 1e15, (0.3, -0.5, 0.2, 0.4, 0.7, -0.3); trise=0.3, t0=0.05)
    force = ForceSource(0.42, 0.0, -1.21; fx=1e13, fz=-2e13, t0=0.1, trise=0.3)
    stations = [Station("A", -1.13, 0.0, -0.33), Station("B", 1.07, 0.0, 0.0; surface=true), Station("C", 0.61, 0.0, -0.71)]
    kw = (; dt=DT, nt=220, T=Float64, interpolation=:sinc)
    obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); kw..., sources=[mt, force], stations, verbose=false)
    J, g, sg = misfit_gradient(m, obs; kw..., sources=[mt, force])
    misfit(srcs, mm=m) = (r = simulate(mm; kw..., sources=srcs, stations=obs.stations, verbose=false); sum(abs2, r.v .- obs.v) / 2)
    @test J ≈ misfit([mt, force])
    fd(f, h) = (misfit(f(h)) - misfit(f(-h))) / 2h
    moved(s, dx, dz) = s isa MomentTensorSource ? MomentTensorSource(s.x + dx, s.y, s.z + dz, s.M0, s.M; t0=s.t0, trise=s.trise) :
                       ForceSource(s.x + dx, s.y, s.z + dz; fx=s.F[1], fy=s.F[2], fz=s.F[3], t0=s.t0, trise=s.trise)
    @test sg[1].x ≈ fd(h -> [moved(mt, h, 0.0), force], 1e-4) rtol = 1e-4
    @test sg[1].z ≈ fd(h -> [moved(mt, 0.0, h), force], 1e-4) rtol = 1e-4
    @test sg[2].x ≈ fd(h -> [mt, moved(force, h, 0.0)], 1e-4) rtol = 1e-4
    i = argmin(abs.(m.x.val[:, 1, 1] .- 1.07))
    for I in (CartesianIndex(110, 1, 75), CartesianIndex(i, 1, findlast(>(0), m.fields.vs[i, 1, :])))
        pert(d) = CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=(A = copy(m.fields.vs); A[I] += d; A))))
        @test g.fields.vs[I] ≈ (misfit([mt, force], pert(1e-4)) - misfit([mt, force], pert(-1e-4))) / 2e-4 rtol = 1e-5
    end
end
