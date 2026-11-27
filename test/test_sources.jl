# Finite faults, OpenSWPC source files, a source time function per source, and the source windows on the device
# (source_window.jl: windows of time steps must give the results of the whole tables).
using SeismicWavePC, GeophysicalModelGenerator, Test
const SWs = SeismicWavePC

@testset "finite faults" begin
    # geometry: strike east, dipping 45° to the south; the subfault centres, moments and rupture times
    ff = finite_fault(; top=(1.0, 2.0, -1.0), strike=90, dip=45, length=8.0, width=4.0, nl=4, nw=2, slip=2.0, rake=90, mu=3e10,
                      hypocenter=(0.0, 2.0), vr=2.0)
    @test length(ff) == 8 && ff isa Vector{MomentTensorSource}
    @test sum(s -> s.M0, ff) ≈ 3e10 * 8e3 * 4e3 * 2.0                          # μ L W D
    s11 = ff[1]                                                                 # along strike -3 km, down dip 1 km
    @test all((s11.x, s11.y, s11.z) .≈ (1.0 - 3.0, 2.0 - sqrt(0.5), -1.0 - sqrt(0.5)))
    @test s11.t0 ≈ hypot(-3.0, 1.0 - 2.0) / 2.0
    d = DoubleCoupleSource(0.0, 0.0, 0.0; M0=1.0, strike=90, dip=45, rake=90)
    @test all(s -> s.M == d.M, ff)
    # one subfault is a double couple; slip, rake, rise times as functions or matrices; zero slip: no source
    one = finite_fault(; top=(0.0, 0.0, -2.0), strike=30, dip=60, length=2.0, width=2.0, nw=1, slip=1.0, rake=70, mu=3e10, t0=0.5, trise=0.8)
    dc = DoubleCoupleSource(one[1].x, one[1].y, one[1].z; M0=3e10 * 4e6, strike=30, dip=60, rake=70, t0=0.5, trise=0.8)
    @test one[1].M == dc.M && one[1].M0 ≈ dc.M0 && one[1].t0 == 0.5
    g = finite_fault(; top=(0.0, 0.0, -1.0), strike=0, dip=90, length=4.0, width=2.0, nl=2, nw=2, slip=[1.0 0.0; 2.0 3.0],
                     rake=(a, b) -> a > 0 ? 180.0 : 0.0, mu=1e10, t0=(a, b) -> b, trise=fill(0.5, 2, 2), stf=(a, b) -> :ricker)
    @test length(g) == 3 && all(s -> s.stf === :ricker && s.trise == 0.5, g)
    @test_throws "matrix needs size" finite_fault(; top=(0.0, 0.0, -1.0), strike=0, dip=90, length=4.0, width=2.0, nl=2, nw=2,
                                                   slip=ones(3, 3), rake=0, mu=1e10, t0=0.0)
    @test_throws "mu" finite_fault(; top=(0.0, 0.0, -1.0), strike=0, dip=90, length=1.0, width=1.0, nw=1, slip=1.0, rake=0, t0=0.0)
    # 2D: a line in the x–z plane, moments per unit length; μ from a model (ρ vs²)
    X2, Y2, Z2 = xyz_grid(-5:0.1:5, 0, -5:0.1:0)
    o = ones(size(X2))
    m2 = CartData(X2, Y2, Z2, (; rho=2500 .* o, vp=5.0 .* o, vs=3.0 .* o))
    l2 = finite_fault(; top=(0.0, 0.0, -1.0), strike=0, dip=30, width=2.0, nw=4, slip=1.0, rake=90, model=m2, vr=2.5)
    @test length(l2) == 4 && all(s -> s.y == 0.0, l2) && l2[1].M0 ≈ 2500 * 3000.0^2 * 500.0
    @test_throws "strikes along y" finite_fault(; top=(0.0, 0.0, -1.0), strike=45, dip=30, width=2.0, nw=4, slip=1.0, rake=90, model=m2, vr=2.5)
end

@testset "OpenSWPC source files" begin
    dc = DoubleCoupleSource(1.0, 2.0, -3.0; M0=1e15, strike=30, dip=60, rake=70)
    m = dc.M                                                    # (x east, y north, z up)
    mo = (m[2], m[1], m[3], -m[5], -m[4], m[6])                 # the same tensor in OpenSWPC's frame (x north, y east, z down)
    dir = mktempdir()
    file(name, lines) = (f = joinpath(dir, name); write(f, "# comment\n\n" * join(lines, "\n") * "\n"); f)
    a = read_sources(file("dc", ["2.0 1.0 3.0 0.5 1.2 1e15 30 60 70"]); format="xym0dc")[1]
    b = read_sources(file("ij", ["2.0 1.0 3.0 0.5 1.2 1e15 " * join(mo, " ")]); format="xym0ij")[1]
    @test (a.x, a.y, a.z, a.t0, a.trise, a.M0) == (1.0, 2.0, -3.0, 0.5, 1.2, 1e15) && a.M == dc.M
    @test all(b.M .≈ dc.M) && b.M0 == 1e15
    @test read_sources(file("mw", ["2.0 1.0 3.0 0 1 4.0 30 60 70"]); format="xymwdc")[1].M0 ≈ 10^(1.5 * 4.0 + 9.05)
    @test read_sources(file("ds", ["2.0 1.0 3.0 0 1 2.0 1e6 30 60 70"]); format="xydsdc", mu=3e10)[1].M0 ≈ 3e10 * 2.0 * 1e6
    # psmeca (GMT, Global CMT): a strike-slip fault striking north, M0 = 10²⁴ dyn cm = 10¹⁷ N m
    p = read_sources(file("meca", ["0.0 0.0 10.0 0 0 0 0 0 -1 24"]); format="psmeca", proj=ProjectionPoint(Lon=0.0, Lat=0.0))[1]
    @test p.M0 ≈ 1e17 && all(isapprox.(p.M, DoubleCoupleSource(0.0, 0.0, 0.0; M0=1.0, strike=0, dip=90, rake=0).M; atol=1e-12))
    # body forces (north, east, down) and geographic positions
    f = read_sources(file("bf", ["2.0 1.0 3.0 0.1 0.4 1e12 2e12 3e12"]); format="xy")[1]
    @test f isa ForceSource && (f.x, f.y, f.z) == (1.0, 2.0, -3.0) && f.F == (2e12, 1e12, -3e12)
    proj = ProjectionPoint(Lon=10.0, Lat=45.0)
    ll = read_sources(file("ll", ["10.1 45.05 4.0 0 1 1e15 30 60 70"]); format="llm0dc", proj)[1]
    @test all((ll.x, ll.y, ll.z) .≈ SWs.project(10.1, 45.05, -4.0, proj))
    @test_throws "unknown format" read_sources(file("x", ["1 2 3"]); format="xyz")
    @test_throws "needs the `proj`" read_sources(file("y", ["1 2 3 0 1 1e15 30 60 70"]); format="llm0dc")
    @test_throws "needs 9" read_sources(file("z", ["1 2 3 0 1"]); format="xym0dc")
end

@testset "source time function per source; windows of time steps" begin
    isdefined(@__MODULE__, :test_model) || include("setup.jl")
    rel(a, b) = maximum(abs, a .- b) / maximum(abs, b)
    m = test_model(:psv)
    kw = (; dt=DT, nt=150, T=Float64, stations=test_stations(:psv), verbose=false)
    src, frc = test_source(), ForceSource(0.3, 0.0, -0.6; fx=1e12, fz=1e12, trise=0.3)
    with(s; kws...) = s isa ForceSource ? ForceSource(s.x, s.y, s.z; fx=s.F[1], fy=s.F[2], fz=s.F[3], t0=s.t0, trise=s.trise, kws...) :
                      MomentTensorSource(s.x, s.y, s.z, s.M0, s.M; t0=s.t0, trise=s.trise, kws...)
    @test simulate(m; kw..., sources=[with(src; stf=:ricker), with(frc; stf=:ricker)]).v == simulate(m; kw..., sources=[src, frc], stf=:ricker).v
    # superposition: a finite fault is the sum of its subfaults (each with its own start time)
    ff = finite_fault(; top=(-0.5, 0.0, -0.6), strike=0, dip=40, width=0.6, nw=3, slip=0.01, rake=90, mu=1e10, vr=1.5, trise=0.2, dims=2)
    sum_ = sum(simulate(m; kw..., sources=[s]).v for s in ff)
    @test rel(simulate(m; kw..., sources=ff).v, sum_) < 1e-12
    # small windows (a block every few steps) give the results of the whole tables: forward and gradients, both solvers
    small(f) = (SWs.SOURCE_BUDGET[] = 30; try f() finally SWs.SOURCE_BUDGET[] = 1 << 24 end)
    for interpolation in (:cell, :sinc)
        k = (; kw..., sources=[src, frc], interpolation)
        @test simulate(m; k...).v == small(() -> simulate(m; k...)).v
        obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); k...)
        kg = Base.structdiff(k, NamedTuple{(:stations, :verbose)})
        (J1, g1, s1), (J2, g2, s2) = misfit_gradient(m, obs; kg...), small(() -> misfit_gradient(m, obs; kg...))
        @test J1 == J2 && g1.fields.vs == g2.fields.vs
        @test all(collect(s2[1].M) .≈ collect(s1[1].M)) && s2[1].t0 ≈ s1[1].t0 && s2[1].stf ≈ s1[1].stf && all(collect(s2[2].F) .≈ collect(s1[2].F))
    end
    mesh = curvilinear_mesh(range(-8, 8, length=61), [x -> 0.6exp(-x^2 / 8), x -> -4.0]; dz=0.25, bottom=-8)
    Z = mesh.z.val
    base = (; rho=ifelse.(Z .< -4, 2900.0, 2600.0), vp=ifelse.(Z .< -4, 6.5, 5.5), vs=ifelse.(Z .< -4, 3.75, 3.2), Qp=100 .+ 0Z, Qs=50 .+ 0Z)
    csrc = [DoubleCoupleSource(-1.0, 0.0, -5.0; Mw=4.0, strike=30, dip=60, rake=70, trise=1.0, stf=:gaussian),
            ForceSource(2.0, 0.0, -3.0; fx=1e14, fz=1e14, trise=0.8)]
    sts = [Station("s$x", Float64(x), 0.0, 0.0; surface=true) for x in -6:3:6]
    r0 = simulate(addfield(mesh, base); tmax=0.05, sources=csrc, T=Float64, verbose=false)
    ck = (; dt=r0.t[2] - r0.t[1], tmax=2.5, sources=csrc, T=Float64, absorber=:pml, na=8)
    @test simulate(addfield(mesh, base); ck..., stations=sts, verbose=false).v == small(() -> simulate(addfield(mesh, base); ck..., stations=sts, verbose=false)).v
    obs = simulate(addfield(mesh, merge(base, (; vs=base.vs .* 0.98))); ck..., stations=sts, verbose=false)
    (J1, g1, s1), (J2, g2, s2) = misfit_gradient(addfield(mesh, base), obs; ck...), small(() -> misfit_gradient(addfield(mesh, base), obs; ck...))
    @test J1 == J2 && g1.fields.vs == g2.fields.vs && s1[1].stf == s2[1].stf && s1[2].F == s2[2].F && s1[1].t0 == s2[1].t0
end

@testset "SRF, CMTSOLUTION, explosions, step and delta" begin
    e = ExplosionSource(1.0, 2.0, -3.0; M0=1e14, t0=0.2)
    @test e.M == (1.0, 1.0, 1.0, 0.0, 0.0, 0.0) && e.M0 == 1e14 && e.t0 == 0.2
    @test ExplosionSource(0.0, 0.0, -1.0; Mw=3.0).M0 ≈ 10^(1.5 * 3 + 9.05)
    @test_throws "M0 or Mw" ExplosionSource(0.0, 0.0, -1.0)
    @test moment_rate(:step, 1.0, 0.5, 1.0) == 1.0 && moment_rate(:step, 0.4, 0.5, 1.0) == 0.0
    @test_throws "impulse" moment_rate(:delta, 0.0, 0.0, 1.0)
    # :delta is one time step: K deltas of weight 1/K at the steps of a boxcar give the boxcar (superposition, exact)
    isdefined(@__MODULE__, :test_model) || include("setup.jl")
    m = test_model(:psv)
    kw = (; dt=DT, nt=120, T=Float64, stations=test_stations(:psv), verbose=false)
    a, K = 20, 6                                              # (forces are injected at t = it dt)
    box = simulate(m; kw..., sources=ForceSource(0.3, 0.0, -0.6; fz=1e12, t0=(a - 0.5) * DT, trise=K * DT, stf=:boxcar))
    deltas = simulate(m; kw..., sources=[ForceSource(0.3, 0.0, -0.6; fz=1e12 / K, t0=(a + k) * DT, stf=:delta) for k in 0:K-1])
    @test maximum(abs, box.v .- deltas.v) < 1e-10 * maximum(abs, box.v)
    # SRF (version 2: S velocity [cm/s] and density [g/cm³] per point; slip [cm], area [cm²], slip velocity [cm/s])
    dir = mktempdir()
    srf = joinpath(dir, "fault.srf")
    write(srf, """2.0
    # a comment
    PLANE 1
      10.00 45.00 2 1 2.0 1.0
      30 60 1.0 0.0 0.5
    POINTS 2
    10.000 45.000 2.0 30 60 1.0e10 0.5 0.1 300000 2.7
    70 100.0 4 0.0 0 0.0 0
    0.0 500.0 500.0 0.0
    10.010 45.000 2.0 30 60 1.0e10 0.9 0.1 300000 2.7
    70 50.0 2 20.0 2 0.0 0
    250.0 250.0
    100.0 100.0
    """)
    proj = ProjectionPoint(Lon=10.0, Lat=45.0)
    f = read_srf(srf; proj)
    μ, A = 2700 * 3000.0^2, 1e6                                    # (1e10 cm² = 1 km²)
    @test length(f) == 3 && f[1].M0 ≈ μ * A * 1.0 && f[2].M0 ≈ μ * A * 0.5 && f[3].M0 ≈ μ * A * 0.2
    @test f[1].t0 == 0.5 && f[2].t0 == 0.9 && all((f[1].x, f[1].y, f[1].z) .≈ SWs.project(10.0, 45.0, -2.0, proj))
    @test f[1].M == DoubleCoupleSource(0.0, 0.0, 0.0; M0=1.0, strike=30, dip=60, rake=70).M
    @test f[3].M == DoubleCoupleSource(0.0, 0.0, 0.0; M0=1.0, strike=30, dip=60, rake=160).M        # (u2: rake + 90°)
    @test sum(f[1].stf.values) * 0.1 ≈ 1                                                          # (slip velocity / slip)
    write(srf, "1.0\nPOINTS 1\n10.0 45.0 2.0 30 60 1e10 0.0 0.1\n70 100.0 2 0.0 0 0.0 0\n1000.0 1000.0\n")
    @test_throws "give `mu` or the `model`" read_srf(srf; proj)
    @test read_srf(srf; proj, mu=3e10)[1].M0 ≈ 3e10 * 1e6 * 1.0
    # CMTSOLUTION: a strike-slip fault striking north (Mtp = -M0), 10²⁵ dyn cm = 10¹⁸ N m
    cmt = joinpath(dir, "CMTSOLUTION")
    write(cmt, """PDE 2020 1 1 0 0 0.0 45.0 10.0 10.0 6.0 6.0 TEST
    event name:     TEST
    time shift:      2.0000
    half duration:   1.5000
    latitude:       45.0000
    longitude:      10.0000
    depth:          10.0000
    Mrr:       0.000000e+00
    Mtt:       0.000000e+00
    Mpp:       0.000000e+00
    Mrt:       0.000000e+00
    Mrp:       0.000000e+00
    Mtp:      -1.000000e+25
    """)
    c = read_cmtsolution(cmt; proj)[1]
    @test c.M0 ≈ 1e18 && c.t0 == 2.0 && all(isapprox.(c.M, DoubleCoupleSource(0.0, 0.0, 0.0; M0=1.0, strike=0, dip=90, rake=0).M; atol=1e-12))
    @test c.z ≈ -10.0 atol = 1e-6
    @test sum(c.stf(t, c.t0, c.trise) for t in -6:0.001:10) * 0.001 ≈ 1 rtol = 1e-6                   # (unit moment rate)
end
