using Test

@testset "source time functions integrate to 1" begin
    dt = 1e-4
    for stf in (:kupper, :boxcar, :triangle, :herrmann, :cosine, :texp)
        t = 0:dt:(stf === :texp ? 20.0 : 3.0)
        @test isapprox(sum(moment_rate.(stf, t, 0.5, 1.0)) * dt, 1; atol=2e-3)
        @test moment_rate(stf, 0.4, 0.5, 1.0) == 0
    end
    @test_throws ErrorException moment_rate(:nonsense, 1.0, 0.0, 1.0)
    t = 0:dt:4.0                                                           # Ricker: integral 0, peak 1/tr at t0 + tr
    @test abs(sum(moment_rate.(:ricker, t, 0.5, 0.2)) * dt) < 1e-6 && moment_rate(:ricker, 0.7, 0.5, 0.2) ≈ 5
end

@testset "double couple" begin
    s = DoubleCoupleSource(1.0, 2.0, -3.0; Mw=4.0, strike=30, dip=60, rake=70)
    xx, yy, zz, yz, xz, xy = s.M
    @test abs(xx + yy + zz) < 1e-12                                       # no volume change
    @test xx^2 + yy^2 + zz^2 + 2(yz^2 + xz^2 + xy^2) ≈ 2                    # unit double couple
    @test s.M0 ≈ 10^(1.5 * 4 + 9.05)
    # vertical strike-slip fault striking north: pure xy couple (x east, y north)
    @test collect(DoubleCoupleSource(0, 0, 0; Mw=1, strike=0, dip=90, rake=0).M) ≈ [0, 0, 0, 0, 0, 1] atol = 1e-12
end

@testset "geographic coordinates (GMG projection)" begin
    proj = ProjectionPoint(Lon=14.0, Lat=40.0)
    st = Station("P", 14.0, 40.0, -1.0, proj)
    @test abs(st.x) < 1e-6 && abs(st.y) < 1e-6 && st.z ≈ -1.0
    st = Station("Q", 14.1, 40.0, 0.0, proj; surface=true)
    @test 8 < st.x < 9 && abs(st.y) < 0.1 && st.surface
    @test DoubleCoupleSource(14.0, 40.1, -2.0, proj; Mw=3, strike=0, dip=45, rake=90).y ≈ 11.1 atol = 0.1
end

@testset "REPL output and errors" begin
    model = test_model(:psv)
    s = test_source()
    @test occursin("MomentTensorSource", sprint(show, s))
    @test occursin("Station(\"A\"", sprint(show, test_stations(:psv)[1]))
    r = simulate(model; dt=DT, nt=50, sources=s, stations=test_stations(:psv), verbose=false)
    txt = sprint(show, MIME("text/plain"), r)
    @test occursin("3 station(s), 50 samples", txt) && occursin("final wavefield", txt)
    @test size(r["B"]) == (50, 2)
    @test_throws ErrorException r["nope"]
    @test r.wavefield isa CartData && size(r.wavefield) == size(model)
    @test redirect_stdout(() -> simulate(model; dt=DT, nt=20, sources=s), devnull) isa Seismograms   # verbose output
    @test_throws ErrorException simulate(model; dt=0.1, nt=10, sources=s, verbose=false)               # unstable
    outside = DoubleCoupleSource(50.0, 0, -1; Mw=1, strike=0, dip=45, rake=0)
    in_air = DoubleCoupleSource(0.0, 0, 0.3; Mw=1, strike=0, dip=45, rake=0)
    @test_throws ErrorException simulate(model; dt=DT, nt=10, sources=[outside, in_air], verbose=false)
    r2 = @test_logs (:warn, r"2 of 3 sources") simulate(model; dt=DT, nt=50, sources=[outside, s, in_air], stations=test_stations(:psv), verbose=false)
    @test r2.v == r.v                                                                                  # ignored, as in OpenSWPC
    @test_throws ErrorException simulate(model; dt=DT, nt=10, sources=s, wave=:love, verbose=false)
    @test_throws ErrorException simulate(model; dt=DT, nt=10, sources=s, nproc=3, verbose=false)       # 200 cells / 3
end

@testset "forces and moment tensors together" begin
    s = test_source()
    f = ForceSource(0.3, 0.0, -1.5; fx=2e13, fz=-1e13, trise=0.3)
    kw = (; dt=DT, nt=300, stations=test_stations(:psv), verbose=false)
    a, b, ab = simulate(test_model(:psv); sources=s, kw...), simulate(test_model(:psv); sources=f, kw...),
               simulate(test_model(:psv); sources=[s, f], kw...)
    @test maximum(abs, ab.v .- a.v .- b.v) < 1e-5 * maximum(abs, ab.v)       # superposition
    @test_throws ErrorException simulate(test_model(:psv); sources=[1.0], kw...)
    @test sprint(show, f) == "ForceSource at (0.3, 0, -1.5) km, F = (2.000e+13, 0.000e+00, -1.000e+13) N, t0 = 0 s, trise = 0.3 s"
end

@testset "optional packages" begin
    isdefined(Main, :NCDatasets) || @test_throws "NCDatasets" read_netcdf("model.nc")
    isdefined(Main, :Enzyme) || @test_throws "Enzyme" SeismicWavePC.ad_driver(nothing, nothing, nothing)   # misfit_gradient(...; method=:ad)
end

@testset "density units" begin
    m = test_model(:sh)
    kw = (; dt=DT, nt=50, sources=test_source(), wave=:sh, verbose=false)
    g = CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; rho=m.fields.rho ./ 1000)))
    @test simulate(g; density_unit=:g_cm3, kw...).v == simulate(m; kw...).v
    @test_throws ErrorException simulate(g; kw...)                          # g/cm³ given as kg/m³
    @test_throws ErrorException simulate(m; density_unit=:g_cm3, kw...)
    @test_throws ErrorException simulate(m; density_unit=:kg, kw...)
end

@testset "callback" begin
    times = Float64[]
    simulate(test_model(:psv); dt=DT, nt=21, sources=test_source(), snapshot_every=10, verbose=false,
             callback=(w, t) -> (@assert w isa CartData && haskey(w.fields, :V); push!(times, t)))
    @test times ≈ [0, 10DT, 20DT]
end

@testset "snapshots" begin
    dir = mktempdir()
    simulate(test_model(:sh); dt=DT, nt=21, sources=test_source(), wave=:sh, snapshot_every=10, snapshot_dir=dir, verbose=false)
    @test isfile(joinpath(dir, "wavefield.pvd"))
    @test count(endswith(".vts"), readdir(dir)) == 3
end

@testset "elastic model without Q, station at the free surface" begin
    m = test_model(:psv)
    elastic = CartData(m.x.val, m.y.val, m.z.val, (; rho=m.fields.rho, vp=m.fields.vp, vs=m.fields.vs))
    r = simulate(elastic; dt=DT, nt=300, sources=test_source(), stations=[Station("S", 0.0, 0.0, 0.0; surface=true)], verbose=false)
    @test all(isfinite, r.v) && maximum(abs, r.v) > 0
end

@testset "checkpoint and restart" begin
    dir = mktempdir()
    file = joinpath(dir, "run")
    kw = (; dt=DT, nt=200, sources=test_source(), stations=test_stations(:p3d), verbose=false)
    model = test_model(:p3d)
    full = simulate(model; kw...)
    simulate(model; kw..., checkpoint=file, checkpoint_every=120)          # writes the state after step 120
    @test isfile(file * ".jls")
    restarted = simulate(model; kw..., checkpoint=file, restart=true)       # continues from step 121
    @test restarted.v == full.v && restarted.strain == full.strain
    @test_throws ErrorException simulate(model; kw..., checkpoint=joinpath(dir, "none"), restart=true)
    @test_throws ErrorException simulate(model; kw..., nt=100, checkpoint=file, restart=true)   # another simulation
end

@testset "plane waves" begin
    # initial field = OpenSWPC's formulas: Vz at the cell centre (average of two staggered points), z up
    pw = PlaneWave(:p; depth=2.5, length=1.0, dip=20.0)
    W = Ref{Any}(nothing)
    simulate(test_model(:psv); dt=DT, nt=1, sources=pw, snapshot_every=1, callback=(w, t) -> (W[] = w), verbose=false)
    m = test_model(:psv)
    x, z = m.x.val[100, 1, 1], m.z.val[1, 1, 30]                     # in the lower layer (vp = 4 km/s, Q = 50)
    vp = 4.0 / SeismicWavePC.visco_chi(SeismicWavePC.relaxation_times(0.05, 5.0), 3 * SeismicWavePC.constq_zeta(0.05, 5.0,
         SeismicWavePC.relaxation_times(0.05, 5.0)) / 50, 1.0)        # unrelaxed P velocity
    vz(dz) = -cosd(20) * moment_rate(:kupper, sind(20) * x + cosd(20) * (-z + dz - 2.5) + DT / 2 * vp, 0.0, 1.0)
    @test -W[].fields.V[3][100, 1, 30] ≈ (vz(0.025) + vz(-0.025)) / 2 rtol = 1e-4
    @test_throws ErrorException PlaneWave(:love; depth=1, length=1)
    @test_throws ErrorException simulate(test_model(:sh); dt=DT, nt=1, sources=pw, wave=:sh, verbose=false)   # no P in SH
end

@testset "teleseismic waves (injected along a box)" begin
    # homogeneous half-space: the free-surface response of a plane P wave (apparent angle ī: sin(ī/2) = β p)
    α, β, dx = 6.0, 6.0 / sqrt(3), 0.1
    x = range(dx / 2, step=dx, length=240); z = range(-12 + dx / 2, step=dx, length=122)
    X, Y, Z = xyz_grid(x, 0, z); air = Z .> 0
    hs = CartData(X, Y, Z, (; rho=ifelse.(air, 1.0, 2700.0), vp=ifelse.(air, 0.0, α), vs=ifelse.(air, 0.0, β)))
    st = [[Station("S$i", xs, 0.0, 0.0; surface=true) for (i, xs) in enumerate(7.0:2.0:17.0)]; Station("O", 3.0, 0.0, 0.0; surface=true)]
    w = TeleseismicWave(:p; angle=30.0, box=(4.0, 20.0, -8.0), trise=1.0, t0=0.3)
    r = simulate(hs; dt=0.005, nt=900, sources=w, stations=st, verbose=false)
    ax, az = vec(maximum(abs, r.v[:, 1, 1:6]; dims=1)), vec(maximum(abs, r.v[:, 2, 1:6]; dims=1))
    @test all(q -> isapprox(q, tan(2asin(β * sind(30) / α)); rtol=2e-3), ax ./ az)
    @test maximum(az) / minimum(az) < 1.001                                        # the same amplitude along the line
    @test maximum(abs, r.v[:, :, 7]) < 2e-3 * maximum(az)                            # outside the box: only leakage
    # the box must keep away from the absorbers; the background must be layered and elastic along the box
    @test_throws ErrorException simulate(hs; dt=0.005, nt=2, verbose=false, sources=TeleseismicWave(:p; angle=0.0, box=(1.0, 20.0, -8.0), trise=1.0))
    lump = CartData(X, Y, Z, merge(hs.fields, (; vs=ifelse.((X .- 4) .^ 2 .+ (Z .+ 3) .^ 2 .< 1, 3.0, hs.fields.vs))))
    @test_throws ErrorException simulate(lump; dt=0.005, nt=2, sources=w, verbose=false)
    @test_throws ErrorException simulate(CartData(X, Y, Z, merge(hs.fields, (; Qp=fill(100.0, size(X)), Qs=fill(50.0, size(X)))));
                                         dt=0.005, nt=2, sources=w, verbose=false)
    @test_throws ErrorException TeleseismicWave(:love; angle=0.0, box=(1, 2, -1), trise=1.0)
end

@testset "reciprocal Green's functions" begin
    model = test_model(:psv)
    p, rcv = (0.5, 0.0, -1.5), (-1.2, 0.0, -0.3)
    src = DoubleCoupleSource(p...; Mw=3.0, strike=30, dip=60, rake=70, trise=0.3)
    forward = simulate(model; dt=DT, nt=NT, sources=src, stations=[Station("R", rcv...)], verbose=false)
    g = reciprocal_green(model, rcv, [Station("P", p...)]; component=:z, trise=0.3, displacement=true, dt=DT, nt=NT, verbose=false)
    @test g.components == (:xx, :zz, :xz, :fx, :fz)
    v = seismogram(g, 1, src)                                         # reciprocity
    @test maximum(abs, v .- forward.v[:, 2, 1]) < 0.06 * maximum(abs, forward.v[:, 2, 1])
    @test occursin("ReciprocalGreen", sprint(show, MIME"text/plain"(), g))
    files = write_sac(g, mktempdir(); station="R")
    @test basename(files[1]) == "swpc__00000001__R__z__mxx__.sac" && basename(files[end]) == "swpc__00000001__R__z__fz___.sac"
    @test_throws ErrorException reciprocal_green(model, rcv, [Station("P", p...)]; component=:w, dt=DT, nt=10, verbose=false)
end

@testset "random media" begin
    x, z = range(0, step=0.05, length=256), range(0, step=0.05, length=256)
    acf(A, l, dim) = (B = dim == 1 ? A[1:end-l, 1, :] .* A[1+l:end, 1, :] : A[:, 1, 1:end-l] .* A[:, 1, 1+l:end];
                      sum(B) / length(B) / (sum(abs2, A) / length(A)))
    xi = random_medium(x, 0.0, z; spectrum=:gaussian, a=(0.5, 1.0), epsilon=0.05, seed=1)
    @test size(xi) == (256, 1, 256) && sqrt(sum(abs2, xi) / length(xi)) ≈ 0.05 && abs(sum(xi)) / length(xi) < 1e-12
    @test acf(xi, 10, 1) ≈ exp(-1) atol = 0.08                                               # Gaussian ACF exp(-r²/a²)
    @test acf(xi, 20, 2) ≈ exp(-1) atol = 0.08                                               # anisotropic: az = 2 ax
    e = random_medium(x, 0.0, z; spectrum=:exponential, a=0.5, epsilon=0.05, seed=1)
    @test acf(e, 10, 1) ≈ exp(-1) atol = 0.08                                                # exponential ACF exp(-r/a)
    @test acf(e, 20, 1) ≈ exp(-2) atol = 0.08
    @test random_medium(x, 0.0, z; a=0.5, epsilon=0.05, seed=3) == random_medium(x, 0.0, z; a=0.5, epsilon=0.05, seed=3)
    @test size(random_medium(1:10, 1:8, 1:6; a=1, epsilon=0.1, seed=1)) == (10, 8, 6)
    @test_throws ErrorException random_medium(x, 0.0, z; spectrum=:pink, a=1, epsilon=0.1)

    m = test_model(:psv)
    xi = random_medium(m.x.val[:, 1, 1], 0.0, m.z.val[1, 1, :]; a=0.3, epsilon=0.05, seed=2)
    p = perturb(m, xi)
    solid = m.fields.vs .> 0
    @test p.fields.vp[solid] ≈ m.fields.vp[solid] .* (1 .+ xi[solid]) && p.fields.vp[.!solid] == m.fields.vp[.!solid]
    @test p.fields.rho[solid] ≈ m.fields.rho[solid] .* (1 .+ 0.8 .* xi[solid]) && haskey(p.fields, :Qp)
    small = reshape(collect(1.0:6.0) ./ 100, 2, 1, 3)                     # repeated periodically from the top
    q = perturb(m, small)
    @test q.fields.vs[3, 1, end-5] ≈ m.fields.vs[3, 1, end-5] * (1 + small[1, 1, 3 - mod1(6, 3) + 1])
    @test simulate(p; dt=DT, nt=50, sources=test_source(), verbose=false) isa Seismograms
end

@testset "free surface: inner corners of the topography" begin
    # a ridge one cell wide on the flat surface: no horizontal motion with the averages of OpenSWPC (the shear stress at the
    # inner corners at its foot is zero), coupled to the ground with free_surface = :corners
    m = test_model(:psv)
    i, k = 110, findlast(>(0), m.fields.vs[110, 1, :]) + 1
    ridge = map(m.fields) do A
        B = copy(A); B[i, 1, k] = A[i, 1, k-1]; B
    end
    mr = CartData(m.x.val, m.y.val, m.z.val, ridge)
    kw = (; dt=DT, nt=300, sources=test_source(), stations=[Station("R", m.x.val[i, 1, 1], 0.0, 0.0; surface=true)], verbose=false)
    ro, rc = simulate(mr; kw...), simulate(mr; kw..., free_surface=:corners)
    @test all(iszero, ro.v[:, 1, 1]) && maximum(abs, rc.v[:, 1, 1]) > 0.1 * maximum(abs, rc.v[:, 2, 1])
    # flat free surface: identical results
    for mode in (:psv, :p3d)
        mf = test_model(mode; step=false)
        kwf = (; dt=DT, nt=100, sources=test_source(), stations=test_stations(mode), verbose=false)
        a, b = simulate(mf; kwf...), simulate(mf; kwf..., free_surface=:corners)
        @test a.v == b.v && a.stress == b.stress
    end
    @test_throws ErrorException simulate(mr; kw..., free_surface=:flat)
end

@testset "multiaxial PML (absorber = :mpml)" begin
    # the reproducer of issue #6, coarse: under a soft surface layer the classical PML grows exponentially, the M-PML decays
    # (air thicker than the z-damping of :pml at the top, na dz, which would otherwise damp the layer in the side zones too)
    function layered(soft; dx=0.2, dz=0.1, L=10.0, depth=3.0, air=2.1)
        X, Y, Z = xyz_grid(collect(dx/2:dx:L), 0.0, collect(-depth + dz/2:dz:air))
        s, solid = (Z .> -soft) .& (Z .<= 0), Z .<= 0
        CartData(X, Y, Z, (; rho=ifelse.(solid, ifelse.(s, 2000.0, 2700.0), 1.0), vp=ifelse.(solid, ifelse.(s, 1.6, 5.8), 0.0),
                           vs=ifelse.(solid, ifelse.(s, 0.35, 3.35), 0.0)))
    end
    src = MomentTensorSource(5.0, 0.0, -1.0, 1e15, (1.0, 0.0, -1.0, 0.0, 1.0, 0.0); trise=1.5)
    kw = (; dt=0.01, nt=10000, sources=src, stations=[Station("pml", 9.75, 0.0, 0.0; surface=true), Station("in", 7.0, 0.0, 0.0; surface=true)],
          record_every=10, verbose=false, wavefield=false)
    growth(r) = maximum(abs, r.v[r.t .>= 80, :, :]) / maximum(abs, r.v[r.t .< 20, :, :])     # last 20 s / first 20 s of 100 s
    @test growth(simulate(layered(0.4); kw...)) > 1e3
    @test growth(simulate(layered(0.4); kw..., absorber=:mpml)) < 0.1
    # reflections: against a domain large enough that nothing comes back within 6 s (rock half-space, source 1 km from the
    # left absorber); the M-PML reflects more than the PML (its reflection grows linearly with mpml_ratio)
    halfspace(pad) = (d = layered(0.0; dx=0.1, dz=0.05, L=6.0 + 2pad, depth=3.0 + pad, air=1.05);
                 CartData(d.x.val .- pad, d.y.val, d.z.val, d.fields))
    kwr = (; dt=0.005, nt=1200, sources=MomentTensorSource(1.05, 0.0, -0.975, 1e15, (1.0, 0.0, -1.0, 0.0, 1.0, 0.0); trise=1.0),
           stations=[Station("S$x", x + 0.05, 0.0, 0.0; surface=true) for x in 0.2:0.4:5.8], verbose=false, wavefield=false)
    ref = simulate(halfspace(19.0); kwr...).v                                                    # 5.8 km/s × 6 s / 2 + 1.6 km
    err(r) = sqrt(sum(abs2, r.v .- ref) / sum(abs2, ref))
    e = map(kw -> err(simulate(halfspace(2.0); kwr..., kw...)), ((; na=20), (; na=20, absorber=:mpml, mpml_ratio=0.02),
                                                                    (; na=20, absorber=:mpml)))
    @test e[1] < 1e-3 && e[1] < e[2] < e[3] < 0.07                    # (2.2e-4, 9.3e-3, 3.5e-2)
    # 3D: a model and a source symmetric under x <-> y give a symmetric wavefield (the profiles of every derivative's point)
    xs = collect(range(-0.975, 0.975, length=40))
    X, Y, Z = xyz_grid(xs, xs, collect(range(-1.45, 0.5, length=40)))
    air = Z .> 0
    m3 = CartData(X, Y, Z, (; rho=ifelse.(air, 1.0, 2700.0), vp=ifelse.(air, 0.0, 5.8), vs=ifelse.(air, 0.0, 3.35)))
    r3 = simulate(m3; dt=0.004, nt=300, sources=MomentTensorSource(-0.425, -0.425, -0.5, 1e15, (1.0, 1.0, 0.5, 0.3, 0.3, 0.2); trise=0.2),
                  absorber=:mpml, mpml_ratio=0.2, na=10, verbose=false)
    V = r3.wavefield.fields.V
    @test maximum(abs, V[1]) > 0 && V[1] == permutedims(V[2], (2, 1, 3)) && V[3] == permutedims(V[3], (2, 1, 3))
    # plane waves pass the side layers of the PML only; the ratio must lie in [0, 1]
    @test_throws "plane waves" simulate(test_model(:psv); dt=DT, nt=1, sources=PlaneWave(:p; depth=2.5, length=1.0), absorber=:mpml, verbose=false)
    @test_throws "mpml_ratio" simulate(test_model(:psv); dt=DT, nt=1, sources=test_source(), absorber=:mpml, mpml_ratio=1.5, verbose=false)
    @test_throws "absorber" simulate(test_model(:psv); dt=DT, nt=1, sources=test_source(), absorber=:mpl, verbose=false)
end

@testset "periodic side boundaries" begin
    # a model uniform along x with periodic sides: moving source and stations together (across the edge) changes nothing
    m = test_model(:psv)                                    # (layered, flat surface; 10 km wide)
    w(x, L) = mod(x + L / 2, L) - L / 2
    src(x) = MomentTensorSource(w(x, 10.0), 0.0, -1.0, 1e15, (1.0, 0.0, -1.0, 0.0, 0.7, 0.0); trise=0.3)
    st(x) = [Station("a", w(x + 1.0, 10.0), 0.0, -0.5), Station("b", w(x - 3.0, 10.0), 0.0, -1.5)]
    kw = (; dt=DT, nt=400, T=Float64, verbose=false, periodic=true)
    @test simulate(m; kw..., sources=src(-0.475), stations=st(-0.475)).v == simulate(m; kw..., sources=src(4.025), stations=st(4.025)).v
    # a plane wave at vertical incidence: the same at all x (no side effects)
    pw = simulate(m; kw..., nt=300, sources=PlaneWave(:p; depth=2.5, length=1.0), stations=[Station("p", x, 0.0, -1.0) for x in (-4.0, 0.0, 3.0)])
    @test pw.v[:, :, 1] == pw.v[:, :, 2] == pw.v[:, :, 3]
    # 3D, periodic in x and y: shifted across both edges
    m3 = test_model(:p3d; step=false)
    src3(x, y) = MomentTensorSource(w(x, 4.0), w(y, 3.0), -1.0, 1e15, (1.0, 0.5, -1.0, 0.3, 0.7, -0.2); trise=0.3)
    st3(x, y) = [Station("a", w(x + 1.0, 4.0), w(y + 0.6, 3.0), -0.5), Station("b", w(x - 1.4, 4.0), w(y - 1.1, 3.0), -1.5)]
    k3 = (; dt=DT, nt=150, T=Float64, verbose=false, periodic=true)
    @test simulate(m3; k3..., sources=src3(-0.475, 0.025), stations=st3(-0.475, 0.025)).v ==
          simulate(m3; k3..., sources=src3(1.525, 1.025), stations=st3(1.525, 1.025)).v
    # periodic in x only (3D), and without periodic sides the absorbing layers
    @test all(isfinite, simulate(m3; k3..., nt=20, periodic=(true, false), sources=src3(0.0, 0.0), stations=st3(0.0, 0.0)).v)
    obs = simulate(m; dt=DT, nt=50, sources=src(0.0), stations=st(0.0), verbose=false)
    @test_throws "periodic boundaries" misfit_gradient(m, obs; dt=DT, nt=50, sources=src(0.0), periodic=true)
end
