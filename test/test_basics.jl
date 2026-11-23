using Test

@testset "source time functions integrate to 1" begin
    dt = 1e-4
    for stf in (:kupper, :boxcar, :triangle, :herrmann, :cosine, :texp)
        t = 0:dt:(stf === :texp ? 20.0 : 3.0)
        @test isapprox(sum(moment_rate.(stf, t, 0.5, 1.0)) * dt, 1; atol=2e-3)
        @test moment_rate(stf, 0.4, 0.5, 1.0) == 0
    end
    @test_throws ErrorException moment_rate(:nonsense, 1.0, 0.0, 1.0)
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
