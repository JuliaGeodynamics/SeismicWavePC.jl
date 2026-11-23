using NCDatasets

# netCDF file as written by OpenSWPC.write_netcdf (OpenSWPC.jl): z top first, rho in g/cm³; mu and lambda or vp and vs
function write_openswpc_netcdf(file, m::CartData; moduli=true)
    is2d = size(m)[2] == 1
    dims = is2d ? ("x", "z") : ("x", "y", "z")
    f = m.fields
    rho = f.rho ./ 1000
    fields = moduli ? (; rho, mu=rho .* f.vs .^ 2, lambda=rho .* (f.vp .^ 2 .- 2 .* f.vs .^ 2), f.Qp, f.Qs) : (; rho, f.vp, f.vs, f.Qp, f.Qs)
    NCDataset(file, "c") do ds
        defDim(ds, "x", size(m)[1]); is2d || defDim(ds, "y", size(m)[2]); defDim(ds, "z", size(m)[3])
        defVar(ds, "x", Float32.(m.x.val[:, 1, 1]), ("x",))
        is2d || defVar(ds, "y", Float32.(m.y.val[1, :, 1]), ("y",))
        defVar(ds, "z", Float32.(reverse(m.z.val[1, 1, :])), ("z",))
        for (name, A) in pairs(fields)
            defVar(ds, String(name), reverse(Float32.(is2d ? A[:, 1, :] : A), dims=ndims(is2d ? A[:, 1, :] : A)), dims)
        end
    end
end

@testset "netCDF model (OpenSWPC velm)" begin
    for (mode, moduli) in ((:psv, true), (:p3d, false))
        m = test_model(mode)
        file = joinpath(mktempdir(), "velocity_model.nc")
        write_openswpc_netcdf(file, m; moduli)
        r = read_netcdf(file)
        @test size(r) == size(m)
        @test r.x.val ≈ m.x.val && r.y.val ≈ m.y.val && r.z.val ≈ m.z.val
        for name in (:rho, :vp, :vs, :Qp, :Qs)
            @test r.fields[name] ≈ m.fields[name] rtol = 1e-5
        end
        kw = (; dt=DT, nt=100, sources=test_source(), stations=test_stations(mode), verbose=false)
        a, b = simulate(m; kw...), simulate(r; kw...)
        @test maximum(abs, a.v .- b.v) <= 1e-4 * maximum(abs, a.v)
    end
end

@testset "SAC output" begin
    r = simulate(test_model(:psv); dt=DT, nt=50, sources=test_source(), stations=test_stations(:psv), verbose=false)
    dir = mktempdir()
    files = write_sac(r, dir; title="test", source=test_source())
    @test length(files) == 6 && basename(files[1]) == "test.psv.A.Vx.sac"
    t, v, h = read_sac(joinpath(dir, "test.psv.B.Vz.sac"))
    @test t ≈ r.t
    @test v ≈ Float32.(r["B"][:, 2] .* 1e9)                    # nm/s
    @test h.kstnm == "B" && h.kcmpnm == "Vz" && h.kevnm == "test" && h.delta ≈ DT
    s = test_source()
    @test h.dist ≈ hypot(1.5 - s.x, -0.4 - s.y) && h.evdp ≈ -s.z
    files = write_sac(r, dir; title="all", quantities=(:v, :u, :stress, :strain))
    @test length(files) == 3 * (2 + 2 + 3 + 3)
    _, u, hu = read_sac(joinpath(dir, "all.psv.C.Uz.sac"))
    _, e, he = read_sac(joinpath(dir, "all.psv.C.Exz.sac"))
    @test u ≈ Float32.(r.u[:, 2, 3] .* 1e9) && e ≈ Float32.(r.strain[:, 3, 3]) && hu.kcmpnm == "Uz" && he.kcmpnm == "Exz"
    @test_throws ErrorException write_sac(r, dir; quantities=(:acceleration,))
end

@testset "layered models (OpenSWPC lhm, lgm, grd)" begin
    dir = mktempdir()
    file = joinpath(dir, "layers.dat")
    write(file, "# depth rho vp vs qp qs\n 1.0  2.0 4.0 2.0 100 50\n\n 3.0  3.0 6.0 3.5 400 200\n")
    x, z = range(-2, 2, length=5), range(-5, 1, length=13)            # z = -5, -4.5, …, 1 (up)
    m = read_lhm(file, x, 0.0, z)
    at(m, f, depth) = m.fields[f][1, 1, findfirst(≈(-depth), z)]
    @test size(m) == (5, 1, 13)
    @test at(m, :vp, -1.0) == 0 && at(m, :rho, -1.0) == 1.0           # air above sea level
    @test at(m, :vp, 0.5) == 1.5 && at(m, :vs, 0.5) == 0 && at(m, :rho, 0.5) == 1000   # sea water above the first layer
    @test at(m, :vp, 1.0) == 4.0 && at(m, :Qs, 2.5) == 50 && at(m, :vs, 3.0) == 3.5 && at(m, :rho, 5.0) == 3000
    @test at(read_lhm(file, x, 0.0, z; vcut=2.5), :vs, 1.5) == 3.5               # the first layer is slower than vcut
    @test at(read_lhm(file, x, 0.0, z; munk=true), :vp, 0.5) ≈ 1.5 * (1 + 0.00737 * (2(500 - 1300) / 1300 - 1 + exp(-2(500 - 1300) / 1300)))
    deep = read_lhm(file, x, 0.0, range(-1000, 0, length=11); earth_flattening=true)
    zs = SeismicWavePC.R_EARTH * (1 - exp(-1000 / SeismicWavePC.R_EARTH))       # spherical depth of the flat depth 1000 km
    @test deep.fields.vp[1, 1, 1] ≈ 6.0 * exp(1000 / SeismicWavePC.R_EARTH) && flatten_depth(zs) ≈ 1000
    flat = flatten_earth(read_lhm(file, x, 0.0, range(-1000, 0, length=1001)))
    @test flat.fields.vp[1, 1, 1] ≈ 6.0 * exp(1000 / SeismicWavePC.R_EARTH) && flat.fields.rho[1, 1, 1] == 3000
    g = read_lgm(file, x, 0.0, z)
    @test at(g, :vp, 2.0) ≈ 5.0 && at(g, :rho, 2.5) ≈ 2750 && at(g, :vp, 5.0) == 6.0   # linear between the layer tops
    src = DoubleCoupleSource(0.0, 0, -4; Mw=1, strike=0, dip=45, rake=0)
    @test simulate(m; dt=0.02, nt=10, sources=src, na=2, verbose=false) isa Seismograms
    @test_throws ErrorException simulate(m; dt=0.02, nt=10, sources=src, verbose=false)     # smaller than the absorber

    # grd: two boundaries on a longitude-latitude grid [m depth]: flat at 1 km, dipping from 2 to 4 km eastwards
    proj = ProjectionPoint(Lon=140.0, Lat=36.0)
    lon, lat = 139.9:0.01:140.1, 35.9:0.01:36.1
    for (name, d) in (("b1.nc", [1000.0 for _ in lon, _ in lat]), ("b2.nc", [2000 + 2000 * (l - 139.9) / 0.2 for l in lon, _ in lat]))
        NCDataset(joinpath(dir, name), "c") do ds
            defVar(ds, "lon", collect(lon), ("lon",)); defVar(ds, "lat", collect(lat), ("lat",)); defVar(ds, "z", d, ("lon", "lat"))
        end
    end
    write(joinpath(dir, "grd.lst"), "b1.nc 2.0 4.0 2.0 100 50 0\nb2.nc 3.0 6.0 3.5 400 200 1\n")
    G = read_grd(joinpath(dir, "grd.lst"), range(-5, 5, length=11), range(-5, 5, length=11), z, proj)
    col(f, i) = G.fields[f][i, 6, :]
    @test col(:vp, 6)[findfirst(≈(-0.5), z)] == 1.5 && col(:vp, 6)[findfirst(≈(-1.5), z)] == 4.0   # water, layer 1
    @test col(:vp, 6)[findfirst(≈(-3.5), z)] == 6.0 && col(:vp, 6)[findfirst(≈(-2.5), z)] == 4.0   # boundary at 3 km in the centre
    @test count(==(6.0), col(:vp, 1)) > count(==(6.0), col(:vp, 11))                              # deeper towards the east
end
