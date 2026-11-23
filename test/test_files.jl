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
end
