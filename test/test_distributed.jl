# A DistributedModel (the model built per subdomain) must give the same results as the same model given as a CartData.
using Test, NCDatasets

"The test model as a `DistributedModel`; `asked` gets the ranges it was asked for."
function distributed(model, asked=Ref{Any}(nothing); attenuation=true, fields=model.fields)
    ax = SeismicWavePC.model_axes(model)
    return DistributedModel(ax.x, ax.y, ax.z; attenuation) do ix, iy
        asked[] = (ix, iy)
        map(A -> A[ix, iy, :], fields)
    end
end

@testset "medium of a subdomain = that part of the whole medium" begin
    H, na = SeismicWavePC.H, 20
    for (mode, stretched, dims) in ((:psv, false, (4, 1)), (:psv, true, (2, 1)), (:p3d, false, (2, 2, 1)), (:p3d, false, (4, 3, 1)))
        N = length(dims)
        model = test_model(mode; stretched)
        m = SeismicWavePC.whole_model(model, :kg_m3)
        g = SeismicWavePC.Grid(SeismicWavePC.model_axes(model), Val(N))
        asked = Tuple[]
        block(S) = (push!(asked, S); SeismicWavePC.model_block(m, S, Val(N)))
        setup(W, windows) = SeismicWavePC.setup_medium(block, g, W, Val(windows), na, 0.05, 5.0, 1.0)
        whole = setup(ntuple(a -> 1:g.n[a]+2H, N), false)
        nl = g.n .÷ dims .+ 2H
        vlayer, attenuated = zero(whole.vlayer), false
        for coords in CartesianIndices(dims)
            off = (Tuple(coords) .- 1) .* (nl .- 2H)
            R = ntuple(a -> off[a]+1:off[a]+nl[a], N)
            W = ntuple(a -> a < N ? (max(first(R[a]) - 3, 1):min(last(R[a]) + 3, g.n[a] + 2H)) : R[a], N)
            part = setup(W, true)
            for f in (:rho, :lam, :mu, :taup, :taus)
                @test part[f][R...] == whole[f][R...]
            end
            for f in (:fs_top, :fs_bot, :ob_top, :ob_bot)
                @test part[f][Base.front(R)...] == whole[f][Base.front(R)...]
            end
            @test part.kfs.a == whole.kfs[part.kfs.r...] && part.kob.a == whole.kob[part.kob.r...]
            @test all(a -> length(asked[end][a]) <= nl[a] + 6, 1:N-1)        # only the cells around the subdomain
            vlayer, attenuated = max.(vlayer, part.vlayer), attenuated | part.attenuated
        end
        @test vlayer == whole.vlayer && attenuated == whole.attenuated && maximum(vlayer) == whole.vmax
    end
end

@testset "DistributedModel = CartData" begin
    for mode in (:psv, :sh, :p3d), stretched in (false, true)
        mode === :p3d && stretched && continue
        model = test_model(mode; stretched)
        asked = Ref{Any}(nothing)
        dm = distributed(model, asked)
        stations = [test_stations(mode); Station("S", 1.4, 0.2, 0.0; surface=true)]
        kw = (; dt=DT, nt=150, sources=[test_source(), ForceSource(0.3, -0.2, -0.6; fz=1e12, trise=0.3)], stations,
              wave=mode === :sh ? :sh : :psv, verbose=false)
        a, b = simulate(model; kw...), simulate(dm; kw..., wavefield=true)
        @test a.v == b.v && a.u == b.u && a.stress == b.stress && a.strain == b.strain
        @test all(i -> a.wavefield.fields.V[i] == b.wavefield.fields.V[i], 1:3)
        @test b.wavefield.x.val == a.wavefield.x.val && b.wavefield.z.val == a.wavefield.z.val
        @test simulate(dm; kw...).wavefield === nothing                       # not collected by default
        @test first(asked[][1]) == 21 && last(asked[][1]) == size(model)[1] - 20   # without the absorbing layers
        c = simulate(dm; kw..., absorber=:cerjan)
        @test c.v == simulate(model; kw..., absorber=:cerjan).v
    end
    model = test_model(:p3d)
    kw = (; dt=DT, nt=20, sources=test_source(), stations=test_stations(:p3d), verbose=false)
    # blocks given as CartData; densities in g/cm³
    ax = SeismicWavePC.model_axes(model)
    cart = DistributedModel(ax.x, ax.y, ax.z) do ix, iy
        CartData(xyz_grid(ax.x[ix], ax.y[iy], ax.z)..., map(A -> A[ix, iy, :], model.fields))
    end
    ref = simulate(model; kw...)
    @test simulate(cart; kw...).v == ref.v
    gcm3 = distributed(model; fields=merge(model.fields, (; rho=model.fields.rho ./ 1000)))
    @test simulate(gcm3; kw..., density_unit=:g_cm3).v ≈ ref.v rtol = 1e-12
    @test_throws "g/cm³" simulate(gcm3; kw...)
    # an elastic model: without Qp, Qs and without memory variables
    elastic = Base.structdiff(model.fields, NamedTuple{(:Qp, :Qs)})
    e = simulate(CartData(model.x.val, model.y.val, model.z.val, elastic); kw...)
    @test simulate(distributed(model; attenuation=false, fields=elastic); kw...).v == e.v
    @test simulate(distributed(model; fields=elastic); kw...).v ≈ e.v rtol = 1e-5      # with (unused) memory variables
    @test_throws "attenuation" simulate(distributed(model; attenuation=false); kw...)
    # what needs the whole model
    @test_throws "CartData" simulate(distributed(model); kw..., interpolation=:sinc)
    @test_throws "CartData" simulate(distributed(test_model(:psv)); kw..., stations=test_stations(:psv), sources=PlaneWave(:s; depth=2.5, length=1.0))
    @test_throws "has size" simulate(DistributedModel((ix, iy) -> map(A -> A[ix, iy, 1:end-1], model.fields), ax.x, ax.y, ax.z); kw...)
    @test_throws "unstable" simulate(distributed(model); kw..., dt=1.0)
    @test occursin("80×60×60", sprint(show, distributed(model)))
end

@testset "DistributedModel: snapshot planes" begin
    model = test_model(:p3d)
    dirs = (mktempdir(), mktempdir())
    for (dir, m) in zip(dirs, (model, distributed(model)))
        simulate(m; dt=DT, nt=101, sources=test_source(), snapshot_every=50, snapshot_dir=dir, verbose=false,
                 snapshots=Snapshots(xy=-1.0, yz=0.1, fs=true, quantities=(:v, :u)))
    end
    @test length(readdir(dirs[1])) == 6
    for f in readdir(dirs[1])
        NCDataset(joinpath(dirs[1], f)) do a
            NCDataset(joinpath(dirs[2], f)) do b
                @test all(v -> a[v][:] == b[v][:], keys(a))
            end
        end
    end
end

@testset "snapshots in pieces" begin
    # every process writes its own points (here: one process, one piece per file); read back they are the single files
    bits(A) = reinterpret(UInt32, vec(Float32.(A)))
    for mode in (:p3d, :psv)
        model = test_model(mode)
        dirs = (mktempdir(), mktempdir())
        planes = mode === :p3d ? (; xy=-1.0, xz=0.1, yz=0.2, fs=true, quantities=(:v, :u, :ps), decimate=(2, 3, 1)) : (; xz=true, quantities=(:v, :u))
        kw = (; dt=DT, nt=101, sources=test_source(), snapshot_every=50, verbose=false)
        simulate(model; kw..., snapshot_dir=dirs[1], snapshots=Snapshots(; planes...))
        simulate(distributed(model); kw..., snapshot_dir=dirs[2], snapshots=Snapshots(; planes..., pieces=true))
        @test length(readdir(dirs[1])) == length(readdir(dirs[2])) == (mode === :p3d ? 12 : 2)
        @test all(f -> endswith(f, ".0000.nc"), readdir(dirs[2]))
        for f in readdir(dirs[1])
            a, b = read_snapshots(joinpath(dirs[1], f)), read_snapshots(joinpath(dirs[2], f))
            @test a.axes == b.axes && a.t == b.t && a.variables == b.variables && length(b.files) == 1
            @test all(v -> bits(a[v]) == bits(b[v]), a.variables)
            @test bits(a[a.variables[4], 2]) == bits(b[b.variables[4], 2]) == bits(b[b.variables[4]][:, :, 2])   # a field with time
            @test_throws "has no time" b["rho", 1]
            NCDataset(joinpath(dirs[1], f)) do ds                         # the reader gives what is in the file
                @test all(v -> bits(a[v]) == bits(Array(ds[v].var)), a.variables)
                @test a.t == ds["t"][:] && a.axes[1][2] == ds[a.axes[1][1]][:]
            end
        end
        mode === :p3d && @test occursin("3 snapshot(s)", sprint(show, read_snapshots(joinpath(dirs[2], "swpc.3d.fs.v.nc"))))
    end
    @test_throws "no snapshot file" read_snapshots(joinpath(mktempdir(), "swpc.3d.fs.v.nc"))
    # files of another run next to the pieces: an error, not a mix
    dir = mktempdir()
    simulate(test_model(:p3d); dt=DT, nt=2, sources=test_source(), snapshot_every=1, snapshot_dir=dir, verbose=false,
             snapshots=Snapshots(xy=-1.0, pieces=true))
    file, piece = joinpath(dir, "swpc.3d.xy.v.nc"), joinpath(dir, "swpc.3d.xy.v.0000.nc")
    @test size(read_snapshots(file)["Vx", 2]) == size(read_snapshots(piece)["Vx", 2])          # one piece read on its own
    cp(piece, joinpath(dir, "swpc.3d.xy.v.0001.nc"))
    @test_throws "overlap" read_snapshots(file)
    mv(joinpath(dir, "swpc.3d.xy.v.0001.nc"), file)
    @test_throws "both" read_snapshots(file)
    rm(file)
    NCDataset(ds -> ds.attrib["whole_size"] = ds.attrib["whole_size"] .+ Int32[1, 0], piece, "a")      # as if a piece were missing
    @test_throws "leave points" read_snapshots(file)
    @test_throws "outside the model" simulate(test_model(:p3d); dt=DT, nt=2, sources=test_source(), snapshot_every=1, snapshot_dir=mktempdir(),
                                               verbose=false, snapshots=Snapshots(xz=50.0, pieces=true))
end
