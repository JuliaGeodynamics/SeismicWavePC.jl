# Started by test_mpi.jl with 4 MPI processes. Exits with an error if a parallel run differs from the serial one.
# The argument selects a part (forward, gradient, born, vti, mpml, other, curvilinear; default all): test_mpi.jl starts each part in fresh processes, so that
# the code compiled for one part does not stay in memory for the next (4 processes compiling everything need ~15 GB).
using MPI
MPI.Init()
include("setup.jl")

rank = MPI.Comm_rank(MPI.COMM_WORLD)
const SELECTED = isempty(ARGS) ? "all" : ARGS[1]                  # (a const: no clash with local names in the loops)
runs(p) = SELECTED == "all" || SELECTED == p

# a DistributedModel: every rank builds the model of its own subdomain only, and gets the results of the whole model
function distributed(model, asked=Ref{Any}(nothing); anisotropy=false)
    ax = SeismicWavePC.model_axes(model)
    return DistributedModel(ax.x, ax.y, ax.z; anisotropy) do ix, iy
        asked[] = (ix, iy)
        map(A -> A[ix, iy, :], model.fields)
    end
end

if runs("forward")
for (mode, nproc, sources) in ((:p3d, (2, 2), test_source()), (:psv, 4, test_source()), (:sh, 4, test_source()),
                               (:psv, 4, PlaneWave(:s; depth=2.5, length=1.0, dip=20.0)))
    kw = (; dt=DT, nt=200, sources, stations=test_stations(mode), wave=mode === :p3d ? :psv : mode, verbose=false)
    model = test_model(mode)
    serial = simulate(model; kw...)                   # every rank runs the full serial model
    parallel = simulate(model; nproc, kw...)
    if rank == 0
        err = maximum(abs, serial.v .- parallel.v) / maximum(abs, serial.v)
        errV = maximum(i -> maximum(abs, serial.wavefield.fields.V[i] .- parallel.wavefield.fields.V[i]), 1:3)
        println("MPI $mode nproc=$nproc: seismogram difference $err, wavefield difference $errV")
        (err < 1e-6 && errV < 1e-12 * maximum(abs, serial.wavefield.fields.V[1]) + 1e-20) || error("MPI result differs for $mode")
    end
end
# free_surface = :corners (shear moduli at the inner corners of the topographic step)
let model = test_model(:p3d), kw = (; dt=DT, nt=200, sources=test_source(), stations=test_stations(:p3d), verbose=false, free_surface=:corners)
    serial, parallel = simulate(model; kw...), simulate(model; nproc=(2, 2), kw...)
    if rank == 0
        err = maximum(abs, serial.v .- parallel.v) / maximum(abs, serial.v)
        println("MPI p3d nproc=(2, 2), free_surface = :corners: seismogram difference $err")
        err < 1e-6 || error("MPI result differs with free_surface = :corners")
    end
end
MPI.Barrier(MPI.COMM_WORLD)

for (mode, nproc) in ((:p3d, (2, 2)), (:psv, 4), (:sh, 4))
    model, asked = test_model(mode), Ref{Any}(nothing)
    stations = [test_stations(mode); Station("S", 1.4, 0.2, 0.0; surface=true)]
    kw = (; dt=DT, nt=200, sources=[test_source(), ForceSource(0.3, -0.2, -0.6; fz=1e12, trise=0.3)], stations,
          wave=mode === :p3d ? :psv : mode, verbose=false)
    serial = simulate(model; kw...)
    parallel = simulate(distributed(model, asked); nproc, kw..., wavefield=true)
    part = prod(length, asked[]) / prod(size(model)[1:2])
    part < 0.5 || error("rank $rank built $(round(100part))% of the columns of the model")
    if rank == 0
        err = maximum(abs, serial.v .- parallel.v) / maximum(abs, serial.v)
        errV = maximum(i -> maximum(abs, serial.wavefield.fields.V[i] .- parallel.wavefield.fields.V[i]), 1:3)
        println("MPI DistributedModel $mode nproc=$nproc: seismogram difference $err, wavefield difference $errV, columns built per rank $(round(100part))%")
        (err < 1e-6 && errV < 1e-12 * maximum(abs, serial.wavefield.fields.V[1]) + 1e-20) || error("MPI result with a DistributedModel differs for $mode")
    end
end
MPI.Barrier(MPI.COMM_WORLD)

# snapshot planes: each rank fills its part, rank 0 writes; the files must equal those of a serial run
using NCDatasets
dirs = (mktempdir(), mktempdir(), mktempdir())
for (dir, nproc, model) in zip(dirs, (1, (2, 2), (2, 2)), (test_model(:p3d), test_model(:p3d), distributed(test_model(:p3d))))
    simulate(model; dt=DT, nt=101, sources=test_source(), nproc, snapshot_every=50, snapshot_dir=dir, verbose=false,
             snapshots=Snapshots(xy=-1.0, xz=0.1, fs=true, quantities=(:v, :u, :ps)))
end
if rank == 0
    for dir in dirs[2:3], f in readdir(dirs[1])
        a, b = NCDataset(joinpath(dirs[1], f)), NCDataset(joinpath(dir, f))
        for v in keys(a)
            a[v][:] == b[v][:] || error("MPI snapshot $f, $v differs from the serial one")
        end
        close(a); close(b)
    end
    println("MPI snapshots: $(length(readdir(dirs[1]))) files identical to the serial run (CartData and DistributedModel)")
end
MPI.Barrier(MPI.COMM_WORLD)

# snapshots in pieces: every rank writes the points of its subdomain; put together they are the files of the serial run
piecedir = MPI.bcast(rank == 0 ? mktempdir() : nothing, 0, MPI.COMM_WORLD)       # one directory for all ranks
simulate(distributed(test_model(:p3d)); dt=DT, nt=101, sources=test_source(), nproc=(2, 2), snapshot_every=50, snapshot_dir=piecedir, verbose=false,
         snapshots=Snapshots(xy=-1.0, xz=0.1, fs=true, quantities=(:v, :u, :ps), pieces=true))
MPI.Barrier(MPI.COMM_WORLD)
if rank == 0
    bits(A) = reinterpret(UInt32, vec(A))
    for f in readdir(dirs[1])
        a, b = read_snapshots(joinpath(dirs[1], f)), read_snapshots(joinpath(piecedir, f))
        (a.axes == b.axes && a.t == b.t && all(v -> bits(a[v]) == bits(b[v]), a.variables)) || error("MPI snapshot pieces of $f differ from the serial file")
    end
    println("MPI snapshot pieces: $(length(readdir(piecedir))) files of 4 ranks, put together identical to the $(length(readdir(dirs[1]))) serial files, bit for bit")
end
MPI.Barrier(MPI.COMM_WORLD)
end

if runs("gradient")
# misfit gradients: every rank gets the gradient of the serial run
for (mode, nproc, stretched) in ((:psv, 4, false), (:p3d, (2, 2), false), (:psv, 4, true))
    m = test_model(mode; stretched)
    kw = (; dt=DT, nt=100, sources=test_source(), T=Float64)
    obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); kw..., stations=test_stations(mode), verbose=false)
    J1, g1 = misfit_gradient(m, obs; kw...)
    J2, g2 = misfit_gradient(m, obs; kw..., nproc)
    err = maximum(f -> maximum(abs, g1.fields[f] .- g2.fields[f]) / maximum(abs, g1.fields[f]), (:rho, :vp, :vs, :Qp, :Qs))
    rank == 0 && println("MPI gradient $mode nproc=$nproc$(stretched ? " stretched" : ""): misfit $(J2 / J1 - 1), gradient difference $err")
    (abs(J2 / J1 - 1) < 1e-12 && err < 1e-10) || error("MPI gradient differs for $mode on rank $rank")
end
MPI.Barrier(MPI.COMM_WORLD)
end

if runs("born")
# Born modelling: the medium perturbation on every rank, the tangents through the halo exchange
for (mode, nproc) in ((:psv, 4), (:p3d, (2, 2)))
    m = test_model(mode)
    X, Y, Z = m.x.val, m.y.val, m.z.val
    b = @. exp(-((X - 0.3)^2 + Y^2 + (Z + 1.0)^2) / 0.4^2) * (m.fields.vp > 0)
    δm = CartData(X, Y, Z, (; rho=200 .* b, vs=0.2 .* b, Qs=5 .* b))
    kw = (; dt=DT, nt=150, sources=test_source(), stations=test_stations(mode), T=Float64)
    d1, δ1 = born(m, δm; kw...)
    d2, δ2 = born(m, δm; kw..., nproc)
    e = (maximum(abs, d1.v .- d2.v) / maximum(abs, d1.v), maximum(abs, δ1.v .- δ2.v) / maximum(abs, δ1.v))
    rank == 0 && println("MPI born $mode nproc=$nproc: background, scattered: $e")
    all(<(1e-12), e) || error("MPI Born modelling differs for $mode on rank $rank")
end
MPI.Barrier(MPI.COMM_WORLD)
end

if runs("vti")
# VTI anisotropy: forward (P-SV, 3D; CartData and DistributedModel) and gradient (P-SV) as in serial
for (mode, nproc) in ((:psv, 4), (:p3d, (2, 2)))
    m0 = test_model(mode)
    m = CartData(m0.x.val, m0.y.val, m0.z.val, merge(m0.fields, (; epsilon=fill(0.2, size(m0)), delta=fill(0.1, size(m0)), gamma=fill(0.15, size(m0)))))
    kw = (; dt=DT, nt=150, sources=test_source(), T=Float64)
    s1, s2 = simulate(m; kw..., stations=test_stations(mode), verbose=false), simulate(m; kw..., stations=test_stations(mode), nproc, verbose=false)
    s3 = simulate(distributed(m; anisotropy=true); kw..., stations=test_stations(mode), nproc, verbose=false)
    e = max(maximum(abs, s1.v .- s2.v), maximum(abs, s1.v .- s3.v)) / maximum(abs, s1.v)
    if mode === :psv
        obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); kw..., stations=test_stations(mode), verbose=false)
        g1, g2 = misfit_gradient(m, obs; kw...)[2], misfit_gradient(m, obs; kw..., nproc)[2]
        e = max(e, maximum(f -> maximum(abs, g1.fields[f] .- g2.fields[f]) / maximum(abs, g1.fields[f]), (:vp, :epsilon, :delta)))
    end
    rank == 0 && println("MPI VTI $mode nproc=$nproc: seismograms (and gradient) difference $e")
    e < 1e-10 || error("MPI result with VTI anisotropy differs for $mode on rank $rank")
end
MPI.Barrier(MPI.COMM_WORLD)
end

if runs("mpml")
# M-PML (absorber = :mpml): coefficients per cell from the profiles sliced per rank; forward (CartData, DistributedModel),
# gradient and Born as in serial
for (mode, nproc) in ((:psv, 4), (:sh, 4), (:p3d, (2, 2)))
    m = test_model(mode)
    kw = (; dt=DT, nt=200, sources=test_source(), stations=test_stations(mode), wave=mode === :p3d ? :psv : mode, verbose=false, absorber=:mpml, mpml_ratio=0.2)
    serial, parallel = simulate(m; kw...), simulate(m; nproc, kw...)
    dist = simulate(distributed(m); nproc, kw..., wavefield=true)
    if rank == 0
        err = max(maximum(abs, serial.v .- parallel.v), maximum(abs, serial.v .- dist.v)) / maximum(abs, serial.v)
        errV = maximum(i -> maximum(abs, serial.wavefield.fields.V[i] .- parallel.wavefield.fields.V[i]), 1:3)
        println("MPI M-PML $mode nproc=$nproc: seismogram difference $err (CartData, DistributedModel), wavefield difference $errV")
        (err < 1e-6 && errV < 1e-12 * maximum(abs, serial.wavefield.fields.V[1]) + 1e-20) || error("MPI result with the M-PML differs for $mode")
    end
end
let m = test_model(:psv), nproc = 4
    kw = (; dt=DT, nt=150, sources=test_source(), T=Float64, absorber=:mpml, mpml_ratio=0.2)
    obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); kw..., stations=test_stations(:psv), verbose=false)
    g1, g2 = misfit_gradient(m, obs; kw...)[2], misfit_gradient(m, obs; kw..., nproc)[2]
    X, Y, Z = m.x.val, m.y.val, m.z.val
    b = @. exp(-((X - 0.3)^2 + (Z + 1.0)^2) / 0.4^2) * (m.fields.vp > 0)
    δm = CartData(X, Y, Z, (; vs=0.2 .* b))
    (_, δ1), (_, δ2) = born(m, δm; kw..., stations=test_stations(:psv)), born(m, δm; kw..., stations=test_stations(:psv), nproc)
    e = (maximum(abs, g1.fields.vs .- g2.fields.vs) / maximum(abs, g1.fields.vs), maximum(abs, δ1.v .- δ2.v) / maximum(abs, δ1.v))
    rank == 0 && println("MPI M-PML gradient, born psv nproc=$nproc: $e")
    all(<(1e-10), e) || error("MPI gradient or Born modelling with the M-PML differs on rank $rank")
end
MPI.Barrier(MPI.COMM_WORLD)
end

if runs("other")
# teleseismic waves injected along a box that spans several ranks: forward and gradient as in serial
let m0 = test_model(:psv), nproc = 4
    m = CartData(m0.x.val, m0.y.val, m0.z.val, Base.structdiff(m0.fields, NamedTuple{(:Qp, :Qs)}))
    w = TeleseismicWave(:p; angle=20.0, box=(-3.5, 3.5, -3.0), trise=0.3, t0=0.05)
    kw = (; dt=DT, nt=300, sources=w, T=Float64)            # long enough for the scattered waves to reach the stations
    s1, s2 = simulate(m; kw..., stations=test_stations(:psv), verbose=false), simulate(m; kw..., stations=test_stations(:psv), nproc, verbose=false)
    vs = copy(m.fields.vs); vs[90:110, 1, 50:60] .*= 1.05
    obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs))); kw..., stations=test_stations(:psv), verbose=false)
    g1, g2 = misfit_gradient(m, obs; kw...)[2], misfit_gradient(m, obs; kw..., nproc)[2]
    e = (maximum(abs, s1.v .- s2.v) / maximum(abs, s1.v), maximum(abs, g1.fields.vs .- g2.fields.vs) / maximum(abs, g1.fields.vs))
    rank == 0 && println("MPI teleseismic psv nproc=$nproc: seismograms, gradient: $e")
    all(<(1e-9), e) || error("MPI result with a TeleseismicWave differs on rank $rank")   # rounding: positions per rank
end
MPI.Barrier(MPI.COMM_WORLD)

# sources and stations between grid points (stencils across ranks)
for (mode, nproc) in ((:psv, 4), (:p3d, (2, 2)))
    m = test_model(mode)
    srcs = [MomentTensorSource(mode === :p3d ? 0.02 : -2.53, 0.03, -1.03, 1e15, (0.3, -0.5, 0.2, 0.4, 0.7, -0.3); trise=0.3, t0=0.05),
            ForceSource(mode === :p3d ? -0.04 : 0.03, -0.02, -1.21; fx=1e13, fy=5e12, fz=-2e13, t0=0.1, trise=0.3)]
    sts = [Station("A", -0.06, 0.04, -0.33), Station("B", mode === :p3d ? 1.47 : 2.47, -0.03, 0.0; surface=true)]
    kw = (; dt=DT, nt=100, T=Float64, interpolation=:sinc, sources=srcs)
    s1, s2 = simulate(m; kw..., stations=sts, verbose=false), simulate(m; kw..., stations=sts, nproc, verbose=false)
    obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); kw..., stations=sts, verbose=false)
    J1, g1, sg1 = misfit_gradient(m, obs; kw...)
    J2, g2, sg2 = misfit_gradient(m, obs; kw..., nproc)
    e = (maximum(abs, s1.v .- s2.v) / maximum(abs, s1.v), maximum(abs, g1.fields.vs .- g2.fields.vs) / maximum(abs, g1.fields.vs),
         abs(sg1[1].x / sg2[1].x - 1))
    rank == 0 && println("MPI sinc $mode nproc=$nproc: seismograms, gradient, source location: $e")
    all(<(1e-10), e) || error("MPI result with interpolation = :sinc differs for $mode on rank $rank")
end
MPI.Barrier(MPI.COMM_WORLD)
end

# curvilinear grids: subdomains in x (2D) and x, y (3D), halos of 4 nodes; a hill, a curved interface (two blocks),
# attenuation, a double couple, a force, stations at the surface and at depth (also across subdomain boundaries)
if runs("curvilinear")
    topo(x) = 1.5 * exp(-x^2 / 18)
    moho(x) = -5.0 + 0.8 * sin(2π * x / 16)
    mesh = curvilinear_mesh(range(-12, 12, length=97), [topo, moho]; dz=0.4, bottom=-15)
    X, Z = mesh.x.val, mesh.z.val
    low = Z .< moho.(X)
    o = ones(size(X))
    m = addfield(mesh, (; rho=ifelse.(low, 2900.0, 2600.0), vp=ifelse.(low, 6.5, 5.5), vs=ifelse.(low, 3.75, 3.2), Qp=200 .* o, Qs=100 .* o))
    srcs = [DoubleCoupleSource(-2.0, 0.0, -7.0; Mw=4.0, strike=30, dip=60, rake=70, trise=1.0), ForceSource(4.5, 0.0, -3.0; fz=1e14, trise=1.0)]
    sts = [Station("s$x", Float64(x), 0.0, 0.0; surface=true) for x in -9:3:9]
    push!(sts, Station("d", 0.1, 0.0, -6.0))
    # the stable time step of the serial run, the same on all ranks (fixed for both runs)
    stable_dt(m, srcs) = MPI.bcast((r = simulate(m; grid=:curvilinear, tmax=0.05, sources=srcs, T=Float64, verbose=false); r.t[2] - r.t[1]), MPI.COMM_WORLD; root=0)
    growing(v) = !all(isfinite, v) || maximum(abs, v[3end÷4:end, :, :]) > maximum(abs, v)
    kw = (; dt=stable_dt(m, srcs), tmax=6.0, sources=srcs, stations=sts, T=Float64, verbose=false)
    s1, s2 = simulate(m; kw...), simulate(m; kw..., nproc=4)
    growing(s1.v) && error("MPI curvilinear 2D test: unstable")
    e = maximum(abs, s1.v .- s2.v) / maximum(abs, s1.v)
    eV = rank == 0 ? maximum(i -> maximum(abs, s1.wavefield.fields.V[i] .- s2.wavefield.fields.V[i]), 1:3) / maximum(abs, s1.wavefield.fields.V[3]) : 0.0
    rank == 0 && println("MPI curvilinear 2D nproc=4: seismograms $e, wavefield $eV")
    (e < 1e-10 && eV < 1e-10) || error("MPI curvilinear 2D result differs on rank $rank")
    # the same model built per process (DistributedModel: the columns of each process), with a PML
    pieces(m) = (ix, iy) -> merge(map(A -> A[ix, iy, :], m.fields), (; Z=m.z.val[ix, iy, :]))
    dmodel(m) = DistributedModel(pieces(m), m.x.val[:, 1, 1], m.y.val[1, :, 1], size(m)[3])
    s1, s2 = simulate(m; kw..., absorber=:pml, na=10, nproc=4), simulate(dmodel(m); kw..., absorber=:pml, na=10, nproc=4)
    e = maximum(abs, s1.v .- s2.v) / maximum(abs, s1.v)
    rank == 0 && println("MPI curvilinear 2D DistributedModel nproc=4: seismograms $e")
    e == 0 || error("MPI curvilinear 2D DistributedModel differs on rank $rank")
    # SH (2D): seismograms and misfit gradient, with a PML
    kws = (; dt=stable_dt(m, srcs), tmax=4.0, sources=srcs, T=Float64, verbose=false, wave=:sh, absorber=:pml, na=10)
    s1, s2 = simulate(m; kws..., stations=sts), simulate(m; kws..., stations=sts, nproc=4)
    e = maximum(abs, s1.v .- s2.v) / maximum(abs, s1.v)
    obs = simulate(addfield(mesh, (; rho=m.fields.rho, vp=m.fields.vp, vs=m.fields.vs .* 0.98, Qp=m.fields.Qp, Qs=m.fields.Qs)); kws..., stations=sts)
    J1, g1, _ = misfit_gradient(m, obs; kws...)
    J2, g2, _ = misfit_gradient(m, obs; kws..., nproc=4)
    e = (e, abs(J1 / J2 - 1), maximum(f -> maximum(abs, g1.fields[f] .- g2.fields[f]) / maximum(abs, g1.fields[f]), (:rho, :vs, :Qs)))
    rank == 0 && println("MPI curvilinear SH nproc=4: seismograms, misfit, gradient $e")
    all(<(1e-10), e) || error("MPI curvilinear SH differs on rank $rank")
    # 3D
    topo3(x, y) = 1.0 * exp(-(x^2 + y^2) / 18)
    moho3(x, y) = -5.0 + 0.6 * sin(2π * x / 16) * cos(2π * y / 20)
    mesh = curvilinear_mesh(range(-8, 8, length=41), range(-7, 7, length=37), [topo3, moho3]; dz=0.5, bottom=-12)
    X, Y, Z = mesh.x.val, mesh.y.val, mesh.z.val
    low = Z .< moho3.(X, Y)
    o = ones(size(X))
    m = addfield(mesh, (; rho=ifelse.(low, 2900.0, 2600.0), vp=ifelse.(low, 6.5, 5.5), vs=ifelse.(low, 3.75, 3.2), Qp=200 .* o, Qs=100 .* o))
    srcs = [DoubleCoupleSource(-1.0, 0.5, -7.0; Mw=4.0, strike=30, dip=60, rake=70, trise=1.0), ForceSource(2.5, -1.5, -3.0; fz=1e14, trise=1.0)]
    sts = [Station("s$x$y", Float64(x), Float64(y), 0.0; surface=true) for x in -6:3:6 for y in (-4, 0, 5)]
    push!(sts, Station("d", 0.1, -0.2, -6.0))
    kw = (; dt=stable_dt(m, srcs), tmax=4.0, sources=srcs, stations=sts, T=Float64, verbose=false)
    s1, s2 = simulate(m; kw...), simulate(m; kw..., nproc=(2, 2))
    growing(s1.v) && error("MPI curvilinear 3D test: unstable")
    e = maximum(abs, s1.v .- s2.v) / maximum(abs, s1.v)
    eV = rank == 0 ? maximum(i -> maximum(abs, s1.wavefield.fields.V[i] .- s2.wavefield.fields.V[i]), 1:3) / maximum(abs, s1.wavefield.fields.V[3]) : 0.0
    rank == 0 && println("MPI curvilinear 3D nproc=(2, 2): seismograms $e, wavefield $eV")
    (e < 1e-10 && eV < 1e-10) || error("MPI curvilinear 3D result differs on rank $rank")
    s1, s2 = simulate(m; kw..., absorber=:pml, na=8, nproc=(2, 2)), simulate(dmodel(m); kw..., absorber=:pml, na=8, nproc=(2, 2))
    e = maximum(abs, s1.v .- s2.v) / maximum(abs, s1.v)
    rank == 0 && println("MPI curvilinear 3D DistributedModel nproc=(2, 2): seismograms $e")
    e == 0 || error("MPI curvilinear 3D DistributedModel differs on rank $rank")
    s1, s2 = simulate(m; kw..., absorber=:mpml, na=8), simulate(m; kw..., absorber=:mpml, na=8, nproc=(2, 2))   # (M-PML)
    e = maximum(abs, s1.v .- s2.v) / maximum(abs, s1.v)
    rank == 0 && println("MPI curvilinear 3D M-PML nproc=(2, 2): seismograms $e")
    e < 1e-10 || error("MPI curvilinear 3D M-PML differs on rank $rank")
    # snapshot planes (netCDF): collected by rank 0 and in pieces, against the serial files
    using NCDatasets
    snapdirs = MPI.bcast(rank == 0 ? (mktempdir(), mktempdir(), mktempdir()) : nothing, 0, MPI.COMM_WORLD)
    for (dir, nproc, pieces) in zip(snapdirs, (1, (2, 2), (2, 2)), (false, false, true))
        (nproc == 1 && rank > 0) || simulate(m; kw..., tmax=1.0, nproc, snapshot_every=25, snapshot_dir=dir,
                                             snapshots=Snapshots(; xy=-1.0, xz=0.4, yz=-0.3, fs=true, quantities=(:v, :u, :ps), pieces))
        MPI.Barrier(MPI.COMM_WORLD)
    end
    if rank == 0
        same(a, b) = size(a) == size(b) && all(((x, y),) -> x == y || (isnan(x) && isnan(y)), zip(a, b))
        for f in readdir(snapdirs[1]), dir in snapdirs[2:3]
            a, b = read_snapshots(joinpath(snapdirs[1], f)), read_snapshots(joinpath(dir, f))
            (a.axes == b.axes && a.t == b.t && all(v -> same(a[v], b[v]), a.variables)) || error("MPI curvilinear snapshot $f ($dir) differs")
        end
        println("MPI curvilinear snapshots: $(length(readdir(snapdirs[1]))) files, collected and in $(length(readdir(snapdirs[3]))) pieces, identical to the serial run")
    end
    MPI.Barrier(MPI.COMM_WORLD)
    # teleseismic wave (TF/SF box across the subdomains): seismograms and gradient
    meshl = curvilinear_mesh(range(-14, 14, length=141), [x -> 0.0, x -> -3.0]; dz=0.2, bottom=-16)
    Xl, Zl = meshl.x.val, meshl.z.val
    lowl = Zl .< -3.0
    bodyl = @. exp(-((Xl - 1)^2 + (Zl + 4)^2) / 1)
    ml = addfield(meshl, (; rho=ifelse.(lowl, 2800.0, 2400.0), vp=ifelse.(lowl, 6.0, 4.0) .* (1 .- 0.05 .* bodyl), vs=ifelse.(lowl, 3.4, 2.3)))
    wt = TeleseismicWave(:p; angle=20.0, box=(-6.0, 6.0, -9.0), backazimuth=270.0, t0=0.3, trise=1.0)
    stsl = [Station("s$x", Float64(x), 0.0, 0.0; surface=true) for x in -5:2:5]
    kwt = (; grid=:curvilinear, dt=stable_dt(ml, [wt]), tmax=5.0, sources=wt, T=Float64, absorber=:pml, na=10)
    s1, s2 = simulate(ml; kwt..., stations=stsl, verbose=false), simulate(ml; kwt..., stations=stsl, verbose=false, nproc=4)
    obs = simulate(addfield(meshl, merge(ml.fields, (; vs=ml.fields.vs .* (1 .- 0.03 .* bodyl)))); kwt..., stations=stsl, verbose=false)
    J1, g1, _ = misfit_gradient(ml, obs; kwt...)
    J2, g2, _ = misfit_gradient(ml, obs; kwt..., nproc=4)
    e = (maximum(abs, s1.v .- s2.v) / maximum(abs, s1.v), abs(J1 / J2 - 1), maximum(f -> maximum(abs, g1.fields[f] .- g2.fields[f]) / maximum(abs, g1.fields[f]), (:rho, :vp, :vs)))
    rank == 0 && println("MPI curvilinear teleseismic nproc=4: seismograms, misfit, gradient $e")
    all(<(1e-10), e) || error("MPI curvilinear teleseismic differs on rank $rank")
    # misfit gradient (2D, attenuation)
    mesh = curvilinear_mesh(range(-12, 12, length=97), [topo, moho]; dz=0.4, bottom=-15)
    X, Z = mesh.x.val, mesh.z.val
    low = Z .< moho.(X)
    base = (; rho=ifelse.(low, 2900.0, 2600.0), vp=ifelse.(low, 6.5, 5.5), vs=ifelse.(low, 3.75, 3.2), Qp=200 .* one.(X), Qs=100 .* one.(X))
    srcs = [DoubleCoupleSource(-2.0, 0.0, -7.0; Mw=4.0, strike=30, dip=60, rake=70, trise=1.0)]
    sts = [Station("s$x", Float64(x), 0.0, 0.0; surface=true) for x in -9:3:9]
    kw = (; dt=stable_dt(addfield(mesh, base), srcs), tmax=4.0, sources=srcs, T=Float64, absorber=:pml, na=10)
    obs = simulate(addfield(mesh, merge(base, (; vs=base.vs .* (1 .+ 0.03 .* exp.(-(X .^ 2 .+ (Z .+ 2) .^ 2) ./ 4))))); kw..., stations=sts, verbose=false)
    J1, g1, sg1 = misfit_gradient(addfield(mesh, base), obs; kw..., illumination=true)
    J2, g2, sg2 = misfit_gradient(addfield(mesh, base), obs; kw..., nproc=4, illumination=true)
    srcdiff(a, b) = maximum(k -> maximum(abs, collect(a[k]) .- collect(b[k])) / maximum(abs, collect(a[k])), (:x, :z, :M, :t0, :stf))
    e = (abs(J1 / J2 - 1), maximum(f -> maximum(abs, g1.fields[f] .- g2.fields[f]) / maximum(abs, g1.fields[f]), (:rho, :vp, :vs, :Qp, :Qs, :illumination)),
         srcdiff(sg1[1], sg2[1]))
    rank == 0 && println("MPI curvilinear gradient (attenuation, PML) nproc=4: misfit, gradient $e")
    all(<(1e-10), e) || error("MPI curvilinear gradient differs on rank $rank")
    kwm = (; kw..., absorber=:mpml)                                                             # (M-PML)
    (J1, g1, _), (J2, g2, _) = misfit_gradient(addfield(mesh, base), obs; kwm...), misfit_gradient(addfield(mesh, base), obs; kwm..., nproc=4)
    e = (abs(J1 / J2 - 1), maximum(f -> maximum(abs, g1.fields[f] .- g2.fields[f]) / maximum(abs, g1.fields[f]), (:rho, :vp, :vs, :Qp, :Qs)))
    rank == 0 && println("MPI curvilinear gradient (attenuation, M-PML) nproc=4: misfit, gradient $e")
    all(<(1e-10), e) || error("MPI curvilinear M-PML gradient differs on rank $rank")
    # misfit gradient (3D, VTI)
    mesh = curvilinear_mesh(range(-8, 8, length=41), range(-7, 7, length=37), [topo3, moho3]; dz=0.5, bottom=-12)
    X, Y, Z = mesh.x.val, mesh.y.val, mesh.z.val
    low = Z .< moho3.(X, Y)
    base = (; rho=ifelse.(low, 2900.0, 2600.0), vp=ifelse.(low, 6.5, 5.5), vs=ifelse.(low, 3.75, 3.2), epsilon=ifelse.(low, 0.05, 0.15),
            delta=ifelse.(low, 0.02, 0.06), gamma=ifelse.(low, 0.03, 0.1))
    srcs = [DoubleCoupleSource(-1.0, 0.5, -7.0; Mw=4.0, strike=30, dip=60, rake=70, trise=1.0)]
    sts = [Station("s$x$y", Float64(x), Float64(y), 0.0; surface=true) for x in -6:3:6 for y in (-4, 0, 5)]
    kw = (; dt=stable_dt(addfield(mesh, base), srcs), tmax=3.0, sources=srcs, T=Float64, absorber=:pml, na=6)
    obs = simulate(addfield(mesh, merge(base, (; vs=base.vs .* (1 .+ 0.03 .* exp.(-(X .^ 2 .+ Y .^ 2 .+ (Z .+ 2) .^ 2) ./ 4))))); kw..., stations=sts, verbose=false)
    J1, g1, sg1 = misfit_gradient(addfield(mesh, base), obs; kw...)
    J2, g2, sg2 = misfit_gradient(addfield(mesh, base), obs; kw..., nproc=(2, 2))
    e = (abs(J1 / J2 - 1), maximum(f -> maximum(abs, g1.fields[f] .- g2.fields[f]) / maximum(abs, g1.fields[f]), (:rho, :vp, :vs, :epsilon, :delta, :gamma)),
         srcdiff(sg1[1], sg2[1]))
    rank == 0 && println("MPI curvilinear 3D gradient (VTI, PML) nproc=(2, 2): misfit, gradient $e")
    all(<(1e-10), e) || error("MPI curvilinear 3D gradient differs on rank $rank")
    # Born modelling (3D)
    δm = addfield(mesh, (; vs=0.1 .* exp.(-((X .- 1) .^ 2 .+ Y .^ 2 .+ (Z .+ 2) .^ 2) ./ 4), rho=zero(X)))
    (b1, d1), (b2, d2) = born(addfield(mesh, base), δm; kw..., stations=sts), born(addfield(mesh, base), δm; kw..., stations=sts, nproc=(2, 2))
    e = (maximum(abs, b1.v .- b2.v) / maximum(abs, b1.v), maximum(abs, d1.v .- d2.v) / maximum(abs, d1.v))
    rank == 0 && println("MPI curvilinear 3D born nproc=(2, 2): background, scattered $e")
    all(<(1e-10), e) || error("MPI curvilinear born differs on rank $rank")
    # an ocean (fluid block, fluid–solid interface): 2D gradient, 3D seismograms
    bathy(x) = -1.5 + 0.4 * tanh(x / 3)
    mesh = curvilinear_mesh(range(-8, 8, length=61), [x -> 0.0, bathy]; dz=0.25, bottom=-8)
    X, Z = mesh.x.val, mesh.z.val
    wet = Z .> bathy.(X) .- 1e-9
    base = (; rho=ifelse.(wet, 1000.0, 2700.0), vp=ifelse.(wet, 1.5, 5.5), vs=ifelse.(wet, 0.0, 3.2), Qp=100 .+ 0 .* X, Qs=50 .+ 0 .* X)
    srcs = [MomentTensorSource(-1.0, 0.0, -4.0, 1e15, (1.0, 0.0, 0.5, 0.0, 0.7, 0.0); trise=1.0)]
    sts = [Station("w$x", Float64(x), 0.0, -0.5) for x in -6:3:6]
    kw = (; dt=stable_dt(addfield(mesh, base), srcs), tmax=3.0, sources=srcs, T=Float64, absorber=:pml, na=8)
    obs = simulate(addfield(mesh, merge(base, (; vs=base.vs .* 0.97))); kw..., stations=sts, verbose=false)
    (J1, g1, _), (J2, g2, _) = misfit_gradient(addfield(mesh, base), obs; kw...), misfit_gradient(addfield(mesh, base), obs; kw..., nproc=4)
    e = (abs(J1 / J2 - 1), maximum(f -> maximum(abs, g1.fields[f] .- g2.fields[f]) / maximum(abs, g1.fields[f]), (:rho, :vp, :vs)))
    rank == 0 && println("MPI curvilinear ocean gradient nproc=4: misfit, gradient $e")
    all(<(1e-10), e) || error("MPI curvilinear ocean gradient differs on rank $rank")
    bathy3(x, y) = -1.5 + 0.4 * tanh((x + y) / 3)
    mesh = curvilinear_mesh(range(-6, 6, length=31), range(-6, 6, length=31), [(x, y) -> 0.0, bathy3]; dz=0.3, bottom=-6)
    wet = mesh.z.val .> bathy3.(mesh.x.val, mesh.y.val) .- 1e-9
    m = addfield(mesh, (; rho=ifelse.(wet, 1000.0, 2700.0), vp=ifelse.(wet, 1.5, 5.5), vs=ifelse.(wet, 0.0, 3.2)))
    src3 = MomentTensorSource(0.5, -0.5, -3.0, 1e15, (1.0, 0.7, 0.5, 0.3, 0.4, 0.2); trise=0.8)
    kw = (; dt=stable_dt(m, [src3]), tmax=3.0, sources=src3, stations=[Station("w", 2.0, 1.0, -0.4), Station("s", -2.0, 2.0, -3.0)], verbose=false, absorber=:pml, na=6)
    s1, s2 = simulate(m; kw...), simulate(m; kw..., nproc=(2, 2))
    e = maximum(abs, s1.v .- s2.v) / maximum(abs, s1.v)
    rank == 0 && println("MPI curvilinear 3D ocean nproc=(2, 2): seismograms $e")
    e < 1e-10 || error("MPI curvilinear 3D ocean differs on rank $rank")
end
MPI.Barrier(MPI.COMM_WORLD)
