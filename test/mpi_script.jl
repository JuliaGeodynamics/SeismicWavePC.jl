# Started by test_mpi.jl with 4 MPI processes. Exits with an error if a parallel run differs from the serial one.
# The argument selects a part (forward, gradient, born, vti, other; default all): test_mpi.jl starts each part in fresh processes, so that
# the code compiled for one part does not stay in memory for the next (4 processes compiling everything need ~15 GB).
using MPI
MPI.Init()
include("setup.jl")

rank = MPI.Comm_rank(MPI.COMM_WORLD)
const SELECTED = isempty(ARGS) ? "all" : ARGS[1]                  # (a const: no clash with local names in the loops)
runs(p) = SELECTED == "all" || SELECTED == p

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

# a DistributedModel: every rank builds the model of its own subdomain only, and gets the results of the whole model
function distributed(model, asked=Ref{Any}(nothing))
    ax = SeismicWavePC.model_axes(model)
    return DistributedModel(ax.x, ax.y, ax.z) do ix, iy
        asked[] = (ix, iy)
        map(A -> A[ix, iy, :], model.fields)
    end
end
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
# VTI anisotropy: forward (P-SV, 3D) and gradient (P-SV) as in serial
for (mode, nproc) in ((:psv, 4), (:p3d, (2, 2)))
    m0 = test_model(mode)
    m = CartData(m0.x.val, m0.y.val, m0.z.val, merge(m0.fields, (; epsilon=fill(0.2, size(m0)), delta=fill(0.1, size(m0)), gamma=fill(0.15, size(m0)))))
    kw = (; dt=DT, nt=150, sources=test_source(), T=Float64)
    s1, s2 = simulate(m; kw..., stations=test_stations(mode), verbose=false), simulate(m; kw..., stations=test_stations(mode), nproc, verbose=false)
    e = maximum(abs, s1.v .- s2.v) / maximum(abs, s1.v)
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
