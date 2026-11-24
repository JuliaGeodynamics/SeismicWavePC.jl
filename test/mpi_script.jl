# Started by test_mpi.jl with 4 MPI processes. Exits with an error if a parallel run differs from the serial one.
using MPI
MPI.Init()
include("setup.jl")

rank = MPI.Comm_rank(MPI.COMM_WORLD)
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
MPI.Barrier(MPI.COMM_WORLD)

# snapshot planes: each rank fills its part, rank 0 writes; the files must equal those of a serial run
using NCDatasets
dirs = (mktempdir(), mktempdir())
for (dir, nproc) in zip(dirs, (1, (2, 2)))
    simulate(test_model(:p3d); dt=DT, nt=101, sources=test_source(), nproc, snapshot_every=50, snapshot_dir=dir, verbose=false,
             snapshots=Snapshots(xy=-1.0, xz=0.1, fs=true, quantities=(:v, :u, :ps)))
end
if rank == 0
    for f in readdir(dirs[1])
        a, b = NCDataset(joinpath(dirs[1], f)), NCDataset(joinpath(dirs[2], f))
        for v in keys(a)
            a[v][:] == b[v][:] || error("MPI snapshot $f, $v differs from the serial one")
        end
        close(a); close(b)
    end
    println("MPI snapshots: $(length(readdir(dirs[1]))) files identical to the serial run")
end
MPI.Barrier(MPI.COMM_WORLD)

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
