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
