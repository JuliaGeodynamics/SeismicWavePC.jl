# Checks of MPI on NVIDIA GPUs (one GPU per process): runs on 4 processes must equal the serial CPU run.
#
#   mpiexecjl -n 4 julia --project=examples examples/check_cuda_mpi.jl
#
using MPI
MPI.Init()
using SeismicWavePC, GeophysicalModelGenerator, CUDA, Printf
include(joinpath(@__DIR__, "..", "test", "setup.jl"))

rank = MPI.Comm_rank(MPI.COMM_WORLD)
reldiff(a, b) = maximum(abs, a .- b) / maximum(abs, a)
ok = true
for (mode, nproc) in ((:psv, 4), (:p3d, (2, 2)))
    m = test_model(mode)
    kw = (; dt=DT, nt=200, sources=test_source(), stations=test_stations(mode), T=Float64, verbose=false)
    serial = simulate(m; kw...)                                                       # CPU, every process
    t = @elapsed par = simulate(m; kw..., nproc, backend=CUDABackend())
    e = reldiff(serial.v, par.v)
    rank == 0 && @printf("MPI forward  %-4s nproc=%s: %.2f s, difference %.1e\n", mode, nproc, t, e)
    e < 1e-10 || (global ok = false)
    obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); kw...)
    kg = (; dt=DT, nt=200, sources=test_source(), T=Float64)
    J1, g1 = misfit_gradient(m, obs; kg...)
    t = @elapsed (J2, g2) = misfit_gradient(m, obs; kg..., nproc, backend=CUDABackend())
    e = maximum(f -> reldiff(g1.fields[f], g2.fields[f]), (:rho, :vp, :vs, :Qp, :Qs))
    rank == 0 && @printf("MPI gradient %-4s nproc=%s: %.2f s, misfit %.1e, gradient difference %.1e\n", mode, nproc, t, abs(J2 / J1 - 1), e)
    e < 1e-8 || (global ok = false)
end
ok = MPI.Allreduce(ok, &, MPI.COMM_WORLD)
rank == 0 && println(ok ? "ALL MPI CHECKS PASSED" : "SOME MPI CHECKS FAILED")
