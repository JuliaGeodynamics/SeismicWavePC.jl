# Timing of the time loop for a few model sizes (homogeneous half-space with an air layer, one source).
#
#   julia -t auto --project=examples examples/benchmark.jl            # CPU, all threads
#   julia --project=examples examples/benchmark.jl cuda               # or amdgpu | metal
#
using SeismicWavePC, GeophysicalModelGenerator, Printf
include("backend.jl")

function halfspace(n)
    dx = 0.05
    x = range(0, step=dx, length=n[1]); y = range(0, step=dx, length=n[2]); z = range(-(n[3] - 11) * dx, step=dx, length=n[3])
    X, Y, Z = xyz_grid(x, n[2] == 1 ? 0.0 : y, z)
    air = Z .> 0
    return CartData(X, Y, Z, (rho=ifelse.(air, 0.001, 2.7), vp=ifelse.(air, 0.0, 5.0), vs=ifelse.(air, 0.0, 2.9),
                              Qp=fill(200.0, size(X)), Qs=fill(100.0, size(X))))
end

cases = [("2D P-SV", (2000, 1, 1000), 1000), ("3D", (128, 128, 128), 500), ("3D", (256, 256, 128), 200)]
@printf("%-8s %-16s %6s %10s %14s   backend %s, %d thread(s)\n", "solver", "cells", "steps", "time [s]", "Mcells/s", backend isa CPU ? "CPU" : nameof(typeof(backend)), Threads.nthreads())
for (name, n, nt) in cases
    model = halfspace(n)
    src = DoubleCoupleSource(n[1] * 0.025, n[2] * 0.025, -2.0; Mw=3, strike=30, dip=60, rake=70, trise=0.3)
    simulate(model; dt=0.004, nt=2, sources=src, backend, verbose=false)                  # compile
    t0 = @elapsed simulate(model; dt=0.004, nt=2, sources=src, backend, verbose=false)    # setup
    t = @elapsed simulate(model; dt=0.004, nt, sources=src, backend, verbose=false)
    loop = t - t0
    @printf("%-8s %-16s %6d %10.2f %14.1f\n", name, join(filter(>(1), n), "×"), nt, loop, prod(n) * nt / loop / 1e6)
end
