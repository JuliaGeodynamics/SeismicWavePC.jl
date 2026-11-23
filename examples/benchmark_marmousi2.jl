# Benchmark on Marmousi2 (elastic 2D P-SV): 5 m grid (3401 × 703 cells including the air), 2 s (5319 steps, Courant number 0.5),
# PML, explosion 10 m below the sea surface, 641 receivers. Time loop only (setup subtracted), the better of two runs.
#
#   julia -t auto --project=examples examples/benchmark_marmousi2.jl [cuda|amdgpu|metal]
#
using SeismicWavePC, GeophysicalModelGenerator, Printf
include("backend.jl")
include("marmousi2_model.jl")

model = marmousi2(5.0)
source = MomentTensorSource(8.5, 0.0, -0.01, 1e13, (1, 0, 1, 0, 0, 0) ./ sqrt(2); trise=0.2)
stations = [Station("R$i", x, 0.0, -0.01) for (i, x) in enumerate(0.5:0.025:16.5)]
dt = 0.5 * 0.005 / (4.7 * sqrt(2))
nt = round(Int, 2.0 / dt)
run(nt) = @elapsed simulate(model; dt, nt, sources=source, stations, stf=:ricker, backend, verbose=false)
run(2)
t0 = minimum(run(2) for _ in 1:2)
t = minimum(run(nt) for _ in 1:2) - t0
n = prod(size(model))
@printf("Marmousi2, %s, %s: %d cells, %d steps: %.2f s, %.0f Mcells/s\n", backend isa CPU ? "CPU" : nameof(typeof(backend)),
        backend isa CPU ? "$(Threads.nthreads()) thread(s)" : "", n, nt, t, n * nt / t / 1e6)
