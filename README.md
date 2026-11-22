# SeismicWavePC.jl

A Julia-only version of the [OpenSWPC](https://openswpc.github.io) finite-difference seismic wave propagation code
(Maeda et al., 2017, EPS, doi:10.1186/s40623-017-0687-2), for 2D (P-SV, SH) and 3D visco-elastic media.

- Velocity models are [GeophysicalModelGenerator.jl](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl) `CartData` objects
  (fields `rho`, `vp`, `vs`, optionally `Qp`, `Qs`); air (`vp = vs = 0`) defines the free surface and topography.
- Runs on CPU and GPU through [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) (CUDA, AMDGPU, Metal, ...).
- Runs on several MPI processes / GPUs through [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl).

**Status: work in progress.** The 2D P-SV solver reproduces OpenSWPC seismograms to single-precision round-off for flat free
surfaces (elastic and visco-elastic); SH, 3D, MPI and GPU runs are still being validated.

```julia
using SeismicWavePC, GeophysicalModelGenerator

x, z = range(-5, 5, length=200), range(-5, 0.5, length=111)
X, Y, Z = xyz_grid(x, 0, z)
air = Z .> 0
model = CartData(X, Y, Z, (rho = ifelse.(air, 0.001, 2.5), vp = ifelse.(air, 0.0, 3.4), vs = ifelse.(air, 0.0, 2.0),
                           Qp = fill(200.0, size(X)), Qs = fill(100.0, size(X))))

src = DoubleCoupleSource(0.0, 0.0, -2.0; Mw=3.0, strike=0, dip=45, rake=90, trise=0.3)
sta = [Station("A", -2.0, 0.0, 0.0; surface=true), Station("B", 2.0, 0.0, 0.0; surface=true)]

r = simulate(model; dt=0.004, nt=1000, sources=src, stations=sta)   # r.v[it, component, station] in m/s
```

Pass `backend=CUDABackend()` (etc.) to run on a GPU, and `nproc=(2, 2)` (3D) or `nproc=4` (2D) with `mpiexec -n 4 julia script.jl`
for MPI runs (CPU, CUDA, AMDGPU).

OpenSWPC is © Takuto Maeda, MIT license; this port keeps the same license.
