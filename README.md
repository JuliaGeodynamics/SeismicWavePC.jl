[![CI](https://github.com/boriskaus/SeismicWavePC.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/boriskaus/SeismicWavePC.jl/actions/workflows/CI.yml)

# SeismicWavePC.jl

**Seismic wave propagation in 2D and 3D visco-elastic media, in pure Julia, on CPUs, GPUs and MPI clusters.**

SeismicWavePC.jl is a Julia version of the finite-difference code [OpenSWPC](https://openswpc.github.io)
(Maeda et al., 2017). It uses the same numerical method: a staggered grid that is 4th order in space and 2nd order in time,
a generalized Zener body for constant Q, free surfaces and topography from an air layer, moment-tensor point sources and a
Cerjan absorbing boundary. It reproduces OpenSWPC seismograms to single-precision round-off; see [Validation](#validation).
Models are [GeophysicalModelGenerator.jl](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl) (GMG) `CartData`
objects, so any GMG model setup can be used directly.

## What is new compared with OpenSWPC / OpenSWPC.jl

| | OpenSWPC (Fortran, via OpenSWPC.jl) | SeismicWavePC.jl |
|---|---|---|
| Code | Fortran binaries, input files | pure Julia: no binaries, no input files |
| Operating systems | Linux, macOS (Windows via WSL) | Linux, macOS, Windows |
| GPUs | NVIDIA only (OpenACC build) | NVIDIA, AMD, Apple (and more) via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) |
| Parallel CPU | MPI processes, single-threaded each | Julia threads, plus MPI |
| Multi-node / multi-GPU | MPI | MPI via [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) (CUDA- and ROCm-aware MPI) |
| Model input | netCDF velocity file written from GMG | the GMG `CartData` itself |
| Output | SAC and netCDF files | seismograms in memory, final wavefield as a `CartData`, ParaView snapshot movies |
| Geographic coordinates | own projection | the same GMG `ProjectionPoint` used to build the model |
| Safety | | stability (Courant) check, sources and stations outside the model are rejected |

## Installation

```julia
julia> ]
pkg> add https://github.com/boriskaus/SeismicWavePC.jl
```
For GPUs also add `CUDA`, `AMDGPU` or `Metal`. To use all CPU cores, start Julia with threads: `julia -t auto`.

## Example

```julia
using SeismicWavePC, GeophysicalModelGenerator

# 2D model (size nx × 1 × nz) in km, z up; vp = vs = 0 marks air above a 1 km high hill
x, z = range(-10, 10, length=400), range(-8, 2, length=201)
model = CartData(xyz_grid(x, 0, z))
Phase, Temp = fill(2, size(model)), zeros(size(model))                  # 2: crust
add_layer!(Phase, Temp, model; zlim=(-2, 2), phase=ConstantPhase(1))     # 1: sediments
X, _, Z = xyz_grid(x, 0, z)
Phase[Z .> exp.(-(X ./ 2) .^ 2)] .= 0                                   # 0: air above a 1 km high hill
rho = [0.001, 2.2, 2.7][Phase .+ 1]; vp = [0.0, 2.5, 5.5][Phase .+ 1]; vs = [0.0, 1.2, 3.2][Phase .+ 1]
model = addfield(model, (; rho, vp, vs, Qp=fill(200.0, size(X)), Qs=fill(100.0, size(X))))

source   = DoubleCoupleSource(-3.0, 0.0, -4.0; Mw=3.5, strike=0, dip=45, rake=90, trise=0.4)
stations = [Station("S$i", xs, 0.0, 0.0; surface=true) for (i, xs) in enumerate(-8:2:8)]

result = simulate(model; dt=0.004, nt=1500, sources=source, stations)
```
```
SeismicWavePC: PSV, 400×201 cells (dx = 0.05013, 0.05 km), 1500 steps of 0.004 s, Courant number 0.617
  1 source(s) (M0 = 1.995e+14 N m), 9 station(s), backend CPU (4 threads), Float32, 1 MPI rank(s)
  step  150/1500  t =   0.6000 s  max|v| = 7.658e+00 m/s
  …
  finished in 0.88 s
Seismograms: 9 station(s), 1500 samples, t = 0.0 … 5.996 s, components x, z
  S1         max |v| = 4.988e+00, 6.250e+00 m/s
  …
  final wavefield: CartData of size (400, 1, 201)
```
`result["S1"]` returns the velocities of a station, `result.v[it, component, station]` all of them, and
`result.wavefield` the velocity field at the end as a `CartData` (`write_paraview(result.wavefield, "final")`).

More examples are in [`examples/`](examples):
- [`2D_PSV_hill.jl`](examples/2D_PSV_hill.jl): layered crust under a hill, with a record section plot.
- [`3D_volcano.jl`](examples/3D_volcano.jl): a GMG volcano (`add_volcano!`) with a magma chamber, and stations in longitude and latitude.
- [`3D_mpi.jl`](examples/3D_mpi.jl): the same on 4 MPI processes.
- [`benchmark.jl`](examples/benchmark.jl): the timings below.

Run them with `julia -t auto --project=examples examples/<name>.jl [cuda|amdgpu|metal]`.

## Usage

- **Model:** a regular GMG `CartData` with fields `rho` [g/cm³] and `vp`, `vs` [km/s]; `Qp`, `Qs` are optional
  (without them the model is elastic).
  - Air has `vp = vs = 0`; the free surface and topography follow from it.
  - Water has `vs = 0`.
  - A `CartData` of size `(nx, 1, nz)` is a 2D model: P-SV by default, `wave=:sh` for SH.
- **Sources:** `DoubleCoupleSource(x, y, z; Mw, strike, dip, rake)` or a general `MomentTensorSource`.
  - Both accept `(lon, lat, z, proj::ProjectionPoint)` instead of `x, y, z`.
  - Source time functions are set with `stf=:kupper | :boxcar | :triangle | :herrmann | :cosine | :texp`.
- **Stations:** `Station(name, x, y, z)`, `Station(name, lon, lat, z, proj)`, with `surface=true` to place a station on the free surface.
- **GPU:** `simulate(model; backend=CUDABackend(), ...)`, or `ROCBackend()`, `MetalBackend()`. Metal requires `T = Float32` (the default).
- **MPI:** start with `mpiexecjl -n 4 julia script.jl`.
  - Pass `nproc=(2, 2)` in 3D (subdomains in x and y) or `nproc=4` in 2D.
  - This works on CPUs and on CUDA or AMD GPUs, one GPU per process; the results are identical to a serial run.
- **Snapshots:** `snapshot_every=100` writes the velocity field to `snapshot_dir` as a ParaView movie (`wavefield.pvd`).

## Validation

The test suite compares seismograms with the Fortran OpenSWPC code (`swpc_psv`, `swpc_sh`, `swpc_3d`; reference traces are in
[`test/reference`](test/reference)):

| Solver | Model | Max. difference to OpenSWPC |
|---|---|---|
| 2D P-SV | two layers, free surface, Q = 50 | 5 × 10⁻⁴ |
| 2D SH | two layers, free surface, Q = 50 | 2 × 10⁻⁶ |
| 3D | two layers, free surface with a topographic step, Q = 50 | 7 × 10⁻⁶ |

The tests also check:
- MPI runs with 4 processes against serial runs (identical);
- CPU against Metal (agree to 10⁻⁶);
- type stability with [JET.jl](https://github.com/aviatesk/JET.jl).

## Performance

Time loop only, in million cell updates per second (higher is better), on an Apple M4 Max laptop (12 performance cores, 40-core GPU).
The model is a half-space with Q and a free surface. Both codes ran through their normal drivers; the times have setup subtracted
and are the minimum of three runs. You can reproduce them with [`examples/benchmark.jl`](examples/benchmark.jl).

| Case | OpenSWPC, 1 core | OpenSWPC, MPI | Julia, 1 thread | Julia, 12 threads | Julia, Metal GPU |
|---|---:|---:|---:|---:|---:|
| 2D P-SV, 2000 × 1000, 500 steps | 57 | 95 (4 ranks) | 95 | 180 | 2252 |
| 3D, 128³, 200 steps | 33 | 52 (12 ranks) | 38 | 77 | 1523 |

- The Julia code is faster than the Fortran code: 1.2–1.7× on one core, and 1.5–1.9× with all cores.
- On the CPU it runs native Julia loops (`Threads.@threads` with `@simd`, vectorized by LLVM), which are 1.5–1.8× faster than the
  same kernels run through KernelAbstractions' CPU backend. GPUs use KernelAbstractions.
- 3D runs on CPUs are limited by memory access from about 8 threads on.
- The Apple GPU is 20–45× faster than one CPU core. NVIDIA (A100) numbers will be added.

The laptop was also busy with other work, so single runs varied a lot; take the numbers as indicative.

## Not (yet) supported

- PML absorbing boundaries: only the Cerjan sponge so far; PML is planned.
- From OpenSWPC: checkpoint/restart, plane-wave and body-force sources, reciprocal Green's functions, Earth flattening and the Munk ocean profile.
- OpenSWPC's random-medium generator; perturb the GMG model yourself instead.
- Near vertical topography steps, the 2D results differ by 7–28% from the `swpc_psv`/`swpc_sh` binaries. They agree to 10⁻⁶ for flat
  surfaces, and for the same step in 3D, so the difference appears to lie in the 2D Fortran codes.

## References and license

Maeda, T., Takemura, S., and Furumura, T. (2017). OpenSWPC: an open-source integrated parallel simulation code for modeling seismic
wave propagation in 3D heterogeneous viscoelastic media. *Earth, Planets and Space*, 69, 102. doi:10.1186/s40623-017-0687-2

MIT license. The numerical algorithms are ported from OpenSWPC, © Takuto Maeda, MIT license.
