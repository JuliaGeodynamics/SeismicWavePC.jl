[![CI](https://github.com/boriskaus/SeismicWavePC.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/boriskaus/SeismicWavePC.jl/actions/workflows/CI.yml)

# SeismicWavePC.jl

**Seismic wave propagation in 2D and 3D visco-elastic media, in pure Julia, on CPUs, GPUs and MPI clusters.**

SeismicWavePC.jl is a Julia version of the finite-difference code [OpenSWPC](https://openswpc.github.io)
(Maeda et al., 2017). It uses the same numerical method: a staggered grid that is 4th order in space and 2nd order in time,
a generalized Zener body for constant Q, free surfaces and topography from an air layer, moment-tensor point sources and a
perfectly matched layer (PML) or Cerjan absorbing boundary. It reproduces OpenSWPC seismograms to single-precision round-off; see [Validation](#validation).
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

1. Install Julia 1.10 or newer, for example with [juliaup](https://github.com/JuliaLang/juliaup):
   `curl -fsSL https://install.julialang.org | sh` (Linux, macOS) or `winget install julia -s msstore` (Windows).
2. Install the package (it is not registered yet):
   ```julia
   julia> ]
   pkg> add https://github.com/boriskaus/SeismicWavePC.jl
   ```
3. To use all CPU cores, start Julia with threads: `julia -t auto`.
4. For GPUs also add `CUDA` (NVIDIA), `AMDGPU` (AMD) or `Metal` (Apple), load it with `using CUDA` and pass `backend=CUDABackend()`
   (or `ROCBackend()`, `MetalBackend()`) to `simulate`.
5. For MPI add `MPI` and install the launcher once with `using MPI; MPI.install_mpiexecjl()`; then run
   `mpiexecjl -n 4 julia script.jl`. On a cluster, use its MPI library with
   [`MPIPreferences.use_system_binary()`](https://juliaparallel.org/MPI.jl/stable/configuration/) (needed for CUDA-aware MPI).
6. To run the examples (Julia 1.11 or newer), clone the repository and install their environment:
   ```
   git clone https://github.com/boriskaus/SeismicWavePC.jl
   cd SeismicWavePC.jl
   julia --project=examples -e 'using Pkg; Pkg.instantiate()'
   julia --project=examples/MagmaThermoKinematics -e 'using Pkg; Pkg.instantiate()'    # MTK + MAGEMin example only
   ```

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
rho = [1.0, 2200.0, 2700.0][Phase .+ 1]; vp = [0.0, 2.5, 5.5][Phase .+ 1]; vs = [0.0, 1.2, 3.2][Phase .+ 1]
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

[`MagmaThermoKinematics/Unzen_2D.jl`](examples/MagmaThermoKinematics/Unzen_2D.jl) (own environment) sends seismic waves through the
magmatic system below Unzen volcano: it runs the 2D Unzen model of [MagmaThermoKinematics.jl](https://github.com/boriskaus/MagmaThermoKinematics.jl)
(about 6 minutes on a laptop at 100 m resolution), computes density, velocities and melt fraction from the temperature with
[MAGEMin](https://github.com/ComputationalThermodynamics/MAGEMin_C.jl) phase diagrams, sets a low Q in partially molten rock,
and makes a movie of the wavefield while the simulation runs (no ParaView files), with seismometers just below the surface:
```
julia -t auto --project=examples/MagmaThermoKinematics examples/MagmaThermoKinematics/Unzen_2D.jl [resolution in m]
```

![Waves through the magmatic system below Unzen](docs/src/assets/Unzen_2D_waves.gif)

*Top: S-wave velocity from MAGEMin for the temperature field of MTK (black: molten, white lines: more than 10 % melt).
Bottom: vertical velocity of a Mw 4 earthquake (star), recorded by seismometers just below the surface (triangles).*

![Seismograms of the Unzen example](docs/src/assets/Unzen_2D_seismograms.png)

*Vertical velocity at the seismometers (left) and peak velocities (right). Stations above the partially molten region (shaded)
record weaker S waves and a longer coda.*

## Usage

- **Model:** a regular GMG `CartData` with fields `rho` [kg/m³] and `vp`, `vs` [km/s]; `Qp`, `Qs` are optional (`density_unit=:g_cm3` for densities in g/cm³, as in OpenSWPC)
  (without them the model is elastic).
  - Air has `vp = vs = 0`; the free surface and topography follow from it.
  - Water has `vs = 0`.
  - A `CartData` of size `(nx, 1, nz)` is a 2D model: P-SV by default, `wave=:sh` for SH.
- **Sources:** `DoubleCoupleSource(x, y, z; Mw, strike, dip, rake)`, a general `MomentTensorSource`, or a point force
  `ForceSource(x, y, z; fx, fy, fz)` [N] (OpenSWPC's body force), or an incoming plane wave
  `PlaneWave(:p or :s; depth, length, strike, dip, rake)` (OpenSWPC's `pw_mode`); all of them can be combined.
  - All accept `(lon, lat, z, proj::ProjectionPoint)` instead of `x, y, z`.
  - Sources outside the model or in the air are ignored with a warning (as in OpenSWPC, e.g. parts of a finite fault).
  - Source time functions are set with `stf=:kupper | :boxcar | :triangle | :herrmann | :cosine | :texp`.
- **Stations:** `Station(name, x, y, z)`, `Station(name, lon, lat, z, proj)`, with `surface=true` to place a station on the free surface.
  Velocity, displacement, stress and strain are recorded (`result.v`, `result.u`, `result.stress`, `result.strain`).
- **GPU:** `simulate(model; backend=CUDABackend(), ...)`, or `ROCBackend()`, `MetalBackend()`. Metal requires `T = Float32` (the default).
- **MPI:** start with `mpiexecjl -n 4 julia script.jl`.
  - Pass `nproc=(2, 2)` in 3D (subdomains in x and y) or `nproc=4` in 2D.
  - This works on CPUs and on CUDA or AMD GPUs, one GPU per process; the results are identical to a serial run.
- **Absorbing boundary:** `absorber=:pml` (default; perfectly matched layer, as recommended for OpenSWPC) or `:cerjan` (sponge),
  `na=20` cells thick at the sides and the bottom. The PML absorbs much better; it costs 15–100% more time, depending on how
  much of the model lies in the absorbing layers.
- **Gradients (for waveform inversion):** with `using Enzyme`, `J, grad = misfit_gradient(model, observed; dt, nt, sources, T=Float64)`
  returns the least-squares misfit of the velocity seismograms and its exact gradient with respect to `rho`, `vp`, `vs`, `Qp`, `Qs`
  (a `CartData`), by reverse-mode automatic differentiation of the time loop with checkpointing. Checked against finite
  differences of whole simulations (2D, 3D, PML, Cerjan, forces, plane waves). CPU only so far, one process; a gradient takes about
  20 (2D) to 50 (3D) times as long as the forward simulation.
- **Random media:** `xi = random_medium(x, y, z; spectrum=:vonkarman, a=0.5, epsilon=0.05, kappa=0.3)` makes fluctuations with a
  Gaussian, exponential or von Kármán spectrum (anisotropic with `a=(ax, ay, az)`), and `perturb(model, xi)` applies them to any
  GMG model as OpenSWPC does (vp, vs × (1 + ξ), ρ × (1 + 0.8 ξ) in solid rock).
- **Reciprocal Green's functions:** `reciprocal_green(model, receiver, points; component=:z)` puts a unit force at the receiver
  and records the displacement gradients at many `points` (OpenSWPC's `green_mode`); `seismogram(green, n, source)` then gives the
  receiver seismogram for a source at any of the points, without a simulation per source.
- **Checkpoints:** `checkpoint="run1", checkpoint_every=1000` saves the state; `restart=true` continues from it (also with MPI).
- **Snapshots:** `snapshot_every=100` writes the velocity field to `snapshot_dir` as a ParaView movie (`wavefield.pvd`).
  With `snapshots=Snapshots(xy=-5.0, xz=0.0, fs=true, quantities=(:v, :u, :ps))` it writes planes (horizontal, vertical, free
  surface, ocean bottom) in OpenSWPC's netCDF format instead, including peak ground velocity maps (needs `using NCDatasets`).
  With `callback=(wavefield, t) -> ...` the wavefield (a `CartData`) is passed to your function instead, e.g. to plot it while the
  simulation runs.

## Coming from OpenSWPC

Velocity models and seismograms in the file formats of OpenSWPC can still be used:

```julia
using SeismicWavePC, NCDatasets
model  = read_netcdf("velocity_model.nc")      # netCDF model of OpenSWPC (vmodel_type = "velm", OpenSWPC.write_netcdf)
model  = read_lhm("layers.dat", x, y, z)       # or a layered model (lhm; read_lgm for lgm) on the grid x, y, z [km];
                                               # options vcut, munk (sea water) and earth_flattening, as in OpenSWPC
model  = read_grd("grd.lst", x, y, z, proj)    # or layer boundaries from GMT grids (grd), placed with a GMG ProjectionPoint
result = simulate(model; dt, nt, sources=source, stations)
write_sac(result, "out/wav"; title="swpc", source, quantities=(:v, :u, :stress, :strain))   # swpc.psv.<station>.Vx.sac, …
t, v, header = read_sac("old_run/wav/swpc.psv.ST01.Vz.sac")   # e.g. to compare with an earlier OpenSWPC run
```

| OpenSWPC | SeismicWavePC.jl |
|---|---|
| input file (`dt`, `nt`, `na`, `fq_min`, `stftype`, …) | keyword arguments of `simulate` (`stf` for `stftype`) |
| velocity model (`vmodel_type`) | any GMG `CartData`; `read_netcdf` (`velm`), `read_lhm` (`lhm`), `read_lgm` (`lgm`), `read_grd` (`grd`); `velm` and `grd` need `using NCDatasets` |
| source and station files | `DoubleCoupleSource`, `MomentTensorSource`, `ForceSource` (`bf_mode`), `PlaneWave` (`pw_mode`), `Station` (in x, y or longitude, latitude) |
| `green_mode` (reciprocal Green's functions) | `reciprocal_green(model, receiver, points; component)`, `seismogram(green, n, source)`, `write_sac(green, dir)` |
| netCDF in general | optional: NCDatasets is only loaded with `using NCDatasets` (package extension) |
| random media (`gen_rmed3d`, `gen_rmed2d`, `*_rmed` models) | `perturb(model, random_medium(x, y, z; spectrum, a, epsilon, kappa))`; `read_random_medium(file)` for existing files |
| `vcut`, `munk_profile`, `earth_flattening` | keywords of `read_lhm`, `read_lgm`, `read_grd`; `flatten_earth(model)` for any model, `flatten_depth` for sources |
| seismograms (`sw_wav_v`, `_u`, `_stress`, `_strain`) | always recorded: `result.v` [m/s], `result.u` [m], `result.stress` [Pa], `result.strain` |
| SAC files in `odir/wav` | `write_sac(result, dir; quantities)`: same file names and units (nm/s, nm, Pa), z positive up |
| snapshots (netCDF, `xy_v_sw`, `fs_ps_sw`, …) | `snapshots=Snapshots(xy, xz, yz, fs, ob; quantities=(:v, :u, :ps))`: same files (needs `using NCDatasets`); or a ParaView movie, or your own plots (`callback`) |
| x north, y east, z down | x east, y north, z up (as in GMG) |
| density in g/cm³ | kg/m³ (`density_unit=:g_cm3` for g/cm³) |

## Validation

The test suite compares seismograms (velocity, displacement, stress and strain) with the Fortran OpenSWPC code (`swpc_psv`,
`swpc_sh`, `swpc_3d`; reference traces are in [`test/reference`](test/reference)):

| Solver | Model | Max. difference to OpenSWPC (Cerjan / PML) |
|---|---|---|
| 2D P-SV | two layers, free surface, Q = 50 | 5 × 10⁻⁴ / 3 × 10⁻⁴ |
| 2D SH | two layers, free surface, Q = 50 | 2 × 10⁻⁶ / 2 × 10⁻⁶ |
| 3D | two layers, free surface with a topographic step, Q = 50 | 7 × 10⁻⁶ / 6 × 10⁻⁶ |

The tests also check:
- MPI runs with 4 processes against serial runs (identical);
- CPU against Metal (agree to 10⁻⁶);
- type stability with [JET.jl](https://github.com/aviatesk/JET.jl).

## Performance

Seconds per simulation for the two examples, on an Apple M4 Max laptop (12 performance cores, 40-core GPU). These are
wall-clock times of a whole run, including setup (and, for OpenSWPC, writing and reading its input files), without snapshots,
with the same absorbing boundary in both codes. Each time is the better of two runs, with OpenSWPC and Julia runs alternating.
The first run in a new Julia session takes 4–10 s more for compilation.

With the PML (the default of both codes):

| Example | OpenSWPC, 1 core | OpenSWPC, MPI | Julia, 1 thread | Julia, MPI | Julia, 12 threads | Julia, Metal GPU |
|---|---:|---:|---:|---:|---:|---:|
| [2D P-SV hill](examples/2D_PSV_hill.jl): 600 × 281 cells, 3000 steps | 6.2 s | 1.7 s (4 ranks) | 4.4 s | 1.9 s (4 ranks) | 1.2 s | 0.28 s |
| [3D volcano](examples/3D_volcano.jl): 160 × 160 × 126 cells, 1000 steps | 109 s | 18 s (8 ranks) | 91 s | 22 s (8 ranks) | 14 s | 4.4 s |

With the Cerjan sponge (`absorber=:cerjan`):

| Example | OpenSWPC, 1 core | OpenSWPC, MPI | Julia, 1 thread | Julia, MPI | Julia, 12 threads | Julia, Metal GPU |
|---|---:|---:|---:|---:|---:|---:|
| 2D P-SV hill | 6.8 s | 1.8 s (4 ranks) | 3.7 s | 1.5 s (4 ranks) | 1.0 s | 0.16 s |
| 3D volcano | 86 s | 17 s (8 ranks) | 68 s | 18 s (8 ranks) | 12 s | 2.9 s |

- On a single computer, Julia threads are faster than MPI with either code. MPI is meant for several GPUs or computers; with
  the same number of MPI ranks, OpenSWPC is somewhat faster with the PML.
- Julia's MPI runs pay a fixed ~0.2–0.5 s for setting up and closing the ImplicitGlobalGrid domain decomposition.

Time loop only, with the Cerjan absorber, in million cell updates per second (higher is better), on an Apple M4 Max laptop (12 performance cores, 40-core GPU).
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

## Differences to OpenSWPC

All features of OpenSWPC are available. Differences:

- Plane waves agree with `swpc_3d` to 10⁻⁵. `swpc_psv` and `swpc_sh` give 1–5% different plane waves (even at vertical incidence),
  while the 2D initial wavefield here equals the analytic formula and uses the same code as 3D; the difference appears to lie in the
  2D Fortran codes. OpenSWPC's 3D P-wave formula for `sxy` has `sd·cd·sf` where `sd²·sf·cf` is correct (only for inclined waves);
  the correct form is used here.
- Near vertical topography steps, the 2D results differ by 7–28% from the `swpc_psv`/`swpc_sh` binaries. They agree to 10⁻⁶ for flat
  surfaces, and for the same step in 3D, so the difference appears to lie in the 2D Fortran codes.

## References and license

Maeda, T., Takemura, S., and Furumura, T. (2017). OpenSWPC: an open-source integrated parallel simulation code for modeling seismic
wave propagation in 3D heterogeneous viscoelastic media. *Earth, Planets and Space*, 69, 102. doi:10.1186/s40623-017-0687-2

MIT license. The numerical algorithms are ported from OpenSWPC, © Takuto Maeda, MIT license.
