# Installation

1. Install Julia 1.10 or newer, for example with [juliaup](https://github.com/JuliaLang/juliaup):
   `curl -fsSL https://install.julialang.org | sh` (Linux, macOS) or `winget install julia -s msstore` (Windows).
2. Install the package (it is not registered yet):
   ```julia
   julia> ]
   pkg> add https://github.com/boriskaus/SeismicWavePC.jl
   ```
3. To use all CPU cores, start Julia with threads: `julia -t auto`. On machines with very many cores that are shared with
   other jobs, a moderate number of threads (e.g. `-t 8`) can be faster.

## GPUs

Add `CUDA` (NVIDIA), `AMDGPU` (AMD) or `Metal` (Apple), load it, and pass its backend to `simulate`:
```julia
using CUDA
result = simulate(model; dt, nt, sources, stations, backend=CUDABackend())    # or ROCBackend(), MetalBackend()
```
Metal supports only `T = Float32` (the default); CUDA and AMD GPUs also `T = Float64`.

## MPI

Add `MPI` and install the launcher once with `using MPI; MPI.install_mpiexecjl()`, then run your script on several processes:
```
mpiexecjl -n 4 julia script.jl
```
On a cluster, use its MPI library with [`MPIPreferences.use_system_binary()`](https://juliaparallel.org/MPI.jl/stable/configuration/)
(needed for CUDA-aware MPI; then also set `IGG_CUDAAWARE_MPI=1`). See [Running simulations](@ref) for `nproc`.

## Optional packages

These are loaded only when you load them yourself (package extensions):
- `NCDatasets`: netCDF velocity models, snapshots and random media in OpenSWPC's formats ([Coming from OpenSWPC](@ref));
- `Enzyme`: gradients by automatic differentiation (`misfit_gradient(...; method=:ad)`);
- `LogDensityProblems`, `ForwardDiff`, `Mooncake`, `ReverseDiff`, `ChainRulesCore`: interfaces to inversion and sampling
  packages such as Turing.jl, AdvancedHMC.jl and Optim.jl ([Inversion and sampling packages](@ref)).

## Example environments

The examples need Julia 1.11 or newer. Clone the repository and install their environments:
```
git clone https://github.com/boriskaus/SeismicWavePC.jl
cd SeismicWavePC.jl
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/MagmaThermoKinematics -e 'using Pkg; Pkg.instantiate()'   # MTK and MAGEMin examples
julia --project=examples/Inversion -e 'using Pkg; Pkg.instantiate()'               # Optim, Turing, AdvancedHMC examples
```
Run them with `julia -t auto --project=examples examples/<name>.jl [cuda|amdgpu|metal]` (see [Examples](examples.md)).
