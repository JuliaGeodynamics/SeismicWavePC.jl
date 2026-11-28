# Running simulations

All options are keyword arguments of [`simulate`](@ref):

```julia
result = simulate(model; dt, nt, sources, stations, stf=:kupper, absorber=:pml, backend=CPU(), T=Float32, nproc=1)
```

- **Time step:** `dt` and `nt`. The Courant number is checked: an unstable `dt` gives an error with the largest stable value.
- **Absorbing boundary:** `absorber = :pml` (default; perfectly matched layer, as recommended for OpenSWPC), `:mpml`
  (multiaxial PML) or `:cerjan` (sponge), `na = 20` cells thick at the sides and the bottom. The PML absorbs much better; it
  costs 15–100 % more time, depending on how much of the model lies in the absorbing layers. Where a soft surface layer (high
  Poisson ratio) reaches into the side layers, the PML can grow exponentially after some tens of seconds. `:mpml` is stable
  there (`mpml_ratio = 0.1`), for a few % more time, but reflects more than the PML (less than `:cerjan`): mostly a slowly
  decaying long-period residual after the direct waves.
- **Periodic sides:** `periodic = true` (or per axis, `(true, false)` for x and y in 3D) makes the sides periodic: waves
  leaving on one side enter on the other (no absorbing layers there; the bottom still absorbs), e.g. for plane waves
  through random media, as SOFI's periodic boundaries. Staggered grid, simulations (with MPI and on GPUs as usual); not yet
  for misfit gradients, Born modelling, a `DistributedModel` or teleseismic waves.
- **Precision:** `T = Float32` (default) or `Float64`.
- **CPU threads:** start Julia with `-t auto` (or `-t n`).
- **GPUs:** `backend = CUDABackend()`, `ROCBackend()` or `MetalBackend()` (after `using CUDA`, `AMDGPU` or `Metal`).
- **MPI:** start with `mpiexecjl -n 4 julia script.jl` and pass `nproc = 4` in 2D (subdomains in x) or `nproc = (2, 2)` in
  3D (x and y). This works on CPUs and on CUDA or AMD GPUs, one GPU per process, and gives the same results as a serial run.
  Every process sets up the whole model; for models too large for that, see [Models built per MPI process](@ref).
- **Checkpoints:** `checkpoint = "run1", checkpoint_every = 1000` saves the state; `restart = true` continues from it (also
  with MPI).
- **Progress:** `verbose = true` (default) prints a summary and the progress.
