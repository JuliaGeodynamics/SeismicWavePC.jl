# Misfit gradients

```julia
using SeismicWavePC, GeophysicalModelGenerator
observed = simulate(true_model; dt, nt, sources=source, stations)      # or recorded data, as Seismograms
J, grad, sgrad = misfit_gradient(start_model, observed; dt, nt, sources=source)
grad.fields.vs                                                          # ∂J/∂vs on the grid of the model (per km/s)
sgrad[1].M, sgrad[1].t0                                                 # ∂J/∂(moment tensor), ∂J/∂(origin time)
```

- **Misfit:** `misfit=L2Misfit()` (default) is `J = ½ Σ (v - v_observed)²` over all samples, components and stations of
  `observed`, whose stations are used.
  - `L2Misfit(weights=w, window=(t1, t2))` weights stations, components or samples and selects time windows (one for all, or
    one per station).
  - `TravelTimeMisfit(; weights, window)` uses the cross-correlation time shifts of the traces.
  - Any function `(syn, obs, t) -> (J, ∂J/∂syn)` of the synthetic and observed velocities `[it, component, station]` works too.
- **Several shots:** `misfit_gradient(model, [obs1, obs2, …]; sources=[src1, src2, …], …)` sums the misfits and model
  gradients of the shots; each shot can have several sources.
- **Model gradient:** `grad` is a `CartData` with the fields `rho`, `vp`, `vs`, `Qp`, `Qs`.
- **Source gradients:** `sgrad` has one entry per source (per shot: a vector):
  - moment tensors: `M` (∂J/∂ of the six components `M0 .* M`) and `M0`;
  - forces: `F` (∂J/∂ of `fx, fy, fz`);
  - both: the location `x, y, z`, `t0`, `trise`, and `stf`, the gradient with respect to the source time function at the
    times `t`.

  The location gradient comes from the spatial derivative of the adjoint field, because the source is injected at a single
  cell; all other source gradients are exact.
- All keywords of `simulate` work: GPUs (`backend`), MPI (`nproc`; every process gets all results), `T`, `absorber`,
  `record_every`, `stf`, ….
- **Method:** adjoint-state method with checkpointing. The forward wavefield is saved every few steps (balancing host and device
  memory) and the steps in between are recomputed; the adjoint simulation runs backwards.
- **Exactness:** the adjoint kernels are the exact transposes of the forward kernels. They include the 2nd-order cells at free
  surfaces and the PML (with adjoint PML memory variables), and adjoint memory variables of attenuation give the `Qp`, `Qs`
  gradients. The result agrees with reverse-mode automatic differentiation to round-off (10⁻¹⁵ with `T = Float64`) in 2D and 3D.
  It is approximate only for the Cerjan sponge, the side extrapolation of plane waves, and the density in the source cell of
  P-SV body forces.
- **Cost:** about 5 forward simulations. On Marmousi2 with 641 receivers a gradient takes 80 s on 12 CPU threads and 30 s on
  an Apple GPU, 5× and 13× faster than [SeismicWaves.jl](https://github.com/GinvLab/SeismicWaves.jl) ([Performance](@ref)).
- `method=:ad` (with `using Enzyme`) differentiates the whole simulation automatically instead: exact in all cases, but 20–50
  forward simulations, on the CPU and one process, with the default misfit and without source gradients.


See [`misfit_gradient`](@ref), [`L2Misfit`](@ref) and [`TravelTimeMisfit`](@ref) for all options.
