# Inversion and sampling packages

A `WaveformProblem` connects the simulations to Julia's optimization and sampling packages:

```julia
params  = ModelParameters(model, :vs; log=true)                # one parameter per solid cell (log vs), or:
params  = ModelParameters(model, :vs; basis=[layer1, layer2])  # a few basis functions (layers, blobs, splines)
problem = WaveformProblem(params, observed; sources, misfit=L2Misfit(weights=1/σ^2), dt, nt, backend)
θ0      = initial_parameters(params)                           # to_model(params, θ) gives the model of θ

# deterministic inversion with Optim.jl (L-BFGS)
using Optim
result = optimize(Optim.NLSolversBase.only_fg!(fg!(problem)), θ0, LBFGS())

# Bayesian inversion with Turing.jl (NUTS): priors in the model, the waveform misfit as likelihood
using Turing
@model function inversion(problem)
    θ ~ MvNormal(zeros(length(problem)), 0.3^2 * I)
    Turing.@addlogprob! -objective(problem, θ)
end
chain = sample(inversion(problem), NUTS(), 1000)
```

- **Objective:** `objective(problem, θ)` is Φ(θ) = misfit − log prior (optional `prior = θ -> (log p, ∇ log p)`), and
  `objective_and_gradient(problem, θ)` adds its gradient from the adjoint-state method. All options of `misfit_gradient`
  work: several shots, misfits, GPUs and MPI.
- **Optim.jl:** `fg!(problem)` is the function `NLSolversBase.only_fg!` expects, for L-BFGS, conjugate gradients, ….
- **Turing.jl:** the AD backends of Turing use the adjoint-state gradient for `objective` instead of differentiating the
  simulation. This works with ForwardDiff (Turing's default), Mooncake, ReverseDiff and Zygote. The gradient is cached,
  so ForwardDiff costs one adjoint simulation per gradient however many parameters there are.
- **LogDensityProblems.jl:** a `WaveformProblem` is a log density with gradient (−Φ), for AdvancedHMC.jl (HMC, NUTS),
  DynamicHMC.jl, Pathfinder.jl and other samplers, without any AD.
- **ChainRulesCore.jl:** a reverse rule for `objective`, for Zygote and other AD tools that use ChainRules.
- **Invalid models:** negative velocities, density or Q, or an unstable time step give Φ = ∞ (log density −∞). Samplers reject
  such proposals and line searches step back.

These connections are package extensions: they load automatically with Optim, Turing, LogDensityProblems, ForwardDiff,
Mooncake, ReverseDiff or ChainRulesCore, and cost nothing otherwise.

Examples, in their own environment [`examples/Inversion`](https://github.com/boriskaus/SeismicWavePC.jl/blob/main/examples/Inversion), invert noisy seismograms of one earthquake
at 11 stations for the shear velocity in three depth bands:
- [`inversion_turing.jl`](https://github.com/boriskaus/SeismicWavePC.jl/blob/main/examples/Inversion/inversion_turing.jl): L-BFGS with Optim, then Hamiltonian Monte Carlo (NUTS)
  with Turing;
- [`inversion_hmc.jl`](https://github.com/boriskaus/SeismicWavePC.jl/blob/main/examples/Inversion/inversion_hmc.jl): Hamiltonian Monte Carlo (NUTS, or classic HMC) with AdvancedHMC
  directly, without Turing and without AD.

```
julia --project=examples/Inversion -e 'using Pkg; Pkg.instantiate()'
julia -t auto --project=examples/Inversion examples/Inversion/inversion_turing.jl [cpu|cuda|metal]
julia -t auto --project=examples/Inversion examples/Inversion/inversion_hmc.jl [cpu|cuda|metal]
```


See [`ModelParameters`](@ref) and [`WaveformProblem`](@ref) for all options.
