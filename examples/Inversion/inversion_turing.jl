# Waveform inversion with SeismicWavePC.jl and Julia's inversion packages: the shear velocity in three depth bands of a 2D
# model, from noisy seismograms of one earthquake at 11 surface stations.
#  1. deterministic: L-BFGS (Optim.jl) from the starting model;
#  2. Bayesian: Hamiltonian Monte Carlo (NUTS) with Turing.jl, giving the posterior distribution.
# The gradients come from the adjoint-state method of SeismicWavePC.jl (also inside Turing's AD backends).
#
#   julia -t auto --project=examples/Inversion examples/Inversion/inversion_turing.jl [cpu|cuda|metal]
#
# About 20 minutes on a laptop CPU. GPUs pay off for larger models (here the model is tiny: 121 × 65 cells).
include("setup.jl")
using Optim, Turing, CairoMakie

# 1. L-BFGS
t = @elapsed result = optimize(Optim.NLSolversBase.only_fg!(fg!(problem)), θ0, LBFGS(), Optim.Options(iterations=20))
θopt = Optim.minimizer(result)
@printf("L-BFGS: θ = %s (true %s), %d iterations, %.0f s\n", round.(θopt, digits=3), θtrue, Optim.iterations(result), t)

# 2. Turing: prior N(0, 0.3²) per band; the waveform misfit is the negative log likelihood
@model function waveform_inversion(problem)
    θ ~ MvNormal(zeros(3), 0.3^2 * I)
    Turing.@addlogprob! -objective(problem, θ)
end
Random.seed!(2)
t = @elapsed chain = sample(waveform_inversion(problem), NUTS(50, 0.8), 150; initial_params=InitFromParams((; θ=θ0)))
S = reshape(Array(chain), size(chain, 1), :)                 # samples × parameters (one chain)
for k in 1:3
    @printf("band %d (%4.1f to %4.1f km): posterior %6.3f ± %.3f km/s, true %6.3f\n", k, abs(bands[k][1]), -bands[k][2],
            mean(S[:, k]), std(S[:, k]), θtrue[k])
end
@printf("NUTS: %d samples in %.0f s\n", size(S, 1), t)

# figure: posterior of each band, with the true value and the L-BFGS result
fig = Figure(size=(900, 300))
for k in 1:3
    ax = Axis(fig[1, k], title=@sprintf("vs change, %.1f–%.1f km depth", abs(bands[k][1]), -bands[k][2]), xlabel="km/s")
    hist!(ax, S[:, k], bins=20, color=(:steelblue, 0.6))
    vlines!(ax, [θtrue[k]], color=:black, linewidth=2, label="true")
    vlines!(ax, [θopt[k]], color=:red, linestyle=:dash, linewidth=2, label="L-BFGS")
    k == 3 && axislegend(ax, position=:rt)
end
save(joinpath(@__DIR__, "inversion_turing.png"), fig)
