# Hamiltonian Monte Carlo with AdvancedHMC.jl directly (without Turing): the same problem as inversion_turing.jl, with the
# prior in the WaveformProblem, which is a LogDensityProblems.jl log density with gradient (adjoint-state method, no AD).
#
#   julia -t auto --project=examples/Inversion examples/Inversion/inversion_hmc.jl [cpu|cuda|metal]
#
include("setup.jl")
using AdvancedHMC, LogDensityProblems

# Gaussian prior N(0, 0.3²) per band: log p and its gradient
prior(θ) = (-sum(abs2, θ) / (2 * 0.3^2), -θ ./ 0.3^2)
posterior = WaveformProblem(params, observed; sources=source, misfit=L2Misfit(weights=1 / σ^2), prior, kw...)

# NUTS with a diagonal mass matrix, adapted (with the step size) during the first 50 iterations.
# Classic HMC with a fixed path: HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(10))).
metric = DiagEuclideanMetric(LogDensityProblems.dimension(posterior))
hamiltonian = Hamiltonian(metric, posterior)
integrator = Leapfrog(find_good_stepsize(hamiltonian, θ0))
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
t = @elapsed samples, stats = sample(hamiltonian, kernel, θ0, 150, adaptor, 50; progress=true)

S = reduce(hcat, samples[51:end])'                  # after adaptation
for k in 1:3
    m, s = sum(S[:, k]) / size(S, 1), sqrt(sum(abs2, S[:, k] .- sum(S[:, k]) / size(S, 1)) / (size(S, 1) - 1))
    @printf("band %d (%4.1f to %4.1f km): posterior %6.3f ± %.3f km/s, true %6.3f\n", k, abs(bands[k][1]), -bands[k][2], m, s, θtrue[k])
end
@printf("AdvancedHMC NUTS: %d samples in %.0f s, mean tree depth %.1f\n", size(S, 1), t, sum(s.tree_depth for s in stats) / length(stats))
