module SeismicWavePCLogDensityProblemsExt

# A WaveformProblem is a LogDensityProblems.jl problem with gradient: log density -Φ(θ) (AdvancedHMC, DynamicHMC, Pathfinder,
# Turing's externalsampler, …).

using SeismicWavePC, LogDensityProblems
const S = SeismicWavePC

LogDensityProblems.capabilities(::Type{<:WaveformProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(prob::WaveformProblem) = length(prob)
LogDensityProblems.logdensity(prob::WaveformProblem, θ) = -objective(prob, θ)
function LogDensityProblems.logdensity_and_gradient(prob::WaveformProblem, θ)
    Φ, g = objective_and_gradient(prob, θ)
    return -Φ, -g
end

end
