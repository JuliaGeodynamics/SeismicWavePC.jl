module SeismicWavePCForwardDiffExt

# ForwardDiff (e.g. Turing's default AD backend) uses the adjoint-state gradient of the objective of a WaveformProblem
# instead of propagating dual numbers through the simulation. The gradient is cached, so the several chunks ForwardDiff
# evaluates for one θ cost one gradient.

using SeismicWavePC, ForwardDiff
using ForwardDiff: Dual, value, partials

function SeismicWavePC.objective(prob::WaveformProblem, θ::AbstractVector{<:Dual{T}}) where {T}
    Φ, g = objective_and_gradient(prob, value.(θ))
    return Dual{T}(Φ, sum(g[i] * partials(θ[i]) for i in eachindex(θ)))
end

end
