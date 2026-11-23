module SeismicWavePCChainRulesCoreExt

# Reverse rules for the objective of a WaveformProblem (gradient by the adjoint-state method), for Zygote and other AD tools.

using SeismicWavePC, ChainRulesCore
const S = SeismicWavePC

for f in (:objective, :objective_value)
    @eval function ChainRulesCore.rrule(::typeof(S.$f), prob::WaveformProblem, θ::AbstractVector)
        Φ, g = objective_and_gradient(prob, θ)
        objective_pullback(ΔΦ) = (NoTangent(), NoTangent(), unthunk(ΔΦ) .* g)
        return Φ, objective_pullback
    end
end

end
