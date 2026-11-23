module SeismicWavePCMooncakeExt

# Mooncake (e.g. Turing with AutoMooncake()) uses the adjoint-state gradient (the ChainRules rule of `objective_value`, which
# `objective` calls with a plain vector); the problem itself is not differentiated.

using SeismicWavePC, Mooncake, ChainRulesCore

Mooncake.tangent_type(::Type{<:WaveformProblem}) = Mooncake.NoTangent
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SeismicWavePC.objective_value),WaveformProblem,Vector{Float64}}

end
