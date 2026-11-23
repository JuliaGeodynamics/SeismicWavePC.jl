module SeismicWavePCReverseDiffExt

# ReverseDiff (e.g. Turing with AutoReverseDiff()) uses the adjoint-state gradient (the ChainRules rule of `objective`).

using SeismicWavePC, ReverseDiff, ChainRulesCore

ReverseDiff.@grad_from_chainrules SeismicWavePC.objective(prob::WaveformProblem, θ::ReverseDiff.TrackedArray)

end
