# Shared setup of the inversion examples: model, parameters, noisy "observed" seismograms and the problem.
using SeismicWavePC, GeophysicalModelGenerator, LinearAlgebra, Random, Printf
include(joinpath(@__DIR__, "..", "backend.jl"))
T = DEVICE == "metal" ? Float32 : Float64

# model: 6 km × 3 km, 50 m cells, free surface at z = 0; starting model with two layers
x, z = range(0, 6, length=121), range(-3, 0.2, length=65)
X, _, Z = xyz_grid(x, 0, z)
air = Z .> 0
vs0 = ifelse.(air, 0.0, 1.8 .+ 0.2 .* (Z .< -1))
model_with(vs) = CartData(X, zero(X), Z, (; rho=ifelse.(air, 1.0, 2400.0), vp=ifelse.(air, 0.0, 1.8 .* vs0 .+ 0.3), vs))
start = model_with(vs0)

# parameters: changes of vs [km/s] in three depth bands (basis functions)
bands = [(0.0, -0.8), (-0.8, -1.6), (-1.6, -3.1)]
basis = [Float64.((Z .<= z1) .& (Z .> z2) .& .!air) for (z1, z2) in bands]
params = ModelParameters(start, :vs; basis)
θtrue = [0.15, -0.10, 0.20]

# "observed" data: the true model, plus noise
source = DoubleCoupleSource(3.0, 0.0, -2.0; Mw=3.0, strike=0, dip=45, rake=90, trise=0.3)
stations = [Station("S$i", xs, 0.0, 0.0; surface=true) for (i, xs) in enumerate(0.5:0.5:5.5)]
kw = (; dt=0.006, nt=400, T, backend)
observed = simulate(to_model(params, θtrue); kw..., sources=source, stations, verbose=false)
σ = 0.02 * maximum(abs, observed.v)                            # noise: 2 % of the largest amplitude
observed.v .+= σ .* randn(MersenneTwister(1), size(observed.v))

# the problem: Gaussian likelihood (weights 1/σ²), without prior (Turing gets it in its model)
problem = WaveformProblem(params, observed; sources=source, misfit=L2Misfit(weights=1 / σ^2), kw...)
θ0 = initial_parameters(params)

