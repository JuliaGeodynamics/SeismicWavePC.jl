# Interfaces to inversion and sampling packages: a parametrization of the model by a vector θ, and a problem object with the
# objective Φ(θ) = misfit - log prior and its gradient. Extensions make it a LogDensityProblems.jl problem (AdvancedHMC,
# DynamicHMC, Pathfinder, Turing) and give it a ChainRulesCore rule (Zygote, …); `fg!` serves Optim.jl.

"""
    ModelParameters(model::CartData, fields::Symbol...; mask=nothing, log=false, smooth=nothing, basis=nothing)

Parametrization of `model` by a vector θ: the listed `fields` (`:vp`, `:vs`, `:rho`, `:Qp`, `:Qs`, and for VTI models
`:epsilon`, `:delta`, `:gamma`) vary, everything else
stays as in `model`.
- Without `basis`: one parameter per cell where `mask` is true (default: solid rock, `vs > 0`), the field value or, with
  `log = true`, its logarithm (keeps it positive, often better scaled for optimization and sampling).
- `smooth = L` [km] (or `(Lx, Ly, Lz)`): the model is the initial one plus a smoothed perturbation, `field = field₀ +
  S (θ - θ₀)` in the mask (with `log`: its logarithm), with `S` a Gaussian smoothing `exp(-(d/L)²)` along each axis. The
  gradient is that of the smoothed model (`Sᵀ g`, exact), which removes the small-scale noise of cell-wise gradients near
  the sources and stations: the usual preconditioning of waveform inversions. θ stays one value per cell.
- With `basis`, a vector of arrays of the size of the model: per field, one parameter per basis function, and
  `field = field₀ + Σₖ θₖ basis[k]` (layers, blobs, splines, …: few parameters, e.g. for Hamiltonian Monte Carlo).

The parameters of the fields follow each other in θ. `initial_parameters(p)` returns θ of `model`, `to_model(p, θ)` the model
of θ.
"""
struct ModelParameters
    model::CartData
    fields::Vector{Symbol}
    mask::Union{Nothing,BitArray{3}}
    log::Bool
    smooth::Union{Nothing,NTuple{3,Matrix{Float64}}}       # smoothing matrices along x, y, z
    basis::Union{Nothing,Vector{Array{Float64,3}}}
end

function ModelParameters(model::CartData, fields::Symbol...; mask=nothing, log=false, smooth=nothing, basis=nothing)
    isempty(fields) && error("give the fields to invert for, e.g. ModelParameters(model, :vs)")
    for f in fields
        haskey(model.fields, f) || error("the model has no field $f")
    end
    if basis === nothing
        mask = mask === nothing ? BitArray(model.fields.vs .> 0) : BitArray(mask)
        size(mask) == size(model) || error("mask must have the size $(size(model)) of the model")
        log && any(f -> any(<=(0), model.fields[f][mask]), fields) && error("log = true needs positive values in the mask")
        return ModelParameters(model, collect(fields), mask, log, smoothing_matrices(model, smooth), nothing)
    end
    log && error("log = true is for cell parameters (without basis)")
    smooth === nothing || error("smooth is for cell parameters (without basis)")
    B = [Float64.(b) for b in basis]
    all(b -> size(b) == size(model), B) || error("the basis functions must have the size $(size(model)) of the model")
    return ModelParameters(model, collect(fields), nothing, false, nothing, B)
end

# shortcut: the coordinates along each axis are those of the first grid line (exact on rectilinear grids, approximate on
# curvilinear meshes with topography)
"Row-normalized Gaussian smoothing matrices `exp(-(d/L)²)` along x, y, z (`nothing`: no smoothing)."
function smoothing_matrices(model, L)
    L === nothing && return nothing
    Ls = L isa Number ? (L, L, L) : Tuple(L)
    length(Ls) == 3 && all(>=(0), Ls) || error("smooth = L or (Lx, Ly, Lz) [km], got $L")
    axes_ = (model.x.val[:, 1, 1], model.y.val[1, :, 1], model.z.val[1, 1, :])
    return map(axes_, Ls) do c, l
        K = l > 0 ? exp.(-((c .- c') ./ l) .^ 2) : Float64.(c .== c')
        K ./ sum(K; dims=2)
    end
end

"Apply the matrices `G[d]` along the dimensions `d` of the 3D array `A`."
function along_axes(G, A)
    n1, n2, n3 = size(A)
    B = reshape(G[1] * reshape(A, n1, :), n1, n2, n3)
    B = stack(B[:, :, k] * transpose(G[2]) for k in 1:n3)
    return reshape(reshape(B, :, n3) * transpose(G[3]), n1, n2, n3)
end

"Number of parameters per field."
nper(p::ModelParameters) = p.basis === nothing ? count(p.mask) : length(p.basis)
Base.length(p::ModelParameters) = nper(p) * length(p.fields)

"θ of the model of the parametrization (cell values, or zero basis coefficients)."
function initial_parameters(p::ModelParameters)
    p.basis === nothing || return zeros(length(p))
    return reduce(vcat, [(v = p.model.fields[f][p.mask]; p.log ? log.(v) : Float64.(v)) for f in p.fields])
end

"The model of the parameters θ."
function to_model(p::ModelParameters, θ::AbstractVector)
    length(θ) == length(p) || error("θ has $(length(θ)) entries, the parametrization $(length(p))")
    n = nper(p)
    fields = map(keys(p.model.fields)) do f
        q = findfirst(==(f), p.fields)
        A = p.model.fields[f]
        q === nothing && return A
        θf = view(θ, (q-1)*n+1:q*n)
        B = Float64.(copy(A))
        if p.smooth !== nothing
            v0 = p.log ? log.(B[p.mask]) : B[p.mask]
            δ = zeros(size(B))
            δ[p.mask] .= θf .- v0
            v = v0 .+ along_axes(p.smooth, δ)[p.mask]
            B[p.mask] .= p.log ? exp.(v) : v
        elseif p.basis === nothing
            B[p.mask] .= p.log ? exp.(θf) : θf
        else
            for (k, b) in enumerate(p.basis)
                B .+= θf[k] .* b
            end
        end
        B
    end
    return CartData(p.model.x.val, p.model.y.val, p.model.z.val, NamedTuple{keys(p.model.fields)}(Tuple(fields)))
end

"Gradient with respect to θ from the gradient `grad` (a `CartData`, as from `misfit_gradient`) at the model `m` of θ."
function parameter_gradient(p::ModelParameters, grad::CartData, m::CartData)
    return reduce(vcat, map(p.fields) do f
        g = grad.fields[f]
        p.basis === nothing || return [sum(g .* b) for b in p.basis]
        gv = p.log ? g .* m.fields[f] : Float64.(g)
        p.smooth === nothing && return gv[p.mask]
        along_axes(transpose.(p.smooth), gv .* p.mask)[p.mask]
    end)
end

"""
    WaveformProblem(params::ModelParameters, observed; sources, misfit=L2Misfit(), prior=nothing, kwargs...)

Waveform inversion problem: the objective (negative log posterior)

    Φ(θ) = J(θ) - log prior(θ)

with the misfit `J` between the seismograms of `to_model(params, θ)` and `observed` (a `Seismograms`, or a vector of them for
several shots with a vector of `sources`), as in [`misfit_gradient`](@ref) (`misfit`, and the `simulate` keywords `dt`, `nt`,
`stf`, `backend`, `nproc`, … as `kwargs`).

`prior` is `nothing` (flat) or a function `θ -> (log p, ∇ log p)`. For a Gaussian likelihood with noise σ use
`misfit = L2Misfit(weights = 1 / σ^2)`.

Use it with
- `objective(prob, θ)`, `objective_and_gradient(prob, θ)`;
- Optim.jl: `optimize(Optim.only_fg!(fg!(prob)), θ₀, LBFGS())`;
- LogDensityProblems.jl (with `using LogDensityProblems`, loaded by the samplers): log density `-Φ` and its gradient, for
  AdvancedHMC.jl (HMC, NUTS), DynamicHMC.jl, Pathfinder.jl, and Turing.jl (`externalsampler`);
- ChainRulesCore.jl: a reverse rule for `objective(prob, θ)` (Zygote and other AD tools).

Models outside the valid range (negative velocities, density or Q, or an unstable time step) have `Φ = Inf` (log density
`-Inf`), which samplers reject and line searches avoid.
"""
struct WaveformProblem{O,S,F,P,K}
    params::ModelParameters
    observed::O
    sources::S
    misfit::F
    prior::P
    kwargs::K
    cache::Base.RefValue{Any}         # last (θ, Φ, ∇Φ): AD tools may ask several times for the same θ
end
WaveformProblem(params::ModelParameters, observed; sources, misfit=L2Misfit(), prior=nothing, kwargs...) =
    WaveformProblem(params, observed, sources, misfit, prior, NamedTuple(kwargs), Ref{Any}(nothing))

Base.length(prob::WaveformProblem) = length(prob.params)

"Is the model physical (and can it be simulated)?"
function valid_model(m::CartData)
    f = m.fields
    all(>=(0), f.vp) && all(>=(0), f.vs) && all(>(0), f.rho) || return false
    haskey(f, :Qp) && !all(>(0), f.Qp) && return false
    haskey(f, :Qs) && !all(>(0), f.Qs) && return false
    return true
end
unstable(e) = e isa ErrorException && occursin("unstable time step", e.msg)

logprior(::Nothing, θ) = (0.0, zero(θ))
logprior(prior, θ) = prior(θ)

"Misfit of the seismograms of the model `m` (summed over the shots)."
function data_misfit(prob::WaveformProblem, m)
    shots = prob.observed isa Seismograms ? ((prob.observed, prob.sources, prob.misfit),) :
            zip(prob.observed, prob.sources, shot_misfits(prob.misfit, length(prob.observed)))
    return sum(shots) do (obs, src, misfit)
        r = simulate(m; Base.structdiff(prob.kwargs, NamedTuple{(:mute,)})..., sources=src, stations=obs.stations, verbose=false)
        first(misfit(r.v, obs.v, r.t))
    end
end

"Objective Φ(θ) = misfit - log prior of the problem (one simulation per shot)."
objective(prob::WaveformProblem, θ::AbstractVector) = objective_value(prob, collect(Float64, θ))

# the function that AD rules (ChainRules, Mooncake) replace by the adjoint-state gradient; θ is a plain vector
function objective_value(prob::WaveformProblem, θ::Vector{Float64})
    c = prob.cache[]
    c !== nothing && c[1] == θ && return c[2]
    m = to_model(prob.params, θ)
    valid_model(m) || return Inf
    J = try
        data_misfit(prob, m)
    catch e
        unstable(e) ? Inf : rethrow()
    end
    return J - first(logprior(prob.prior, θ))
end

"Objective Φ(θ) and its gradient (adjoint-state method, about 5 simulations per shot)."
function objective_and_gradient(prob::WaveformProblem, θ::AbstractVector)
    c = prob.cache[]
    c !== nothing && c[1] == θ && return c[2], copy(c[3])
    Φ, g = objective_and_gradient_(prob, θ)
    prob.cache[] = (collect(Float64, θ), Φ, g)
    return Φ, copy(g)
end

function objective_and_gradient_(prob::WaveformProblem, θ::AbstractVector)
    m = to_model(prob.params, θ)
    valid_model(m) || return Inf, zeros(length(θ))
    res = try
        misfit_gradient(m, prob.observed; prob.kwargs..., sources=prob.sources, misfit=prob.misfit)
    catch e
        unstable(e) ? nothing : rethrow()
    end
    res === nothing && return Inf, zeros(length(θ))
    lp, dlp = logprior(prob.prior, θ)
    return res[1] - lp, parameter_gradient(prob.params, res[2], m) .- dlp
end

"""
    fg!(prob::WaveformProblem)

Function `(F, G, θ)` for `Optim.only_fg!`: returns Φ(θ) and writes its gradient into `G` (if not `nothing`).
"""
function fg!(prob::WaveformProblem)
    return function (F, G, θ)
        G === nothing && return objective(prob, θ)
        Φ, g = objective_and_gradient(prob, θ)
        G .= g
        return Φ
    end
end
