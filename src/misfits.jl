# Misfit functions for `misfit_gradient`: callables `(syn, obs, t) -> (J, ∂J/∂syn)` on velocity seismograms
# `syn[it, component, station]`, `obs` (same size) [m/s] and the sample times `t` [s]. Any function of this form can be used.

"""
    L2Misfit(; weights=1, window=nothing, invcov=nothing)

Least-squares misfit `J = ½ Σ w (v - v_observed)²` of the samples in the time window.
- `weights`: a number, one per station (vector), per component and station (matrix `nc × nstations`) or per sample (array
  like the seismograms).
- `window`: `nothing` (all samples), `(t1, t2)` [s] for all stations, or a vector of `(t1, t2)`, one per station.
- `invcov`: an inverse covariance matrix of the samples of a trace (`nt × nt`, symmetric), for correlated noise:
  `J = ½ Σ_traces w rᵀ invcov r` with the residuals `r` in the window (`weights` per trace then).
"""
Base.@kwdef struct L2Misfit{W,T,C}
    weights::W = 1.0
    window::T = nothing
    invcov::C = nothing
end

"""
    TravelTimeMisfit(; weights=1, window=nothing)

Cross-correlation travel-time misfit `J = ½ Σ w τ²` over the traces (station and component), with `τ` the time shift
[s] of the synthetic against the observed trace (positive: synthetic late) at the maximum of their cross-correlation in the
time window, refined to a fraction of a sample by a parabola. `weights`: a number, one per station or per component and
station (`nc × nstations`); `window` as for [`L2Misfit`](@ref). Traces without signal are skipped.
"""
Base.@kwdef struct TravelTimeMisfit{W,T}
    weights::W = 1.0
    window::T = nothing
end

"Weights of the shape `(1 or nt, nc, nstations)` from a number, a vector (stations), a matrix (nc × stations) or an array."
expand_weights(w::Number, sz) = fill(Float64(w), 1, sz[2], sz[3])
expand_weights(w::AbstractVector, sz) = (length(w) == sz[3] || error("weights: $(sz[3]) stations, got $(length(w)) values");
                                         repeat(reshape(Float64.(w), 1, 1, :), 1, sz[2], 1))
expand_weights(w::AbstractMatrix, sz) = (size(w) == (sz[2], sz[3]) || error("weights must be $(sz[2]) × $(sz[3]) (components × stations)");
                                         reshape(Float64.(w), 1, sz...))
expand_weights(w::AbstractArray{<:Any,3}, sz) = (size(w) == sz || error("weights must have the size $sz of the seismograms"); Float64.(w))

"Time-window mask `(nt, 1, nstations)`."
window_mask(::Nothing, t, ns) = ones(length(t), 1, ns)
window_mask(w::Tuple, t, ns) = repeat(Float64.(w[1] .<= t .<= w[2]), 1, 1, ns)
window_mask(w::AbstractVector, t, ns) = (length(w) == ns || error("window: one (t1, t2) per station ($ns)");
                                         reshape(reduce(hcat, [Float64.(wi[1] .<= t .<= wi[2]) for wi in w]), length(t), 1, ns))

function (m::L2Misfit)(syn, obs, t)
    mask = window_mask(m.window, t, size(syn, 3))
    r = syn .- obs
    if m.invcov === nothing
        w = expand_weights(m.weights, size(syn)) .* mask
        return sum(w .* r .^ 2) / 2, w .* r
    end
    C = m.invcov
    size(C) == (size(syn, 1), size(syn, 1)) || error("invcov must be $(size(syn, 1)) × $(size(syn, 1)) (samples of a trace)")
    m.weights isa AbstractArray{<:Any,3} && error("L2Misfit with invcov: weights per trace (number, vector or matrix)")
    w = expand_weights(m.weights, size(syn))
    J, adj = 0.0, zero(syn)
    for n in axes(syn, 3), c in axes(syn, 2)
        rm = r[:, c, n] .* mask[:, 1, n]
        Cr = C * rm
        J += w[1, c, n] * sum(rm .* Cr) / 2
        adj[:, c, n] .= w[1, c, n] .* mask[:, 1, n] .* Cr
    end
    return J, adj
end

function (m::TravelTimeMisfit)(syn, obs, t)
    m.weights isa AbstractArray{<:Any,3} && error("TravelTimeMisfit: weights per trace (number, vector or matrix)")
    w = expand_weights(m.weights, size(syn))
    mask = window_mask(m.window, t, size(syn, 3))
    dt = length(t) > 1 ? t[2] - t[1] : 1.0
    J, adj = 0.0, zero(syn)
    for n in axes(syn, 3), c in axes(syn, 2)
        s, o = syn[:, c, n] .* mask[:, 1, n], obs[:, c, n] .* mask[:, 1, n]
        τ, dτ = cc_shift(s, o, dt)
        τ === nothing && continue
        J += w[1, c, n] * τ^2 / 2
        adj[:, c, n] .= w[1, c, n] * τ .* dτ .* mask[:, 1, n]
    end
    return J, adj
end

"""
Time shift `τ` of `s` against `o` at the maximum of their cross-correlation `C(k) = Σₜ s[t+k] o[t]`, refined by a parabola
through `C(k-1), C(k), C(k+1)`, and `∂τ/∂s`; `(nothing, nothing)` without signal.
"""
function cc_shift(s, o, dt)
    n = length(s)
    (all(iszero, s) || all(iszero, o)) && return nothing, nothing
    L = nextpow(2, 2n)
    C = real.(ifft(fft([s; zeros(L - n)]) .* conj.(fft([o; zeros(L - n)]))))
    lag(k) = C[mod(k, L)+1]
    k = argmax(k -> lag(k), -(n - 1):(n - 1))
    Cl(j) = sum(s[t+j] * o[t] for t in max(1, 1 - j):min(n, n - j); init=0.0)    # exact C(j)
    dC(j) = [1 <= u - j <= n ? o[u-j] : 0.0 for u in 1:n]                       # ∂C(j)/∂s
    if abs(k) < n - 1
        cm, c0, cp = Cl(k - 1), Cl(k), Cl(k + 1)
        den = cm - 2c0 + cp
        if den < 0
            δ = (cm - cp) / 2den
            if abs(δ) <= 1
                dδ = ((dC(k - 1) .- dC(k + 1)) .- 2δ .* (dC(k - 1) .- 2 .* dC(k) .+ dC(k + 1))) ./ 2den
                return (k + δ) * dt, dδ .* dt
            end
        end
    end
    return k * dt, zeros(n)
end
