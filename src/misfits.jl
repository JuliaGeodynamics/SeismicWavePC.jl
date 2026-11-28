# Misfit functions for `misfit_gradient`: callables `(syn, obs, t) -> (J, ∂J/∂syn)` on velocity seismograms
# `syn[it, component, station]`, `obs` (same size) [m/s] and the sample times `t` [s]. Any function of this form can be used.

"""
    L2Misfit(; weights=1, window=nothing, invcov=nothing, normalize=false)

Least-squares misfit `J = ½ Σ w (v - v_observed)²` of the samples in the time window.
- `weights`: a number, one per station (vector), per component and station (matrix `nc × nstations`) or per sample (array
  like the seismograms).
- `window`: `nothing` (all samples), `(t1, t2)` [s] for all stations, or a vector of `(t1, t2)`, one per station.
- `invcov`: an inverse covariance matrix of the samples of a trace (`nt × nt`, symmetric), for correlated noise:
  `J = ½ Σ_traces w rᵀ invcov r` with the residuals `r` in the window (`weights` per trace then).
- `normalize = true`: divide the misfit of each trace by the energy `Σ v_observed²` of the observed trace in the window
  (traces without signal are skipped), so that distant and near stations count alike.
"""
Base.@kwdef struct L2Misfit{W,T,C}
    weights::W = 1.0
    window::T = nothing
    invcov::C = nothing
    normalize::Bool = false
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
    w = expand_weights(m.weights, size(syn))
    if m.normalize
        E = sum((obs .* mask) .^ 2; dims=1)
        w = w .* ifelse.(E .> 0, 1 ./ E, 0.0)
    end
    if m.invcov === nothing
        w = w .* mask
        return sum(w .* r .^ 2) / 2, w .* r
    end
    C = m.invcov
    size(C) == (size(syn, 1), size(syn, 1)) || error("invcov must be $(size(syn, 1)) × $(size(syn, 1)) (samples of a trace)")
    m.weights isa AbstractArray{<:Any,3} && error("L2Misfit with invcov: weights per trace (number, vector or matrix)")
    J, adj = 0.0, zero(syn)
    for n in axes(syn, 3), c in axes(syn, 2)
        rm = r[:, c, n] .* mask[:, 1, n]
        Cr = C * rm
        J += w[1, c, n] * sum(rm .* Cr) / 2
        adj[:, c, n] .= w[1, c, n] .* mask[:, 1, n] .* Cr
    end
    return J, adj
end

"""
`J = Σ_traces w f(s, o)` and `∂J/∂syn` from the misfit `f(s, o, dt) -> (J, ∂J/∂s)` (or `(0, nothing)` to skip the trace) of
each trace `s`, `o` in its time window, with weights per trace.
"""
function per_trace(f, weights, window, syn, obs, t, name)
    weights isa AbstractArray{<:Any,3} && error("$name: weights per trace (number, vector or matrix)")
    w = expand_weights(weights, size(syn))
    mask = window_mask(window, t, size(syn, 3))
    dt = length(t) > 1 ? t[2] - t[1] : 1.0
    J, adj = 0.0, zero(syn)
    for n in axes(syn, 3), c in axes(syn, 2)
        Jt, a = f(syn[:, c, n] .* mask[:, 1, n], obs[:, c, n] .* mask[:, 1, n], dt)
        a === nothing && continue
        J += w[1, c, n] * Jt
        adj[:, c, n] .= w[1, c, n] .* a .* mask[:, 1, n]
    end
    return J, adj
end

(m::TravelTimeMisfit)(syn, obs, t) = per_trace(m.weights, m.window, syn, obs, t, "TravelTimeMisfit") do s, o, dt
    τ, dτ = cc_shift(s, o, dt)
    τ === nothing ? (0.0, nothing) : (τ^2 / 2, τ .* dτ)
end

"""
    EnvelopeMisfit(; weights=1, window=nothing)

Envelope misfit `J = ½ Σ w (E - E_observed)²` (Bozdağ et al., 2011) with the envelopes `E = √(v² + H[v]²)` (`H`: Hilbert
transform) of the traces in the time window. Insensitive to the phase: robust against cycle skipping (surface waves, poor
starting models), but with little resolution; often used first, followed by [`PhaseMisfit`](@ref) or [`L2Misfit`](@ref).
`weights` per trace (number, per station, or `nc × nstations`), `window` as for [`L2Misfit`](@ref).
"""
Base.@kwdef struct EnvelopeMisfit{W,T}
    weights::W = 1.0
    window::T = nothing
end

(m::EnvelopeMisfit)(syn, obs, t) = per_trace(m.weights, m.window, syn, obs, t, "EnvelopeMisfit") do s, o, dt
    hs = hilbert(s)
    Es = sqrt.(s .^ 2 .+ hs .^ 2)
    r = Es .- sqrt.(o .^ 2 .+ hilbert(o) .^ 2)
    q = ifelse.(Es .> 0, r ./ Es, 0.0)
    sum(abs2, r) / 2, q .* s .- hilbert(q .* hs)       # Hᵀ = -H
end

"""
    PhaseMisfit(; weights=1, window=nothing, waterlevel=0.05)

Instantaneous-phase misfit (Bozdağ et al., 2011) in the exponentiated form of Yuan et al. (2020), free of phase wrapping:
`J = ½ Σ w [(v/E - v_obs/E_obs)² + (H[v]/E - H[v_obs]/E_obs)²]`, the squared difference of `exp(iφ)` of the analytic
signals, with the envelopes `E = √(v² + H[v]² + ε²)` and `ε = waterlevel · max E_observed` of the trace. Independent of
amplitudes (source strength, attenuation, site effects). `weights` and `window` as for [`EnvelopeMisfit`](@ref).
"""
Base.@kwdef struct PhaseMisfit{W,T}
    weights::W = 1.0
    window::T = nothing
    waterlevel::Float64 = 0.05
end

(m::PhaseMisfit)(syn, obs, t) = per_trace(m.weights, m.window, syn, obs, t, "PhaseMisfit") do s, o, dt
    hs, ho = hilbert(s), hilbert(o)
    ε² = (m.waterlevel * sqrt(maximum(o .^ 2 .+ ho .^ 2)))^2
    ε² > 0 || return 0.0, nothing
    Es, Eo = sqrt.(s .^ 2 .+ hs .^ 2 .+ ε²), sqrt.(o .^ 2 .+ ho .^ 2 .+ ε²)
    ra, rb = s ./ Es .- o ./ Eo, hs ./ Es .- ho ./ Eo
    q = (ra .* s .+ rb .* hs) ./ Es .^ 3
    (sum(abs2, ra) + sum(abs2, rb)) / 2, (ra ./ Es .- q .* s) .- hilbert(rb ./ Es .- q .* hs)
end

"""
    TFPhaseMisfit(; width, band=nothing, weights=1, window=nothing, waterlevel=0.01)

Time-frequency phase misfit (Fichtner et al., 2008): `J = ½ Σ w Σ_τ,f W² Δφ(τ, f)²` with the phase difference
`Δφ = arg(S Õ*)` of the Gabor transforms `S`, `Õ` of the synthetic and observed trace (Gaussian windows `exp(-(t-τ)²/2width²)`
[s] every `width/2`), weighted by `W = |Õ| / max |Õ|` to the energetic parts of the observed trace. Only frequencies in
`band = (f1, f2)` [Hz] (default: all) and points with `|S| > waterlevel · max |S|` count. Like a travel time, but per
time and frequency; meant for phase differences below π (a good starting model or [`Filtered`](@ref) data). `width` ≈ the
dominant period. `weights` and `window` as for [`EnvelopeMisfit`](@ref).
"""
Base.@kwdef struct TFPhaseMisfit{W,T,B}
    width::Float64
    band::B = nothing
    weights::W = 1.0
    window::T = nothing
    waterlevel::Float64 = 0.01
end

(m::TFPhaseMisfit)(syn, obs, t) = per_trace(m.weights, m.window, syn, obs, t, "TFPhaseMisfit") do s, o, dt
    n = length(s)
    L = nextpow(2, 2n)
    f = (0:L-1) ./ (L * dt)
    keep = (f .> 0) .& (f .< 1 / 2dt) .& (m.band === nothing ? true : m.band[1] .<= f .<= m.band[2])
    τs = 1:max(1, round(Int, m.width / 2dt)):n
    g = [exp.(-((1:n) .- j) .^ 2 .* dt^2 ./ 2m.width^2) for j in τs]
    S = [fft([s .* gj; zeros(L - n)])[keep] for gj in g]
    O = [fft([o .* gj; zeros(L - n)])[keep] for gj in g]
    Omax, Smax = maximum(maximum.(abs, O)), maximum(maximum.(abs, S))
    (Omax > 0 && Smax > 0) || return 0.0, nothing
    J, a = 0.0, zeros(n)
    for (gj, Sj, Oj) in zip(g, S, O)
        W² = (abs.(Oj) ./ Omax) .^ 2 .* (abs.(Sj) .> m.waterlevel * Smax)
        Δφ = angle.(Sj .* conj.(Oj))
        J += sum(W² .* Δφ .^ 2) / 2
        Y = zeros(ComplexF64, L)
        Y[keep] .= W² .* Δφ ./ ifelse.(W² .> 0, Sj, one(eltype(Sj)))       # ∂Δφ/∂s(t) = Im(g(t) e^{-iωt} / S)
        a .+= gj .* imag.(fft(Y)[1:n])
    end
    J, a
end

"Hilbert transform of a trace, zero-padded to twice its length so that the operator is antisymmetric (`Hᵀ = -H`)."
function hilbert(s)
    n = length(s)
    S = rfft([s; zeros(n)])
    S .*= -im
    S[1] = S[end] = 0                                    # DC and Nyquist
    return irfft(S, 2n)[1:n]
end

"""
    bandpass(v, dt, band; order=4)

Zero-phase Butterworth band-pass filter of seismograms `v[it, …]` (along the first dimension, time step `dt` [s]) with the
corner frequencies `band = (f1, f2)` [Hz]; `f1 = 0` gives a low-pass, `f2 = Inf` a high-pass filter. The squared
Butterworth amplitude (forward and backward filtering) is applied in the frequency domain after zero padding to twice the
length: the filter is symmetric, its own adjoint.
"""
function bandpass(v::AbstractArray, dt, band; order=4)
    f1, f2 = band
    n = size(v, 1)
    f = (0:n) ./ (2n * dt)
    H = @. (f1 > 0 ? 1 / (1 + (f1 / f)^(2order)) : 1.0) * (isfinite(f2) ? 1 / (1 + (f / f2)^(2order)) : 1.0)
    V = reshape(Float64.(v), n, :)
    return reshape(irfft(H .* rfft([V; zeros(size(V))], 1), 2n, 1)[1:n, :], size(v))
end

"""
    Filtered(misfit, band; order=4)

The misfit `misfit` of the [`bandpass`](@ref) filtered synthetic and observed seismograms, for example
`Filtered(L2Misfit(), (0.0, 0.5))`. Inverting first at low frequencies and then with wider bands (frequency continuation)
avoids cycle skipping.
"""
struct Filtered{M,B}
    misfit::M
    band::B
    order::Int
end
Filtered(misfit, band; order=4) = Filtered(misfit, band, order)

function (m::Filtered)(syn, obs, t)
    dt = t[2] - t[1]
    J, adj = m.misfit(bandpass(syn, dt, m.band; m.order), bandpass(obs, dt, m.band; m.order), t)
    return J, bandpass(adj, dt, m.band; m.order)
end

"""
    geographic_weights(points, L)

Weights `wᵢ = 1 / Σⱼ exp(-(dᵢⱼ/L)²)` (Ruan et al., 2019) of points at the distances `dᵢⱼ` [km], normalized to a mean of 1:
clusters of stations (or sources) count about as much as single ones. `points`: stations, sources or anything with the
fields `x, y, z` [km]. Use them as `L2Misfit(weights = geographic_weights(observed.stations, 20))`, or for the shots as a
vector of misfits, one per shot, with weights from the sources.
"""
function geographic_weights(points, L)
    w = [1 / sum(q -> exp(-((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2) / L^2), points) for p in points]
    return w .* (length(w) / sum(w))
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
