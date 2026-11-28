# Point moment-tensor sources and source time functions (ported from OpenSWPC `m_fdtool.f90`).

"""
    MomentTensorSource(x, y, z, M0, M; t0=0.0, trise=1.0, stf=nothing)

Point source at model coordinates `x, y, z` [km] (the `CartData` frame: x east, y north, z up).
`M0` is the scalar moment [N m] (per unit length [N m/m] in 2D) and
`M = (xx, yy, zz, yz, xz, xy)` the moment tensor in the same frame, normalized so that `M0` sets the size.
`t0` is the rupture start time and `trise` the rise time [s]. `stf`: the source time function of this source (as the `stf`
keyword of [`simulate`](@ref), e.g. a [`SampledSTF`](@ref)); `nothing`: the one of the simulation.
"""
struct MomentTensorSource
    x::Float64
    y::Float64
    z::Float64
    M0::Float64
    M::NTuple{6,Float64}
    t0::Float64
    trise::Float64
    stf::Any
end
MomentTensorSource(x, y, z, M0, M::NTuple{6,Float64}, t0, trise) = MomentTensorSource(x, y, z, M0, M, t0, trise, nothing)
MomentTensorSource(x, y, z, M0, M; t0=0.0, trise=1.0, stf=nothing) = MomentTensorSource(x, y, z, M0, Tuple(Float64.(M)), t0, trise, stf)
"The source time function of source `s`: its own or the simulation's `stf`."
source_stf(s, stf) = something(s.stf, stf)

function Base.show(io::IO, s::MomentTensorSource)
    @printf(io, "MomentTensorSource at (%g, %g, %g) km, M0 = %.3e N m (Mw %.2f), t0 = %g s, trise = %g s\n", s.x, s.y, s.z,
            s.M0, (log10(s.M0) - 9.05) / 1.5, s.t0, s.trise)
    @printf(io, "  M (xx, yy, zz, yz, xz, xy) = (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)", s.M...)
end

"""
    ForceSource(x, y, z; fx=0.0, fy=0.0, fz=0.0, t0=0.0, trise=1.0, stf=nothing)
    ForceSource(lon, lat, z, proj::ProjectionPoint; kwargs...)

Point force (the body-force source of OpenSWPC, `bf_mode`) at model coordinates [km] or at a geographic location.
`fx, fy, fz` [N] point east, north and up; the force history is the source time function (`stf`, unit area) times the force.
Forces and moment tensor sources can be used together. `stf`: the source time function of this force (`nothing`: the
simulation's).
"""
struct ForceSource
    x::Float64
    y::Float64
    z::Float64
    F::NTuple{3,Float64}
    t0::Float64
    trise::Float64
    stf::Any
end
ForceSource(x, y, z, F::NTuple{3,Float64}, t0, trise) = ForceSource(x, y, z, F, t0, trise, nothing)
ForceSource(x, y, z; fx=0.0, fy=0.0, fz=0.0, t0=0.0, trise=1.0, stf=nothing) = ForceSource(x, y, z, Float64.((fx, fy, fz)), t0, trise, stf)
ForceSource(lon, lat, z, proj::ProjectionPoint; kwargs...) = ForceSource(project(lon, lat, z, proj)...; kwargs...)

function Base.show(io::IO, s::ForceSource)
    @printf(io, "ForceSource at (%g, %g, %g) km, F = (%.3e, %.3e, %.3e) N, t0 = %g s, trise = %g s", s.x, s.y, s.z, s.F..., s.t0, s.trise)
end

"Model coordinates [km] of a geographic point, using the projection of the model."
function project(lon, lat, z, proj::ProjectionPoint)
    A(v) = fill(Float64(v), 2, 1, 1)    # two copies: GeophysicalModelGenerator 0.10 fails on single-point UTMData
    c = convert2CartData(GeoData(A(lon), A(lat), A(z), (z=A(z),)), proj)
    return c.x.val[1], c.y.val[1], c.z.val[1]
end

"""
    DoubleCoupleSource(x, y, z; Mw, strike, dip, rake, t0=0.0, trise=1.0, stf=nothing)
    DoubleCoupleSource(lon, lat, z, proj::ProjectionPoint; kwargs...)

Double-couple source with moment magnitude `Mw` and fault angles in degrees (strike measured clockwise from north = +y),
at model coordinates [km], or at a geographic location projected with the `ProjectionPoint` of the model.
"""
DoubleCoupleSource(lon, lat, z, proj::ProjectionPoint; kwargs...) = DoubleCoupleSource(project(lon, lat, z, proj)...; kwargs...)

function DoubleCoupleSource(x, y, z; Mw=nothing, M0=nothing, strike, dip, rake, t0=0.0, trise=1.0, stf=nothing)
    (Mw === nothing) == (M0 === nothing) && error("DoubleCoupleSource: give Mw or M0")
    # Aki & Richards components in (north, east, down)
    sd, cd, s2d, c2d = sind(dip), cosd(dip), sind(2dip), cosd(2dip)
    sl, cl = sind(rake), cosd(rake)
    sf, cf, s2f, c2f = sind(strike), cosd(strike), sind(2strike), cosd(2strike)
    mnn = -(sd * cl * s2f + s2d * sl * sf^2)
    mne = sd * cl * c2f + s2d * sl * s2f / 2
    mnd = -(cd * cl * cf + c2d * sl * sf)
    mee = sd * cl * s2f - s2d * sl * cf^2
    med = -(cd * cl * sf - c2d * sl * cf)
    mdd = s2d * sl
    # to (east, north, up)
    M = (mee, mnn, mdd, -mnd, -med, mne)
    return MomentTensorSource(x, y, z, M0 === nothing ? seismic_moment(Mw) : Float64(M0), M, t0, trise, stf)
end

"""
    ExplosionSource(x, y, z; M0 or Mw, t0=0.0, trise=1.0, stf=nothing)
    ExplosionSource(lon, lat, z, proj::ProjectionPoint; kwargs...)

An explosion (isotropic moment tensor) with the diagonal components `M0` [N m] (2D: [N m/m]), or the moment of magnitude `Mw`.
"""
function ExplosionSource(x, y, z; M0=nothing, Mw=nothing, t0=0.0, trise=1.0, stf=nothing)
    (Mw === nothing) == (M0 === nothing) && error("ExplosionSource: give M0 or Mw")
    return MomentTensorSource(x, y, z, M0 === nothing ? seismic_moment(Mw) : Float64(M0), (1.0, 1.0, 1.0, 0.0, 0.0, 0.0), t0, trise, stf)
end
ExplosionSource(lon, lat, z, proj::ProjectionPoint; kwargs...) = ExplosionSource(project(lon, lat, z, proj)...; kwargs...)

"The source time function as injected at the time step of time `t` (`dt`): `moment_rate`, and for `:delta` 1/dt in the
step that contains `t0`."
source_rate(stf, t, t0, tr, dt) = stf === :delta ? (t - dt / 2 <= t0 < t + dt / 2 ? 1 / dt : 0.0) : moment_rate(stf, t, t0, tr)

"Scalar moment [N m] from moment magnitude (same relation as OpenSWPC)."
seismic_moment(Mw) = 10^(1.5Mw + 9.05)

"""
    moment_rate(stf, t, t0, tr)

Normalized moment-rate function (source time function, integral 1) at time `t` of a source with origin time `t0` and rise
time `tr`. `stf` is
- one of `:kupper` (default in OpenSWPC), `:boxcar`, `:triangle`, `:herrmann`, `:cosine`, `:texp`;
- `:ricker`: a Ricker wavelet with the peak frequency `1/tr`, centred at `t0 + tr` (integral 0, peak value `1/tr`; common in
  benchmarks);
- `:gaussian`: a Gaussian centred at `t0 + tr` with width `tr` (`√π/tr · exp(-(π(t - t0 - tr)/tr)²)`, integral 1), and
  `:gaussian_derivative`: its time derivative times `tr/π` (integral 0);
- `:fuchs_mueller`: the Fuchs-Müller wavelet of SOFI, `sin(2πfτ) - sin(4πfτ)/2` for `τ = t - t0` in `[0, 1/f]`, `f = 1/tr`
  (integral 0; times `1/tr`, as `:ricker`); `:sin3` is `:kupper` (SOFI's sin³ wavelet);
- [`BerlageSTF`](@ref) and [`KlauderSTF`](@ref) (SOFI's Berlage and Klauder wavelets, with their parameters);
- `:step`: a unit step at `t0` (e.g. a force switched on: a static load; for moment tensors the moment grows linearly);
- `:delta`: a unit impulse at `t0` (the moment steps from 0 to `M0`; a force gives a unit impulse), in simulations the time
  step that contains `t0` (`1/dt` there);
- any function `stf(t, t0, tr)`, e.g. a [`SampledSTF`](@ref) of sampled values.
"""
moment_rate(stf, t, t0, tr) = stf(t, t0, tr)
function moment_rate(stf::Symbol, t, t0, tr)
    τ = t - t0
    if stf === :gaussian || stf === :gaussian_derivative
        u = π * (τ - tr) / tr
        g = sqrt(π) / tr * exp(-u^2)
        return stf === :gaussian ? g : -2u * g
    end
    if stf === :texp
        return τ >= 0 ? (2π)^2 * τ / tr^2 * exp(-2π * τ / tr) : 0.0
    end
    if stf === :ricker
        a = (π * (τ - tr) / tr)^2
        return (1 - 2a) * exp(-a) / tr
    end
    (0 <= τ <= tr) || return 0.0
    stf === :step && return τ >= 0 ? 1.0 : 0.0
    stf === :sin3 && return moment_rate(:kupper, t, t0, tr)
    if stf === :fuchs_mueller
        (τ < 0 || τ > tr) && return 0.0
        return (sin(2π * τ / tr) - sin(4π * τ / tr) / 2) / tr
    end
    stf === :delta && error("moment_rate(:delta, …) is an impulse: use it as `stf` of a simulation (one time step, 1/dt)")
    stf === :kupper && return 3π * sin(π * τ / tr)^3 / (4tr)
    stf === :boxcar && return 1 / tr
    stf === :cosine && return (1 - cos(2π * τ / tr)) / tr
    stf === :triangle && return τ <= tr / 2 ? 4τ / tr^2 : 4 * (tr - τ) / tr^2
    if stf === :herrmann
        τ < tr / 4 && return 16τ^2 / tr^3
        τ < 3tr / 4 && return -2 * (8 * (τ^2 - τ * tr) + tr^2) / tr^3
        return 16 * (tr - τ)^2 / tr^3
    end
    error("unknown source time function :$stf")
end

"""
    BerlageSTF(; n=1.5, α, φ0=-90)

The Berlage wavelet (Aldridge, 1990; as in SOFI) as a source time function: `τⁿ exp(-ατ) cos(2πfτ + φ0)` for `τ = t - t0 ≥ 0`
with the centre frequency `f = 1/trise` and `φ0` in degrees, scaled so that its envelope peaks at `1/trise`. Use it as `stf`.
"""
Base.@kwdef struct BerlageSTF
    n::Float64 = 1.5
    α::Float64
    φ0::Float64 = -90.0
end
function (b::BerlageSTF)(t, t0, tr)
    τ = t - t0
    τ < 0 && return 0.0
    peak = (b.n / b.α)^b.n * exp(-b.n)                    # (of the envelope τⁿ exp(-ατ))
    return τ^b.n * exp(-b.α * τ) * cosd(360 * τ / tr + b.φ0) / (peak * tr)
end

"""
    KlauderSTF(; fmin, fmax, tsweep, ntwidth=5)

The Klauder wavelet (the autocorrelation of a linear sweep from `fmin` to `fmax` [Hz] of duration `tsweep` [s]; as in
SOFI): `sin(πkτ(T - τ)) / (πkτ T) cos(2π fc τ)`, `k = (fmax - fmin)/T`, `fc = (fmax + fmin)/2`, centred `ntwidth/fc` after
`t0`, zero beyond `ntwidth` centre periods on either side, with cosine tapers over the first and last 20 %; peak `fc`.
`trise` is not used (give `trise = 1/fmax` for the absorbing layers). Use it as `stf`.
"""
Base.@kwdef struct KlauderSTF
    fmin::Float64
    fmax::Float64
    tsweep::Float64
    ntwidth::Float64 = 5.0
end
function (w::KlauderSTF)(t, t0, tr)
    fc, T = (w.fmax + w.fmin) / 2, w.tsweep
    k = (w.fmax - w.fmin) / T
    half = w.ntwidth / fc
    τ = t - t0 - half
    abs(τ) > half && return 0.0
    a = abs(τ) < 1e-12 ? 1.0 : sin(π * k * τ * (T - τ)) / (π * k * τ * T)
    x = (τ + half) / (2half)                              # (0 … 1 across the window)
    taper = x < 0.2 ? (1 - cos(π * x / 0.2)) / 2 : x > 0.8 ? (1 - cos(π * (1 - x) / 0.2)) / 2 : 1.0
    return fc * a * cos(2π * fc * τ) * taper
end

"""
    SampledSTF(t, values)

Source time function (moment rate, or force history) given by `values` at the times `t` [s] after the origin time `t0` of
the source, linearly interpolated, zero outside `t`. For moment tensors and forces the integral should be 1 (`M0` and the
force set the size). Use it as `simulate(...; stf=SampledSTF(t, values))`; the gradient of the misfit with respect to it is
`source_gradients[n].stf` of [`misfit_gradient`](@ref).
"""
struct SampledSTF
    t::Vector{Float64}
    values::Vector{Float64}
    function SampledSTF(t, values)
        length(t) == length(values) >= 2 || error("SampledSTF: t and values must have the same length (at least 2)")
        issorted(t) || error("SampledSTF: the times must be increasing")
        return new(collect(Float64, t), collect(Float64, values))
    end
end
function (s::SampledSTF)(t, t0, tr)
    τ = t - t0
    (τ < s.t[1] || τ > s.t[end]) && return 0.0
    i = clamp(searchsortedlast(s.t, τ), 1, length(s.t) - 1)
    a = (τ - s.t[i]) / (s.t[i+1] - s.t[i])
    return (1 - a) * s.values[i] + a * s.values[i+1]
end
