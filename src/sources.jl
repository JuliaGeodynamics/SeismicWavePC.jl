# Point moment-tensor sources and source time functions (ported from OpenSWPC `m_fdtool.f90`).

"""
    MomentTensorSource(x, y, z, M0, M; t0=0.0, trise=1.0)

Point source at model coordinates `x, y, z` [km] (the `CartData` frame: x east, y north, z up).
`M0` is the scalar moment [N m] (per unit length [N m/m] in 2D) and
`M = (xx, yy, zz, yz, xz, xy)` the moment tensor in the same frame, normalized so that `M0` sets the size.
`t0` is the rupture start time and `trise` the rise time [s].
"""
struct MomentTensorSource
    x::Float64
    y::Float64
    z::Float64
    M0::Float64
    M::NTuple{6,Float64}
    t0::Float64
    trise::Float64
end
MomentTensorSource(x, y, z, M0, M; t0=0.0, trise=1.0) = MomentTensorSource(x, y, z, M0, Tuple(Float64.(M)), t0, trise)

function Base.show(io::IO, s::MomentTensorSource)
    @printf(io, "MomentTensorSource at (%g, %g, %g) km, M0 = %.3e N m (Mw %.2f), t0 = %g s, trise = %g s\n", s.x, s.y, s.z,
            s.M0, (log10(s.M0) - 9.05) / 1.5, s.t0, s.trise)
    @printf(io, "  M (xx, yy, zz, yz, xz, xy) = (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)", s.M...)
end

"""
    ForceSource(x, y, z; fx=0.0, fy=0.0, fz=0.0, t0=0.0, trise=1.0)
    ForceSource(lon, lat, z, proj::ProjectionPoint; kwargs...)

Point force (the body-force source of OpenSWPC, `bf_mode`) at model coordinates [km] or at a geographic location.
`fx, fy, fz` [N] point east, north and up; the force history is the source time function (`stf`, unit area) times the force.
Forces and moment tensor sources can be used together.
"""
struct ForceSource
    x::Float64
    y::Float64
    z::Float64
    F::NTuple{3,Float64}
    t0::Float64
    trise::Float64
end
ForceSource(x, y, z; fx=0.0, fy=0.0, fz=0.0, t0=0.0, trise=1.0) = ForceSource(x, y, z, Float64.((fx, fy, fz)), t0, trise)
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
    DoubleCoupleSource(x, y, z; Mw, strike, dip, rake, t0=0.0, trise=1.0)
    DoubleCoupleSource(lon, lat, z, proj::ProjectionPoint; kwargs...)

Double-couple source with moment magnitude `Mw` and fault angles in degrees (strike measured clockwise from north = +y),
at model coordinates [km], or at a geographic location projected with the `ProjectionPoint` of the model.
"""
DoubleCoupleSource(lon, lat, z, proj::ProjectionPoint; kwargs...) = DoubleCoupleSource(project(lon, lat, z, proj)...; kwargs...)

function DoubleCoupleSource(x, y, z; Mw, strike, dip, rake, t0=0.0, trise=1.0)
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
    return MomentTensorSource(x, y, z, seismic_moment(Mw), M, t0, trise)
end

"Scalar moment [N m] from moment magnitude (same relation as OpenSWPC)."
seismic_moment(Mw) = 10^(1.5Mw + 9.05)

"""
    moment_rate(stf::Symbol, t, t0, tr)

Normalized moment-rate function (integral 1). `stf` is one of
`:kupper` (default in OpenSWPC), `:boxcar`, `:triangle`, `:herrmann`, `:cosine`, `:texp`.
"""
function moment_rate(stf::Symbol, t, t0, tr)
    τ = t - t0
    if stf === :texp
        return τ >= 0 ? (2π)^2 * τ / tr^2 * exp(-2π * τ / tr) : 0.0
    end
    (0 <= τ <= tr) || return 0.0
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
