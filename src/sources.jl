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

"""
    DoubleCoupleSource(x, y, z; Mw, strike, dip, rake, t0=0.0, trise=1.0)

Double-couple source with moment magnitude `Mw` and fault angles in degrees (strike measured clockwise from north = +y).
"""
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
