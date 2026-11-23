# Plane-wave initial condition (OpenSWPC pw_mode, m_source pw_setup).

"""
    PlaneWave(wave; depth, length, strike=0.0, dip=0.0, rake=0.0)

Plane P (`wave = :p`) or S (`wave = :s`) wave, as in OpenSWPC (`pw_mode`); use it in `sources`. At the start time the pulse
lies between `depth` and `depth + length` [km] (measured along its direction of propagation) and travels upwards, inclined by
`dip` [°] from the vertical, towards the azimuth `strike` + 180° [° clockwise from north]. `rake` [°] sets the polarization of
an S wave. The shape is the source time function `stf` of the simulation (with a "rise time" of `length`), and the
velocity amplitude is of the order of 1/`length` (in m/s, the OpenSWPC convention).

In 2D the wave travels in the x-z plane (OpenSWPC fixes `strike = 90°`, and `rake = 90°` (P-SV) or `0°` (SH, S waves only)).
Use the PML absorber (the default): its side boundaries let the plane wave pass.
"""
struct PlaneWave
    wave::Symbol
    depth::Float64
    length::Float64
    strike::Float64
    dip::Float64
    rake::Float64
    function PlaneWave(wave::Symbol; depth, length, strike=0.0, dip=0.0, rake=0.0)
        wave in (:p, :s) || error("plane wave must be :p or :s, got :$wave")
        length > 0 || error("plane wave length must be positive")
        return new(wave, depth, length, strike, dip, rake)
    end
end

function Base.show(io::IO, p::PlaneWave)
    @printf(io, "PlaneWave(:%s) from %g km depth, length %g km, strike %g°, dip %g°, rake %g°", p.wave, p.depth, p.length,
            p.strike, p.dip, p.rake)
end

"""
Fields of the plane wave `p` on the local arrays (host): `fields[name][i, (j,) k]` for the names of `F`, from OpenSWPC's
formulas in its frame (x north, y east, z down; in 3D our x is its y). Values are divided by the output `scale`.
"""
function plane_wave_fields(p::PlaneWave, ::Val{mode}, med, g::Grid{N}, off, nl, stf, dt, scale) where {mode,N}
    strike = mode === :p3d ? p.strike : 90.0
    rake = mode === :p3d ? p.rake : mode === :psv ? 90.0 : 0.0
    mode === :sh && p.wave === :p && error("SH simulations have no plane P waves")
    sd, cd, sf, cf, sl, cl = sind(p.dip), cosd(p.dip), sind(strike), cosd(strike), sind(rake), cosd(rake)
    c2d, c2f = cosd(2p.dip), cosd(2strike)
    f(ξ) = moment_rate(stf, ξ, 0.0, p.length)
    names = mode === :psv ? (:Vx, :Vz, :Sxx, :Szz, :Sxz) : mode === :sh ? (:Vy, :Syz, :Sxy) :
            (:Vx, :Vy, :Vz, :Sxx, :Syy, :Szz, :Syz, :Sxz, :Sxy)
    out = Dict(n => zeros(nl...) for n in names)
    d = g.d
    for I in CartesianIndices(nl)
        G = Tuple(I) .+ off                       # global padded index
        ρ, λ, μ = med.rho[G...], med.lam[G...], med.mu[G...]
        v = sqrt(max(p.wave === :p ? (λ + 2μ) / ρ : μ / ρ, 0.0))
        v < eps(Float32) && continue
        c = ntuple(a -> g.beg[a] + (G[a] - H - 0.5) * d[a], N)          # cell centre (x, [y,] depth)
        z0, z1 = c[N] - p.depth, c[N] - p.depth + d[N] / 2
        # OpenSWPC frame: xF north, yF east. 2D: xF = x. 3D: xF = y, yF = x.
        xF0, xF1, yF0, yF1 = N == 3 ? (c[2], c[2] + d[2] / 2, c[1], c[1] + d[1] / 2) : (c[1], c[1] + d[1] / 2, 0.0, 0.0)
        ph(x, y, z) = sd * sf * x - sd * cf * y + cd * z
        shift = dt / 2 * v
        ii, vxF, vyF, vz = f(ph(xF0, yF0, z0)), f(ph(xF1, yF0, z0) + shift), f(ph(xF0, yF1, z0) + shift), f(ph(xF0, yF0, z1) + shift)
        yz, xz, xy = f(ph(xF0, yF1, z1)), f(ph(xF1, yF0, z1)), f(ph(xF1, yF1, z0))
        if p.wave === :p
            V = (-sd * sf * vxF, sd * cf * vyF, -cd * vz)
            S = (-(λ + 2μ * sd^2 * sf^2) * ii, -(λ + 2μ * sd^2 * cf^2) * ii, -(λ + 2μ * cd^2) * ii,
                 2μ * sd * cd * cf * yz, -2μ * sd * cd * sf * xz, 2μ * sd^2 * sf * cf * xy) ./ v   # OpenSWPC: sd*cd*sf in sxy
        else
            V = ((cl * cf + sl * cd * sf) * vxF, (cl * sf - sl * cd * cf) * vyF, -sl * sd * vz)
            S = (2μ * sd * sf * (cl * cf + sl * cd * sf) * ii, -2μ * sd * cf * (cl * sf - sl * cd * cf) * ii, -2μ * cd * sl * sd * ii,
                 μ * (cl * cd * sf - sl * c2d * cf) * yz, μ * (cl * cd * cf + sl * c2d * sf) * xz,
                 -μ * (cl * sd * c2f + 2sl * sd * cd * sf * cf) * xy) ./ v
        end
        # (Vx, Vy, Vz, Sxx, Syy, Szz, Syz, Sxz, Sxy) in the OpenSWPC frame -> ours
        vals = mode === :psv ? (Vx=V[1], Vz=V[3], Sxx=S[1], Szz=S[3], Sxz=S[5]) :
               mode === :sh ? (Vy=V[2], Syz=S[4], Sxy=S[6]) :
               (Vx=V[2], Vy=V[1], Vz=V[3], Sxx=S[2], Syy=S[1], Szz=S[3], Syz=S[5], Sxz=S[4], Sxy=S[6])
        for n in names
            out[n][I] = vals[n] / scale
        end
    end
    return out
end

"Velocity of the plane wave at its top in the middle of the model, over its length: the highest frequency (for the PML)."
function plane_wave_fcut(p::PlaneWave, med, g::Grid{N}) where {N}
    G = (ntuple(a -> (g.n[a] + 1) ÷ 2 + H, N - 1)..., clamp(cell_index(g, N, p.depth), H + 1, g.n[N] + H))
    ρ, λ, μ = med.rho[G...], med.lam[G...], med.mu[G...]
    return sqrt(max(p.wave === :p ? (λ + 2μ) / ρ : μ / ρ, 0.0)) / p.length
end

"""
Linear extrapolation into the ghost cells at the sides of the model (x, and y in 3D) on the ranks at the model edge, so that
plane waves pass the side boundaries (OpenSWPC set_vel_boundary / set_stress_boundary, PML only).
"""
function extrapolate_sides!(arrays, edges)
    for A in arrays, (a, (lo, hi)) in enumerate(edges)
        n = size(A, a)
        lo && (selectdim(A, a, H) .= 2 .* selectdim(A, a, H + 1) .- selectdim(A, a, H + 2))
        hi && (selectdim(A, a, n - H + 1) .= 2 .* selectdim(A, a, n - H) .- selectdim(A, a, n - H - 1))
    end
end
