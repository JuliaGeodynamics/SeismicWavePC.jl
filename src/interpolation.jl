# Sources and stations between grid points: Kaiser-windowed sinc interpolation (Hicks, 2002, Geophysics 67, 156–166), with
# a cut-off radius of 4 grid points and the Kaiser shape 6.31 (as in SeismicWaves.jl). Every quantity is spread onto (or
# sampled from) its own staggered grid. Points where the quantity is not defined are dropped and the other weights scaled to
# the same sum: in the air (velocities, normal stresses) and where a cell of a shear stress point has μ = 0 (air, water),
# as a stress injected there would never relax. With no shear stress point left (a source or station in water), the shear
# stress is zero: nothing is injected or recorded.

const SINC_R, SINC_B = 4, 6.31

"Modified Bessel function I₀ (power series)."
function besseli0(x)
    s, t, k = 1.0, 1.0, 1
    while t > 1e-17 * s
        t *= (x / 2k)^2
        s += t
        k += 1
    end
    return s
end
const I0B = besseli0(SINC_B)
"Kaiser-windowed sinc at the distance `x` [grid points]."
kaiser_sinc(x) = abs(x) >= SINC_R ? 0.0 : sinc(x) * besseli0(SINC_B * sqrt(1 - (x / SINC_R)^2)) / I0B

# The staggered grids: the cells that define a point (offsets from its index), whose mean is its position, and whether it
# is a shear stress point (needs μ > 0 in all of them) or another quantity (needs no air cell).
const C2, C3 = CartesianIndex{2}, CartesianIndex{3}
stagger(::Val{:psv}) = (velocity=((C2(0, 0), C2(1, 0)), (C2(0, 0), C2(0, 1))),                       # Vx, Vz
                        stress=((C2(0, 0),), (C2(0, 0),), (C2(0, 0), C2(0, 1), C2(1, 0), C2(1, 1))),  # Sxx, Szz, Sxz
                        shear=(false, false, true), mcols=(1, 3, 5), fcols=(1, 3))
stagger(::Val{:sh}) = (velocity=((C2(0, 0),),),                                                       # Vy
                       stress=((C2(0, 0), C2(0, 1)), (C2(0, 0), C2(1, 0))),                           # Syz, Sxy
                       shear=(true, true), mcols=(4, 6), fcols=(2,))
stagger(::Val{:p3d}) = (velocity=((C3(0, 0, 0), C3(1, 0, 0)), (C3(0, 0, 0), C3(0, 1, 0)), (C3(0, 0, 0), C3(0, 0, 1))),
                        stress=((C3(0, 0, 0),), (C3(0, 0, 0),), (C3(0, 0, 0),),
                                (C3(0, 0, 0), C3(0, 1, 0), C3(0, 0, 1), C3(0, 1, 1)),               # Syz
                                (C3(0, 0, 0), C3(1, 0, 0), C3(0, 0, 1), C3(1, 0, 1)),               # Sxz
                                (C3(0, 0, 0), C3(1, 0, 0), C3(0, 1, 0), C3(1, 1, 0))),              # Sxy
                        shear=(false, false, false, true, true, true), mcols=(1, 2, 3, 4, 5, 6), fcols=(1, 2, 3))

"""
Points (global array indices), weights and their derivatives with respect to the position (per km, internal axes x, [y,]
depth) of the interpolation at `pos` (internal coordinates [km]) on the grid of points defined by the cells `cells`.
"""
function sinc_points(g::Grid{N}, med, pos::NTuple{N,Float64}, cells, shear::Bool) where {N}
    o = ntuple(a -> sum(c[a] for c in cells) / length(cells), N)
    sz = size(med.mu)
    air(J) = med.lam[J] == 0 && med.mu[J] == 0
    inside(J) = all(c -> all(1 .<= Tuple(J + c) .<= sz), cells)
    valid(J) = inside(J) && (shear ? all(c -> med.mu[J+c] > 0, cells) : !any(c -> air(J + c), cells))
    cidx(p) = ntuple(a -> (p[a] - g.beg[a]) / g.d[a] + H + 0.5 - o[a], N)            # continuous index of the point grid
    c0 = cidx(pos)
    R = CartesianIndices(ntuple(a -> floor(Int, c0[a])-SINC_R+1:floor(Int, c0[a])+SINC_R, N))
    pts = [J for J in R if valid(J)]
    function weights(p)
        c = cidx(p)
        w(J) = prod(a -> kaiser_sinc(J[a] - c[a]), 1:N)
        S0, S = sum(w, R), sum(w, pts; init=0.0)
        S == 0 && shear && return Float64[]          # shear stress in water: zero (not recorded, nothing injected)
        S == 0 && error("a source or station at $pos (internal coordinates) lies in the air")
        return [w(J) * S0 / S for J in pts]
    end
    h = 1e-6 .* g.d
    dw = ntuple(a -> (weights(pos .+ h[a] .* ntuple(==(a), N)) .- weights(pos .- h[a] .* ntuple(==(a), N))) ./ 2h[a], N)
    return pts, weights(pos), dw
end

"""
Device tables of the points of this rank (`off`, local size `nl`; only its interior points, others belong to neighbours) of
the interpolation of `items` (internal positions) for every component: `ix` (linear indices) and `w` (weights) per
component, matrices `item × point`, and the derivatives `dw` (per component a tuple over axes, times the grid spacing).
Returns the tables and the items with points on this rank.
"""
function sinc_tables(g::Grid{N}, med, positions, spec, off, nl, dev, ::Type{T}; derivatives=false, items=nothing) where {N,T}
    comps = spec.cells
    raw = [[sinc_points(g, med, p, cells, sh) for (cells, sh) in zip(comps, spec.shear)] for p in positions]
    local_(J) = J - CartesianIndex(off)
    owned(J) = all(a -> H < local_(J)[a] <= nl[a] - H, 1:N)
    keep = [[findall(owned, r[1]) for r in rp] for rp in raw]
    items = something(items, [n for n in eachindex(positions) if any(!isempty, keep[n])])
    Q = max(1, maximum((length(k) for n in items for k in keep[n]); init=1))
    L = LinearIndices(nl)
    function table(c, f)
        A = zeros(T, max(length(items), 1), Q)
        for (r, n) in enumerate(items), (q, m) in enumerate(keep[n][c])
            A[r, q] = f(raw[n][c], m)
        end
        return dev(A)
    end
    ix = ntuple(c -> (A = ones(Int32, max(length(items), 1), Q);
                      for (r, n) in enumerate(items), (q, m) in enumerate(keep[n][c])
                          A[r, q] = L[local_(raw[n][c][1][m])]
                      end; dev(A)), length(comps))
    w = ntuple(c -> table(c, (rc, m) -> rc[2][m]), length(comps))
    dw = derivatives ? ntuple(c -> ntuple(a -> table(c, (rc, m) -> rc[3][a][m] * g.d[a]), N), length(comps)) : nothing
    return (; ix, w, dw), items
end

# ------------------------------------------------------------------ kernels
"Σ w A over the points of item `n` (component tables `ix`, `w`)."
Base.@propagate_inbounds function sinc_sample(A, ix, w, n)
    s = zero(eltype(A))
    for q in axes(w, 2)
        s += w[n, q] * A[ix[n, q]]
    end
    return s
end

"Moment tensor sources: the stress glut on the stress grids (as `inject_*!`)."
Base.@propagate_inbounds function inject_sinc!(Σ, S, cols, it)
    for n in 1:size(S.M, 1), c in eachindex(Σ)
        m = S.M[n, cols[c]] * S.s[n, it]
        A, ix, w = Σ[c], S.ix[c], S.w[c]
        for q in axes(w, 2)
            A[ix[n, q]] -= m * w[n, q]
        end
    end
end

"Buoyancy at the velocity point `J` of the component staggered along `e` (`nothing`: cell centred, SH)."
@inline buoyancy(ρ, J, ::Nothing) = @inbounds 1 / ρ[J]
@inline buoyancy(ρ, J, e) = @inbounds 2 / (ρ[J] + ρ[J+e])

"Forces: added to the velocities, times the buoyancy at the points (as `force_*!`)."
Base.@propagate_inbounds function force_sinc!(V, ρ, es, S, cols, it)
    for n in 1:size(S.F, 1), c in eachindex(V)
        f = S.F[n, cols[c]] * S.s[n, it]
        A, ix, w = V[c], S.ix[c], S.w[c]
        C = CartesianIndices(A)
        for q in axes(w, 2)
            A[ix[n, q]] += f * w[n, q] * buoyancy(ρ, C[ix[n, q]], es[c])
        end
    end
end

"Adjoint of the velocity recording: q -= B w (sign) r at the points of every station."
Base.@propagate_inbounds function residual_sinc!(V, ρ, es, signs, St, res, itw, nown)
    for n in 1:nown, c in eachindex(V)
        r = signs[c] * res[itw, c, n]
        A, ix, w = V[c], St.vi[c], St.vw[c]
        C = CartesianIndices(A)
        for q in axes(w, 2)
            A[ix[n, q]] -= buoyancy(ρ, C[ix[n, q]], es[c]) * w[n, q] * r
        end
    end
end

"""
Source sensitivities (as `source_sens!`): `W[l, col, n]` = Σ w f (f = -η of the stress components for moment tensors,
-q of the velocities for forces) and `X[l, col, ±a]` = ± Σₙ s Σ dw f (derivative of the weights: exact location gradient).
"""
Base.@propagate_inbounds function source_sens_sinc!(W, X, fields, sgn, S, cols, n)
    N = length(S.dw[1])
    for l in 1:size(S.s, 1), c in eachindex(fields)
        A, ix, w = fields[c], S.ix[c], S.w[c]
        W[l, cols[c], n] = sgn * sinc_sample(A, ix, w, l)
        s = S.s[l, n]
        for a in 1:N
            d = sgn * s * sinc_sample(A, ix, S.dw[c][a], l)
            X[l, cols[c], 2a-1] += d
            X[l, cols[c], 2a] -= d
        end
    end
end

"Does the interpolation of `pos` have points on this rank (any component of `spec`)?"
function sinc_on_rank(g::Grid{N}, med, pos, spec, off, nl) where {N}
    any(zip(spec.cells, spec.shear)) do (cells, sh)
        any(J -> all(a -> H < J[a] - off[a] <= nl[a] - H, 1:N), sinc_points(g, med, pos, cells, sh)[1])
    end
end

# kernels of the modes (the stress and velocity components in the order of `stagger`)
inject_sinc_psv!(F, S, it) = inject_sinc!((F.Sxx, F.Szz, F.Sxz), S, (1, 3, 5), it)
inject_sinc_sh!(F, S, it) = inject_sinc!((F.Syz, F.Sxy), S, (4, 6), it)
inject_sinc_3d!(F, S, it) = inject_sinc!((F.Sxx, F.Syy, F.Szz, F.Syz, F.Sxz, F.Sxy), S, (1, 2, 3, 4, 5, 6), it)
force_sinc_psv!(F, M, S, it) = force_sinc!((F.Vx, F.Vz), M.rho, (C2(1, 0), C2(0, 1)), S, (1, 3), it)
force_sinc_sh!(F, M, S, it) = force_sinc!((F.Vy,), M.rho, (nothing,), S, (2,), it)
force_sinc_3d!(F, M, S, it) = force_sinc!((F.Vx, F.Vy, F.Vz), M.rho, (C3(1, 0, 0), C3(0, 1, 0), C3(0, 0, 1)), S, (1, 2, 3), it)
residual_sinc_psv!(Q, M, St, res, itw, nown) = residual_sinc!((Q.Vx, Q.Vz), M.rho, (C2(1, 0), C2(0, 1)), (1, -1), St, res, itw, nown)
residual_sinc_sh!(Q, M, St, res, itw, nown) = residual_sinc!((Q.Vy,), M.rho, (nothing,), (1,), St, res, itw, nown)
residual_sinc_3d!(Q, M, St, res, itw, nown) =
    residual_sinc!((Q.Vx, Q.Vy, Q.Vz), M.rho, (C3(1, 0, 0), C3(0, 1, 0), C3(0, 0, 1)), (1, 1, -1), St, res, itw, nown)
sinc_kernels(mode) = mode === :psv ? (inject_sinc_psv!, force_sinc_psv!, residual_sinc_psv!) :
                     mode === :sh ? (inject_sinc_sh!, force_sinc_sh!, residual_sinc_sh!) : (inject_sinc_3d!, force_sinc_3d!, residual_sinc_3d!)
