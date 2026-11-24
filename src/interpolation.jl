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
function sinc_points(g::Grid{N}, med, pos::NTuple{N,Float64}, cells, shear::Bool; source=false) where {N}
    o = ntuple(a -> sum(c[a] for c in cells) / length(cells), Val(N))     # position of the points relative to their cell
    c0 = point_index(g, pos, o)
    R = CartesianIndices(ntuple(a -> floor(Int, c0[a])-SINC_R+1:floor(Int, c0[a])+SINC_R, Val(N)))
    pts = [J for J in R if valid_point(med, J, cells, shear)]
    return pts, sinc_weights(g, pos, o, R, pts, shear, source), weight_derivatives(g, pos, o, R, pts, shear, source)
end
# (top-level functions, no nested closures: inferable on all Julia versions)

"Is the point `J` (cells `J + cells`) defined: no air cell, or for shear stresses μ > 0 in all of them?"
function valid_point(med, J, cells, shear)
    sz = size(med.mu)
    all(c -> all(1 .<= Tuple(J + c) .<= sz), cells) || return false
    return shear ? all(c -> med.mu[J+c] > 0, cells) : !any(c -> med.lam[J+c] == 0 && med.mu[J+c] == 0, cells)
end

"Continuous index of the position `p` on the grid of points offset by `o` (in depth on a stretched grid: linear between the cell centres)."
point_index(g::Grid{N}, p, o) where {N} =
    ntuple(a -> a == N && !g.uniform ? depth_index(g, p[a]) - o[a] : (p[a] - g.beg[a]) / g.d[a] + H + 0.5 - o[a], Val(N))

"Derivatives of the weights with respect to the position, per axis (central differences), unrolled over the axes."
@generated weight_derivatives(g::Grid{N}, pos, o, R, pts, shear, source) where {N} =
    Expr(:tuple, (:(weight_derivative(g, pos, o, R, pts, shear, source, $a)) for a in 1:N)...)
function weight_derivative(g::Grid{N}, pos, o, R, pts, shear, source, a) where {N}
    h = 1e-6 * g.d[a]
    e = ntuple(==(a), Val(N))
    return (sinc_weights(g, pos .+ h .* e, o, R, pts, shear, source) .- sinc_weights(g, pos .- h .* e, o, R, pts, shear, source)) ./ 2h
end

"Kaiser-windowed sinc weight of the point `J` for the continuous index `c`."
kaiser_weight(J, c) = prod(a -> kaiser_sinc(J[a] - c[a]), 1:length(c))

"Weights of the points `pts` (of the stencil `R`) at the position `p`, scaled to the sum of the full stencil."
function sinc_weights(g::Grid{N}, p, o, R, pts, shear, source) where {N}
    c = point_index(g, p, o)
    S0, S = sum(J -> kaiser_weight(J, c), R), sum(J -> kaiser_weight(J, c), pts; init=0.0)
    S == 0 && shear && return Float64[]          # shear stress in water: zero (not recorded, nothing injected)
    S == 0 && error("a source or station at $p (internal coordinates) lies in the air")
    # a point source in depth on a stretched grid: δ(z - zₛ) = δ(ζ - ζₛ) / z'(ζₛ), relative to the nominal spacing
    f = source && !g.uniform ? g.d[N] / depth_spacing(g, p[N]) : 1.0
    return [kaiser_weight(J, c) * S0 / S * f for J in pts]
end

"Continuous (padded) index of the depth `z` on a stretched grid: `k` at the centre of cell `k`, linear in between."
function depth_index(g::Grid, z)
    k = clamp(searchsortedlast(g.zc, z), 1, length(g.zc) - 1)
    return k + (z - g.zc[k]) / (g.zc[k+1] - g.zc[k])
end
"Spacing dz/dζ at the depth `z` (the cell thickness, linear between the cell centres)."
function depth_spacing(g::Grid, z)
    ζ = depth_index(g, z)
    k = clamp(floor(Int, ζ), 2, length(g.zc) - 1)
    return thickness(g, k) + (ζ - k) * (thickness(g, k + 1) - thickness(g, k))
end

"""
Device tables of the points of this rank (`off`, local size `nl`; only its interior points, others belong to neighbours) of
the interpolation of `items` (internal positions) for every component: `ix` (linear indices) and `w` (weights) per
component, matrices `item × point`, and the derivatives `dw` (per component a tuple over axes, times the grid spacing).
Returns the tables and the items with points on this rank.
"""
function sinc_tables(g::Grid{N}, med, positions, spec, off, nl, dev, ::Type{T}; derivatives::Val{D}=Val(false), items=nothing) where {N,T,D}
    comps = spec.cells
    raw = [component_points(g, med, p, comps, spec.shear, D) for p in positions]    # per item, a tuple over components
    local_(J) = J - CartesianIndex(off)
    owned(J) = all(a -> H < local_(J)[a] <= nl[a] - H, 1:N)
    keep = [map(r -> findall(owned, r[1]), rp) for rp in raw]
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
    dw = D ? ntuple(c -> ntuple(a -> table(c, (rc, m) -> rc[3][a][m] * g.d[a]), N), length(comps)) : nothing
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

# Over the components (tuples of different types), unrolled: inferable on all Julia versions.
"""
The cells of a point as 4 cells (1 or 2 cells repeated: the same mean and validity), so that `sinc_points` has a single type
for all components (Julia 1.10 does not infer it for cell tuples of different lengths in one call tree).
"""
four(c::Tuple) = ntuple(i -> c[mod1(i, length(c))], Val(4))
"`sinc_points` of every component."
@generated component_points(g, med, p, cells::Tuple, shear::Tuple, source) =
    Expr(:tuple, (:(sinc_points(g, med, p, four(cells[$c]), shear[$c]; source)) for c in 1:length(cells.parameters))...)

"Does the interpolation of `pos` have points on this rank (any component of `spec`)?"
sinc_on_rank(g::Grid, med, pos, spec, off, nl) = any_on_rank(g, med, pos, spec.cells, spec.shear, off, nl)
@generated any_on_rank(g, med, pos, cells::Tuple, shear::Tuple, off, nl) =
    foldr((c, rest) -> :(on_rank(g, med, pos, cells[$c], shear[$c], off, nl) || $rest), 1:length(cells.parameters); init=false)
"Has the interpolation of `pos` on the grid of `cells` points on this rank?"
on_rank(g::Grid{N}, med, pos, cells, shear, off, nl) where {N} =
    any(J -> all(a -> H < J[a] - off[a] <= nl[a] - H, 1:N), sinc_points(g, med, pos, four(cells), shear)[1])

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
