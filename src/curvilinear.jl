# 2D P-SV on a terrain-following (curvilinear) grid: summation-by-parts (SBP) finite differences with simultaneous
# approximation terms (SAT). A deformed 2D CartData (vertical columns, a grid line along the surface and along every
# major interface) is split into blocks at the interfaces (found from the material). In every block: collocated
# velocity-stress, dual-pairing upwind SBP operators (SummationByPartsOperators.jl, see sbp_coefficients.jl): the
# divergence (momentum) with D-, the gradient
# (stress) with D+, which are adjoint in the SBP norm, so that the semi-discrete energy changes only through the boundary
# terms; SAT terms make the top a free surface, couple the blocks (continuity of velocity and traction, energy-neutral)
# and absorb at the sides and the bottom (impedance, σ·n = -Z v). Leapfrog in time (velocities at half steps), the time
# step from the spectral radius of the operator. Attenuation as in the staggered solver (generalized Zener body).

include("sbp_coefficients.jl")

# ------------------------------------------------------------------ SBP operators on a 1D grid of n points

"Nodes of the SBP grid of `n` points on [a, b] and its interior spacing h."
function sbp_grid(a, b, n)
    n >= SBP_NMIN || error("an SBP grid needs at least $SBP_NMIN points")
    h = (b - a) / (2SBP_P[end] + n - 2SBP_NP + 1)
    x = [a + h * (j <= SBP_NP ? SBP_P[j] : SBP_P[end] + (j - SBP_NP)) for j in 1:n]
    for j in 1:SBP_NP
        x[n+1-j] = b - h * SBP_P[j]
    end
    return x, h
end
"Weights of the SBP norm (quadrature) on a grid of `n` points with interior spacing `h`."
sbp_weights(n, h) = [h * (j <= SBP_NP ? SBP_W[j] : j > n - SBP_NP ? SBP_W[n+1-j] : 1.0) for j in 1:n]

@assert length(SBP_OM) == 8 && length(SBP_OP) == 8                  # (the unrolled interior sums below)
"h × the boundary row `i` (1…NB) of D₋ (`tab = :LM`) or D₊ (`:LP`) applied to `g`: an if-chain over the rows with the
coefficients as literals (a runtime index into the tables would put them on the stack at every call; also for GPUs)."
@generated function boundary_row(::Val{tab}, g::G, i, ::Type{T}) where {tab,G,T}
    rows = tab === :LM ? SBP_LM : SBP_LP
    ex = :(zero(T))
    for r in length(rows):-1:1
        terms = [:($(T(rows[r][j])) * g($j)) for j in eachindex(rows[r]) if rows[r][j] != 0]
        ex = :(i == $r ? $(Expr(:call, :+, terms...)) : $ex)
    end
    return ex
end
"h (D₋ g)(i) on `n` points; `g(j)` is the value at node j."
@inline function dminus(g::G, i, n) where {G}
    T = typeof(g(i))
    i <= SBP_NB && return boundary_row(Val(:LM), g, i, T)
    i > n - SBP_NB && return -boundary_row(Val(:LP), j -> g(n + 1 - j), n + 1 - i, T)
    return dminus_int(g, i, T)
end
"h (D₊ g)(i) on `n` points."
@inline function dplus(g::G, i, n) where {G}
    T = typeof(g(i))
    i <= SBP_NB && return boundary_row(Val(:LP), g, i, T)
    i > n - SBP_NB && return -boundary_row(Val(:LM), j -> g(n + 1 - j), n + 1 - i, T)
    return dplus_int(g, i, T)
end

"h (D₋ g)(i) and h (D₊ g)(i) away from the boundaries (no branches: the interior of a block vectorizes)."
@inline function dminus_int(g::G, i, ::Type{T}) where {G,T}
    s = zero(T)
    Base.Cartesian.@nexprs 8 m -> (s += T(SBP_CM[m]) * g(i + SBP_OM[m]))
    return s
end
@inline function dplus_int(g::G, i, ::Type{T}) where {G,T}
    s = zero(T)
    Base.Cartesian.@nexprs 8 m -> (s += T(SBP_CP[m]) * g(i + SBP_OP[m]))
    return s
end

# ------------------------------------------------------------------ the mesh

"""
    mesh = curvilinear_mesh(x, surfaces; dz, bottom)

A deformed 2D `CartData` (size nx × 1 × nz, km) for `simulate` on a terrain-following grid: one grid line follows each
of the `surfaces`, top first: the topography, then interfaces with a jump in the material (e.g. a Moho), each a function
z(x) [km] or a surface `CartData` (nx × 1 × 1, linearly interpolated). The bottom is flat at `bottom` [km]; between two
surfaces z is blended linearly. The columns are vertical, at the nodes of the SBP operator over `first(x)`…`last(x)` with
`length(x)` points (equidistant except the outermost few); between two surfaces at least 13 nodes, with about `dz` [km]
spacing where the layer is thickest.

Fill the fields `rho` [kg/m³], `vp`, `vs` [km/s] (and optionally `Qp`, `Qs`) at the nodes with the usual GMG tools,
e.g. layers with `Z .< moho.(X)` and a magma chamber with `add_ellipsoid!`, then `simulate(addfield(mesh, (; rho, vp, vs)); ...)`.
"""
function curvilinear_mesh(x, surfaces; dz, bottom)
    nx = length(x)
    gx, _ = sbp_grid(Float64(first(x)), Float64(last(x)), nx)
    surf(f::Function) = f.(gx)
    function surf(c::CartData)                                          # linear interpolation of a surface CartData
        xs, zs = c.x.val[:, 1, 1], c.z.val[:, 1, 1]
        p = sortperm(xs)
        xs, zs = xs[p], zs[p]
        lin(v) = (j = clamp(searchsortedlast(xs, v), 1, length(xs) - 1); zs[j] + (zs[j+1] - zs[j]) * (v - xs[j]) / (xs[j+1] - xs[j]))
        return lin.(gx)
    end
    S = [[surf(f) for f in surfaces]; [fill(Float64(bottom), nx)]]       # top to bottom
    Zs = Matrix{Float64}[]
    for b in length(S)-1:-1:1                                          # bottom first (z increases with the index)
        top, bot = S[b], S[b+1]
        all(top .> bot) || error("curvilinear_mesh: surfaces $b and $(b+1) touch or cross")
        nz = max(SBP_NMIN, ceil(Int, maximum(top .- bot) / dz) + 1)
        η, _ = sbp_grid(0.0, 1.0, nz)
        Zb = [bot[i] + (top[i] - bot[i]) * e for i in 1:nx, e in η]
        Zb[:, end] .= top                                              # exactly on the surface
        push!(Zs, isempty(Zs) ? Zb : Zb[:, 2:end])                     # the shared row once
    end
    Z2 = reduce(hcat, Zs)
    nz = size(Z2, 2)
    X = [gx[i] for i in 1:nx, _ in 1:1, _ in 1:nz]
    Z = reshape(Z2, nx, 1, nz)
    return CartData(X, zero(X), Z, (; z=copy(Z)))
end

"Is the 2D `CartData` a deformed (terrain-following) mesh, i.e. do its rows of nodes vary in depth along x?"
function is_curvilinear(model::CartData)
    size(model)[2] == 1 || return false
    Z = model.z.val
    return any(k -> maximum(view(Z, :, 1, k)) - minimum(view(Z, :, 1, k)) > 1e-9 * (maximum(Z) - minimum(Z)), axes(Z, 3))
end

"""
    rows, below = find_blocks(Z, vp, vs; jump=0.05)

Interfaces of a deformed mesh (nx × nz, z increasing with the index) from its material: a row r is an interface if vp or
vs jumps along the full width, in every column between r - 1 and r or between r and r + 1; a jump is a relative change of
more than `jump` and more than 3 times the changes between the neighbouring rows. Jumps over part of the width (a magma
chamber) and gradients (also steep ones) are no interfaces. If all columns jump between the
same two rows, the mesh decides (the shape of the vertical spacing changes at the grid line that follows the interface);
if it does not, the interface is the upper row (GMG: a node exactly on a surface is not below it). Returns the rows, and
for each whether each column's node has the material below the interface (true) or above it.
"""
function find_blocks(Z, vp, vs; jump=0.05)
    rel(a, b) = abs(a - b) / max(abs(a), abs(b))
    nx, nz = size(vp)
    # a jump: larger than `jump` and than 3 times the changes between the neighbouring rows (not a steep gradient)
    ch(A, i, k) = 1 <= k <= nz - 1 ? rel(A[i, k], A[i, k+1]) : 0.0
    isjump(A, i, k) = ch(A, i, k) > jump && ch(A, i, k) > 3max(ch(A, i, k - 1), ch(A, i, k + 1))
    J = [isjump(vp, i, k) || isjump(vs, i, k) for i in 1:nx, k in 1:nz-1]
    cand = [r for r in 2:nz-1 if all(i -> J[i, r-1] || J[i, r], 1:nx)]
    shape(k) = (d = Z[:, k+1] .- Z[:, k]; d ./ sum(d))
    kink(k) = 2 <= k <= nz - 1 ? maximum(abs, shape(k) .- shape(k - 1)) * nx : 0.0
    rows = Int[]
    for r in cand
        (r in rows || r - 1 in rows) && continue
        push!(rows, r + 1 in cand && all(i -> J[i, r], 1:nx) ? (kink(r) > 3kink(r + 1) ? r : r + 1) : r)
    end
    return rows, [[J[i, r] for i in 1:nx] for r in rows]
end

# ------------------------------------------------------------------ kernels (one node (i, k) of a block)

@inline orzero(x, I) = x[I]
@inline orzero(::Nothing, I) = false
@inline cmemvar!(R, I, f, vc, dt) = memvar!(R, I, f, vc, dt)
@inline cmemvar!(R, I, f, ::Nothing, dt) = zero(f)
@inline cmemvar!(::Nothing, I, f, ::Nothing, dt) = zero(f)

# Momentum: ρ J ∂t v_i = Dξ(Jξ_j σ_ij) + Dη(Jη_j σ_ij), Jξ = (J, 0), Jη = (Jx, 1), with SAT terms on the boundary nodes:
# the traction removed (free surface on top; free or absorbing at the sides and bottom), half the traction mismatch at
# interfaces (`Bd`, `Bu`: the blocks below and above, `nothing` at the bottom and top). The impedance damping `a` is
# implicit (trapezoidal): unconditionally stable. `damp = false` leaves it out (power iteration).
# `Val(true)`: a node away from the sides (interior stencils in x, no side terms): the boundary rows then vectorize.
@inline Base.@propagate_inbounds function momentum_cv!(B, Bd, Bu, dt, damp, ::Val{xint}, i, k) where {xint}
    (; V1, V2, S11, S22, S12, J, Jx, iρJ, a1, a2, hξ, hη) = B
    nx, nz = size(V1)
    T = eltype(V1)
    w = T(SBP_W[1])
    g1, g2 = j -> @inbounds(J[j, k] * S11[j, k]), j -> @inbounds(J[j, k] * S12[j, k])
    d1 = (xint ? dminus_int(g1, i, T) : dminus(g1, i, nx)) / hξ + dminus(j -> @inbounds(Jx[i, j] * S11[i, j] + S12[i, j]), k, nz) / hη
    d2 = (xint ? dminus_int(g2, i, T) : dminus(g2, i, nx)) / hξ + dminus(j -> @inbounds(Jx[i, j] * S12[i, j] + S22[i, j]), k, nz) / hη
    if xint
    elseif i == 1
        d1 += J[i, k] * S11[i, k] / (w * hξ); d2 += J[i, k] * S12[i, k] / (w * hξ)
    elseif i == nx
        d1 -= J[i, k] * S11[i, k] / (w * hξ); d2 -= J[i, k] * S12[i, k] / (w * hξ)
    end
    if k == nz
        g1, g2 = Jx[i, k] * S11[i, k] + S12[i, k], Jx[i, k] * S12[i, k] + S22[i, k]
        if Bu === nothing
            d1 -= g1 / (w * hη); d2 -= g2 / (w * hη)
        else
            d1 -= (g1 - (Bu.Jx[i, 1] * Bu.S11[i, 1] + Bu.S12[i, 1])) / (2w * hη)
            d2 -= (g2 - (Bu.Jx[i, 1] * Bu.S12[i, 1] + Bu.S22[i, 1])) / (2w * hη)
        end
    end
    if k == 1
        g1, g2 = Jx[i, k] * S11[i, k] + S12[i, k], Jx[i, k] * S12[i, k] + S22[i, k]
        if Bd === nothing
            d1 += g1 / (w * hη); d2 += g2 / (w * hη)
        else
            n = size(Bd.V1, 2)
            d1 -= ((Bd.Jx[i, n] * Bd.S11[i, n] + Bd.S12[i, n]) - g1) / (2w * hη)
            d2 -= ((Bd.Jx[i, n] * Bd.S12[i, n] + Bd.S22[i, n]) - g2) / (2w * hη)
        end
    end
    b1, b2 = damp ? (a1[i, k] * dt / 2, a2[i, k] * dt / 2) : (zero(T), zero(T))
    V1[i, k] = ((1 - b1) * V1[i, k] + dt * iρJ[i, k] * d1) / (1 + b1)
    V2[i, k] = ((1 - b2) * V2[i, k] + dt * iρJ[i, k] * d2) / (1 + b2)
end

# The interior of a block (no boundary closures, SAT terms or damping): the same updates without branches.
@inline Base.@propagate_inbounds function momentum_cv_int!(B, dt, i, k)
    (; V1, V2, S11, S22, S12, J, Jx, iρJ, hξ, hη) = B
    T = eltype(V1)
    d1 = dminus_int(j -> @inbounds(J[j, k] * S11[j, k]), i, T) / hξ + dminus_int(j -> @inbounds(Jx[i, j] * S11[i, j] + S12[i, j]), k, T) / hη
    d2 = dminus_int(j -> @inbounds(J[j, k] * S12[j, k]), i, T) / hξ + dminus_int(j -> @inbounds(Jx[i, j] * S12[i, j] + S22[i, j]), k, T) / hη
    V1[i, k] += dt * iρJ[i, k] * d1
    V2[i, k] += dt * iρJ[i, k] * d2
end
@inline Base.@propagate_inbounds function stress_cv_int!(B, vc, dt, i, k)
    (; V1, V2, S11, S22, S12, J, Jx, lam, mu, hξ, hη) = B
    T = eltype(V1)
    a = dplus_int(j -> @inbounds(V1[j, k]), i, T) / hξ
    b = dplus_int(j -> @inbounds(V1[i, j]), k, T) / hη
    c = dplus_int(j -> @inbounds(V2[j, k]), i, T) / hξ
    e = dplus_int(j -> @inbounds(V2[i, j]), k, T) / hη
    stress_update_cv!(B, vc, dt, i, k, a, b, c, e)
end

# the interior as a region of the same kernels (one launch for all regions of a block, see foreach_region!)
@inline Base.@propagate_inbounds momentum_cv!(B, Bd, Bu, dt, damp, ::Val{:int}, i, k) = momentum_cv_int!(B, dt, i, k)
@inline Base.@propagate_inbounds stress_cv!(B, Bd, Bu, vc, dt, ::Val{:int}, i, k) = stress_cv_int!(B, vc, dt, i, k)

# Stress: the gradient ∂x = Dξ + (Jx / J) Dη, ∂z = Dη / J with D₊ (half the velocity mismatch at interfaces added to Dη v),
# and the stress update of the staggered solver (memory variables of the generalized Zener body if `vc !== nothing`).
@inline Base.@propagate_inbounds function stress_cv!(B, Bd, Bu, vc, dt, ::Val{xint}, i, k) where {xint}
    (; V1, V2, hξ, hη) = B
    nx, nz = size(V1)
    T = eltype(V1)
    w = T(SBP_W[1])
    g1, g2 = j -> @inbounds(V1[j, k]), j -> @inbounds(V2[j, k])
    a = (xint ? dplus_int(g1, i, T) : dplus(g1, i, nx)) / hξ
    b = dplus(j -> @inbounds(V1[i, j]), k, nz) / hη
    c = (xint ? dplus_int(g2, i, T) : dplus(g2, i, nx)) / hξ
    e = dplus(j -> @inbounds(V2[i, j]), k, nz) / hη
    if k == nz && Bu !== nothing
        b -= (V1[i, k] - Bu.V1[i, 1]) / (2w * hη); e -= (V2[i, k] - Bu.V2[i, 1]) / (2w * hη)
    end
    if k == 1 && Bd !== nothing
        n = size(Bd.V1, 2)
        b -= (Bd.V1[i, n] - V1[i, k]) / (2w * hη); e -= (Bd.V2[i, n] - V2[i, k]) / (2w * hη)
    end
    stress_update_cv!(B, vc, dt, i, k, a, b, c, e)
end
"The stress update from Dξ v1, Dη v1, Dξ v2, Dη v2 (as the staggered solver: generalized Zener body if `vc !== nothing`)."
@inline Base.@propagate_inbounds function stress_update_cv!(B, vc, dt, i, k, a, b, c, e)
    (; S11, S22, S12, J, Jx, lam, mu) = B
    I = CartesianIndex(i, k)
    q = Jx[I] / J[I]
    exx, ezz, exz = a + q * b, e / J[I], b / J[I] + c + q * e
    mu2 = 2mu[I]
    l2m = lam[I] + mu2
    tp, ts = orzero(B.tp, I), orzero(B.ts, I)
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    div = exx + ezz
    Rx = cmemvar!(memory(B, Val(:R11)), I, l2m * tp * div - mu2 * ts * ezz, vc, dt)
    Rz = cmemvar!(memory(B, Val(:R22)), I, l2m * tp * div - mu2 * ts * exx, vc, dt)
    Rxz = cmemvar!(memory(B, Val(:R12)), I, mu[I] * ts * exz, vc, dt)
    S11[I] += (l2m * tp1 * div - mu2 * ts1 * ezz + Rx) * dt
    S22[I] += (l2m * tp1 * div - mu2 * ts1 * exx + Rz) * dt
    S12[I] += (mu[I] * ts1 * exz + Rxz) * dt
end

# Sources (lists of the nodes they reach: block, node, weight, source): forces on the velocities after the momentum
# update, moment tensors (stress glut) on the stresses after the stress update. Sequential: sources may share nodes.
Base.@propagate_inbounds function force_cv!(blocks, S, it, dt)
    for m in 1:length(S.i)
        B = blocks[S.b[m]]
        i, k, n = S.i[m], S.k[m], S.n[m]
        s = dt * S.c[m] * S.s[n, it] * B.iρJ[i, k]
        B.V1[i, k] += S.F[n, 1] * s
        B.V2[i, k] += S.F[n, 2] * s
    end
end
Base.@propagate_inbounds function moment_cv!(blocks, S, it)
    for m in 1:length(S.i)
        B = blocks[S.b[m]]
        i, k, n = S.i[m], S.k[m], S.n[m]
        s = S.c[m] * S.s[n, it]
        B.S11[i, k] -= S.M[n, 1] * s
        B.S22[i, k] -= S.M[n, 2] * s
        B.S12[i, k] -= S.M[n, 3] * s
    end
end

# Stations: Lagrange interpolation (6 × 6 nodes of their block).
Base.@propagate_inbounds function sample_cv(A, St, n)
    s = zero(eltype(A))
    for c in 1:6, a in 1:6
        s += St.wx[a, n] * St.wz[c, n] * A[St.i0[n]+a-1, St.k0[n]+c-1]
    end
    return s
end
"v^{n-1/2} at the stations (before the momentum update of a recording step)."
Base.@propagate_inbounds function vprev_cv!(prev, blocks, St, n)
    B = blocks[St.b[n]]
    prev[1, n] = sample_cv(B.V1, St, n)
    prev[2, n] = sample_cv(B.V2, St, n)
end
"After the momentum update: record v(tₙ) = (v^{n-1/2} + v^{n+1/2})/2, u(tₙ), σⁿ and the strain (from σ, unrelaxed moduli) at the
recording steps; integrate the displacement u += v^{n+1/2} dt."
Base.@propagate_inbounds function record_cv!(rec, acc, prev, blocks, St, itw, dt, n)
    B = blocks[St.b[n]]
    v1, v2 = sample_cv(B.V1, St, n), sample_cv(B.V2, St, n)
    if itw > 0
        s11, s22, s12 = sample_cv(B.S11, St, n), sample_cv(B.S22, St, n), sample_cv(B.S12, St, n)
        l, m = sample_cv(B.lam, St, n), sample_cv(B.mu, St, n)
        d = 4m * (l + m)
        q = ((prev[1, n] + v1) / 2, (prev[2, n] + v2) / 2, acc[1, n], acc[2, n], s11, s22, s12,
             ((l + 2m) * s11 - l * s22) / d, ((l + 2m) * s22 - l * s11) / d, s12 / (2m))
        for j in 1:10
            rec[itw, j, n] = q[j]
        end
    end
    acc[1, n] += v1 * dt
    acc[2, n] += v2 * dt
end

# ------------------------------------------------------------------ the solver

"Lagrange weights of the `np` nodes of the grid `g` around `x0`: first index and weights."
function lagrange_weights(g, x0; np=6)
    i0 = clamp(searchsortedlast(g, x0) - np ÷ 2 + 1, 1, length(g) - np + 1)
    return i0, [prod((x0 - g[m]) / (g[j] - g[m]) for m in i0:i0+np-1 if m != j) for j in i0:i0+np-1]
end
"A discrete delta at `x0` on the grid `g` with the norm `H`: Kaiser-windowed sinc where the grid is uniform around it,
otherwise Lagrange weights (moments up to degree 5). Returns (nodes, weights / H)."
function delta_weights(g, H, h, x0)
    c = clamp(searchsortedlast(g, x0), 1, length(g) - 1)
    pos = c + (x0 - g[c]) / (g[c+1] - g[c])
    idx = max(1, ceil(Int, pos - SINC_R)):min(length(g), floor(Int, pos + SINC_R))
    if first(idx) > SBP_NP + 1 && last(idx) < length(g) - SBP_NP
        return collect(idx), [kaiser_sinc(pos - j) / H[j] for j in idx]
    end
    i0, w = lagrange_weights(g, x0)
    return collect(i0:i0+5), [w[a] / H[i0+a-1] for a in 1:6]
end

"""
    simulate_curvilinear(model; tmax or nt, sources, ...)

2D P-SV on a deformed (terrain-following) `CartData` (see [`curvilinear_mesh`](@ref)); called by `simulate` for such a
model. Keywords as in `simulate`, and: `dt = nothing` (from the spectral radius of the operator: 90 % of the leapfrog
limit), `tmax` (duration) instead of `nt`, `absorber = :impedance` (or `:none`: free sides and bottom), `jump = 0.05`
(relative jump of vp or vs that makes a grid row an interface, see `find_blocks`).
"""
function simulate_curvilinear(model::CartData; dt=nothing, nt=nothing, tmax=nothing, sources, stations=Station[], stf=:kupper,
                              tbeg=0.0, absorber=:impedance, fq_min=0.05, fq_max=5.0, fq_ref=1.0, record_every=1,
                              density_unit=:kg_m3, snapshot_every=0, snapshot_dir="snapshots", callback=nothing, backend=CPU(),
                              T::Type{FT}=Float32, verbose=true, wavefield=true, jump=0.05, wave=:psv, interpolation=:sinc,
                              free_surface=:openswpc, kwargs...) where {FT}
    isempty(kwargs) || error("simulate on a curvilinear grid: not (yet) supported: $(join(keys(kwargs), ", "))")
    wave === :psv || error("curvilinear grids: P-SV only (wave = :psv)")
    absorber in (:impedance, :none) || error("curvilinear grids absorb with impedance boundary conditions: absorber = :impedance (or :none)")
    srcs = sources isa AbstractVector ? sources : [sources]
    all(s -> s isa Union{MomentTensorSource,ForceSource}, srcs) || error("curvilinear grids: MomentTensorSource and ForceSource only")
    wall = time()
    dev(A) = (B = KernelAbstractions.allocate(backend, eltype(A), size(A)); copyto!(B, A); B)
    # the mesh (SI) and the material
    X, Z = model.x.val, model.z.val
    nx, nzt = size(X, 1), size(X, 3)
    all(k -> view(X, :, 1, k) == view(X, :, 1, 1), 1:nzt) || error("curvilinear grid: the columns of the mesh must be vertical")
    gξ = 1e3 .* X[:, 1, 1]
    gx, hξ = sbp_grid(gξ[1], gξ[end], nx)
    maximum(abs, gx .- gξ) < 1e-6 * (gξ[end] - gξ[1]) ||
        error("curvilinear grid: x must be the nodes of the SBP grid (make the mesh with curvilinear_mesh)")
    Zm = 1e3 .* Z[:, 1, :]
    all(>(0), diff(Zm; dims=2)) || error("curvilinear grid: z must increase with the row index (and rows must not touch)")
    f = model.fields
    haskey(f, :rho) && haskey(f, :vp) && haskey(f, :vs) || error("the model needs fields rho, vp and vs")
    ρ = f.rho[:, 1, :] .* (density_unit === :g_cm3 ? 1e3 : 1.0)
    vp, vs = 1e3 .* f.vp[:, 1, :], 1e3 .* f.vs[:, 1, :]
    minimum(vs) > 0 || error("curvilinear grids: solids only (vs > 0 everywhere)")
    qp = haskey(f, :Qp) ? Float64.(f.Qp[:, 1, :]) : fill(Inf, nx, nzt)
    qs = haskey(f, :Qs) ? Float64.(f.Qs[:, 1, :]) : fill(Inf, nx, nzt)
    attenuation = any(isfinite, qp) || any(isfinite, qs)
    ts = relaxation_times(fq_min, fq_max)
    zeta = constq_zeta(fq_min, fq_max, ts)
    # blocks: split at the interfaces; at an interface row each block takes the material of its side
    rows, below = find_blocks(Zm, vp, vs; jump)
    edges = [1; rows; nzt]
    any(b -> edges[b+1] - edges[b] + 1 < SBP_NMIN, 1:length(edges)-1) &&
        error("curvilinear grid: every block (between interfaces) needs at least $SBP_NMIN rows; interfaces at rows $rows")
    Hξ = sbp_weights(nx, hξ)
    hostblocks = map(1:length(edges)-1) do b
        r = edges[b]:edges[b+1]
        m = map(A -> copy(A[:, r]), (ρ, vp, vs, qp, qs))
        for (j, side, src) in ((b - 1, false, 2), (b, true, length(r) - 1))
            1 <= j <= length(rows) || continue
            kk = side ? length(r) : 1
            for i in 1:nx
                below[j][i] == side && continue
                foreach(A -> A[i, kk] = A[i, src], m)
            end
        end
        bρ, bvp, bvs, bqp, bqs = m
        nz = length(r)
        _, hη = sbp_grid(0.0, 1.0, nz)
        Zb = Zm[:, r]
        J = [dminus(j -> Zb[i, j], k, nz) / hη for i in 1:nx, k in 1:nz]
        Jx = [-dminus(j -> Zb[j, k], i, nx) / hξ for i in 1:nx, k in 1:nz]
        minimum(J) > 0 || error("curvilinear grid: the mesh folds in block $b")
        taup, taus = NM * zeta ./ bqp, NM * zeta ./ bqs
        mu = @. bρ * bvs^2 / visco_chi((ts,), taus, fq_ref)^2
        lam = @. bρ * bvp^2 / visco_chi((ts,), taup, fq_ref)^2 - 2mu
        # impedance damping at the sides and (bottom block) the bottom: a = |J·| Z / (H ρ J)
        a1, a2 = zeros(nx, nz), zeros(nx, nz)
        if absorber === :impedance
            for i in (1, nx)
                @. a1[i, :] += bvp[i, :] / Hξ[i]; @. a2[i, :] += bvs[i, :] / Hξ[i]
            end
            if b == 1
                Hη1 = SBP_W[1] * hη
                @. a1[:, 1] += sqrt(Jx[:, 1]^2 + 1) * bvs[:, 1] / (Hη1 * J[:, 1])
                @. a2[:, 1] += sqrt(Jx[:, 1]^2 + 1) * bvp[:, 1] / (Hη1 * J[:, 1])
            end
        end
        (; nz, hη, Z=Zb, J, Jx, ρ=bρ, lam, mu, taup, taus, a1, a2, vp=bvp)
    end
    nb = length(hostblocks)
    blocks = Tuple(map(hostblocks) do H
        z() = KernelAbstractions.zeros(backend, FT, nx, H.nz)
        mem = attenuation ? (; R11=KernelAbstractions.zeros(backend, FT, NM, nx, H.nz), R22=KernelAbstractions.zeros(backend, FT, NM, nx, H.nz),
                               R12=KernelAbstractions.zeros(backend, FT, NM, nx, H.nz)) : (;)
        merge((; V1=z(), V2=z(), S11=z(), S22=z(), S12=z(), J=dev(FT.(H.J)), Jx=dev(FT.(H.Jx)), iρJ=dev(FT.(1 ./ (H.ρ .* H.J))),
               lam=dev(FT.(H.lam)), mu=dev(FT.(H.mu)), tp=attenuation ? dev(FT.(H.taup)) : nothing,
               ts=attenuation ? dev(FT.(H.taus)) : nothing, a1=dev(FT.(H.a1)), a2=dev(FT.(H.a2)), hξ=FT(hξ), hη=FT(H.hη)), mem)
    end)
    below_(b) = b > 1 ? blocks[b-1] : nothing
    above_(b) = b < nb ? blocks[b+1] : nothing
    # the regions of a block, each with its kernel variant: the interior (branch-free), the top and bottom rows away from
    # the sides (closures in η only: vectorize) and the side columns (closures in both directions). All regions of a block
    # in one parallel loop on the CPU (one launch per kernel and block: many small launches cost much on a loaded machine).
    m = SBP_NB
    regions = map(hostblocks) do H
        nz = H.nz
        (((m, m), (nx - 2m, nz - 2m), Val(:int)), ((m, 0), (nx - 2m, m), Val(true)), ((m, nz - m), (nx - 2m, m), Val(true)),
         ((0, 0), (m, nz), Val(false)), ((nx - m, 0), (m, nz), Val(false)))
    end
    momentum!(dtT, damp) = for b in 1:nb
        foreach_region!(backend, momentum_cv!, (blocks[b], below_(b), above_(b), dtT, damp), regions[b])
    end
    stress!(vc, dtT) = for b in 1:nb
        foreach_region!(backend, stress_cv!, (blocks[b], below_(b), above_(b), vc, dtT), regions[b])
    end
    # the time step: power iteration for the largest eigenvalue λ of v -> M(S(v)); leapfrog is stable for dt < 2/√λ
    if dt === nothing
        foreach(B -> (copyto!(B.V1, FT.(randn(size(B.V1)))); copyto!(B.V2, FT.(randn(size(B.V2))))), blocks)
        λ = 0.0
        for _ in 1:100
            foreach(B -> (fill!(B.S11, 0); fill!(B.S22, 0); fill!(B.S12, 0)), blocks)
            stress!(nothing, one(FT))
            nrm0 = sqrt(sum(B -> sum(abs2, B.V1) + sum(abs2, B.V2), blocks))
            foreach(B -> (fill!(B.V1, 0); fill!(B.V2, 0)), blocks)
            momentum!(one(FT), false)
            KernelAbstractions.synchronize(backend)
            nrm = sqrt(sum(B -> sum(abs2, B.V1) + sum(abs2, B.V2), blocks))
            λ = nrm / nrm0
            foreach(B -> (B.V1 ./= FT(nrm); B.V2 ./= FT(nrm)), blocks)
        end
        foreach(B -> foreach(a -> fill!(a, 0), (B.V1, B.V2, B.S11, B.S22, B.S12)), blocks)
        dt = 0.9 * 2 / sqrt(λ)
    end
    nt = something(nt, tmax === nothing ? nothing : round(Int, tmax / dt))
    nt === nothing && error("give the duration tmax (or nt, with dt)")
    dtT = FT(dt)
    vc = attenuation ? memory_coefficients(ts, dt, FT) : nothing
    # locate points: block and η̂ (Lagrange interpolation of the mesh columns, bisection)
    function locate(x, z)
        i0, w = lagrange_weights(gx, x)
        for (b, H) in enumerate(hostblocks)
            col = [sum(w[a] * H.Z[i0+a-1, k] for a in 1:6) for k in 1:H.nz]
            (col[1] - 1e-9 <= z <= col[end] + 1e-9) || continue
            gη, _ = sbp_grid(0.0, 1.0, H.nz)
            zof(η) = let (j0, wj) = lagrange_weights(gη, η)
                sum(wj[c] * col[j0+c-1] for c in 1:6)
            end
            lo, hi = 0.0, 1.0
            for _ in 1:60
                mid = (lo + hi) / 2
                zof(mid) < z ? (lo = mid) : (hi = mid)
            end
            return b, (lo + hi) / 2
        end
        error("the point ($(x / 1e3), $(z / 1e3)) km is outside the model")
    end
    # sources: the nodes they reach in their block
    function source_table(list, coef)
        tab = (b=Int[], i=Int[], k=Int[], c=Float64[], n=Int[])
        for (n, s) in enumerate(list)
            b, η = locate(1e3s.x, 1e3s.z)
            H = hostblocks[b]
            gη, hη = sbp_grid(0.0, 1.0, H.nz)
            iξ, wξ = delta_weights(gx, Hξ, hξ, 1e3s.x)
            iη, wη = delta_weights(gη, sbp_weights(H.nz, hη), hη, η)
            for (a, i) in enumerate(iξ), (c, k) in enumerate(iη)
                push!(tab.b, b); push!(tab.i, i); push!(tab.k, k); push!(tab.c, wξ[a] * wη[c] * coef(H, i, k)); push!(tab.n, n)
            end
        end
        return tab
    end
    forces = filter(s -> s isa ForceSource, srcs)
    moments = filter(s -> s isa MomentTensorSource, srcs)
    ft = source_table(forces, (H, i, k) -> 1.0)
    mt = source_table(moments, (H, i, k) -> 1 / H.J[i, k])
    FS = (; b=dev(ft.b), i=dev(ft.i), k=dev(ft.k), c=dev(FT.(ft.c)), n=dev(ft.n),
          F=dev(FT[s.F[c] for s in forces, c in (1, 3)]),
          s=dev(FT[moment_rate(stf, tbeg + (it - 1) * dt, s.t0, s.trise) for s in forces, it in 1:nt]))
    MS = (; b=dev(mt.b), i=dev(mt.i), k=dev(mt.k), c=dev(FT.(mt.c)), n=dev(mt.n),
          M=dev(FT[s.M0 * s.M[c] for s in moments, c in (1, 3, 5)]),
          s=dev(FT[moment_rate(stf, tbeg + (it - 0.5) * dt, s.t0, s.trise) * dt for s in moments, it in 1:nt]))
    # stations
    nst = length(stations)
    st = map(stations) do s
        if s.surface
            b = nb
            η = 1.0
        else
            b, η = locate(1e3s.x, 1e3s.z)
        end
        gη, _ = sbp_grid(0.0, 1.0, hostblocks[b].nz)
        i0, wx = lagrange_weights(gx, 1e3s.x)
        k0, wz = lagrange_weights(gη, η)
        (b, i0, k0, wx, wz)
    end
    St = (; b=dev([s[1] for s in st]), i0=dev([s[2] for s in st]), k0=dev([s[3] for s in st]),
          wx=dev(FT[s[4][a] for a in 1:6, s in st]), wz=dev(FT[s[5][a] for a in 1:6, s in st]))
    nrec = cld(nt, record_every)
    rec = KernelAbstractions.zeros(backend, FT, nrec, 10, nst)
    acc = KernelAbstractions.zeros(backend, FT, 2, nst)
    prev = KernelAbstractions.zeros(backend, FT, 2, nst)
    # the wavefield on the mesh (the shared interface rows from the block above)
    function velocity_field()
        cols = [k == 1 ? (1:H.nz) : (2:H.nz) for (k, H) in enumerate(hostblocks)]
        v1 = reduce(hcat, [Array(blocks[b].V1)[:, cols[b]] for b in 1:nb])
        v2 = reduce(hcat, [Array(blocks[b].V2)[:, cols[b]] for b in 1:nb])
        r(A) = reshape(Float64.(A), nx, 1, nzt)
        return CartData(model.x.val, model.y.val, model.z.val, (; V=(r(v1), zero(r(v1)), r(v2))))
    end
    movie = (snapshot_every > 0 && callback === nothing) ? movie_paraview(name=joinpath(mkpath(snapshot_dir), "wavefield")) : nothing
    if verbose
        @printf("SeismicWavePC: PSV on a curvilinear grid, %d × %d nodes in %d block(s) (interfaces at rows %s), %d steps of %.4g s%s%s\n",
                nx, nzt, nb, isempty(rows) ? "–" : join(rows, ", "), nt, dt, attenuation ? ", attenuation" : "",
                absorber === :impedance ? ", absorbing sides and bottom" : "")
    end
    # time loop: v^{n+1/2} = v^{n-1/2} + dt M(σⁿ, tₙ), σ^{n+1} = σⁿ + dt S(v^{n+1/2})
    for it in 1:nt
        n = it - 1
        itw = n % record_every == 0 ? n ÷ record_every + 1 : 0
        itw > 0 && nst > 0 && foreach_item!(backend, vprev_cv!, (prev, blocks, St), nst)
        momentum!(dtT, true)
        length(forces) > 0 && run_once!(backend, force_cv!, (blocks, FS, it, dtT))
        nst > 0 && foreach_item!(backend, record_cv!, (rec, acc, prev, blocks, St, itw, dtT), nst)
        stress!(vc, dtT)
        length(moments) > 0 && run_once!(backend, moment_cv!, (blocks, MS, it))
        if snapshot_every > 0 && n % snapshot_every == 0
            KernelAbstractions.synchronize(backend)
            data = velocity_field()
            callback === nothing ? (movie = write_paraview(data, "wavefield_$(lpad(n, 6, '0'))"; directory=snapshot_dir, pvd=movie,
                                                            time=tbeg + n * dt, verbose=false)) : callback(data, tbeg + n * dt)
        end
        if verbose && (it % max(1, nt ÷ 10) == 0 || it == nt)
            KernelAbstractions.synchronize(backend)
            vmax = maximum(B -> max(maximum(abs, B.V1), maximum(abs, B.V2)), blocks)
            isfinite(vmax) || error("curvilinear grid: the simulation became unstable at step $it")
            @printf("  step %6d/%d  t = %8.4f s  max|v| = %.3e m/s\n", it, nt, tbeg + it * dt, vmax)
        end
    end
    KernelAbstractions.synchronize(backend)
    movie === nothing || movie_paraview(movie)
    verbose && @printf("  finished in %.2f s\n", time() - wall)
    traces = Float64.(Array(rec))
    return Seismograms(collect(tbeg .+ (0:nrec-1) .* (record_every * dt)), traces[:, 1:2, :], traces[:, 3:4, :],
                       traces[:, 5:7, :], traces[:, 8:10, :], collect(stations), (:x, :z), (:xx, :zz, :xz),
                       wavefield ? velocity_field() : nothing)
end
