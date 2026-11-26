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

"""
    mesh = curvilinear_mesh(x, y, surfaces; dz, bottom)

The 3D version: columns at the SBP nodes in x and y, `surfaces` functions z(x, y) [km] or surface `CartData`
(nx × ny × 1, bilinearly interpolated).
"""
function curvilinear_mesh(x, y, surfaces; dz, bottom)
    nx, ny = length(x), length(y)
    gx, _ = sbp_grid(Float64(first(x)), Float64(last(x)), nx)
    gy, _ = sbp_grid(Float64(first(y)), Float64(last(y)), ny)
    surf(f::Function) = [f(a, b) for a in gx, b in gy]
    function surf(c::CartData)                                          # bilinear interpolation of a surface CartData
        xs, ys, zs = c.x.val[:, 1, 1], c.y.val[1, :, 1], c.z.val[:, :, 1]
        ix(v, g) = (q = clamp(searchsortedlast(g, v), 1, length(g) - 1); (q, (v - g[q]) / (g[q+1] - g[q])))
        return [(((p, s), (q, t)) = (ix(a, xs), ix(b, ys));
                 (1 - s) * (1 - t) * zs[p, q] + s * (1 - t) * zs[p+1, q] + (1 - s) * t * zs[p, q+1] + s * t * zs[p+1, q+1]) for a in gx, b in gy]
    end
    S = [[surf(f) for f in surfaces]; [fill(Float64(bottom), nx, ny)]]
    Zs = Array{Float64,3}[]
    for b in length(S)-1:-1:1
        top, bot = S[b], S[b+1]
        all(top .> bot) || error("curvilinear_mesh: surfaces $b and $(b+1) touch or cross")
        nz = max(SBP_NMIN, ceil(Int, maximum(top .- bot) / dz) + 1)
        η, _ = sbp_grid(0.0, 1.0, nz)
        Zb = [bot[i, j] + (top[i, j] - bot[i, j]) * e for i in 1:nx, j in 1:ny, e in η]
        Zb[:, :, end] .= top
        push!(Zs, isempty(Zs) ? Zb : Zb[:, :, 2:end])
    end
    Z = cat(Zs...; dims=3)
    nz = size(Z, 3)
    X = [gx[i] for i in 1:nx, _ in 1:ny, _ in 1:nz]
    Y = [gy[j] for _ in 1:nx, j in 1:ny, _ in 1:nz]
    return CartData(X, Y, Z, (; z=copy(Z)))
end

"Is the `CartData` a deformed (terrain-following) mesh, i.e. do its rows of nodes vary in depth?"
function is_curvilinear(model::CartData)
    Z = model.z.val
    return any(k -> maximum(view(Z, :, :, k)) - minimum(view(Z, :, :, k)) > 1e-9 * (maximum(Z) - minimum(Z)), axes(Z, 3))
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
    (; V1, V2, S11, S22, S12, J, Jx, iρJ, a1, a2, hξ, hη, ox, nxg) = B
    nz = size(V1, 2)
    T = eltype(V1)
    w = T(SBP_W[1])
    # x: local index i, global i + ox of nxg (MPI subdomains); the boundary closures in global indices
    g1, g2 = j -> @inbounds(J[j, k] * S11[j, k]), j -> @inbounds(J[j, k] * S12[j, k])
    d1 = (xint ? dminus_int(g1, i, T) : dminus(q -> g1(q - ox), i + ox, nxg)) / hξ + dminus(j -> @inbounds(Jx[i, j] * S11[i, j] + S12[i, j]), k, nz) / hη
    d2 = (xint ? dminus_int(g2, i, T) : dminus(q -> g2(q - ox), i + ox, nxg)) / hξ + dminus(j -> @inbounds(Jx[i, j] * S12[i, j] + S22[i, j]), k, nz) / hη
    if xint
    elseif i + ox == 1
        d1 += J[i, k] * S11[i, k] / (w * hξ); d2 += J[i, k] * S12[i, k] / (w * hξ)
    elseif i + ox == nxg
        d1 -= J[i, k] * S11[i, k] / (w * hξ); d2 -= J[i, k] * S12[i, k] / (w * hξ)
    end
    if k == nz
        t1, t2 = Jx[i, k] * S11[i, k] + S12[i, k], Jx[i, k] * S12[i, k] + S22[i, k]
        if Bu === nothing
            d1 -= t1 / (w * hη); d2 -= t2 / (w * hη)
        else
            d1 -= (t1 - (Bu.Jx[i, 1] * Bu.S11[i, 1] + Bu.S12[i, 1])) / (2w * hη)
            d2 -= (t2 - (Bu.Jx[i, 1] * Bu.S12[i, 1] + Bu.S22[i, 1])) / (2w * hη)
        end
    end
    if k == 1
        t1, t2 = Jx[i, k] * S11[i, k] + S12[i, k], Jx[i, k] * S12[i, k] + S22[i, k]
        if Bd === nothing
            d1 += t1 / (w * hη); d2 += t2 / (w * hη)
        else
            n = size(Bd.V1, 2)
            d1 -= ((Bd.Jx[i, n] * Bd.S11[i, n] + Bd.S12[i, n]) - t1) / (2w * hη)
            d2 -= ((Bd.Jx[i, n] * Bd.S12[i, n] + Bd.S22[i, n]) - t2) / (2w * hη)
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
    ox, nxg = B.ox, B.nxg
    a = (xint ? dplus_int(g1, i, T) : dplus(q -> g1(q - ox), i + ox, nxg)) / hξ
    b = dplus(j -> @inbounds(V1[i, j]), k, nz) / hη
    c = (xint ? dplus_int(g2, i, T) : dplus(q -> g2(q - ox), i + ox, nxg)) / hξ
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
    I = CartesianIndex(i, k)
    J = B.J[I]
    q = B.Jx[I] / J
    stress_update_cv!(anisotropic(B), B, vc, dt, I, a + q * b, e / J, b / J + c + q * e)
end
@inline Base.@propagate_inbounds function stress_update_cv!(::Val{false}, B, vc, dt, I, exx, ezz, exz)
    (; S11, S22, S12, lam, mu) = B
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
# VTI (anisotropy.jl): normal stresses w (w·e) relaxing with τp and the rest (s13) with τs; σxz with C55 (`mu`)
@inline Base.@propagate_inbounds function stress_update_cv!(::Val{true}, B, vc, dt, I, exx, ezz, exz)
    (; S11, S22, S12, wx, wz, s13, mu) = B
    tp, ts = orzero(B.tp, I), orzero(B.ts, I)
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    w1, w3, s = wx[I], wz[I], s13[I]
    pe = w1 * exx + w3 * ezz
    Rx = cmemvar!(memory(B, Val(:R11)), I, w1 * pe * tp + s * ts * ezz, vc, dt)
    Rz = cmemvar!(memory(B, Val(:R22)), I, w3 * pe * tp + s * ts * exx, vc, dt)
    Rxz = cmemvar!(memory(B, Val(:R12)), I, mu[I] * ts * exz, vc, dt)
    S11[I] += (w1 * pe * tp1 + s * ts1 * ezz + Rx) * dt
    S22[I] += (w3 * pe * tp1 + s * ts1 * exx + Rz) * dt
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

# ------------------------------------------------------------------ MPI: subdomains in x (and y in 3D); columns stay whole

const CV_HALO = 4                       # the reach of the interior stencils
"The nodes of part `c` (1-based) of `p` of `n` nodes."
part_range(n, p, c) = fld((c - 1) * n, p)+1:fld(c * n, p)

"MPI for the curvilinear solver: `nothing` on one process, else the Cartesian communicator, rank, dims, coordinates (1-based)
and neighbours (left, right) of the `N` decomposed directions."
function cv_mpi(nproc, N)
    dims = ntuple(a -> a <= length(nproc) ? Int(nproc[a]) : 1, N)
    prod(dims) == 1 && return nothing
    MPI.Initialized() || MPI.Init()
    np = MPI.Comm_size(MPI.COMM_WORLD)
    np == prod(dims) || error("nproc = $nproc needs $(prod(dims)) MPI processes, julia runs on $np")
    comm = MPI.Cart_create(MPI.COMM_WORLD, collect(dims); periodic=fill(false, N), reorder=false)
    me = MPI.Comm_rank(comm)
    coords = Tuple(Int.(MPI.Cart_coords(comm, me))) .+ 1
    nb = ntuple(a -> MPI.Cart_shift(comm, a - 1, 1), N)
    return (; comm, me, dims, coords, nb)
end

"Exchange the halos (width CV_HALO) of the decomposed directions of `arrays` (all the same size) with the neighbours."
function cv_exchange!(par, arrays)
    par === nothing && return nothing
    nd, h = ndims(arrays[1]), CV_HALO
    for a in 1:length(par.dims)
        left, right = par.nb[a]
        n = size(arrays[1], a)
        sel(r) = ntuple(b -> b == a ? r : Colon(), nd)
        pack(r) = reduce(vcat, (vec(Array(view(A, sel(r)...))) for A in arrays))    # one message per neighbour
        sl, sr = pack(h+1:2h), pack(n-2h+1:n-h)
        rl, rr = similar(sl), similar(sr)
        MPI.Waitall([MPI.Irecv!(rl, par.comm; source=left, tag=0), MPI.Irecv!(rr, par.comm; source=right, tag=1),
                     MPI.Isend(sl, par.comm; dest=left, tag=1), MPI.Isend(sr, par.comm; dest=right, tag=0)])
        m = length(sl) ÷ length(arrays)
        for (q, A) in enumerate(arrays), (buf, r, nbr) in ((rl, 1:h, left), (rr, n-h+1:n, right))
            nbr == MPI.PROC_NULL && continue
            dest = view(A, sel(r)...)
            d = similar(A, length(dest))
            copyto!(d, view(buf, (q-1)*m+1:q*m))
            dest .= reshape(d, size(dest))
        end
    end
    return nothing
end
cv_sum(par, x) = par === nothing ? x : MPI.Allreduce(x, +, par.comm)
cv_max(par, x) = par === nothing ? x : MPI.Allreduce(x, max, par.comm)

"""
    simulate_curvilinear(model; tmax or nt, sources, ...)

2D P-SV on a deformed (terrain-following) `CartData` (see [`curvilinear_mesh`](@ref)); called by `simulate` for such a
model. Keywords as in `simulate`, and: `dt = nothing` (from the spectral radius of the operator: 90 % of the leapfrog
limit), `tmax` (duration) instead of `nt`, `absorber = :impedance` (or `:none`: free sides and bottom), `jump = 0.05`
(relative jump of vp or vs that makes a grid row an interface, see `find_blocks`), `nproc` MPI processes along x (2D) or
`(npx, npy)` (3D): start julia with `mpiexec`; the columns are not split.
"""
function simulate_curvilinear(model::CartData; dt=nothing, nt=nothing, tmax=nothing, sources, stations=Station[], stf=:kupper,
                              tbeg=0.0, absorber=:impedance, fq_min=0.05, fq_max=5.0, fq_ref=1.0, record_every=1,
                              density_unit=:kg_m3, snapshot_every=0, snapshot_dir="snapshots", callback=nothing, snapshots=nothing, checkpoint=nothing, checkpoint_every=0,
                              restart=false, backend=CPU(),
                              T::Type{FT}=Float32, verbose=true, wavefield=true, jump=0.05, wave=:psv, interpolation=:sinc,
                              free_surface=:openswpc, nproc=1, driver=nothing, kwargs...) where {FT}
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
    # MPI: this process holds the nodes ia:ib in x and CV_HALO halo nodes on either side (local = global - ox)
    par = cv_mpi(nproc, 1)
    me = par === nothing ? 0 : par.me
    ia, ib = par === nothing ? (1, nx) : extrema(part_range(nx, par.dims[1], par.coords[1]))
    par === nothing || ib - ia + 1 >= SBP_NB + CV_HALO || error("curvilinear grid: too many processes in x for $nx nodes")
    hx = par === nothing ? 0 : CV_HALO
    ox = ia - hx - 1
    xr = clamp.(ia-hx:ib+hx, 1, nx)                   # the local nodes (beyond the domain ends: copies, not used)
    nl, no = length(xr), ib - ia + 1
    f = model.fields
    haskey(f, :rho) && haskey(f, :vp) && haskey(f, :vs) || error("the model needs fields rho, vp and vs")
    ρ = f.rho[:, 1, :] .* (density_unit === :g_cm3 ? 1e3 : 1.0)
    vp, vs = 1e3 .* f.vp[:, 1, :], 1e3 .* f.vs[:, 1, :]
    minimum(vs) > 0 || error("curvilinear grids: solids only (vs > 0 everywhere)")
    qp = haskey(f, :Qp) ? Float64.(f.Qp[:, 1, :]) : fill(Inf, nx, nzt)
    qs = haskey(f, :Qs) ? Float64.(f.Qs[:, 1, :]) : fill(Inf, nx, nzt)
    attenuation = any(isfinite, qp) || any(isfinite, qs)
    aniso = has_anisotropy(model)                               # VTI: Thomsen ε, δ, γ
    thomsen(k) = haskey(f, k) ? Float64.(f[k][:, 1, :]) : zeros(nx, nzt)
    ε, δ, γ = thomsen(:epsilon), thomsen(:delta), thomsen(:gamma)
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
        m = map(A -> copy(A[:, r]), (ρ, vp, vs, qp, qs, ε, δ, γ))
        kmap = [r[k] for i in 1:nx, k in eachindex(r)]       # the model row of the material of each node (gradients)
        for (j, side, src) in ((b - 1, false, 2), (b, true, length(r) - 1))
            1 <= j <= length(rows) || continue
            kk = side ? length(r) : 1
            for i in 1:nx
                below[j][i] == side && continue
                foreach(A -> A[i, kk] = A[i, src], m)
                kmap[i, kk] = kmap[i, src]
            end
        end
        bρ, bvp, bvs, bqp, bqs, bε, bδ, bγ = m
        nz = length(r)
        _, hη = sbp_grid(0.0, 1.0, nz)
        Zb = Zm[:, r]
        J = [dminus(j -> Zb[i, j], k, nz) / hη for i in 1:nx, k in 1:nz]
        Jx = [-dminus(j -> Zb[j, k], i, nx) / hξ for i in 1:nx, k in 1:nz]
        minimum(J) > 0 || error("curvilinear grid: the mesh folds in block $b")
        taup, taus = NM * zeta ./ bqp, NM * zeta ./ bqs
        mu = @. bρ * bvs^2 / visco_chi((ts,), taus, fq_ref)^2
        lam = @. bρ * bvp^2 / visco_chi((ts,), taup, fq_ref)^2 - 2mu
        # the moduli in the VTI form (also of isotropic media: the adjoint)
        V = vti_moduli.(bρ, bvp, bvs, bε, bδ, bγ, visco_chi.((ts,), taup, fq_ref), visco_chi.((ts,), taus, fq_ref))
        wx, wz, s13, mu55, mu66 = ntuple(q -> getindex.(V, q), 5)
        # impedance damping at the sides and (bottom block) the bottom: a = |J·| Z / (H ρ J) with the speeds of normal
        # incidence (VTI: horizontal qP vp √(1+2ε)), and its derivatives (vp, vs, ε, γ) per component (gradients)
        a1, a2 = zeros(nx, nz), zeros(nx, nz)
        da = ntuple(_ -> ntuple(_ -> zeros(nx, nz), 4), 2)
        if absorber === :impedance
            for i in (1, nx)
                h = 1 / Hξ[i]
                @. a1[i, :] += bvp[i, :] * sqrt(1 + 2bε[i, :]) * h; @. a2[i, :] += bvs[i, :] * h
                @. da[1][1][i, :] += sqrt(1 + 2bε[i, :]) * h; @. da[1][3][i, :] += bvp[i, :] / sqrt(1 + 2bε[i, :]) * h
                da[2][2][i, :] .+= h
            end
            if b == 1
                Hη1 = SBP_W[1] * hη
                q = @. sqrt(Jx[:, 1]^2 + 1) / (Hη1 * J[:, 1])
                @. a1[:, 1] += q * bvs[:, 1]; @. a2[:, 1] += q * bvp[:, 1]
                da[1][2][:, 1] .+= q; da[2][1][:, 1] .+= q
            end
        end
        (; nz, hη, Z=Zb, J, Jx, ρ=bρ, lam, mu, taup, taus, a1, a2, vp=bvp, vs=bvs, ε=bε, δ=bδ, γ=bγ, wx, wz, s13, mu55, mu66, kmap, da)
    end
    nb = length(hostblocks)
    blocks = Tuple(map(hostblocks) do H
        z() = KernelAbstractions.zeros(backend, FT, nl, H.nz)
        loc(A) = dev(FT.(A[xr, :]))
        mem = attenuation ? (; R11=KernelAbstractions.zeros(backend, FT, NM, nl, H.nz), R22=KernelAbstractions.zeros(backend, FT, NM, nl, H.nz),
                               R12=KernelAbstractions.zeros(backend, FT, NM, nl, H.nz)) : (;)
        vti = aniso ? (; wx=loc(H.wx), wz=loc(H.wz), s13=loc(H.s13)) : (;)
        merge((; V1=z(), V2=z(), S11=z(), S22=z(), S12=z(), J=loc(H.J), Jx=loc(H.Jx), iρJ=loc(1 ./ (H.ρ .* H.J)), lam=loc(H.lam),
               mu=loc(aniso ? H.mu55 : H.mu), tp=attenuation ? loc(H.taup) : nothing, ts=attenuation ? loc(H.taus) : nothing, a1=loc(H.a1),
               a2=loc(H.a2), hξ=FT(hξ), hη=FT(H.hη), ox, nxg=nx), mem, vti)
    end)
    below_(b) = b > 1 ? blocks[b-1] : nothing
    above_(b) = b < nb ? blocks[b+1] : nothing
    # the regions of a block, each with its kernel variant: the interior (branch-free), the top and bottom rows away from
    # the sides (closures in η only: vectorize) and the side columns (closures in both directions). All regions of a block
    # in one parallel loop on the CPU (one launch per kernel and block: many small launches cost much on a loaded machine).
    # (with MPI: the global segmentation in x, intersected with the nodes of this process)
    m = SBP_NB
    regions = map(hostblocks) do H
        nz = H.nz
        R = Any[]
        for (lo, hi, xint) in ((1, m, false), (m + 1, nx - m, true), (nx - m + 1, nx, false))
            a, b = max(lo, ia), min(hi, ib)
            a > b && continue
            l0, w = a - ox - 1, b - a + 1
            xint ? push!(R, ((l0, m), (w, nz - 2m), Val(:int)), ((l0, 0), (w, m), Val(true)), ((l0, nz - m), (w, m), Val(true))) :
                   push!(R, ((l0, 0), (w, nz), Val(false)))
        end
        Tuple(R)
    end
    # (on `blks`: the blocks, or blocks of other arrays with the same mesh: the operators of the adjoint)
    momentum!(dtT, damp, blks=blocks) = for b in 1:nb
        foreach_region!(backend, momentum_cv!, (blks[b], b > 1 ? blks[b-1] : nothing, b < nb ? blks[b+1] : nothing, dtT, damp), regions[b])
    end
    stress!(vc, dtT, blks=blocks) = for b in 1:nb
        foreach_region!(backend, stress_cv!, (blks[b], b > 1 ? blks[b-1] : nothing, b < nb ? blks[b+1] : nothing, vc, dtT), regions[b])
    end
    exchange_v!() = foreach(B -> cv_exchange!(par, (B.V1, B.V2)), blocks)
    exchange_s!() = foreach(B -> cv_exchange!(par, (B.S11, B.S22, B.S12)), blocks)
    owned(A) = view(A, hx+1:hx+no, :)
    # the time step: power iteration for the largest eigenvalue λ of v -> M(S(v)); leapfrog is stable for dt < 2/√λ
    if dt === nothing
        rng = Random.Xoshiro(20261126 + me)                 # (seeded: the same time step in every run)
        foreach(B -> (copyto!(B.V1, FT.(randn(rng, size(B.V1)))); copyto!(B.V2, FT.(randn(rng, size(B.V2))))), blocks)
        exchange_v!()
        λ = 0.0
        for _ in 1:100
            foreach(B -> (fill!(B.S11, 0); fill!(B.S22, 0); fill!(B.S12, 0)), blocks)
            stress!(nothing, one(FT))
            exchange_s!()
            nrm0 = sqrt(cv_sum(par, value(sum(B -> sum(abs2, owned(B.V1)) + sum(abs2, owned(B.V2)), blocks))))
            foreach(B -> (fill!(B.V1, 0); fill!(B.V2, 0)), blocks)
            momentum!(one(FT), false)
            KernelAbstractions.synchronize(backend)
            exchange_v!()
            nrm = sqrt(cv_sum(par, value(sum(B -> sum(abs2, owned(B.V1)) + sum(abs2, owned(B.V2)), blocks))))
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
    # (`δ`: shifted by δ [km]: the derivatives of the weights for the gradient with respect to the location)
    function source_table(list, coef, δ=(0.0, 0.0))
        tab = (b=Int[], i=Int[], k=Int[], c=Float64[], n=Int[])
        for (n, s) in enumerate(list)
            x, z = s.x + δ[1], s.z + δ[2]
            b, η = locate(1e3x, 1e3z)
            H = hostblocks[b]
            gη, hη = sbp_grid(0.0, 1.0, H.nz)
            iξ, wξ = delta_weights(gx, Hξ, hξ, 1e3x)
            iη, wη = delta_weights(gη, sbp_weights(H.nz, hη), hη, η)
            for (a, i) in enumerate(iξ), (c, k) in enumerate(iη)
                push!(tab.b, b); push!(tab.i, i); push!(tab.k, k); push!(tab.c, wξ[a] * wη[c] * coef(H, i, k)); push!(tab.n, n)
            end
        end
        return tab
    end
    forces = filter(s -> s isa ForceSource, srcs)
    moments = filter(s -> s isa MomentTensorSource, srcs)
    # the nodes of this process, in local indices
    mine(t) = (q = findall(i -> ia <= i <= ib, t.i); (b=t.b[q], i=t.i[q] .- ox, k=t.k[q], c=t.c[q], n=t.n[q]))
    fcoef, mcoef = (H, i, k) -> 1.0, (H, i, k) -> 1 / H.J[i, k]
    tables(δ) = (mine(source_table(forces, fcoef, δ)), mine(source_table(moments, mcoef, δ)))
    ft, mt = tables((0.0, 0.0))
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
    st_all = st
    own = findall(s -> ia <= s[2] + 2 <= ib, st)        # the stations of this process (the centre of their stencil)
    st = st[own]
    nst = length(st)
    St = (; b=dev([s[1] for s in st]), i0=dev([s[2] - ox for s in st]), k0=dev([s[3] for s in st]),
          wx=dev(FT[s[4][a] for a in 1:6, s in st]), wz=dev(FT[s[5][a] for a in 1:6, s in st]))
    nrec = cld(nt, record_every)
    rec = KernelAbstractions.zeros(backend, FT, nrec, 10, nst)
    acc = KernelAbstractions.zeros(backend, FT, 2, nst)
    prev = KernelAbstractions.zeros(backend, FT, 2, nst)
    # the wavefield on the mesh (the shared interface rows from the block below)
    # (with MPI: gathered on rank 0, `nothing` elsewhere)
    function velocity_field()
        cols = [k == 1 ? (1:H.nz) : (2:H.nz) for (k, H) in enumerate(hostblocks)]
        v1 = reduce(hcat, [Array(owned(blocks[b].V1))[:, cols[b]] for b in 1:nb])
        v2 = reduce(hcat, [Array(owned(blocks[b].V2))[:, cols[b]] for b in 1:nb])
        if par !== nothing
            parts = MPI.gather((ia, ib, v1, v2), par.comm; root=0)
            me == 0 || return nothing
            v1, v2 = zeros(nx, nzt), zeros(nx, nzt)
            for (a, b, p1, p2) in parts
                v1[a:b, :] .= p1; v2[a:b, :] .= p2
            end
        end
        r(A) = reshape(Float64.(A), nx, 1, nzt)
        return CartData(model.x.val, model.y.val, model.z.val, (; V=(r(v1), zero(r(v1)), r(v2))))
    end
    planes = snapshots === nothing ? nothing :
             setup_planes_cv(snapshots, 2, (; hostblocks, nx, ny=1, gx, gy=nothing, Zm, ia, ib, ja=1, jb=1, oy=0, ox, dev, backend, FT, dir=snapshot_dir,
                                              me, parallel=par !== nothing))
    movie = (snapshot_every > 0 && callback === nothing && snapshots === nothing && me == 0) ?
            movie_paraview(name=joinpath(mkpath(snapshot_dir), "wavefield")) : nothing
    verbose &= me == 0
    if verbose
        @printf("SeismicWavePC: PSV on a curvilinear grid, %d × %d nodes in %d block(s) (interfaces at rows %s), %d steps of %.4g s%s%s\n",
                nx, nzt, nb, isempty(rows) ? "–" : join(rows, ", "), nt, dt, attenuation ? ", attenuation" : "",
                absorber === :impedance ? ", absorbing sides and bottom" : "")
    end
    # one step: v^{n+1/2} = v^{n-1/2} + dt M(σⁿ, tₙ), σ^{n+1} = σⁿ + dt S(v^{n+1/2}) (n = it - 1); recording if `record`
    function step!(it, record)
        n = it - 1
        itw = record && n % record_every == 0 ? n ÷ record_every + 1 : 0
        itw > 0 && nst > 0 && foreach_item!(backend, vprev_cv!, (prev, blocks, St), nst)
        momentum!(dtT, true)
        length(forces) > 0 && run_once!(backend, force_cv!, (blocks, FS, it, dtT))
        KernelAbstractions.synchronize(backend)
        exchange_v!()
        record && nst > 0 && foreach_item!(backend, record_cv!, (rec, acc, prev, blocks, St, itw, dtT), nst)
        stress!(vc, dtT)
        length(moments) > 0 && run_once!(backend, moment_cv!, (blocks, MS, it))
        KernelAbstractions.synchronize(backend)
        exchange_s!()
    end
    "the seismograms of all stations (velocity, displacement, stress, strain) on every process (`part`: of Born numbers)"
    function all_traces(part=identity)
        traces = zeros(nrec, 10, length(stations))
        traces[:, :, own] .= part.(Array(rec))
        par === nothing || MPI.Allreduce!(traces, +, par.comm)
        return traces
    end
    seismograms(traces, wf=nothing) = Seismograms(collect(tbeg .+ (0:nrec-1) .* (record_every * dt)), traces[:, 1:2, :], traces[:, 3:4, :],
                                                  traces[:, 5:7, :], traces[:, 8:10, :], collect(stations), (:x, :z), (:xx, :zz, :xz), wf)
    # a driver (misfit gradients) takes over
    driver === nothing || return drive_curvilinear(driver, (; backend, FT, blocks, hostblocks, nb, nx, nzt, nl, no, hx, ox, xr, ia, ib,
        par, me, dt, dtT, nt, nrec, record_every, momentum!, stress!, step!, exchange_v!, exchange_s!, owned, St, nst, FS, forces,
        Hξ, hξ, tbeg, attenuation, dev, model, all_traces, stations, density_unit, vc, own_stations=own, nsrc=length(srcs), st_all,
        seismograms, ts, fq_ref, loc=A -> A[xr, :], ftab=ft, mtab=mt, MS, moments, srcs, stf, tables))
    # checkpoints (one file per process): the wavefield and memory variables, the recording and the plane accumulators
    ckp = checkpoint === nothing ? nothing : par !== nothing ? "$(checkpoint)_rank$me.jls" : "$checkpoint.jls"
    restart && (ckp === nothing || !isfile(ckp)) && error("restart = true needs the checkpoint file $ckp")
    fields_of(B) = Tuple(getfield(B, k) for k in keys(B) if occursin(r"^[VSR][0-9]+$", string(k)))
    state = (Iterators.flatten(map(fields_of, blocks))..., rec, acc, prev,
             (planes === nothing ? () : Iterators.flatten((p.U, p.PK) for p in planes))...)
    it0 = restart ? load_state!(state, ckp, nt) + 1 : 1
    restart && verbose && @printf("  restart from step %d\n", it0 - 1)
    for it in it0:nt
        n = it - 1
        step!(it, true)
        if ckp !== nothing && checkpoint_every > 0 && it % checkpoint_every == 0 && it < nt
            KernelAbstractions.synchronize(backend)
            save_state(ckp, it, nt, state)
        end
        if planes !== nothing                          # netCDF planes: displacement and peaks every step
            store = snapshot_every > 0 && n % snapshot_every == 0
            for p in planes
                isempty(p.own) || foreach_item!(backend, plane_cv!, (p.B, p.U, p.PK, blocks, p.pts, dtT, store, Val(2)), length(p.own))
            end
            if store
                KernelAbstractions.synchronize(backend)
                write_planes(planes, :psv, n ÷ snapshot_every + 1, tbeg + n * dt, 1.0, par !== nothing, me)
            end
        end
        if snapshot_every > 0 && n % snapshot_every == 0 && (planes === nothing || callback !== nothing)
            data = velocity_field()
            if me == 0
                callback === nothing ? (movie = write_paraview(data, "wavefield_$(lpad(n, 6, '0'))"; directory=snapshot_dir, pvd=movie,
                                                                time=tbeg + n * dt, verbose=false)) : callback(data, tbeg + n * dt)
            end
        end
        if (verbose || par !== nothing) && (it % max(1, nt ÷ 10) == 0 || it == nt)
            vmax = cv_max(par, Float64(maximum(B -> max(maximum(abs, owned(B.V1)), maximum(abs, owned(B.V2))), blocks)))
            isfinite(vmax) || error("curvilinear grid: the simulation became unstable at step $it")
            verbose && @printf("  step %6d/%d  t = %8.4f s  max|v| = %.3e m/s\n", it, nt, tbeg + it * dt, vmax)
        end
    end
    KernelAbstractions.synchronize(backend)
    movie === nothing || movie_paraview(movie)
    planes === nothing || write_peaks(planes, snapshots.quantities, :psv, 1.0, par !== nothing, me)
    verbose && @printf("  finished in %.2f s\n", time() - wall)
    return seismograms(all_traces(), wavefield ? velocity_field() : nothing)
end
