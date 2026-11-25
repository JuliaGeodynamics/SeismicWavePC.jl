# Immersed free surface (option `topography` of `simulate`): the free surface is given as a function or heights, at its
# true position between grid points, instead of as a staircase of air cells (OpenSWPC, the default).
#
# Every staggered point is solid or air by its side of the surface. The 4th-order stencils of solid points that reach into
# the air read "ghost" values there: extrapolations of the solid values with the traction-free condition built in. For
# each ghost point, a quadratic polynomial per field is fitted (weighted least squares) to the solid values around it,
# subject to the boundary condition at three points along the surface (the nearest point and one cell to either side):
# σ n = 0 for the stresses, its rate (C : ε̇) n = 0 for the velocities. The ghost value is that polynomial at the ghost
# point: a fixed linear combination of solid values, computed once (as the coefficient lists of the immersed interface
# Stokes solver of W. Landry, gitlab.com/wlandry/discontinuous_stokes). Each time step fills them with one sparse gather.
# The other air points near the surface are set to zero, and the air cells near the surface get the material of the
# nearest solid cell (the solid points next to them use it), so that the kernels are the same as without topography.

const GHOST_BAND = 6        # air cells within this many cells of a solid point are overwritten every step
const GHOST_FIT = Ref(3.0)  # the fit uses the solid points within this many cells of the point one cell below the surface

"""
The surface as depth [km] (internal axes: depth down) of the x coordinate, flat outside `[xa, xb]` (the absorbing layers
at the sides, whose medium continues the columns next to them).
"""
struct Surface{F}
    zs::F
    xa::Float64
    xb::Float64
end
surface_depth(s::Surface, x) = -Float64(s.zs(clamp(x, s.xa, s.xb)))

"The surface of the `topography` option for the grid `g` (absorbing layers of `na` cells at the sides)."
const SURFACE_FLAT = Ref(0)   # the surface is also flat this many cells next to the absorbing layers
immersed_surface(topography, x, g::Grid{2}, na) =
    Surface(topography_function(topography, x), g.beg[1] + (na + SURFACE_FLAT[] + 0.5) * g.d[1], g.beg[1] + (g.n[1] - na - SURFACE_FLAT[] - 0.5) * g.d[1])

"The topography as a function of x [km] (z up): a function, or the heights at the model's x coordinates."
topography_function(f::Function, x) = f
function topography_function(z::AbstractVector, x)
    length(z) == length(x) || error("topography: $(length(z)) heights for $(length(x)) x coordinates")
    xs, zs, dx = collect(Float64, x), collect(Float64, z), step(range(x[1], x[end], length=length(x)))
    return function (xq)                                       # cubic (Catmull-Rom) between the given points
        u = clamp((xq - xs[1]) / dx, 0, length(xs) - 1 - 1e-12)
        i = floor(Int, u) + 1
        t = u - (i - 1)
        p(j) = zs[clamp(j, 1, length(zs))]
        p0, p1, p2, p3 = p(i - 1), p(i), p(i + 1), p(i + 2)
        return p1 + 0.5t * (p2 - p0 + t * (2p0 - 5p1 + 4p2 - p3 + t * (3(p1 - p2) + p3 - p0)))
    end
end

"Nearest point (x, depth) of the surface to the point `q`, and the unit normal there (pointing into the air)."
function nearest_surface_point(s::Surface, q)
    δ = 1e-6
    d(x) = surface_depth(s, x)
    d1(x) = (d(x + δ) - d(x - δ)) / 2δ
    d2(x) = (d(x + δ) - 2d(x) + d(x - δ)) / δ^2
    x = q[1]
    for _ in 1:50                                              # Newton on the distance squared
        f = (x - q[1]) + (d(x) - q[2]) * d1(x)
        df = 1 + d1(x)^2 + (d(x) - q[2]) * d2(x)
        step = f / (df > 0.1 ? df : 1.0)
        x -= step
        abs(step) < 1e-12 && break
    end
    s1 = d1(x)
    nrm = sqrt(1 + s1^2)
    return (x, d(x)), (s1 / nrm, -1 / nrm)                     # φ = d(x) - depth > 0 in the air: n = ∇φ / |∇φ|
end

"Is the point (x, depth) in the air?"
in_air(s::Surface, p) = p[2] < surface_depth(s, p[1])

const GHOST_MARGIN = Ref(0.3)  # solid points closer than this (in cells) to the surface are extrapolated too (small cut cells)

"Is the point (x, depth) a ghost point: in the air or within `GHOST_MARGIN` cells of the surface?"
function ghosted(s::Surface, p, h)
    in_air(s, p) && return true
    GHOST_MARGIN[] > 0 || return false
    P, _ = nearest_surface_point(s, p)
    return hypot(p[1] - P[1], p[2] - P[2]) < GHOST_MARGIN[] * h
end

# ------------------------------------------------------------------ the points of the staggered grid (P-SV)

"Offsets of the velocity and stress points of P-SV within their cell (x, depth): Vx, Vz and Sxx, Szz, Sxz."
immersed_offsets(::Val{:psv}) = (velocity=((0.5, 0.0), (0.0, 0.5)), stress=((0.0, 0.0), (0.0, 0.0), (0.5, 0.5)))

"Position (x, depth) [km] of the point with offset `o` of the cell `I` (padded indices)."
point_position(g::Grid{2}, I, o) = (g.beg[1] + (I[1] - H - 0.5 + o[1]) * g.d[1], o[2] == 0 ? g.zc[I[2]] : g.zf[I[2]])

"""
Points that the update of a point of field `f` reads: for velocities the stresses (field, offset), for stresses the
velocities (`kernels.jl`).
"""
stencil_reads(::Val{:psv}, ::Val{:velocity}, f) =
    f == 1 ? ([(1, (di, 0)) for di in -1:2]..., [(3, (0, dk)) for dk in -2:1]...) :      # Vx: Sxx in x, Sxz in depth
             ([(2, (0, dk)) for dk in -1:2]..., [(3, (di, 0)) for di in -2:1]...)        # Vz: Szz in depth, Sxz in x
stencil_reads(::Val{:psv}, ::Val{:stress}, f) =
    f <= 2 ? ([(1, (di, 0)) for di in -2:1]..., [(2, (0, dk)) for dk in -2:1]...) :      # Sxx, Szz: Vx in x, Vz in depth
             ([(2, (di, 0)) for di in -1:2]..., [(1, (0, dk)) for dk in -1:2]...)        # Sxz: Vz in x, Vx in depth

# ------------------------------------------------------------------ the extrapolation

const GHOST_DEGREE = Ref(3)  # degree of the fitted polynomials
const GHOST_WMAX = Ref(Inf)  # largest sum of the absolute weights of a ghost value (else a lower degree)
const FILTER = Ref(0.02)      # strength of the filter near the surface: the grid-scale mode decays by this per step
const FILTER_KIND = Ref(:projection)   # :projection (towards the local polynomial fit) or :difference (Kreiss-Oliger)
const FILTER_ORDER = Ref(1)  # lowest half-order m of the filter (difference of order 2m) next to the surface
const FILTER_BAND = 3.0       # it acts on the solid points within this many cells of the surface
const GHOST_EDGE = Ref(4)    # quadratic polynomials in this many columns next to the absorbing layers (stable there)

"Exponents (a, b) of the monomials u^a v^b of degree ≤ `deg`."
exponents(deg) = [(a, d - a) for d in 0:deg for a in d:-1:0]
"Monomials of degree ≤ `deg` at the scaled offset (u, v), and their derivatives with respect to u and v."
monomials(u, v, deg) = [u^a * v^b for (a, b) in exponents(deg)]
monomials_du(u, v, deg) = [a == 0 ? 0.0 : a * u^(a - 1) * v^b for (a, b) in exponents(deg)]
monomials_dv(u, v, deg) = [b == 0 ? 0.0 : b * u^a * v^(b - 1) for (a, b) in exponents(deg)]

"Stiffness (C11, C13, C33, C55) of the cell `I` (unrelaxed): isotropic or VTI."
function cell_stiffness(med, I)
    haskey(med, :wx) && return (med.wx[I]^2, med.wx[I] * med.wz[I] + med.s13[I], med.wz[I]^2, med.mu[I])
    l2m = med.lam[I] + 2med.mu[I]
    return (l2m, med.lam[I], l2m, med.mu[I])
end

"""
Boundary condition rows (traction zero) at the surface point `p` with normal `n`, for the polynomial coefficients (`nm`
monomials per field) of the fields of the group around `P` (scaled by `h`).
"""
function bc_rows(::Val{:stress}, p, n, P, h, deg, _)
    m = monomials((p[1] - P[1]) / h, (p[2] - P[2]) / h, deg)
    z = zeros(length(m))
    # fields Sxx, Szz, Sxz: traction (Sxx n_x + Sxz n_z, Sxz n_x + Szz n_z)
    return [hcat((n[1] .* m)', z', (n[2] .* m)'); hcat(z', (n[2] .* m)', (n[1] .* m)')]
end
function bc_rows(::Val{:velocity}, p, n, P, h, deg, C)
    C11, C13, C33, C55 = C
    u, v = (p[1] - P[1]) / h, (p[2] - P[2]) / h
    dx, dz = monomials_du(u, v, deg) ./ h, monomials_dv(u, v, deg) ./ h
    # fields Vx, Vz: σxx = C11 ∂x vx + C13 ∂z vz, σzz = C13 ∂x vx + C33 ∂z vz, σxz = C55 (∂z vx + ∂x vz)
    sxx = (C11 .* dx, C13 .* dz)
    szz = (C13 .* dx, C33 .* dz)
    sxz = (C55 .* dz, C55 .* dx)
    row(a, b) = hcat((a[1] .* n[1] .+ b[1] .* n[2])', (a[2] .* n[1] .+ b[2] .* n[2])')
    return [row(sxx, sxz); row(sxz, szz)]
end

"""
Solid points (field, cell, position) of the group with offsets `offs` that the fit for the ghost point at `q` (cell `I`)
uses: those within `GHOST_FIT` cells of the point one cell below the surface under `q`.
"""
function ghost_data(s::Surface, g::Grid{2}, I, q, offs, sz, isghost=(J, o, p) -> ghosted(s, p, g.d[1]))
    P, n = nearest_surface_point(s, q)
    h = g.d[1]
    c = (P[1] - h * n[1], P[2] - h * n[2])
    R = CartesianIndices(ntuple(a -> max(I[a] - ceil(Int, GHOST_FIT[]) - 2, 1):min(I[a] + ceil(Int, GHOST_FIT[]) + 2, sz[a]), 2))
    return [(fd, J, p) for J in R for (fd, od) in enumerate(offs) for p in (point_position(g, J, od),)
            if !isghost(J, od, p) && hypot(p[1] - c[1], p[2] - c[2]) <= GHOST_FIT[] * h]
end

"""
Weights of the ghost value at the point `q` (field `f` of `group`) in terms of the solid points `data` (field, cell,
position): the polynomial fit (of degree `GHOST_DEGREE`, lower if the data do not determine it) with the boundary
condition at degree + 1 surface points.
"""
function ghost_weights(group, s::Surface, q, f, data, h, C; degree=GHOST_DEGREE[])
    nf = group === Val(:stress) ? 3 : 2
    P, n = nearest_surface_point(s, q)
    t = (-n[2], n[1])
    for deg in degree:-1:1
        nm = length(exponents(deg))
        colloc = [m == 0 ? (P, n) : nearest_surface_point(s, (P[1] + m * h * t[1], P[2] + m * h * t[2])) for m in -deg÷2:deg-deg÷2]
        B = reduce(vcat, [bc_rows(group, p, nn, P, h, deg, C) for (p, nn) in colloc])
        A = zeros(length(data), nm * nf)
        w = zeros(length(data))
        for (j, (fj, _, pj)) in enumerate(data)
            A[j, nm*(fj-1)+1:nm*fj] .= monomials((pj[1] - P[1]) / h, (pj[2] - P[2]) / h, deg)
            w[j] = exp(-((pj[1] - q[1])^2 + (pj[2] - q[2])^2) / (8h^2))     # nearer points count more
        end
        F = svd(B; full=true)
        rank = count(>(1e-10 * F.S[1]), F.S)
        N = F.V[:, rank+1:end]                                 # coefficients that satisfy the boundary condition
        W = sqrt.(w)
        M = W .* (A * N)
        deg == 1 || LinearAlgebra.rank(M; rtol=1e-8) == size(M, 2) || continue  # not determined: lower degree
        sol = pinv(M; rtol=1e-10) .* W'                        # least squares in the null space
        e = zeros(nm * nf)
        e[nm*(f-1)+1:nm*f] .= monomials((q[1] - P[1]) / h, (q[2] - P[2]) / h, deg)
        w = vec((e' * N) * sol)
        (deg == 1 || sum(abs, w) <= GHOST_WMAX[]) && return w   # large weights (an ill-posed extrapolation): lower degree
    end
end

"""
Ghost tables of the immersed free surface: per group (velocity, stress) the target points (field, linear index), and per
target the source points and weights (CSR); targets without sources are set to zero. Cells whose 4th-order differences
would reach into the air use 2nd-order differences (the zones `fs_top`, `fs_bot` of `med` are set accordingly, as next to
the staircase surface): the ghost points are then at most about a cell away from the surface, where the extrapolation is
stable. In the absorbing layers at the sides (`na` cells) the surface is the staircase of the cells (as without
topography): the ghost values satisfy the traction-free condition of the elastic equations, not of the PML's stretched
ones, which is unstable there.
"""
function ghost_tables(s::Surface, g::Grid{2}, med, mode::Val{:psv}, FT, dev, na)
    offs = immersed_offsets(mode)
    sz = size(med.rho)
    h = g.d[1]
    interior = H+na+1:H+g.n[1]-na                             # padded columns between the absorbing layers
    staircase_absorbers!(med, s, g, interior)
    cache = Dict{Tuple{CartesianIndex{2},NTuple{2,Float64}},Bool}()      # (the nearest surface points are costly)
    air(I, o) = I[1] in interior && get!(() -> ghosted(s, point_position(g, I, o), h), cache, (I, o))
    inside(I) = all(1 .<= Tuple(I) .<= sz)
    # cells with a solid point, and the air points near them (the band that is overwritten)
    cellsolid = [any(o -> !air(I, o), (offs.velocity..., offs.stress...)) for I in CartesianIndices(sz)]
    near = falses(sz)
    for I in findall(cellsolid)
        for J in CartesianIndices(ntuple(a -> max(I[a] - GHOST_BAND, 1):min(I[a] + GHOST_BAND, sz[a]), 2))
            near[J] = true
        end
    end
    # 2nd-order differences in the cells whose 4th-order differences reach into the air
    reads(other, fo, two) = (r = stencil_reads(mode, Val(other), fo); two ? (r[2], r[3], r[6], r[7]) : r)
    flagged = map(CartesianIndices(sz)) do I
        cellsolid[I] || return false
        any((:velocity, :stress)) do other
            grp = other === :velocity ? :stress : :velocity
            any(((fo, oo),) -> !air(I, oo) && any(((fr, d),) -> (J = I + CartesianIndex(d); inside(J) && air(J, offs[grp][fr])),
                                                      reads(other, fo, false)), enumerate(offs[other]))
        end
    end
    for i in interior
        ks = findall(view(flagged, i, :))
        med.fs_top[i], med.fs_bot[i] = isempty(ks) ? (0, -1) : (1, maximum(ks))
    end
    # the ghost points: air points read by solid points
    aircell(J) = !inside(J) || in_air(s, point_position(g, J, (0.0, 0.0)))     # cell centre above the surface
    # a point that is computed: in the interior below the surface, in the absorbing layers not among air cells only
    computed(grp, f, I) = I[1] in interior ? !air(I, offs[grp][f]) : !all(c -> aircell(I + c), stagger(mode)[grp][f])
    # the stiffness for the boundary condition at the surface point P: of the solid cell half a cell below it
    stiffness(P) = cell_stiffness(med, CartesianIndex(clamp(cell_index(g, 1, P[1]), 1, sz[1]), clamp(cell_index(g, 2, P[2] + 0.5h), 1, sz[2])))
    tables = map((:velocity, :stress)) do grp
        other = grp === :velocity ? :stress : :velocity       # the solid points of `other` read the ghosts of `grp`
        needed = Set{Tuple{Int,CartesianIndex{2}}}()
        for I in CartesianIndices(sz), (fo, oo) in enumerate(offs[other])
            (cellsolid[I] && computed(other, fo, I)) || continue
            for (fr, d) in reads(other, fo, flagged[I])
                J = I + CartesianIndex(d)
                inside(J) && air(J, offs[grp][fr]) && push!(needed, (fr, J))
            end
        end
        tf, ti, ptr, sf, si, sw = Int8[], Int32[], Int32[1], Int8[], Int32[], FT[]
        L = LinearIndices(sz)
        # in the absorbing layers next to the interior: the points among air cells only (zero in the staircase scheme; here
        # they would take up the extrapolated values next to them)
        cells_of = stagger(mode)[grp]
        side(I) = !(I[1] in interior) && min(abs(I[1] - first(interior)), abs(I[1] - last(interior))) <= GHOST_BAND
        for I in CartesianIndices(sz), (f, o) in enumerate(offs[grp])
            (near[I] && air(I, o)) || (side(I) && all(c -> aircell(I + c), cells_of[f])) || continue
            push!(tf, f); push!(ti, L[I])
            if (f, I) in needed
                q = point_position(g, I, o)
                data = ghost_data(s, g, I, q, offs[grp], sz, (J, od, _) -> get!(() -> ghosted(s, point_position(g, J, od), h), cache, (J, od)))
                P, _ = nearest_surface_point(s, q)
                C = stiffness(P)
                edge = min(I[1] - first(interior), last(interior) - I[1]) < GHOST_EDGE[]   # next to the absorbing layers
                wts = ghost_weights(Val(grp), s, q, f, data, h, C; degree=edge ? min(2, GHOST_DEGREE[]) : GHOST_DEGREE[])
                for ((fd, J, _), wj) in zip(data, wts)
                    abs(wj) > 1e-12 || continue
                    push!(sf, fd); push!(si, L[J]); push!(sw, FT(wj))
                end
            end
            push!(ptr, length(si) + 1)
        end
        # the filter: solid points near the surface, in x and in depth, of the highest order (6, 4, 2) whose points are
        # solid or ghost points in use
        usable(f, J) = inside(J) && (J[1] in interior ? (!air(J, offs[grp][f]) || (f, J) in needed) : !all(c -> aircell(J + c), cells_of[f]))
        ff, fi, fptr, fsf, fsi, fsw = Int8[], Int32[], Int32[1], Int8[], Int32[], FT[]
        if FILTER[] > 0
            for I in CartesianIndices(sz), (f, o) in enumerate(offs[grp])
                (I[1] in interior && !air(I, o)) || continue
                p = point_position(g, I, o)
                P, _ = nearest_surface_point(s, p)
                hypot(p[1] - P[1], p[2] - P[2]) <= FILTER_BAND * h || continue
                n0 = length(fsi)
                if FILTER_KIND[] === :projection
                    # relax towards the polynomial fit of the points around (without the point itself, with the boundary
                    # condition): smooth fields are fitted to O((kh)^4), the grid-scale mode not at all
                    data = filter(d -> !(d[1] == f && d[2] == I),
                                  ghost_data(s, g, I, p, offs[grp], sz, (J, od, _) -> get!(() -> ghosted(s, point_position(g, J, od), h), cache, (J, od))))
                    w = ghost_weights(Val(grp), s, p, f, data, h, stiffness(P))
                    for ((fd, J, _), wj) in zip(data, w)
                        abs(wj) > 1e-12 || continue
                        push!(fsf, fd); push!(fsi, L[J]); push!(fsw, FT(FILTER[] * wj))
                    end
                    push!(fsf, f); push!(fsi, L[I]); push!(fsw, FT(-FILTER[]))
                    push!(ff, f); push!(fi, L[I]); push!(fptr, length(fsi) + 1)
                    continue
                end
                for a in 1:2
                    e = CartesianIndex(ntuple(==(a), 2))
                    m = 0
                    while m < 3 && usable(f, I + (m + 1) * e) && usable(f, I - (m + 1) * e)
                        m += 1
                    end
                    m < FILTER_ORDER[] && continue                 # (lower orders next to the surface damp the waves)
                    for j in -m:m                                # -(-1)^j binomial(2m, m + j) / 4^m: damps (-1)^j by FILTER
                        push!(fsf, f); push!(fsi, L[I+j*e]); push!(fsw, FT(-FILTER[] * (-1)^j * binomial(2m, m + j) / 4^m))
                    end
                end
                length(fsi) > n0 || continue
                push!(ff, f); push!(fi, L[I]); push!(fptr, length(fsi) + 1)
            end
        end
        (; tf=dev(tf), ti=dev(ti), ptr=dev(ptr), sf=dev(sf), si=dev(si), sw=dev(sw), n=length(ti),
           filter=(; ff=dev(ff), fi=dev(fi), ptr=dev(fptr), sf=dev(fsf), si=dev(fsi), sw=dev(fsw), buf=dev(zeros(FT, length(fi))), n=length(fi)))
    end
    return (; velocity=tables[1], stress=tables[2])
end

"""
The absorbing layers at the sides with the staircase surface: their cells above the surface are air, with the 2nd-order
zones of the staircase (`surface_zones`).
"""
function staircase_absorbers!(med, s::Surface, g::Grid{2}, interior)
    sz = size(med.rho)
    for i in 1:sz[1], k in 1:sz[2]
        i in interior && continue
        in_air(s, point_position(g, CartesianIndex(i, k), (0.0, 0.0))) || continue
        med.rho[i, k], med.lam[i, k], med.mu[i, k] = 0.001, 0.0, 0.0
        haskey(med, :wx) && (med.wx[i, k] = med.wz[i, k] = med.s13[i, k] = med.mu66[i, k] = 0.0)
    end
    z = surface_zones(med.lam, med.mu, g.n, map(n -> 1:n, sz))
    for i in 1:sz[1]
        i in interior && continue
        med.fs_top[i], med.fs_bot[i] = z.fs_top[i], z.fs_bot[i]
    end
end

"The medium for the interpolation tables (`interpolation = :sinc`): the cells whose centre is in the air are air."
function sinc_medium(med, s::Surface, g::Grid{2})
    lam, mu = copy(med.lam), copy(med.mu)
    for I in CartesianIndices(lam)
        in_air(s, point_position(g, I, (0.0, 0.0))) && (lam[I] = mu[I] = 0)
    end
    return merge(med, (; lam, mu))
end

"Ghost value `n` of the table `T` for the fields `A` (a tuple of arrays of one group)."
Base.@propagate_inbounds function fill_ghost!(A, T, n)
    acc = zero(eltype(A[1]))
    for j in T.ptr[n]:T.ptr[n+1]-1
        acc += T.sw[j] * pick(A, T.sf[j], T.si[j])
    end
    place!(A, T.tf[n], T.ti[n], acc)
end
@inline pick(A::NTuple{2}, f, i) = @inbounds ifelse(f == 1, A[1][i], A[2][i])
@inline pick(A::NTuple{3}, f, i) = @inbounds f == 1 ? A[1][i] : f == 2 ? A[2][i] : A[3][i]
@inline place!(A::Tuple, f, i, v) = @inbounds (f == 1 ? (A[1][i] = v) : f == 2 ? (A[2][i] = v) : (A[3][i] = v); nothing)

"Filter increment of the point `n` of the filter table `T` (into `T.buf`)."
Base.@propagate_inbounds function filter_increment!(A, T, n)
    acc = zero(eltype(A[1]))
    for j in T.ptr[n]:T.ptr[n+1]-1
        acc += T.sw[j] * pick(A, T.sf[j], T.si[j])
    end
    T.buf[n] = acc
end
Base.@propagate_inbounds filter_apply!(A, T, n) = place!(A, T.ff[n], T.fi[n], pick(A, T.ff[n], T.fi[n]) + T.buf[n])

"Fill the ghost points of the group of fields `A`, after the filter near the surface."
function fill_ghosts!(backend, A, T)
    if T.filter.n > 0
        foreach_item!(backend, filter_increment!, (A, T.filter), T.filter.n)
        foreach_item!(backend, filter_apply!, (A, T.filter), T.filter.n)
    end
    T.n > 0 && foreach_item!(backend, fill_ghost!, (A, T), T.n)
end
fill_ghosts!(backend, A, ::Nothing) = nothing

"""
The model arrays `m` (model axes x, y, z up) with the free surface `s`: cells above it are air, and those near it get the
material of the nearest solid cell; solid cells below it that are air in the model get it too.
"""
function immersed_model(m, s::Surface)
    nx, nz = length(m.x), length(m.z)
    solid = [!in_air(s, (m.x[i], -m.z[k])) for i in 1:nx, k in 1:nz]
    isrock(i, k) = solid[i, k] && !(m.vp[i, 1, k] == 0 && m.vs[i, 1, k] == 0)
    src = fill(CartesianIndex(0, 0), nx, nz)                   # the cell whose material a cell gets (multi-source BFS)
    queue = CartesianIndex{2}[]
    for I in CartesianIndices((nx, nz))
        isrock(I[1], I[2]) && (src[I] = I; push!(queue, I))
    end
    head = 1
    while head <= length(queue)
        I = queue[head]; head += 1
        for d in (CartesianIndex(1, 0), CartesianIndex(-1, 0), CartesianIndex(0, 1), CartesianIndex(0, -1))
            J = I + d
            checkbounds(Bool, src, J) && src[J] == CartesianIndex(0, 0) || continue
            (solid[J] || maximum(abs, Tuple(J - src[I])) <= GHOST_BAND + 2) || continue
            src[J] = src[I]; push!(queue, J)
        end
    end
    any(i -> !any(k -> isrock(i, k), 1:nz), 1:nx) && error("topography: a column of the model has no solid cell below the surface")
    any(I -> solid[I] && src[I] == CartesianIndex(0, 0), CartesianIndices(solid)) && error("topography: solid cells without material")
    function field(A, airvalue)
        B = similar(A)
        for i in 1:nx, k in 1:nz
            S = src[i, k]
            B[i, 1, k] = S == CartesianIndex(0, 0) ? airvalue : A[S[1], 1, S[2]]
        end
        return B
    end
    air = (; rho=0.001, vp=0.0, vs=0.0, Qp=Inf, Qs=Inf, epsilon=0.0, delta=0.0, gamma=0.0)
    return merge(m, NamedTuple{filter(k -> haskey(air, k), keys(m))}(map(k -> field(m[k], air[k]), filter(k -> haskey(air, k), keys(m)))))
end
