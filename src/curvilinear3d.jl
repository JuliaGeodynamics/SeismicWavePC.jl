# 3D on a terrain-following (curvilinear) grid: the scheme of curvilinear.jl with a second horizontal direction. The mesh
# has vertical columns (x = ξ, y = ψ on SBP nodes, z = Z(ξ, ψ, η)); metric: J = z_η, Jξ = (J, 0, 0), Jψ = (0, J, 0),
# Jη = (Jx, Jy, 1) = (-z_ξ, -z_ψ, 1). Momentum: ρ J ∂t v_c = Dξ(J σ_c1) + Dψ(J σ_c2) + Dη(Jx σ_c1 + Jy σ_c2 + σ_c3); gradient:
# ∂x = Dξ + (Jx/J) Dη, ∂y = Dψ + (Jy/J) Dη, ∂z = Dη/J. SAT terms, blocks, time stepping as in 2D.

"`flag ? interior : general` SBP derivative at the local index i (global i + o of n: MPI subdomains; the boundary closures
in global indices). No closure over the element type: the kernels stay type stable."
@inline dmin(g::G, i, o, n, ::Val{int}, ::Type{T}) where {G,int,T} = int ? dminus_int(g, i, T) : dminus(q -> g(q - o), i + o, n)
@inline dpl(g::G, i, o, n, ::Val{int}, ::Type{T}) where {G,int,T} = int ? dplus_int(g, i, T) : dplus(q -> g(q - o), i + o, n)

# The tractions across η = const (scaled), G_c = Jx σ_c1 + Jy σ_c2 + σ_c3, in a pointwise pass before the momentum update:
# the η stencils then read one array each (7× faster on CPUs: 8 planes of three arrays did not stay in cache).
@inline Base.@propagate_inbounds function traction3_cv!(B, i, j, k)
    jx, jy = B.Jx[i, j, k], B.Jy[i, j, k]
    B.G1[i, j, k] = jx * B.S11[i, j, k] + jy * B.S12[i, j, k] + B.S13[i, j, k]
    B.G2[i, j, k] = jx * B.S12[i, j, k] + jy * B.S22[i, j, k] + B.S23[i, j, k]
    B.G3[i, j, k] = jx * B.S13[i, j, k] + jy * B.S23[i, j, k] + B.S33[i, j, k]
end

# Momentum at node (i, j, k); the region flags (fx, fy, fz): interior in x, y, η (no closures, no SAT terms there).
@inline Base.@propagate_inbounds function momentum3_cv!(B, Bd, Bu, dt, damp, ::Val{flags}, i, j, k) where {flags}
    (; V1, V2, V3, S11, S22, S33, S23, S13, S12, G1, G2, G3, J, iρJ, a1, a2, a3, hξ, hψ, hη, ox, oy, nxg, nyg) = B
    fx, fy, fz = Val(flags[1]), Val(flags[2]), Val(flags[3])
    nx, ny, nz = size(V1)
    T = eltype(V1)
    w = T(SBP_W[1])
    # J σ_c1 along ξ, J σ_c2 along ψ, G_c along η
    d1 = dmin(q -> @inbounds(J[q, j, k] * S11[q, j, k]), i, ox, nxg, fx, T) / hξ + dmin(q -> @inbounds(J[i, q, k] * S12[i, q, k]), j, oy, nyg, fy, T) / hψ +
         dmin(q -> @inbounds(G1[i, j, q]), k, 0, nz, fz, T) / hη
    d2 = dmin(q -> @inbounds(J[q, j, k] * S12[q, j, k]), i, ox, nxg, fx, T) / hξ + dmin(q -> @inbounds(J[i, q, k] * S22[i, q, k]), j, oy, nyg, fy, T) / hψ +
         dmin(q -> @inbounds(G2[i, j, q]), k, 0, nz, fz, T) / hη
    d3 = dmin(q -> @inbounds(J[q, j, k] * S13[q, j, k]), i, ox, nxg, fx, T) / hξ + dmin(q -> @inbounds(J[i, q, k] * S23[i, q, k]), j, oy, nyg, fy, T) / hψ +
         dmin(q -> @inbounds(G3[i, j, q]), k, 0, nz, fz, T) / hη
    if !flags[1] && (i + ox == 1 || i + ox == nxg)       # sides x: the traction J σ_c1 removed
        s = (i + ox == 1 ? 1 : -1) * J[i, j, k] / (w * hξ)
        d1 += s * S11[i, j, k]; d2 += s * S12[i, j, k]; d3 += s * S13[i, j, k]
    end
    if !flags[2] && (j + oy == 1 || j + oy == nyg)       # sides y: J σ_c2
        s = (j + oy == 1 ? 1 : -1) * J[i, j, k] / (w * hψ)
        d1 += s * S12[i, j, k]; d2 += s * S22[i, j, k]; d3 += s * S23[i, j, k]
    end
    if !flags[3] && (k == 1 || k == nz)                 # free surface, bottom, interfaces
        g1, g2, g3 = G1[i, j, k], G2[i, j, k], G3[i, j, k]
        if k == nz
            if Bu === nothing
                d1 -= g1 / (w * hη); d2 -= g2 / (w * hη); d3 -= g3 / (w * hη)
            else
                d1 -= (g1 - Bu.G1[i, j, 1]) / (2w * hη); d2 -= (g2 - Bu.G2[i, j, 1]) / (2w * hη); d3 -= (g3 - Bu.G3[i, j, 1]) / (2w * hη)
            end
        end
        if k == 1
            if Bd === nothing
                d1 += g1 / (w * hη); d2 += g2 / (w * hη); d3 += g3 / (w * hη)
            else
                n = size(Bd.V1, 3)
                d1 -= (Bd.G1[i, j, n] - g1) / (2w * hη); d2 -= (Bd.G2[i, j, n] - g2) / (2w * hη); d3 -= (Bd.G3[i, j, n] - g3) / (2w * hη)
            end
        end
    end
    r = iρJ[i, j, k]
    if flags[1] && flags[2] && flags[3]
        V1[i, j, k] += dt * r * d1; V2[i, j, k] += dt * r * d2; V3[i, j, k] += dt * r * d3
    else
        b1, b2, b3 = damp ? (a1[i, j, k] * dt / 2, a2[i, j, k] * dt / 2, a3[i, j, k] * dt / 2) : (zero(T), zero(T), zero(T))
        V1[i, j, k] = ((1 - b1) * V1[i, j, k] + dt * r * d1) / (1 + b1)
        V2[i, j, k] = ((1 - b2) * V2[i, j, k] + dt * r * d2) / (1 + b2)
        V3[i, j, k] = ((1 - b3) * V3[i, j, k] + dt * r * d3) / (1 + b3)
    end
end

# Stress at node (i, j, k): gradients with D₊ (half the velocity mismatch at interfaces added to Dη v), and the stress
# update of the staggered 3D solver (generalized Zener body).
@inline Base.@propagate_inbounds function stress3_cv!(B, Bd, Bu, vc, dt, ::Val{flags}, i, j, k) where {flags}
    (; V1, V2, V3, S11, S22, S33, S23, S13, S12, J, Jx, Jy, lam, mu, hξ, hψ, hη, ox, oy, nxg, nyg) = B
    fx, fy, fz = Val(flags[1]), Val(flags[2]), Val(flags[3])
    nx, ny, nz = size(V1)
    T = eltype(V1)
    w = T(SBP_W[1])
    # (no local helper closures: one capturing the type T would make the kernel allocate)
    x1 = dpl(q -> @inbounds(V1[q, j, k]), i, ox, nxg, fx, T) / hξ
    x2 = dpl(q -> @inbounds(V2[q, j, k]), i, ox, nxg, fx, T) / hξ
    x3 = dpl(q -> @inbounds(V3[q, j, k]), i, ox, nxg, fx, T) / hξ
    y1 = dpl(q -> @inbounds(V1[i, q, k]), j, oy, nyg, fy, T) / hψ
    y2 = dpl(q -> @inbounds(V2[i, q, k]), j, oy, nyg, fy, T) / hψ
    y3 = dpl(q -> @inbounds(V3[i, q, k]), j, oy, nyg, fy, T) / hψ
    z1 = dpl(q -> @inbounds(V1[i, j, q]), k, 0, nz, fz, T) / hη
    z2 = dpl(q -> @inbounds(V2[i, j, q]), k, 0, nz, fz, T) / hη
    z3 = dpl(q -> @inbounds(V3[i, j, q]), k, 0, nz, fz, T) / hη
    if !flags[3]
        if k == nz && Bu !== nothing
            z1 -= (V1[i, j, k] - Bu.V1[i, j, 1]) / (2w * hη); z2 -= (V2[i, j, k] - Bu.V2[i, j, 1]) / (2w * hη)
            z3 -= (V3[i, j, k] - Bu.V3[i, j, 1]) / (2w * hη)
        end
        if k == 1 && Bd !== nothing
            n = size(Bd.V1, 3)
            z1 -= (Bd.V1[i, j, n] - V1[i, j, k]) / (2w * hη); z2 -= (Bd.V2[i, j, n] - V2[i, j, k]) / (2w * hη)
            z3 -= (Bd.V3[i, j, n] - V3[i, j, k]) / (2w * hη)
        end
    end
    I = CartesianIndex(i, j, k)
    iJ = 1 / J[I]
    qx, qy = Jx[I] * iJ, Jy[I] * iJ
    dxV1, dyV2, dzV3 = x1 + qx * z1, y2 + qy * z2, z3 * iJ
    eyz = z2 * iJ + y3 + qy * z3
    exz = z1 * iJ + x3 + qx * z3
    exy = y1 + qy * z1 + x2 + qx * z2
    mu2 = 2mu[I]
    l2m = lam[I] + mu2
    tp, ts = orzero(B.tp, I), orzero(B.ts, I)
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    div = dxV1 + dyV2 + dzV3
    Rx = cmemvar!(memory(B, Val(:R11)), I, l2m * tp * div - mu2 * ts * (dyV2 + dzV3), vc, dt)
    Ry = cmemvar!(memory(B, Val(:R22)), I, l2m * tp * div - mu2 * ts * (dxV1 + dzV3), vc, dt)
    Rz = cmemvar!(memory(B, Val(:R33)), I, l2m * tp * div - mu2 * ts * (dxV1 + dyV2), vc, dt)
    S11[I] += (l2m * tp1 * div - mu2 * ts1 * (dyV2 + dzV3) + Rx) * dt
    S22[I] += (l2m * tp1 * div - mu2 * ts1 * (dxV1 + dzV3) + Ry) * dt
    S33[I] += (l2m * tp1 * div - mu2 * ts1 * (dxV1 + dyV2) + Rz) * dt
    m = mu[I]
    Ryz = cmemvar!(memory(B, Val(:R23)), I, m * ts * eyz, vc, dt)
    Rxz = cmemvar!(memory(B, Val(:R13)), I, m * ts * exz, vc, dt)
    Rxy = cmemvar!(memory(B, Val(:R12)), I, m * ts * exy, vc, dt)
    S23[I] += (m * ts1 * eyz + Ryz) * dt
    S13[I] += (m * ts1 * exz + Rxz) * dt
    S12[I] += (m * ts1 * exy + Rxy) * dt
end

Base.@propagate_inbounds function force3_cv!(blocks, S, it, dt)
    for m in 1:length(S.i)
        B = blocks[S.b[m]]
        i, j, k, n = S.i[m], S.j[m], S.k[m], S.n[m]
        s = dt * S.c[m] * S.s[n, it] * B.iρJ[i, j, k]
        B.V1[i, j, k] += S.F[n, 1] * s; B.V2[i, j, k] += S.F[n, 2] * s; B.V3[i, j, k] += S.F[n, 3] * s
    end
end
Base.@propagate_inbounds function moment3_cv!(blocks, S, it)
    for m in 1:length(S.i)
        B = blocks[S.b[m]]
        i, j, k, n = S.i[m], S.j[m], S.k[m], S.n[m]
        s = S.c[m] * S.s[n, it]
        B.S11[i, j, k] -= S.M[n, 1] * s; B.S22[i, j, k] -= S.M[n, 2] * s; B.S33[i, j, k] -= S.M[n, 3] * s
        B.S23[i, j, k] -= S.M[n, 4] * s; B.S13[i, j, k] -= S.M[n, 5] * s; B.S12[i, j, k] -= S.M[n, 6] * s
    end
end
Base.@propagate_inbounds function sample3_cv(A, St, n)
    s = zero(eltype(A))
    for c in 1:6, b in 1:6, a in 1:6
        s += St.wx[a, n] * St.wy[b, n] * St.wz[c, n] * A[St.i0[n]+a-1, St.j0[n]+b-1, St.k0[n]+c-1]
    end
    return s
end
Base.@propagate_inbounds function vprev3_cv!(prev, blocks, St, n)
    B = blocks[St.b[n]]
    prev[1, n] = sample3_cv(B.V1, St, n); prev[2, n] = sample3_cv(B.V2, St, n); prev[3, n] = sample3_cv(B.V3, St, n)
end
Base.@propagate_inbounds function record3_cv!(rec, acc, prev, blocks, St, itw, dt, n)
    B = blocks[St.b[n]]
    v = (sample3_cv(B.V1, St, n), sample3_cv(B.V2, St, n), sample3_cv(B.V3, St, n))
    if itw > 0
        σ = (sample3_cv(B.S11, St, n), sample3_cv(B.S22, St, n), sample3_cv(B.S33, St, n), sample3_cv(B.S23, St, n),
             sample3_cv(B.S13, St, n), sample3_cv(B.S12, St, n))
        l, m = sample3_cv(B.lam, St, n), sample3_cv(B.mu, St, n)
        tr = (σ[1] + σ[2] + σ[3]) * l / (3l + 2m)
        e = ((σ[1] - tr) / 2m, (σ[2] - tr) / 2m, (σ[3] - tr) / 2m, σ[4] / 2m, σ[5] / 2m, σ[6] / 2m)
        for c in 1:3
            rec[itw, c, n] = (prev[c, n] + v[c]) / 2
            rec[itw, 3+c, n] = acc[c, n]
        end
        for c in 1:6
            rec[itw, 6+c, n] = σ[c]
            rec[itw, 12+c, n] = e[c]
        end
    end
    for c in 1:3
        acc[c, n] += v[c] * dt
    end
end

"""
    simulate_curvilinear3d(model; tmax or nt, sources, ...)

3D on a deformed (terrain-following) `CartData` (vertical columns; see [`curvilinear_mesh`](@ref)); called by `simulate`.
Keywords as [`SeismicWavePC.simulate_curvilinear`](@ref).
"""
function simulate_curvilinear3d(model::CartData; dt=nothing, nt=nothing, tmax=nothing, sources, stations=Station[], stf=:kupper,
                                tbeg=0.0, absorber=:impedance, fq_min=0.05, fq_max=5.0, fq_ref=1.0, record_every=1,
                                density_unit=:kg_m3, snapshot_every=0, snapshot_dir="snapshots", callback=nothing, backend=CPU(),
                                T::Type{FT}=Float32, verbose=true, wavefield=true, jump=0.05, wave=:psv, interpolation=:sinc,
                                free_surface=:openswpc, nproc=1, kwargs...) where {FT}
    isempty(kwargs) || error("simulate on a curvilinear grid: not (yet) supported: $(join(keys(kwargs), ", "))")
    absorber in (:impedance, :none) || error("curvilinear grids absorb with impedance boundary conditions: absorber = :impedance (or :none)")
    srcs = sources isa AbstractVector ? sources : [sources]
    all(s -> s isa Union{MomentTensorSource,ForceSource}, srcs) || error("curvilinear grids: MomentTensorSource and ForceSource only")
    wall = time()
    dev(A) = (B = KernelAbstractions.allocate(backend, eltype(A), size(A)); copyto!(B, A); B)
    X, Y, Z = model.x.val, model.y.val, model.z.val
    nx, ny, nzt = size(X)
    all(k -> view(X, :, :, k) == view(X, :, :, 1), 1:nzt) && all(k -> view(Y, :, :, k) == view(Y, :, :, 1), 1:nzt) ||
        error("curvilinear grid: the columns of the mesh must be vertical")
    all(j -> view(X, :, j, 1) == view(X, :, 1, 1), 1:ny) && all(i -> view(Y, i, :, 1) == view(Y, 1, :, 1), 1:nx) ||
        error("curvilinear grid: x must not depend on y and y not on x")
    gξ, gψ = 1e3 .* X[:, 1, 1], 1e3 .* Y[1, :, 1]
    gx, hξ = sbp_grid(gξ[1], gξ[end], nx)
    gy, hψ = sbp_grid(gψ[1], gψ[end], ny)
    (maximum(abs, gx .- gξ) < 1e-6 * (gξ[end] - gξ[1]) && maximum(abs, gy .- gψ) < 1e-6 * (gψ[end] - gψ[1])) ||
        error("curvilinear grid: x and y must be the nodes of the SBP grid (make the mesh with curvilinear_mesh)")
    Zm = 1e3 .* Z
    all(>(0), diff(Zm; dims=3)) || error("curvilinear grid: z must increase with the row index (and rows must not touch)")
    # MPI: this process holds the nodes ia:ib in x, ja:jb in y, and CV_HALO halo nodes around them (local = global - offset)
    par = cv_mpi(nproc, 2)
    me = par === nothing ? 0 : par.me
    ia, ib = par === nothing ? (1, nx) : extrema(part_range(nx, par.dims[1], par.coords[1]))
    ja, jb = par === nothing ? (1, ny) : extrema(part_range(ny, par.dims[2], par.coords[2]))
    par === nothing || min(ib - ia, jb - ja) + 1 >= SBP_NB + CV_HALO || error("curvilinear grid: too many processes for $nx × $ny nodes")
    hx = par === nothing ? 0 : CV_HALO
    ox, oy = ia - hx - 1, ja - hx - 1
    xr, yr = clamp.(ia-hx:ib+hx, 1, nx), clamp.(ja-hx:jb+hx, 1, ny)
    nlx, nly, nox, noy = length(xr), length(yr), ib - ia + 1, jb - ja + 1
    f = model.fields
    haskey(f, :rho) && haskey(f, :vp) && haskey(f, :vs) || error("the model needs fields rho, vp and vs")
    ρ = f.rho .* (density_unit === :g_cm3 ? 1e3 : 1.0)
    vp, vs = 1e3 .* f.vp, 1e3 .* f.vs
    minimum(vs) > 0 || error("curvilinear grids: solids only (vs > 0 everywhere)")
    qp = haskey(f, :Qp) ? Float64.(f.Qp) : fill(Inf, size(X))
    qs = haskey(f, :Qs) ? Float64.(f.Qs) : fill(Inf, size(X))
    attenuation = any(isfinite, qp) || any(isfinite, qs)
    ts = relaxation_times(fq_min, fq_max)
    zeta = constq_zeta(fq_min, fq_max, ts)
    # blocks: from the columns (flattened)
    flat(A) = reshape(A, nx * ny, nzt)
    rows, below = find_blocks(flat(Zm), flat(vp), flat(vs); jump)
    edges = [1; rows; nzt]
    any(b -> edges[b+1] - edges[b] + 1 < SBP_NMIN, 1:length(edges)-1) &&
        error("curvilinear grid: every block (between interfaces) needs at least $SBP_NMIN rows; interfaces at rows $rows")
    Hξ, Hψ = sbp_weights(nx, hξ), sbp_weights(ny, hψ)
    hostblocks = map(1:length(edges)-1) do b
        r = edges[b]:edges[b+1]
        m = map(A -> copy(A[:, :, r]), (ρ, vp, vs, qp, qs))
        for (q, side, src) in ((b - 1, false, 2), (b, true, length(r) - 1))
            1 <= q <= length(rows) || continue
            kk = side ? length(r) : 1
            for c in 1:nx*ny
                below[q][c] == side && continue
                i, j = mod1(c, nx), fld1(c, nx)
                foreach(A -> A[i, j, kk] = A[i, j, src], m)
            end
        end
        bρ, bvp, bvs, bqp, bqs = m
        nz = length(r)
        _, hη = sbp_grid(0.0, 1.0, nz)
        Zb = Zm[:, :, r]
        J = [dminus(q -> Zb[i, j, q], k, nz) / hη for i in 1:nx, j in 1:ny, k in 1:nz]
        Jx = [-dminus(q -> Zb[q, j, k], i, nx) / hξ for i in 1:nx, j in 1:ny, k in 1:nz]
        Jy = [-dminus(q -> Zb[i, q, k], j, ny) / hψ for i in 1:nx, j in 1:ny, k in 1:nz]
        minimum(J) > 0 || error("curvilinear grid: the mesh folds in block $b")
        taup, taus = NM * zeta ./ bqp, NM * zeta ./ bqs
        mu = @. bρ * bvs^2 / visco_chi((ts,), taus, fq_ref)^2
        lam = @. bρ * bvp^2 / visco_chi((ts,), taup, fq_ref)^2 - 2mu
        a1, a2, a3 = zeros(nx, ny, nz), zeros(nx, ny, nz), zeros(nx, ny, nz)
        if absorber === :impedance                      # a = |J·| Z / (H ρ J): normal vp, tangential vs
            for i in (1, nx)
                @. a1[i, :, :] += bvp[i, :, :] / Hξ[i]; @. a2[i, :, :] += bvs[i, :, :] / Hξ[i]; @. a3[i, :, :] += bvs[i, :, :] / Hξ[i]
            end
            for j in (1, ny)
                @. a1[:, j, :] += bvs[:, j, :] / Hψ[j]; @. a2[:, j, :] += bvp[:, j, :] / Hψ[j]; @. a3[:, j, :] += bvs[:, j, :] / Hψ[j]
            end
            if b == 1
                g = @. sqrt(Jx[:, :, 1]^2 + Jy[:, :, 1]^2 + 1) / (SBP_W[1] * hη * J[:, :, 1])
                @. a1[:, :, 1] += g * bvs[:, :, 1]; @. a2[:, :, 1] += g * bvs[:, :, 1]; @. a3[:, :, 1] += g * bvp[:, :, 1]
            end
        end
        (; nz, hη, Z=Zb, J, Jx, Jy, ρ=bρ, lam, mu, taup, taus, a1, a2, a3)
    end
    nb = length(hostblocks)
    blocks = Tuple(map(hostblocks) do H
        z() = KernelAbstractions.zeros(backend, FT, nlx, nly, H.nz)
        R() = KernelAbstractions.zeros(backend, FT, NM, nlx, nly, H.nz)
        loc(A) = dev(FT.(A[xr, yr, :]))
        mem = attenuation ? (; R11=R(), R22=R(), R33=R(), R23=R(), R13=R(), R12=R()) : (;)
        merge((; V1=z(), V2=z(), V3=z(), S11=z(), S22=z(), S33=z(), S23=z(), S13=z(), S12=z(), G1=z(), G2=z(), G3=z(), J=loc(H.J),
               Jx=loc(H.Jx), Jy=loc(H.Jy), iρJ=loc(1 ./ (H.ρ .* H.J)), lam=loc(H.lam), mu=loc(H.mu),
               tp=attenuation ? loc(H.taup) : nothing, ts=attenuation ? loc(H.taus) : nothing, a1=loc(H.a1), a2=loc(H.a2),
               a3=loc(H.a3), hξ=FT(hξ), hψ=FT(hψ), hη=FT(H.hη), ox, oy, nxg=nx, nyg=ny), mem)
    end)
    below_(b) = b > 1 ? blocks[b-1] : nothing
    above_(b) = b < nb ? blocks[b+1] : nothing
    # up to 27 regions per block: low / interior / high in x, y and η, with their kernel flags (in x and y: the global
    # segments intersected with the nodes of this process, in local indices)
    m = SBP_NB
    function seg(n, a0, b0, o)
        S = Tuple{Int,Int,Bool}[]
        for (lo, hi, f) in ((1, m, false), (m + 1, n - m, true), (n - m + 1, n, false))
            a, b = max(lo, a0), min(hi, b0)
            a <= b && push!(S, (a - o - 1, b - a + 1, f))
        end
        S
    end
    regions = map(hostblocks) do H
        Tuple(((a[1], b[1], c[1]), (a[2], b[2], c[2]), Val((a[3], b[3], c[3]))) for a in seg(nx, ia, ib, ox), b in seg(ny, ja, jb, oy),
              c in seg(H.nz, 1, H.nz, 0))
    end
    function momentum!(dtT, damp)
        for b in 1:nb                                  # the tractions of all blocks first (interfaces read the neighbours')
            foreach_cell!(backend, traction3_cv!, (blocks[b],), (0, 0, 0), (nlx, nly, hostblocks[b].nz))
        end
        for b in 1:nb
            foreach_region!(backend, momentum3_cv!, (blocks[b], below_(b), above_(b), dtT, damp), regions[b])
        end
    end
    stress!(vc, dtT) = for b in 1:nb
        foreach_region!(backend, stress3_cv!, (blocks[b], below_(b), above_(b), vc, dtT), regions[b])
    end
    fields(B) = (B.V1, B.V2, B.V3, B.S11, B.S22, B.S33, B.S23, B.S13, B.S12)
    exchange_v!() = foreach(B -> cv_exchange!(par, (B.V1, B.V2, B.V3)), blocks)
    exchange_s!() = foreach(B -> cv_exchange!(par, (B.S11, B.S22, B.S33, B.S23, B.S13, B.S12)), blocks)
    owned(A) = view(A, hx+1:hx+nox, hx+1:hx+noy, :)
    vnorm2() = cv_sum(par, sum(B -> sum(abs2, owned(B.V1)) + sum(abs2, owned(B.V2)) + sum(abs2, owned(B.V3)), blocks))
    if dt === nothing
        rng = Random.Xoshiro(20261126 + me)                 # (seeded: the same time step in every run)
        foreach(B -> foreach(A -> copyto!(A, FT.(randn(rng, size(A)))), (B.V1, B.V2, B.V3)), blocks)
        exchange_v!()
        λ = 0.0
        for _ in 1:100
            foreach(B -> foreach(A -> fill!(A, 0), (B.S11, B.S22, B.S33, B.S23, B.S13, B.S12)), blocks)
            stress!(nothing, one(FT))
            exchange_s!()
            nrm0 = sqrt(vnorm2())
            foreach(B -> foreach(A -> fill!(A, 0), (B.V1, B.V2, B.V3)), blocks)
            momentum!(one(FT), false)
            KernelAbstractions.synchronize(backend)
            exchange_v!()
            nrm = sqrt(vnorm2())
            λ = nrm / nrm0
            foreach(B -> foreach(A -> (A ./= FT(nrm)), (B.V1, B.V2, B.V3)), blocks)
        end
        foreach(B -> foreach(A -> fill!(A, 0), fields(B)), blocks)
        dt = 0.9 * 2 / sqrt(λ)
    end
    nt = something(nt, tmax === nothing ? nothing : round(Int, tmax / dt))
    nt === nothing && error("give the duration tmax (or nt, with dt)")
    dtT = FT(dt)
    vc = attenuation ? memory_coefficients(ts, dt, FT) : nothing
    # locate points: block and η̂ (Lagrange interpolation of the columns, bisection)
    function locate(x, y, z)
        i0, wx = lagrange_weights(gx, x)
        j0, wy = lagrange_weights(gy, y)
        for (b, H) in enumerate(hostblocks)
            col = [sum(wx[a] * wy[c] * H.Z[i0+a-1, j0+c-1, k] for a in 1:6, c in 1:6) for k in 1:H.nz]
            (col[1] - 1e-9 <= z <= col[end] + 1e-9) || continue
            gη, _ = sbp_grid(0.0, 1.0, H.nz)
            zof(η) = let (q0, wq) = lagrange_weights(gη, η)
                sum(wq[c] * col[q0+c-1] for c in 1:6)
            end
            lo, hi = 0.0, 1.0
            for _ in 1:60
                mid = (lo + hi) / 2
                zof(mid) < z ? (lo = mid) : (hi = mid)
            end
            return b, (lo + hi) / 2
        end
        error("the point ($(x / 1e3), $(y / 1e3), $(z / 1e3)) km is outside the model")
    end
    function source_table(list, coef)
        tab = (b=Int[], i=Int[], j=Int[], k=Int[], c=Float64[], n=Int[])
        for (n, s) in enumerate(list)
            b, η = locate(1e3s.x, 1e3s.y, 1e3s.z)
            H = hostblocks[b]
            gη, hη = sbp_grid(0.0, 1.0, H.nz)
            iξ, wξ = delta_weights(gx, Hξ, hξ, 1e3s.x)
            iψ, wψ = delta_weights(gy, Hψ, hψ, 1e3s.y)
            iη, wη = delta_weights(gη, sbp_weights(H.nz, hη), hη, η)
            for (a, i) in enumerate(iξ), (c, j) in enumerate(iψ), (d, k) in enumerate(iη)
                push!(tab.b, b); push!(tab.i, i); push!(tab.j, j); push!(tab.k, k); push!(tab.n, n)
                push!(tab.c, wξ[a] * wψ[c] * wη[d] * coef(H, i, j, k))
            end
        end
        return tab
    end
    forces = filter(s -> s isa ForceSource, srcs)
    moments = filter(s -> s isa MomentTensorSource, srcs)
    mine(t) = (q = findall(n -> ia <= t.i[n] <= ib && ja <= t.j[n] <= jb, eachindex(t.i));
               (b=t.b[q], i=t.i[q] .- ox, j=t.j[q] .- oy, k=t.k[q], c=t.c[q], n=t.n[q]))
    ft = mine(source_table(forces, (H, i, j, k) -> 1.0))
    mt = mine(source_table(moments, (H, i, j, k) -> 1 / H.J[i, j, k]))
    FS = (; b=dev(ft.b), i=dev(ft.i), j=dev(ft.j), k=dev(ft.k), c=dev(FT.(ft.c)), n=dev(ft.n), F=dev(FT[s.F[c] for s in forces, c in 1:3]),
          s=dev(FT[moment_rate(stf, tbeg + (it - 1) * dt, s.t0, s.trise) for s in forces, it in 1:nt]))
    MS = (; b=dev(mt.b), i=dev(mt.i), j=dev(mt.j), k=dev(mt.k), c=dev(FT.(mt.c)), n=dev(mt.n), M=dev(FT[s.M0 * s.M[c] for s in moments, c in 1:6]),
          s=dev(FT[moment_rate(stf, tbeg + (it - 0.5) * dt, s.t0, s.trise) * dt for s in moments, it in 1:nt]))
    nst = length(stations)
    st = map(stations) do s
        b, η = s.surface ? (nb, 1.0) : locate(1e3s.x, 1e3s.y, 1e3s.z)
        gη, _ = sbp_grid(0.0, 1.0, hostblocks[b].nz)
        i0, wx = lagrange_weights(gx, 1e3s.x)
        j0, wy = lagrange_weights(gy, 1e3s.y)
        k0, wz = lagrange_weights(gη, η)
        (b, i0, j0, k0, wx, wy, wz)
    end
    own = findall(s -> ia <= s[2] + 2 <= ib && ja <= s[3] + 2 <= jb, st)     # the stations of this process
    st = st[own]
    nst = length(st)
    St = (; b=dev([s[1] for s in st]), i0=dev([s[2] - ox for s in st]), j0=dev([s[3] - oy for s in st]), k0=dev([s[4] for s in st]),
          wx=dev(FT[s[5][a] for a in 1:6, s in st]), wy=dev(FT[s[6][a] for a in 1:6, s in st]), wz=dev(FT[s[7][a] for a in 1:6, s in st]))
    nrec = cld(nt, record_every)
    rec = KernelAbstractions.zeros(backend, FT, nrec, 18, nst)
    acc = KernelAbstractions.zeros(backend, FT, 3, nst)
    prev = KernelAbstractions.zeros(backend, FT, 3, nst)
    function velocity_field()                       # (with MPI: gathered on rank 0, `nothing` elsewhere)
        keep(b) = b == 1 ? (1:hostblocks[b].nz) : (2:hostblocks[b].nz)
        r(c) = Float64.(cat((Array(owned(getfield(blocks[b], c)))[:, :, keep(b)] for b in 1:nb)...; dims=3))
        V = (r(:V1), r(:V2), r(:V3))
        if par !== nothing
            parts = MPI.gather((ia, ib, ja, jb, V), par.comm; root=0)
            me == 0 || return nothing
            V = ntuple(_ -> zeros(nx, ny, nzt), 3)
            for (a, b, c, d, p) in parts, q in 1:3
                V[q][a:b, c:d, :] .= p[q]
            end
        end
        return CartData(model.x.val, model.y.val, model.z.val, (; V))
    end
    movie = (snapshot_every > 0 && callback === nothing && me == 0) ? movie_paraview(name=joinpath(mkpath(snapshot_dir), "wavefield")) : nothing
    verbose &= me == 0
    verbose && @printf("SeismicWavePC: 3D on a curvilinear grid, %d × %d × %d nodes in %d block(s) (interfaces at rows %s), %d steps of %.4g s%s%s\n",
                       nx, ny, nzt, nb, isempty(rows) ? "–" : join(rows, ", "), nt, dt, attenuation ? ", attenuation" : "",
                       absorber === :impedance ? ", absorbing sides and bottom" : "")
    for it in 1:nt
        n = it - 1
        itw = n % record_every == 0 ? n ÷ record_every + 1 : 0
        itw > 0 && nst > 0 && foreach_item!(backend, vprev3_cv!, (prev, blocks, St), nst)
        momentum!(dtT, true)
        length(forces) > 0 && run_once!(backend, force3_cv!, (blocks, FS, it, dtT))
        KernelAbstractions.synchronize(backend)
        exchange_v!()
        nst > 0 && foreach_item!(backend, record3_cv!, (rec, acc, prev, blocks, St, itw, dtT), nst)
        stress!(vc, dtT)
        length(moments) > 0 && run_once!(backend, moment3_cv!, (blocks, MS, it))
        KernelAbstractions.synchronize(backend)
        exchange_s!()
        if snapshot_every > 0 && n % snapshot_every == 0
            data = velocity_field()
            if me == 0
                callback === nothing ? (movie = write_paraview(data, "wavefield_$(lpad(n, 6, '0'))"; directory=snapshot_dir, pvd=movie,
                                                                time=tbeg + n * dt, verbose=false)) : callback(data, tbeg + n * dt)
            end
        end
        if (verbose || par !== nothing) && (it % max(1, nt ÷ 10) == 0 || it == nt)
            vmax = cv_max(par, Float64(maximum(B -> max(maximum(abs, owned(B.V1)), maximum(abs, owned(B.V2)), maximum(abs, owned(B.V3))), blocks)))
            isfinite(vmax) || error("curvilinear grid: the simulation became unstable at step $it")
            verbose && @printf("  step %6d/%d  t = %8.4f s  max|v| = %.3e m/s\n", it, nt, tbeg + it * dt, vmax)
        end
    end
    KernelAbstractions.synchronize(backend)
    movie === nothing || movie_paraview(movie)
    verbose && @printf("  finished in %.2f s\n", time() - wall)
    traces = zeros(nrec, 18, length(stations))
    traces[:, :, own] .= Float64.(Array(rec))
    par === nothing || MPI.Allreduce!(traces, +, par.comm)
    return Seismograms(collect(tbeg .+ (0:nrec-1) .* (record_every * dt)), traces[:, 1:3, :], traces[:, 4:6, :],
                       traces[:, 7:12, :], traces[:, 13:18, :], collect(stations), (:x, :y, :z), (:xx, :yy, :zz, :yz, :xz, :xy),
                       wavefield ? velocity_field() : nothing)
end
