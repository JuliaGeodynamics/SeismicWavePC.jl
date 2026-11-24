# Velocity-stress staggered-grid kernels: 4th order in space, 2nd order in time, visco-elastic
# (generalized Zener body), ported from OpenSWPC `m_kernel.f90` (3D, P-SV and SH).
# Near free surfaces / ocean bottoms 2nd-order differences are used. The Cerjan sponge is folded
# into the updates: field = (field + increment) * damping.
#
# `F` holds the wavefields, `M` the medium, sponge (gx_c, gx_b, ...) and surface zones.
# `c` = (cx, [cy,] cz) with c* = (9/8/d, 1/24/d, 1/d), `vc` = memory-variable coefficients.

@inline second_order(k, ft, fb, ot, ob) = ((k - ft) * (fb - k) >= 0) | ((k - ot) * (ob - k) >= 0)
@inline fd(c, two) = (ifelse(two, c[3], c[1]), ifelse(two, zero(c[2]), c[2]))   # branch-free: vectorizes
@inline havg2(a, b) = 2a * b / (a + b + eps(typeof(a)))
@inline havg4(a, b, c, d) = 4a * b * c * d / (a * b * c + a * b * d + a * c * d + b * c * d + eps(typeof(a)))

"Crank-Nicolson update of the NM memory variables of one stress component; returns their weighted sum."
Base.@propagate_inbounds function memvar!(R, I, f, vc, dt)
    c1, c2, d1, _ = vc
    s = zero(f)
    # unrolled, so that the loop over cells is the innermost loop and can be vectorized
    Base.Cartesian.@nexprs 3 m -> begin
        r = c1[m] * R[m, I] - c2[m] * f * dt
        R[m, I] = r
        s += d1[m] * r
    end
    return s
end
# Without attenuation (Q = ∞ everywhere) `vc` is `nothing` and the memory variables are not allocated: no work, no memory.
@inline memvar!(::Nothing, I, f, ::Nothing, dt) = zero(f)
"Memory variables `F.s`, `nothing` if not allocated (resolved at compile time)."
@inline memory(F::NamedTuple{K}, ::Val{s}) where {K,s} = s in K ? getfield(F, s) : nothing
"Unrelaxed factor 1 + τ (1 + d₂) of the moduli, and d₂."
@inline unrelaxed(t, vc) = 1 + t * (1 + vc[4])
@inline unrelaxed(t, ::Nothing) = one(t)
@inline d2(vc) = vc[4]
@inline d2(::Nothing) = false

# ------------------------------------------------------------------ PML (OpenSWPC m_absorb_p: ADE CFS-PML, Zhang & Shen, 2010)
# The kernels take a zone `P`: `nothing` in the interior (no PML terms), or a PML zone with the first cell `lo`, memory
# variables `mem.ax*`, `mem.ay*`, `mem.az*` (one per derivative) and damping profiles `g?c`, `g?e` (cell centres and edges).

"Derivative `d` along a PML axis: stretched with the profile `P.g[i]` and the memory variable `P.a[J]` (which is updated)."
@inline ade(::Nothing, ::Symbol, ::Symbol, J, i, d, dt) = d
Base.@propagate_inbounds function ade(P, a::Symbol, g::Symbol, J, i, d, dt)
    A, γ = getfield(P.mem, a), getfield(P, g)[i]
    m = A[J]
    A[J] = γ[3] * m + γ[4] * d * dt
    return γ[1] * d + γ[2] * m
end
@inline zone_index(::Nothing, I) = I
@inline zone_index(P, I) = I - P.lo
@inline edge(::Nothing, i, j...) = false                       # 2nd order at the outer two cells of the PML (as OpenSWPC)
Base.@propagate_inbounds edge(P, i) = P.ox[i]
Base.@propagate_inbounds edge(P, i, j) = P.ox[i] | P.oy[j]

"""
Rows of the interior without 2nd-order differences (no free surface or ocean bottom near, no PML): the kernels use the
4th-order coefficients there without deciding it per cell (a third faster). Otherwise as the interior (`nothing`).
"""
struct Plain end
@inline zone_index(::Plain, I) = I
@inline ade(::Plain, ::Symbol, ::Symbol, J, i, d, dt) = d
"2nd-order differences at the cell (next to free surfaces and ocean bottoms, at the outer cells of the PML)?"
Base.@propagate_inbounds twoflag(P, M, k, i) = second_order(k, M.fs_top[i], M.fs_bot[i], M.ob_top[i], M.ob_bot[i]) | edge(P, i)
Base.@propagate_inbounds twoflag(P, M, k, i, j) = second_order(k, M.fs_top[i, j], M.fs_bot[i, j], M.ob_top[i, j], M.ob_bot[i, j]) | edge(P, i, j)
@inline twoflag(::Plain, M, k, i...) = false

# ------------------------------------------------------------------ P-SV (x, z)

Base.@propagate_inbounds function vel_psv!(F, M, c, dt, P, i, k)
    (; Vx, Vz, Sxx, Szz, Sxz) = F
    two = twoflag(P, M, k, i)
    ax, bx = fd(c[1], two)
    azc, bzc = fd(coefat(c[2], CartesianIndex(i, k)), two)    # depth derivatives of fields at cell centres (at faces)
    azf, bzf = fd(coefat(c[3], CartesianIndex(i, k)), two)    # and of fields at faces (at cell centres)
    J = zone_index(P, CartesianIndex(i, k))
    dxSxx = ade(P, :axSxx, :gxe, J, i, (Sxx[i+1, k] - Sxx[i, k]) * ax - (Sxx[i+2, k] - Sxx[i-1, k]) * bx, dt)
    dzSzz = ade(P, :azSzz, :gze, J, k, (Szz[i, k+1] - Szz[i, k]) * azc - (Szz[i, k+2] - Szz[i, k-1]) * bzc, dt)
    dxSxz = ade(P, :axSxz, :gxc, J, i, (Sxz[i, k] - Sxz[i-1, k]) * ax - (Sxz[i+1, k] - Sxz[i-2, k]) * bx, dt)
    dzSxz = ade(P, :azSxz, :gzc, J, k, (Sxz[i, k] - Sxz[i, k-1]) * azf - (Sxz[i, k+1] - Sxz[i, k-2]) * bzf, dt)
    vel_update_psv!(F, M, dt, i, k, dxSxx + dzSxz, dxSxz + dzSzz)
end

"Velocity update from the stress divergence `sx`, `sz` (shared with the adjoint kernels)."
Base.@propagate_inbounds function vel_update_psv!(F, M, dt, i, k, sx, sz)
    F.Vx[i, k] = (F.Vx[i, k] + 2 / (M.rho[i, k] + M.rho[i+1, k]) * sx * dt) * M.gx_b[i] * M.gz_c[k]
    F.Vz[i, k] = (F.Vz[i, k] + 2 / (M.rho[i, k] + M.rho[i, k+1]) * sz * dt) * M.gx_c[i] * M.gz_b[k]
end

Base.@propagate_inbounds function stress_psv!(F, M, c, vc, dt, P, i, k)
    (; Vx, Vz, Sxx, Szz, Sxz) = F
    two = twoflag(P, M, k, i)
    ax, bx = fd(c[1], two)
    azc, bzc = fd(coefat(c[2], CartesianIndex(i, k)), two)    # depth derivatives of fields at cell centres (at faces)
    azf, bzf = fd(coefat(c[3], CartesianIndex(i, k)), two)    # and of fields at faces (at cell centres)
    J = zone_index(P, CartesianIndex(i, k))
    dxVx = ade(P, :axVx, :gxc, J, i, (Vx[i, k] - Vx[i-1, k]) * ax - (Vx[i+1, k] - Vx[i-2, k]) * bx, dt)
    dzVz = ade(P, :azVz, :gzc, J, k, (Vz[i, k] - Vz[i, k-1]) * azf - (Vz[i, k+1] - Vz[i, k-2]) * bzf, dt)
    dxVz = ade(P, :axVz, :gxe, J, i, (Vz[i+1, k] - Vz[i, k]) * ax - (Vz[i+2, k] - Vz[i-1, k]) * bx, dt)
    dzVx = ade(P, :azVx, :gze, J, k, (Vx[i, k+1] - Vx[i, k]) * azc - (Vx[i, k+2] - Vx[i, k-1]) * bzc, dt)
    stress_update_psv!(F, M, vc, dt, i, k, dxVx, dzVz, dxVz + dzVx)
end

"Stress update from the strain rates (shared with the adjoint kernels)."
Base.@propagate_inbounds function stress_update_psv!(F, M, vc, dt, i, k, dxVx, dzVz, e)
    (; Sxx, Szz, Sxz) = F
    I = CartesianIndex(i, k)
    mu2 = 2M.mu[I]
    l2m = M.lam[I] + mu2
    tp, ts = M.taup[I], M.taus[I]
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    div = dxVx + dzVz
    Rx = memvar!(memory(F, Val(:Rxx)), I, l2m * tp * div - mu2 * ts * dzVz, vc, dt)
    Rz = memvar!(memory(F, Val(:Rzz)), I, l2m * tp * div - mu2 * ts * dxVx, vc, dt)
    gc = M.gx_c[i] * M.gz_c[k]
    Sxx[I] = (Sxx[I] + (l2m * tp1 * div - mu2 * ts1 * dzVz + Rx) * dt) * gc
    Szz[I] = (Szz[I] + (l2m * tp1 * div - mu2 * ts1 * dxVx + Rz) * dt) * gc
    muxz = havg4(M.mu[i, k], M.mu[i, k+1], M.mu[i+1, k], M.mu[i+1, k+1])
    Rxz = memvar!(memory(F, Val(:Rxz)), I, muxz * ts * e, vc, dt)
    Sxz[I] = (Sxz[I] + (muxz * ts1 * e + Rxz) * dt) * M.gx_b[i] * M.gz_b[k]
end

Base.@propagate_inbounds function inject_psv!(F, S, it)
    for n in 1:length(S.i)
        i, k, s = S.i[n], S.k[n], S.s[n, it]
        F.Sxx[i, k] -= S.M[n, 1] * s
        F.Szz[i, k] -= S.M[n, 3] * s
        q = S.M[n, 5] * s / 4
        F.Sxz[i, k] -= q; F.Sxz[i, k-1] -= q; F.Sxz[i-1, k] -= q; F.Sxz[i-1, k-1] -= q
    end
end

# body forces (OpenSWPC source__bodyforce), added to the velocities after their update
Base.@propagate_inbounds function force_psv!(F, M, S, it)
    for n in 1:length(S.i)
        i, k, s = S.i[n], S.k[n], S.s[n, it]
        b = s / M.rho[i, k] / 2
        F.Vx[i, k] += S.F[n, 1] * b; F.Vx[i-1, k] += S.F[n, 1] * b
        F.Vz[i, k] += S.F[n, 3] * b; F.Vz[i, k-1] += S.F[n, 3] * b
    end
end


# ------------------------------------------------------------------ SH (x, z), out-of-plane y

Base.@propagate_inbounds function vel_sh!(F, M, c, dt, P, i, k)
    (; Vy, Syz, Sxy) = F
    two = twoflag(P, M, k, i)
    ax, bx = fd(c[1], two)
    azc, bzc = fd(coefat(c[2], CartesianIndex(i, k)), two)    # depth derivatives of fields at cell centres (at faces)
    azf, bzf = fd(coefat(c[3], CartesianIndex(i, k)), two)    # and of fields at faces (at cell centres)
    J = zone_index(P, CartesianIndex(i, k))
    dzSyz = ade(P, :azSyz, :gzc, J, k, (Syz[i, k] - Syz[i, k-1]) * azf - (Syz[i, k+1] - Syz[i, k-2]) * bzf, dt)
    dxSxy = ade(P, :axSxy, :gxc, J, i, (Sxy[i, k] - Sxy[i-1, k]) * ax - (Sxy[i+1, k] - Sxy[i-2, k]) * bx, dt)
    vel_update_sh!(F, M, dt, i, k, dxSxy + dzSyz)
end

Base.@propagate_inbounds vel_update_sh!(F, M, dt, i, k, s) = F.Vy[i, k] = (F.Vy[i, k] + s / M.rho[i, k] * dt) * M.gx_c[i] * M.gz_c[k]

Base.@propagate_inbounds function stress_sh!(F, M, c, vc, dt, P, i, k)
    (; Vy, Syz, Sxy) = F
    two = twoflag(P, M, k, i)
    ax, bx = fd(c[1], two)
    azc, bzc = fd(coefat(c[2], CartesianIndex(i, k)), two)    # depth derivatives of fields at cell centres (at faces)
    azf, bzf = fd(coefat(c[3], CartesianIndex(i, k)), two)    # and of fields at faces (at cell centres)
    J = zone_index(P, CartesianIndex(i, k))
    dxVy = ade(P, :axVy, :gxe, J, i, (Vy[i+1, k] - Vy[i, k]) * ax - (Vy[i+2, k] - Vy[i-1, k]) * bx, dt)
    dzVy = ade(P, :azVy, :gze, J, k, (Vy[i, k+1] - Vy[i, k]) * azc - (Vy[i, k+2] - Vy[i, k-1]) * bzc, dt)
    stress_update_sh!(F, M, vc, dt, i, k, dzVy, dxVy)
end

Base.@propagate_inbounds function stress_update_sh!(F, M, vc, dt, i, k, dzVy, dxVy)
    (; Syz, Sxy) = F
    I = CartesianIndex(i, k)
    ts = M.taus[I]
    ts1 = unrelaxed(ts, vc)
    muxy = havg2(M.mu[i, k], M.mu[i+1, k])
    muyz = havg2(M.mu[i, k], M.mu[i, k+1])
    Ryz = memvar!(memory(F, Val(:Ryz)), I, muyz * ts * dzVy, vc, dt)
    Rxy = memvar!(memory(F, Val(:Rxy)), I, muxy * ts * dxVy, vc, dt)
    Syz[I] = (Syz[I] + (muyz * ts1 * dzVy + Ryz) * dt) * M.gx_c[i] * M.gz_b[k]
    Sxy[I] = (Sxy[I] + (muxy * ts1 * dxVy + Rxy) * dt) * M.gx_b[i] * M.gz_c[k]
end

Base.@propagate_inbounds function inject_sh!(F, S, it)
    for n in 1:length(S.i)
        i, k, s = S.i[n], S.k[n], S.s[n, it]
        F.Sxy[i, k] -= S.M[n, 6] * s / 2; F.Sxy[i-1, k] -= S.M[n, 6] * s / 2
        F.Syz[i, k] -= S.M[n, 4] * s / 2; F.Syz[i, k-1] -= S.M[n, 4] * s / 2
    end
end

Base.@propagate_inbounds function force_sh!(F, M, S, it)
    for n in 1:length(S.i)
        i, k = S.i[n], S.k[n]
        F.Vy[i, k] += S.F[n, 2] * S.s[n, it] / M.rho[i, k]
    end
end


# ------------------------------------------------------------------ 3D (x, y, z)

Base.@propagate_inbounds function vel_3d!(F, M, c, dt, P, i, j, k)
    (; Vx, Vy, Vz, Sxx, Syy, Szz, Syz, Sxz, Sxy) = F
    two = twoflag(P, M, k, i, j)
    ax, bx = fd(c[1], two)
    ay, by = fd(c[2], two)
    azc, bzc = fd(coefat(c[3], CartesianIndex(i, j, k)), two)
    azf, bzf = fd(coefat(c[4], CartesianIndex(i, j, k)), two)
    J = zone_index(P, CartesianIndex(i, j, k))
    dSx = ade(P, :axSxx, :gxe, J, i, (Sxx[i+1, j, k] - Sxx[i, j, k]) * ax - (Sxx[i+2, j, k] - Sxx[i-1, j, k]) * bx, dt) +
          ade(P, :aySxy, :gyc, J, j, (Sxy[i, j, k] - Sxy[i, j-1, k]) * ay - (Sxy[i, j+1, k] - Sxy[i, j-2, k]) * by, dt) +
          ade(P, :azSxz, :gzc, J, k, (Sxz[i, j, k] - Sxz[i, j, k-1]) * azf - (Sxz[i, j, k+1] - Sxz[i, j, k-2]) * bzf, dt)
    dSy = ade(P, :axSxy, :gxc, J, i, (Sxy[i, j, k] - Sxy[i-1, j, k]) * ax - (Sxy[i+1, j, k] - Sxy[i-2, j, k]) * bx, dt) +
          ade(P, :aySyy, :gye, J, j, (Syy[i, j+1, k] - Syy[i, j, k]) * ay - (Syy[i, j+2, k] - Syy[i, j-1, k]) * by, dt) +
          ade(P, :azSyz, :gzc, J, k, (Syz[i, j, k] - Syz[i, j, k-1]) * azf - (Syz[i, j, k+1] - Syz[i, j, k-2]) * bzf, dt)
    dSz = ade(P, :axSxz, :gxc, J, i, (Sxz[i, j, k] - Sxz[i-1, j, k]) * ax - (Sxz[i+1, j, k] - Sxz[i-2, j, k]) * bx, dt) +
          ade(P, :aySyz, :gyc, J, j, (Syz[i, j, k] - Syz[i, j-1, k]) * ay - (Syz[i, j+1, k] - Syz[i, j-2, k]) * by, dt) +
          ade(P, :azSzz, :gze, J, k, (Szz[i, j, k+1] - Szz[i, j, k]) * azc - (Szz[i, j, k+2] - Szz[i, j, k-1]) * bzc, dt)
    vel_update_3d!(F, M, dt, i, j, k, dSx, dSy, dSz)
end

Base.@propagate_inbounds function vel_update_3d!(F, M, dt, i, j, k, dSx, dSy, dSz)
    (; Vx, Vy, Vz) = F
    ρ = M.rho
    Vx[i, j, k] = (Vx[i, j, k] + 2 / (ρ[i, j, k] + ρ[i+1, j, k]) * dSx * dt) * M.gx_b[i] * M.gy_c[j] * M.gz_c[k]
    Vy[i, j, k] = (Vy[i, j, k] + 2 / (ρ[i, j, k] + ρ[i, j+1, k]) * dSy * dt) * M.gx_c[i] * M.gy_b[j] * M.gz_c[k]
    Vz[i, j, k] = (Vz[i, j, k] + 2 / (ρ[i, j, k] + ρ[i, j, k+1]) * dSz * dt) * M.gx_c[i] * M.gy_c[j] * M.gz_b[k]
end

Base.@propagate_inbounds function stress_3d!(F, M, c, vc, dt, P, i, j, k)
    (; Vx, Vy, Vz, Sxx, Syy, Szz, Syz, Sxz, Sxy) = F
    two = twoflag(P, M, k, i, j)
    ax, bx = fd(c[1], two)
    ay, by = fd(c[2], two)
    azc, bzc = fd(coefat(c[3], CartesianIndex(i, j, k)), two)
    azf, bzf = fd(coefat(c[4], CartesianIndex(i, j, k)), two)
    J = zone_index(P, CartesianIndex(i, j, k))
    dxVx = ade(P, :axVx, :gxc, J, i, (Vx[i, j, k] - Vx[i-1, j, k]) * ax - (Vx[i+1, j, k] - Vx[i-2, j, k]) * bx, dt)
    dyVy = ade(P, :ayVy, :gyc, J, j, (Vy[i, j, k] - Vy[i, j-1, k]) * ay - (Vy[i, j+1, k] - Vy[i, j-2, k]) * by, dt)
    dzVz = ade(P, :azVz, :gzc, J, k, (Vz[i, j, k] - Vz[i, j, k-1]) * azf - (Vz[i, j, k+1] - Vz[i, j, k-2]) * bzf, dt)
    exy = ade(P, :axVy, :gxe, J, i, (Vy[i+1, j, k] - Vy[i, j, k]) * ax - (Vy[i+2, j, k] - Vy[i-1, j, k]) * bx, dt) +
          ade(P, :ayVx, :gye, J, j, (Vx[i, j+1, k] - Vx[i, j, k]) * ay - (Vx[i, j+2, k] - Vx[i, j-1, k]) * by, dt)
    exz = ade(P, :axVz, :gxe, J, i, (Vz[i+1, j, k] - Vz[i, j, k]) * ax - (Vz[i+2, j, k] - Vz[i-1, j, k]) * bx, dt) +
          ade(P, :azVx, :gze, J, k, (Vx[i, j, k+1] - Vx[i, j, k]) * azc - (Vx[i, j, k+2] - Vx[i, j, k-1]) * bzc, dt)
    eyz = ade(P, :ayVz, :gye, J, j, (Vz[i, j+1, k] - Vz[i, j, k]) * ay - (Vz[i, j+2, k] - Vz[i, j-1, k]) * by, dt) +
          ade(P, :azVy, :gze, J, k, (Vy[i, j, k+1] - Vy[i, j, k]) * azc - (Vy[i, j, k+2] - Vy[i, j, k-1]) * bzc, dt)
    stress_update_3d!(F, M, vc, dt, i, j, k, dxVx, dyVy, dzVz, eyz, exz, exy)
end

Base.@propagate_inbounds function stress_update_3d!(F, M, vc, dt, i, j, k, dxVx, dyVy, dzVz, eyz, exz, exy)
    (; Sxx, Syy, Szz, Syz, Sxz, Sxy) = F
    I = CartesianIndex(i, j, k)
    mu = M.mu
    mu2 = 2mu[I]
    l2m = M.lam[I] + mu2
    tp, ts = M.taup[I], M.taus[I]
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    div = dxVx + dyVy + dzVz
    Rx = memvar!(memory(F, Val(:Rxx)), I, l2m * tp * div - mu2 * ts * (dyVy + dzVz), vc, dt)
    Ry = memvar!(memory(F, Val(:Ryy)), I, l2m * tp * div - mu2 * ts * (dxVx + dzVz), vc, dt)
    Rz = memvar!(memory(F, Val(:Rzz)), I, l2m * tp * div - mu2 * ts * (dxVx + dyVy), vc, dt)
    gc = M.gx_c[i] * M.gy_c[j] * M.gz_c[k]
    Sxx[I] = (Sxx[I] + (l2m * tp1 * div - mu2 * ts1 * (dyVy + dzVz) + Rx) * dt) * gc
    Syy[I] = (Syy[I] + (l2m * tp1 * div - mu2 * ts1 * (dxVx + dzVz) + Ry) * dt) * gc
    Szz[I] = (Szz[I] + (l2m * tp1 * div - mu2 * ts1 * (dxVx + dyVy) + Rz) * dt) * gc
    muxz = havg4(mu[i, j, k], mu[i, j, k+1], mu[i+1, j, k], mu[i+1, j, k+1])
    muxy = havg4(mu[i, j, k], mu[i+1, j, k], mu[i, j+1, k], mu[i+1, j+1, k])
    muyz = havg4(mu[i, j, k], mu[i, j, k+1], mu[i, j+1, k], mu[i, j+1, k+1])
    Ryz = memvar!(memory(F, Val(:Ryz)), I, muyz * ts * eyz, vc, dt)
    Rxz = memvar!(memory(F, Val(:Rxz)), I, muxz * ts * exz, vc, dt)
    Rxy = memvar!(memory(F, Val(:Rxy)), I, muxy * ts * exy, vc, dt)
    Syz[I] = (Syz[I] + (muyz * ts1 * eyz + Ryz) * dt) * M.gx_c[i] * M.gy_b[j] * M.gz_b[k]
    Sxz[I] = (Sxz[I] + (muxz * ts1 * exz + Rxz) * dt) * M.gx_b[i] * M.gy_c[j] * M.gz_b[k]
    Sxy[I] = (Sxy[I] + (muxy * ts1 * exy + Rxy) * dt) * M.gx_b[i] * M.gy_b[j] * M.gz_c[k]
end

Base.@propagate_inbounds function inject_3d!(F, S, it)
    for n in 1:length(S.i)
        i, j, k, s = S.i[n], S.j[n], S.k[n], S.s[n, it]
        F.Sxx[i, j, k] -= S.M[n, 1] * s
        F.Syy[i, j, k] -= S.M[n, 2] * s
        F.Szz[i, j, k] -= S.M[n, 3] * s
        q = S.M[n, 4] * s / 4
        F.Syz[i, j, k] -= q; F.Syz[i, j, k-1] -= q; F.Syz[i, j-1, k] -= q; F.Syz[i, j-1, k-1] -= q
        q = S.M[n, 5] * s / 4
        F.Sxz[i, j, k] -= q; F.Sxz[i, j, k-1] -= q; F.Sxz[i-1, j, k] -= q; F.Sxz[i-1, j, k-1] -= q
        q = S.M[n, 6] * s / 4
        F.Sxy[i, j, k] -= q; F.Sxy[i, j-1, k] -= q; F.Sxy[i-1, j, k] -= q; F.Sxy[i-1, j-1, k] -= q
    end
end

Base.@propagate_inbounds function force_3d!(F, M, S, it)
    ρ = M.rho
    for n in 1:length(S.i)
        i, j, k, s = S.i[n], S.j[n], S.k[n], S.s[n, it] / 2
        r = ρ[i, j, k]
        F.Vx[i, j, k] += 2 / (r + ρ[i+1, j, k]) * S.F[n, 1] * s; F.Vx[i-1, j, k] += 2 / (r + ρ[i-1, j, k]) * S.F[n, 1] * s
        F.Vy[i, j, k] += 2 / (r + ρ[i, j+1, k]) * S.F[n, 2] * s; F.Vy[i, j-1, k] += 2 / (r + ρ[i, j-1, k]) * S.F[n, 2] * s
        F.Vz[i, j, k] += 2 / (r + ρ[i, j, k+1]) * S.F[n, 3] * s; F.Vz[i, j, k-1] += 2 / (r + ρ[i, j, k-1]) * S.F[n, 3] * s
    end
end


# ------------------------------------------------------------------ stations (OpenSWPC m_wav)
# Every step: displacement and strain are integrated from the velocity (`acc`). At the recording steps (`itw > 0`) velocity,
# displacement, stress and strain at the cell centre are stored in `rec[itw, :, n]` (z positive up: z-shear components flip).

"4th-order difference of `A` along `e` (unit `CartesianIndex`) at `I + e/2`; `ca` are the coefficients of that axis."
Base.@propagate_inbounds Dp(A, I, e, ca) = (A[I+e] - A[I]) * ca[1] - (A[I+2e] - A[I-e]) * ca[2]
"Derivative along `eb` at the cell centre `I` of a component staggered along `ea`."
Base.@propagate_inbounds Dm(A, I, ea, eb, cb) = (Dp(A, I, eb, coefat(cb, I)) + Dp(A, I - ea, eb, coefat(cb, I - ea)) +
                                                  Dp(A, I - eb, eb, coefat(cb, I - eb)) + Dp(A, I - ea - eb, eb, coefat(cb, I - ea - eb))) / 4
Base.@propagate_inbounds function store!(rec, itw, n, q)
    for m in eachindex(q)
        rec[itw, m, n] = q[m]
    end
end

# Stations between grid points (`st.vw`: interpolation weights, see interpolation.jl): velocity and stress interpolated, the
# strain at the nearest cell, counted on the rank that owns it (`st.own`); other stations: all at their cell.
"Velocity or stress component `m` of station `n`: interpolated (`ix`, `w`) or `cell()`."
@inline station_value(st, ::Val{:v}, F, m, n, cell) = haskey(st, :vw) ? @inbounds(sinc_sample(F, st.vi[m], st.vw[m], n)) : cell()
@inline station_value(st, ::Val{:s}, F, m, n, cell) = haskey(st, :sw) ? @inbounds(sinc_sample(F, st.si[m], st.sw[m], n)) : cell()
@inline owner(st, n) = haskey(st, :own) ? @inbounds(st.own[n]) : true

Base.@propagate_inbounds function record_psv!(rec, acc, F, st, c, dt, itw, n)
    (; Vx, Vz, Sxx, Szz, Sxz) = F
    I, ex, ez = CartesianIndex(st.i[n], st.k[n]), CartesianIndex(1, 0), CartesianIndex(0, 1)
    v = (station_value(st, Val(:v), Vx, 1, n, () -> @inbounds((Vx[I] + Vx[I-ex]) / 2)),
         -station_value(st, Val(:v), Vz, 2, n, () -> @inbounds((Vz[I] + Vz[I-ez]) / 2)))
    e = (Dp(Vx, I - ex, ex, c[1]), Dp(Vz, I - ez, ez, coefat(c[3], I)), -(Dm(Vz, I, ez, ex, c[1]) + Dm(Vx, I, ex, ez, c[2])) / 2) .* owner(st, n)
    a = ntuple(m -> acc[m, n] + (m <= 2 ? v[m] : e[m-2]) * dt, Val(5))
    for m in 1:5
        acc[m, n] = a[m]
    end
    σ = (station_value(st, Val(:s), Sxx, 1, n, () -> @inbounds(Sxx[I])), station_value(st, Val(:s), Szz, 2, n, () -> @inbounds(Szz[I])),
         -station_value(st, Val(:s), Sxz, 3, n, () -> @inbounds((Sxz[I] + Sxz[I-ex] + Sxz[I-ez] + Sxz[I-ex-ez]) / 4)))
    itw > 0 && store!(rec, itw, n, (v..., a[1], a[2], σ..., a[3], a[4], a[5]))
end

Base.@propagate_inbounds function record_sh!(rec, acc, F, st, c, dt, itw, n)
    (; Vy, Syz, Sxy) = F
    I, ex, ez = CartesianIndex(st.i[n], st.k[n]), CartesianIndex(1, 0), CartesianIndex(0, 1)
    v = station_value(st, Val(:v), Vy, 1, n, () -> @inbounds(Vy[I]))
    eyz = -(Dp(Vy, I, ez, coefat(c[2], I)) + Dp(Vy, I - ez, ez, coefat(c[2], I - ez))) / 4 * owner(st, n)
    exy = (Dp(Vy, I, ex, c[1]) + Dp(Vy, I - ex, ex, c[1])) / 4 * owner(st, n)
    a = (acc[1, n] + v * dt, acc[2, n] + eyz * dt, acc[3, n] + exy * dt)
    acc[1, n], acc[2, n], acc[3, n] = a
    σ = (-station_value(st, Val(:s), Syz, 1, n, () -> @inbounds((Syz[I] + Syz[I-ez]) / 2)),
         station_value(st, Val(:s), Sxy, 2, n, () -> @inbounds((Sxy[I] + Sxy[I-ex]) / 2)))
    itw > 0 && store!(rec, itw, n, (v, a[1], σ..., a[2], a[3]))
end

Base.@propagate_inbounds function record_3d!(rec, acc, F, st, c, dt, itw, n)
    (; Vx, Vy, Vz, Sxx, Syy, Szz, Syz, Sxz, Sxy) = F
    I = CartesianIndex(st.i[n], st.j[n], st.k[n])
    ex, ey, ez = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    v = (station_value(st, Val(:v), Vx, 1, n, () -> @inbounds((Vx[I] + Vx[I-ex]) / 2)),
         station_value(st, Val(:v), Vy, 2, n, () -> @inbounds((Vy[I] + Vy[I-ey]) / 2)),
         -station_value(st, Val(:v), Vz, 3, n, () -> @inbounds((Vz[I] + Vz[I-ez]) / 2)))
    e = (Dp(Vx, I - ex, ex, c[1]), Dp(Vy, I - ey, ey, c[2]), Dp(Vz, I - ez, ez, coefat(c[4], I)),
         -(Dm(Vz, I, ez, ey, c[2]) + Dm(Vy, I, ey, ez, c[3])) / 2,
         -(Dm(Vz, I, ez, ex, c[1]) + Dm(Vx, I, ex, ez, c[3])) / 2,
         (Dm(Vy, I, ey, ex, c[1]) + Dm(Vx, I, ex, ey, c[2])) / 2) .* owner(st, n)
    a = ntuple(m -> acc[m, n] + (m <= 3 ? v[m] : e[m-3]) * dt, Val(9))
    for m in 1:9
        acc[m, n] = a[m]
    end
    avg(S, e1, e2) = @inbounds (S[I] + S[I-e1] + S[I-e2] + S[I-e1-e2]) / 4
    σ = (station_value(st, Val(:s), Sxx, 1, n, () -> @inbounds(Sxx[I])), station_value(st, Val(:s), Syy, 2, n, () -> @inbounds(Syy[I])),
         station_value(st, Val(:s), Szz, 3, n, () -> @inbounds(Szz[I])), -station_value(st, Val(:s), Syz, 4, n, () -> avg(Syz, ey, ez)),
         -station_value(st, Val(:s), Sxz, 5, n, () -> avg(Sxz, ex, ez)), station_value(st, Val(:s), Sxy, 6, n, () -> avg(Sxy, ex, ey)))
    itw > 0 && store!(rec, itw, n, (v..., a[1], a[2], a[3], σ..., a[4], a[5], a[6], a[7], a[8], a[9]))
end

# ------------------------------------------------------------------ snapshot planes (OpenSWPC m_snap)
# Every step at every point of a plane: displacement (U) and peak velocities (PK: vertical, horizontal, vector). At snapshot
# steps (`store`): velocity, displacement and divergence / rotation (2nd order, as OpenSWPC; zero in air / fluid) into B.
# z positive up: the rotation is that of the (x east, y north, z up) frame.

Base.@propagate_inbounds function store_row!(B, n, q)
    for m in eachindex(q)
        B[n, m] = q[m]
    end
end
Base.@propagate_inbounds function peaks!(PK, n, vz, vh, va)
    PK[n, 1] = max(PK[n, 1], abs(vz)); PK[n, 2] = max(PK[n, 2], vh); PK[n, 3] = max(PK[n, 3], va)
end

Base.@propagate_inbounds function plane_psv!(B, U, PK, F, M, c, pts, dt, store, n)
    (; Vx, Vz) = F
    I, ex, ez = CartesianIndex(pts.i[n], pts.k[n]), CartesianIndex(1, 0), CartesianIndex(0, 1)
    vx, vz = (Vx[I] + Vx[I-ex]) / 2, -(Vz[I] + Vz[I-ez]) / 2
    U[n, 1] += vx * dt; U[n, 2] += vz * dt
    peaks!(PK, n, vz, abs(vx), hypot(vx, vz))
    if store
        cx, czf, czc = c[1][3], coefat(c[3], I)[3], coefat(c[2], I)[3]
        div = ((Vx[I] - Vx[I-ex]) * cx + (Vz[I] - Vz[I-ez]) * czf) * (M.lam[I] + 2M.mu[I] > 0)
        rot = -((Vx[I+ez] - Vx[I]) * czc - (Vz[I+ex] - Vz[I]) * cx) * (M.mu[I] > 0)
        store_row!(B, n, (vx, vz, U[n, 1], U[n, 2], div, rot))
    end
end

Base.@propagate_inbounds function plane_sh!(B, U, PK, F, M, c, pts, dt, store, n)
    vy = F.Vy[pts.i[n], pts.k[n]]
    U[n, 1] += vy * dt
    peaks!(PK, n, zero(vy), abs(vy), abs(vy))
    store && store_row!(B, n, (vy, U[n, 1]))
end

Base.@propagate_inbounds function plane_3d!(B, U, PK, F, M, c, pts, dt, store, n)
    (; Vx, Vy, Vz) = F
    I = CartesianIndex(pts.i[n], pts.j[n], pts.k[n])
    ex, ey, ez = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    vx, vy, vz = (Vx[I] + Vx[I-ex]) / 2, (Vy[I] + Vy[I-ey]) / 2, -(Vz[I] + Vz[I-ez]) / 2
    U[n, 1] += vx * dt; U[n, 2] += vy * dt; U[n, 3] += vz * dt
    peaks!(PK, n, vz, hypot(vx, vy), hypot(hypot(vx, vy), vz))   # hypot: no underflow of tiny values
    if store
        cx, cy, czf, czc = c[1][3], c[2][3], coefat(c[4], I)[3], coefat(c[3], I)[3]
        solid = M.mu[I] > 0
        div = ((Vx[I] - Vx[I-ex]) * cx + (Vy[I] - Vy[I-ey]) * cy + (Vz[I] - Vz[I-ez]) * czf) * (M.lam[I] + 2M.mu[I] > 0)
        rx = -((Vz[I+ey] - Vz[I]) * cy - (Vy[I+ez] - Vy[I]) * czc) * solid
        ry = -((Vx[I+ez] - Vx[I]) * czc - (Vz[I+ex] - Vz[I]) * cx) * solid
        rz = ((Vy[I+ex] - Vy[I]) * cx - (Vx[I+ey] - Vx[I]) * cy) * solid
        store_row!(B, n, (vx, vy, vz, U[n, 1], U[n, 2], U[n, 3], div, rx, ry, rz))
    end
end

# ------------------------------------------------------------------ launching
# The physics lives in the @inline cell functions above. On GPUs they are wrapped in one generic
# KernelAbstractions kernel; on the CPU plain threaded/SIMD loops are used (much faster than POCL).

@kernel inbounds = true function cells!(f, args, lo)
    I = @index(Global, NTuple)
    f(args..., (I .+ lo)...)
end

@kernel inbounds = true function items!(f, args)
    f(args..., @index(Global))
end

@kernel function serial!(f, args)
    f(args...)
end

"Apply `f(args..., i, [j,] k)` to the cells `lo .+ (1:ndr[1], …)` (array indices)."
function foreach_cell!(backend, f, args, lo, ndr)
    all(>(0), ndr) && cells!(backend, workgroup(ndr))(f, args, lo; ndrange=ndr)
    return nothing
end

# 2D tiles in the fastest two directions keep stencil neighbours in cache
workgroup(::NTuple{2}) = (32, 8)
workgroup(::NTuple{3}) = (32, 8, 1)

# Every task gets at least `MINWORK` cells: small loops (small models, thin PML zones) use few tasks, as waking many threads
# (and waiting for the slowest, e.g. on a busy machine or on efficiency cores) costs more than their work.
const MINWORK = (1 << 11, 1 << 11)                 # 2D, 3D (more work per cell)
ntasks(ndr, maxtasks) = clamp(prod(ndr) ÷ MINWORK[length(ndr)-1], 1, min(Threads.nthreads(), maxtasks))

# The part `t` of `nt` of the cells `lo .+ (1:ndr[1], …)`. 2D: jobs of rows; with fewer rows than tasks (thin zones: the PML,
# rows next to the surface) the rows are cut along x.
function part!(f::F, args, lo, ndr::NTuple{2}, t, nt) where {F}
    any(==(0), ndr) && return nothing
    nc = clamp(cld(nt, ndr[2]), 1, max(ndr[1] ÷ 64, 1))       # pieces per row
    nj = ndr[2] * nc
    for j in (t - 1) * nj ÷ nt + 1:t * nj ÷ nt
        k, q = divrem(j - 1, nc)
        kk = k + 1 + lo[2]
        rargs = @inbounds atrow(args, kk)                     # depth coefficients of the row, loaded once
        @inbounds @simd ivdep for i in q * ndr[1] ÷ nc + 1:(q + 1) * ndr[1] ÷ nc
            f(rargs..., i + lo[1], kk)
        end
    end
    return nothing
end

# 3D: blocks of `JB` rows in y and `KB` in z, so the stencil neighbours in y and z stay in cache, and thin zones (the PML)
# still give every task work.
const JB, KB = 8, 16
function part!(f::F, args, lo, ndr::NTuple{3}, t, nt) where {F}
    any(==(0), ndr) && return nothing
    nbj = cld(ndr[2], JB)
    nb = nbj * cld(ndr[3], KB)
    for b in (t - 1) * nb ÷ nt:t * nb ÷ nt - 1
        jb, kb = b % nbj, b ÷ nbj
        for k in kb*KB+1:min((kb + 1) * KB, ndr[3])
            rargs = @inbounds atrow(args, k + lo[3])             # depth coefficients of the row, loaded once
            for j in jb*JB+1:min((jb + 1) * JB, ndr[2])
                @inbounds @simd ivdep for i in 1:ndr[1]
                    f(rargs..., i + lo[1], j + lo[2], k + lo[3])
                end
            end
        end
    end
    return nothing
end

function foreach_cell!(::CPU, f::F, args, lo, ndr) where {F}
    nt = ntasks(ndr, typemax(Int))
    nt == 1 ? part!(f, args, lo, ndr, 1, 1) : Threads.@threads :static for t in 1:nt
        part!(f, args, lo, ndr, t, nt)
    end
    return nothing
end

"""
Apply `f(front..., P, i, [j,] k)` to the cells of all `regions = ((lo, ndr, P), …)`: on the CPU in one parallel loop (every
thread takes its part of every region; one launch per kernel instead of one per region), on GPUs one kernel per region.
"""
foreach_region!(backend, f, front, regions) = (map(((lo, ndr, P),) -> foreach_cell!(backend, f, (front..., P), lo, ndr), regions); nothing)
function foreach_region!(::CPU, f::F, front, regions) where {F}
    N = length(regions[1][2])
    nt = clamp(sum(r -> prod(r[2]), regions) ÷ MINWORK[N-1], 1, Threads.nthreads())
    nt == 1 ? map(((lo, ndr, P),) -> part!(f, (front..., P), lo, ndr, 1, 1), regions) : Threads.@threads :static for t in 1:nt
        map(((lo, ndr, P),) -> part!(f, (front..., P), lo, ndr, t, nt), regions)
    end
    return nothing
end

"Apply `f(args..., n)` for `n = 1:count` (stations)."
foreach_item!(backend, f, args, count) = items!(backend)(f, args; ndrange=count)
foreach_item!(::CPU, f::F, args, count) where {F} = @inbounds for n in 1:count
    f(args..., n)
end

"Run `f(args...)` once (source injection: sequential, as several sources may share a cell)."
run_once!(backend, f, args) = serial!(backend)(f, args; ndrange=1)
run_once!(::CPU, f, args) = @inbounds f(args...)
