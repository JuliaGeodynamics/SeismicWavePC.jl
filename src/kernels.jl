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

# ------------------------------------------------------------------ P-SV (x, z)

Base.@propagate_inbounds function vel_psv!(F, M, c, dt, i, k)
    (; Vx, Vz, Sxx, Szz, Sxz) = F
    two = second_order(k, M.fs_top[i], M.fs_bot[i], M.ob_top[i], M.ob_bot[i])
    ax, bx = fd(c[1], two)
    az, bz = fd(c[2], two)
    dxSxx = (Sxx[i+1, k] - Sxx[i, k]) * ax - (Sxx[i+2, k] - Sxx[i-1, k]) * bx
    dzSzz = (Szz[i, k+1] - Szz[i, k]) * az - (Szz[i, k+2] - Szz[i, k-1]) * bz
    dxSxz = (Sxz[i, k] - Sxz[i-1, k]) * ax - (Sxz[i+1, k] - Sxz[i-2, k]) * bx
    dzSxz = (Sxz[i, k] - Sxz[i, k-1]) * az - (Sxz[i, k+1] - Sxz[i, k-2]) * bz
    Vx[i, k] = (Vx[i, k] + 2 / (M.rho[i, k] + M.rho[i+1, k]) * (dxSxx + dzSxz) * dt) * M.gx_b[i] * M.gz_c[k]
    Vz[i, k] = (Vz[i, k] + 2 / (M.rho[i, k] + M.rho[i, k+1]) * (dxSxz + dzSzz) * dt) * M.gx_c[i] * M.gz_b[k]
end

Base.@propagate_inbounds function stress_psv!(F, M, c, vc, dt, i, k)
    (; Vx, Vz, Sxx, Szz, Sxz) = F
    two = second_order(k, M.fs_top[i], M.fs_bot[i], M.ob_top[i], M.ob_bot[i])
    ax, bx = fd(c[1], two)
    az, bz = fd(c[2], two)
    dxVx = (Vx[i, k] - Vx[i-1, k]) * ax - (Vx[i+1, k] - Vx[i-2, k]) * bx
    dzVz = (Vz[i, k] - Vz[i, k-1]) * az - (Vz[i, k+1] - Vz[i, k-2]) * bz
    dxVz = (Vz[i+1, k] - Vz[i, k]) * ax - (Vz[i+2, k] - Vz[i-1, k]) * bx
    dzVx = (Vx[i, k+1] - Vx[i, k]) * az - (Vx[i, k+2] - Vx[i, k-1]) * bz
    I = CartesianIndex(i, k)
    mu2 = 2M.mu[I]
    l2m = M.lam[I] + mu2
    tp, ts = M.taup[I], M.taus[I]
    tp1, ts1 = 1 + tp * (1 + vc[4]), 1 + ts * (1 + vc[4])
    div = dxVx + dzVz
    Rx = memvar!(F.Rxx, I, l2m * tp * div - mu2 * ts * dzVz, vc, dt)
    Rz = memvar!(F.Rzz, I, l2m * tp * div - mu2 * ts * dxVx, vc, dt)
    gc = M.gx_c[i] * M.gz_c[k]
    Sxx[I] = (Sxx[I] + (l2m * tp1 * div - mu2 * ts1 * dzVz + Rx) * dt) * gc
    Szz[I] = (Szz[I] + (l2m * tp1 * div - mu2 * ts1 * dxVx + Rz) * dt) * gc
    muxz = havg4(M.mu[i, k], M.mu[i, k+1], M.mu[i+1, k], M.mu[i+1, k+1])
    e = dxVz + dzVx
    Rxz = memvar!(F.Rxz, I, muxz * ts * e, vc, dt)
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


# ------------------------------------------------------------------ SH (x, z), out-of-plane y

Base.@propagate_inbounds function vel_sh!(F, M, c, dt, i, k)
    (; Vy, Syz, Sxy) = F
    two = second_order(k, M.fs_top[i], M.fs_bot[i], M.ob_top[i], M.ob_bot[i])
    ax, bx = fd(c[1], two)
    az, bz = fd(c[2], two)
    dzSyz = (Syz[i, k] - Syz[i, k-1]) * az - (Syz[i, k+1] - Syz[i, k-2]) * bz
    dxSxy = (Sxy[i, k] - Sxy[i-1, k]) * ax - (Sxy[i+1, k] - Sxy[i-2, k]) * bx
    Vy[i, k] = (Vy[i, k] + (dxSxy + dzSyz) / M.rho[i, k] * dt) * M.gx_c[i] * M.gz_c[k]
end

Base.@propagate_inbounds function stress_sh!(F, M, c, vc, dt, i, k)
    (; Vy, Syz, Sxy) = F
    two = second_order(k, M.fs_top[i], M.fs_bot[i], M.ob_top[i], M.ob_bot[i])
    ax, bx = fd(c[1], two)
    az, bz = fd(c[2], two)
    dxVy = (Vy[i+1, k] - Vy[i, k]) * ax - (Vy[i+2, k] - Vy[i-1, k]) * bx
    dzVy = (Vy[i, k+1] - Vy[i, k]) * az - (Vy[i, k+2] - Vy[i, k-1]) * bz
    I = CartesianIndex(i, k)
    ts = M.taus[I]
    ts1 = 1 + ts * (1 + vc[4])
    muxy = havg2(M.mu[i, k], M.mu[i+1, k])
    muyz = havg2(M.mu[i, k], M.mu[i, k+1])
    Ryz = memvar!(F.Ryz, I, muyz * ts * dzVy, vc, dt)
    Rxy = memvar!(F.Rxy, I, muxy * ts * dxVy, vc, dt)
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


# ------------------------------------------------------------------ 3D (x, y, z)

Base.@propagate_inbounds function vel_3d!(F, M, c, dt, i, j, k)
    (; Vx, Vy, Vz, Sxx, Syy, Szz, Syz, Sxz, Sxy) = F
    two = second_order(k, M.fs_top[i, j], M.fs_bot[i, j], M.ob_top[i, j], M.ob_bot[i, j])
    ax, bx = fd(c[1], two)
    ay, by = fd(c[2], two)
    az, bz = fd(c[3], two)
    dSx = (Sxx[i+1, j, k] - Sxx[i, j, k]) * ax - (Sxx[i+2, j, k] - Sxx[i-1, j, k]) * bx +
          (Sxy[i, j, k] - Sxy[i, j-1, k]) * ay - (Sxy[i, j+1, k] - Sxy[i, j-2, k]) * by +
          (Sxz[i, j, k] - Sxz[i, j, k-1]) * az - (Sxz[i, j, k+1] - Sxz[i, j, k-2]) * bz
    dSy = (Sxy[i, j, k] - Sxy[i-1, j, k]) * ax - (Sxy[i+1, j, k] - Sxy[i-2, j, k]) * bx +
          (Syy[i, j+1, k] - Syy[i, j, k]) * ay - (Syy[i, j+2, k] - Syy[i, j-1, k]) * by +
          (Syz[i, j, k] - Syz[i, j, k-1]) * az - (Syz[i, j, k+1] - Syz[i, j, k-2]) * bz
    dSz = (Sxz[i, j, k] - Sxz[i-1, j, k]) * ax - (Sxz[i+1, j, k] - Sxz[i-2, j, k]) * bx +
          (Syz[i, j, k] - Syz[i, j-1, k]) * ay - (Syz[i, j+1, k] - Syz[i, j-2, k]) * by +
          (Szz[i, j, k+1] - Szz[i, j, k]) * az - (Szz[i, j, k+2] - Szz[i, j, k-1]) * bz
    ρ = M.rho
    Vx[i, j, k] = (Vx[i, j, k] + 2 / (ρ[i, j, k] + ρ[i+1, j, k]) * dSx * dt) * M.gx_b[i] * M.gy_c[j] * M.gz_c[k]
    Vy[i, j, k] = (Vy[i, j, k] + 2 / (ρ[i, j, k] + ρ[i, j+1, k]) * dSy * dt) * M.gx_c[i] * M.gy_b[j] * M.gz_c[k]
    Vz[i, j, k] = (Vz[i, j, k] + 2 / (ρ[i, j, k] + ρ[i, j, k+1]) * dSz * dt) * M.gx_c[i] * M.gy_c[j] * M.gz_b[k]
end

Base.@propagate_inbounds function stress_3d!(F, M, c, vc, dt, i, j, k)
    (; Vx, Vy, Vz, Sxx, Syy, Szz, Syz, Sxz, Sxy) = F
    two = second_order(k, M.fs_top[i, j], M.fs_bot[i, j], M.ob_top[i, j], M.ob_bot[i, j])
    ax, bx = fd(c[1], two)
    ay, by = fd(c[2], two)
    az, bz = fd(c[3], two)
    dxVx = (Vx[i, j, k] - Vx[i-1, j, k]) * ax - (Vx[i+1, j, k] - Vx[i-2, j, k]) * bx
    dyVy = (Vy[i, j, k] - Vy[i, j-1, k]) * ay - (Vy[i, j+1, k] - Vy[i, j-2, k]) * by
    dzVz = (Vz[i, j, k] - Vz[i, j, k-1]) * az - (Vz[i, j, k+1] - Vz[i, j, k-2]) * bz
    exy = (Vy[i+1, j, k] - Vy[i, j, k]) * ax - (Vy[i+2, j, k] - Vy[i-1, j, k]) * bx +
          (Vx[i, j+1, k] - Vx[i, j, k]) * ay - (Vx[i, j+2, k] - Vx[i, j-1, k]) * by
    exz = (Vz[i+1, j, k] - Vz[i, j, k]) * ax - (Vz[i+2, j, k] - Vz[i-1, j, k]) * bx +
          (Vx[i, j, k+1] - Vx[i, j, k]) * az - (Vx[i, j, k+2] - Vx[i, j, k-1]) * bz
    eyz = (Vz[i, j+1, k] - Vz[i, j, k]) * ay - (Vz[i, j+2, k] - Vz[i, j-1, k]) * by +
          (Vy[i, j, k+1] - Vy[i, j, k]) * az - (Vy[i, j, k+2] - Vy[i, j, k-1]) * bz
    I = CartesianIndex(i, j, k)
    mu = M.mu
    mu2 = 2mu[I]
    l2m = M.lam[I] + mu2
    tp, ts = M.taup[I], M.taus[I]
    tp1, ts1 = 1 + tp * (1 + vc[4]), 1 + ts * (1 + vc[4])
    div = dxVx + dyVy + dzVz
    Rx = memvar!(F.Rxx, I, l2m * tp * div - mu2 * ts * (dyVy + dzVz), vc, dt)
    Ry = memvar!(F.Ryy, I, l2m * tp * div - mu2 * ts * (dxVx + dzVz), vc, dt)
    Rz = memvar!(F.Rzz, I, l2m * tp * div - mu2 * ts * (dxVx + dyVy), vc, dt)
    gc = M.gx_c[i] * M.gy_c[j] * M.gz_c[k]
    Sxx[I] = (Sxx[I] + (l2m * tp1 * div - mu2 * ts1 * (dyVy + dzVz) + Rx) * dt) * gc
    Syy[I] = (Syy[I] + (l2m * tp1 * div - mu2 * ts1 * (dxVx + dzVz) + Ry) * dt) * gc
    Szz[I] = (Szz[I] + (l2m * tp1 * div - mu2 * ts1 * (dxVx + dyVy) + Rz) * dt) * gc
    muxz = havg4(mu[i, j, k], mu[i, j, k+1], mu[i+1, j, k], mu[i+1, j, k+1])
    muxy = havg4(mu[i, j, k], mu[i+1, j, k], mu[i, j+1, k], mu[i+1, j+1, k])
    muyz = havg4(mu[i, j, k], mu[i, j, k+1], mu[i, j+1, k], mu[i, j+1, k+1])
    Ryz = memvar!(F.Ryz, I, muyz * ts * eyz, vc, dt)
    Rxz = memvar!(F.Rxz, I, muxz * ts * exz, vc, dt)
    Rxy = memvar!(F.Rxy, I, muxy * ts * exy, vc, dt)
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


# ------------------------------------------------------------------ stations (OpenSWPC m_wav)
# Every step: displacement and strain are integrated from the velocity (`acc`). At the recording steps (`itw > 0`) velocity,
# displacement, stress and strain at the cell centre are stored in `rec[itw, :, n]` (z positive up: z-shear components flip).

"4th-order difference of `A` along `e` (unit `CartesianIndex`) at `I + e/2`; `ca` are the coefficients of that axis."
Base.@propagate_inbounds Dp(A, I, e, ca) = (A[I+e] - A[I]) * ca[1] - (A[I+2e] - A[I-e]) * ca[2]
"Derivative along `eb` at the cell centre `I` of a component staggered along `ea`."
Base.@propagate_inbounds Dm(A, I, ea, eb, cb) = (Dp(A, I, eb, cb) + Dp(A, I - ea, eb, cb) + Dp(A, I - eb, eb, cb) + Dp(A, I - ea - eb, eb, cb)) / 4
Base.@propagate_inbounds function store!(rec, itw, n, q)
    for m in eachindex(q)
        rec[itw, m, n] = q[m]
    end
end

Base.@propagate_inbounds function record_psv!(rec, acc, F, st, c, dt, itw, n)
    (; Vx, Vz, Sxx, Szz, Sxz) = F
    I, ex, ez = CartesianIndex(st.i[n], st.k[n]), CartesianIndex(1, 0), CartesianIndex(0, 1)
    v = ((Vx[I] + Vx[I-ex]) / 2, -(Vz[I] + Vz[I-ez]) / 2)
    e = (Dp(Vx, I - ex, ex, c[1]), Dp(Vz, I - ez, ez, c[2]), -(Dm(Vz, I, ez, ex, c[1]) + Dm(Vx, I, ex, ez, c[2])) / 2)
    a = ntuple(m -> acc[m, n] + (m <= 2 ? v[m] : e[m-2]) * dt, Val(5))
    for m in 1:5
        acc[m, n] = a[m]
    end
    itw > 0 && store!(rec, itw, n, (v..., a[1], a[2], Sxx[I], Szz[I], -(Sxz[I] + Sxz[I-ex] + Sxz[I-ez] + Sxz[I-ex-ez]) / 4, a[3], a[4], a[5]))
end

Base.@propagate_inbounds function record_sh!(rec, acc, F, st, c, dt, itw, n)
    (; Vy, Syz, Sxy) = F
    I, ex, ez = CartesianIndex(st.i[n], st.k[n]), CartesianIndex(1, 0), CartesianIndex(0, 1)
    v = Vy[I]
    eyz = -(Dp(Vy, I, ez, c[2]) + Dp(Vy, I - ez, ez, c[2])) / 4
    exy = (Dp(Vy, I, ex, c[1]) + Dp(Vy, I - ex, ex, c[1])) / 4
    a = (acc[1, n] + v * dt, acc[2, n] + eyz * dt, acc[3, n] + exy * dt)
    acc[1, n], acc[2, n], acc[3, n] = a
    itw > 0 && store!(rec, itw, n, (v, a[1], -(Syz[I] + Syz[I-ez]) / 2, (Sxy[I] + Sxy[I-ex]) / 2, a[2], a[3]))
end

Base.@propagate_inbounds function record_3d!(rec, acc, F, st, c, dt, itw, n)
    (; Vx, Vy, Vz, Sxx, Syy, Szz, Syz, Sxz, Sxy) = F
    I = CartesianIndex(st.i[n], st.j[n], st.k[n])
    ex, ey, ez = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    v = ((Vx[I] + Vx[I-ex]) / 2, (Vy[I] + Vy[I-ey]) / 2, -(Vz[I] + Vz[I-ez]) / 2)
    e = (Dp(Vx, I - ex, ex, c[1]), Dp(Vy, I - ey, ey, c[2]), Dp(Vz, I - ez, ez, c[3]),
         -(Dm(Vz, I, ez, ey, c[2]) + Dm(Vy, I, ey, ez, c[3])) / 2,
         -(Dm(Vz, I, ez, ex, c[1]) + Dm(Vx, I, ex, ez, c[3])) / 2,
         (Dm(Vy, I, ey, ex, c[1]) + Dm(Vx, I, ex, ey, c[2])) / 2)
    a = ntuple(m -> acc[m, n] + (m <= 3 ? v[m] : e[m-3]) * dt, Val(9))
    for m in 1:9
        acc[m, n] = a[m]
    end
    avg(S, e1, e2) = (S[I] + S[I-e1] + S[I-e2] + S[I-e1-e2]) / 4
    itw > 0 && store!(rec, itw, n, (v..., a[1], a[2], a[3], Sxx[I], Syy[I], Szz[I], -avg(Syz, ey, ez), -avg(Sxz, ex, ez), avg(Sxy, ex, ey),
                                    a[4], a[5], a[6], a[7], a[8], a[9]))
end

# ------------------------------------------------------------------ launching
# The physics lives in the @inline cell functions above. On GPUs they are wrapped in one generic
# KernelAbstractions kernel; on the CPU plain threaded/SIMD loops are used (much faster than POCL).

@kernel inbounds = true function cells!(f, args)
    I = @index(Global, NTuple)
    f(args..., (I .+ H)...)
end

@kernel inbounds = true function items!(f, args)
    f(args..., @index(Global))
end

@kernel function serial!(f, args)
    f(args...)
end

"Apply `f(args..., i, [j,] k)` to every interior cell (`ndr` cells per direction)."
foreach_cell!(backend, f, args, ndr) = cells!(backend, workgroup(ndr))(f, args; ndrange=ndr)

# 2D tiles in the fastest two directions keep stencil neighbours in cache
workgroup(::NTuple{2}) = (32, 8)
workgroup(::NTuple{3}) = (32, 8, 1)

function foreach_cell!(::CPU, f::F, args, ndr::NTuple{2}) where {F}
    Threads.@threads for k in 1:ndr[2]
        @inbounds @simd ivdep for i in 1:ndr[1]
            f(args..., i + H, k + H)
        end
    end
end

# Threads work on strips of `JB` rows in y that sweep through all z, so the stencil neighbours
# in y and z stay in cache.
const JB = 8
function foreach_cell!(::CPU, f::F, args, ndr::NTuple{3}) where {F}
    Threads.@threads :dynamic for jb in 1:cld(ndr[2], JB)
        for k in 1:ndr[3], j in (jb-1)*JB+1:min(jb * JB, ndr[2])
            @inbounds @simd ivdep for i in 1:ndr[1]
                f(args..., i + H, j + H, k + H)
            end
        end
    end
end

"Apply `f(args..., n)` for `n = 1:count` (stations)."
foreach_item!(backend, f, args, count) = items!(backend)(f, args; ndrange=count)
foreach_item!(::CPU, f::F, args, count) where {F} = @inbounds for n in 1:count
    f(args..., n)
end

"Run `f(args...)` once (source injection: sequential, as several sources may share a cell)."
run_once!(backend, f, args) = serial!(backend)(f, args; ndrange=1)
run_once!(::CPU, f, args) = @inbounds f(args...)
