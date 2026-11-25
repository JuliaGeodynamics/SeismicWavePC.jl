# Vertical transverse isotropy (VTI), given by the Thomsen parameters `epsilon`, `delta`, `gamma` as fields of the model
# (vp, vs: vertical velocities). The symmetry axis is the z axis of the grid: on the staggered grid every normal stress
# needs only the normal strain rates at its point and every shear stress its own, as in the isotropic case (no
# interpolation). A model without these fields runs the isotropic kernels (chosen at compile time: no cost).
#
# Stiffness and attenuation. In the isotropic scheme (OpenSWPC) the normal stresses are (λ + 2μ) div − 2μ (other strains):
# the P modulus relaxes with Qp, the shear part with Qs. VTI splits the normal stiffness in the same way, symmetrically:
#     C = w wᵀ (Qp)  +  the rest (Qs),   w = (√C11, [√C11,] √C33),
# so the rest has the off-diagonal terms C13 − √(C11 C33) (and C12 − C11 = −2 C66 in 3D) and a zero diagonal. Isotropic
# media give exactly the isotropic scheme (w wᵀ = (λ + 2μ) 1 1ᵀ, C13 − √(C11 C33) = −2μ); qP waves along the axes relax
# with Qp, S waves with Qs, and the stiffness stays symmetric (a requirement of the exact adjoint).
#
# Stored per cell (unrelaxed, at the reference frequency as the isotropic moduli): wx = √C11/χp, wz = √C33/χp,
# s13 = (C13 − √(C11 C33))/χs², mu = C44 = C55 and mu66 = C66 (/χs²), and lam = wz² − 2mu (λ of vertical P waves: air,
# water, free surfaces, snapshots).

"Is the medium VTI (`Val(true)`) or isotropic? From the fields of the NamedTuple (a compile-time constant)."
@generated anisotropic(::NamedTuple{K}) where {K} = :wx in K ? :(Val(true)) : :(Val(false))

"Does the model have VTI fields (`epsilon`, `delta`, `gamma`)?"
has_anisotropy(model) = any(f -> haskey(model.fields, f), (:epsilon, :delta, :gamma))
has_anisotropy(model::DistributedModel) = model.anisotropy

@inline pos(x) = ifelse(x > zero(x), x, zero(x))

"""
The stored moduli `(wx, wz, s13, mu, mu66)` of a cell from density, vertical velocities, Thomsen parameters and the
dispersion factors χp, χs of the attenuation. Generic: with tangent numbers it gives their derivatives (Born, gradient).
"""
function vti_moduli(ρ, vp, vs, ε, δ, γ, χp, χs)
    c33, c55 = ρ * vp * vp, ρ * vs * vs
    c11, c66 = c33 * (1 + 2ε), c55 * (1 + 2γ)
    c13 = sqrt(pos((c33 - c55) * (c33 * (1 + 2δ) - c55))) - c55
    return (sqrt(c11) / χp, sqrt(c33) / χp, (c13 - sqrt(c11 * c33)) / (χs * χs), c55 / (χs * χs), c66 / (χs * χs))
end

"Largest qP phase velocity of a VTI cell over all directions (unrelaxed: for the Courant number)."
function vti_vmax(ρ, wx, wz, s13, mu)
    c11, c33, c13 = wx^2, wz^2, wx * wz + s13
    v2 = maximum(0:5:90) do θ
        s, c = sind(θ)^2, cosd(θ)^2
        (c11 * s + c33 * c + mu + sqrt(((c11 - mu) * s - (c33 - mu) * c)^2 + 4(c13 + mu)^2 * s * c)) / 2
    end
    return sqrt(v2 / ρ)
end

"The VTI part of the medium (padded arrays), from the padded model fields."
function vti_medium(rho, vp, vs, ε, δ, γ, taup, taus, ts, fq_ref)
    χp, χs = visco_chi.((ts,), taup, fq_ref), visco_chi.((ts,), taus, fq_ref)
    C = vti_moduli.(rho, vp, vs, ε, δ, γ, χp, χs)
    wx, wz, s13, mu, mu66 = ntuple(q -> getindex.(C, q), 5)
    lam = wz .^ 2 .- 2mu
    vpmax = vti_vmax.(rho, wx, wz, s13, mu)
    return (; lam, mu, wx, wz, s13, mu66, vpmax)
end

# ------------------------------------------------------------------ kernels: the normal stresses with w wᵀ (τp) and the rest (τs)

Base.@propagate_inbounds function stress_update_psv!(::Val{true}, F, M, vc, dt, i, k, dxVx, dzVz, e)
    (; Sxx, Szz, Sxz) = F
    I = CartesianIndex(i, k)
    wx, wz, s13 = M.wx[I], M.wz[I], M.s13[I]
    tp, ts = M.taup[I], M.taus[I]
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    pe = wx * dxVx + wz * dzVz
    Rx = memvar!(memory(F, Val(:Rxx)), I, wx * pe * tp + s13 * ts * dzVz, vc, dt)
    Rz = memvar!(memory(F, Val(:Rzz)), I, wz * pe * tp + s13 * ts * dxVx, vc, dt)
    gc = M.gx_c[i] * M.gz_c[k]
    Sxx[I] = (Sxx[I] + (wx * pe * tp1 + s13 * ts1 * dzVz + Rx) * dt) * gc
    Szz[I] = (Szz[I] + (wz * pe * tp1 + s13 * ts1 * dxVx + Rz) * dt) * gc
    muxz = shear_xz(cutcells(M), M, i, k)                                               # C55
    Rxz = memvar!(memory(F, Val(:Rxz)), I, muxz * ts * e, vc, dt)
    Sxz[I] = (Sxz[I] + (muxz * ts1 * e + Rxz) * dt) * M.gx_b[i] * M.gz_b[k]
end

Base.@propagate_inbounds function stress_update_sh!(::Val{true}, F, M, vc, dt, i, k, dzVy, dxVy)
    (; Syz, Sxy) = F
    I = CartesianIndex(i, k)
    ts = M.taus[I]
    ts1 = unrelaxed(ts, vc)
    muxy = havg2(M.mu66[i, k], M.mu66[i+1, k])                                     # C66
    muyz = havg2(M.mu[i, k], M.mu[i, k+1])                                         # C44
    Ryz = memvar!(memory(F, Val(:Ryz)), I, muyz * ts * dzVy, vc, dt)
    Rxy = memvar!(memory(F, Val(:Rxy)), I, muxy * ts * dxVy, vc, dt)
    Syz[I] = (Syz[I] + (muyz * ts1 * dzVy + Ryz) * dt) * M.gx_c[i] * M.gz_b[k]
    Sxy[I] = (Sxy[I] + (muxy * ts1 * dxVy + Rxy) * dt) * M.gx_b[i] * M.gz_c[k]
end

Base.@propagate_inbounds function stress_update_3d!(::Val{true}, F, M, vc, dt, i, j, k, dxVx, dyVy, dzVz, eyz, exz, exy)
    (; Sxx, Syy, Szz, Syz, Sxz, Sxy) = F
    I = CartesianIndex(i, j, k)
    mu, m66 = M.mu, M.mu66
    wx, wz, s13, n66 = M.wx[I], M.wz[I], M.s13[I], -2m66[I]                        # C12 − C11 = −2 C66
    tp, ts = M.taup[I], M.taus[I]
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    pe = wx * (dxVx + dyVy) + wz * dzVz
    ox, oy, oz = n66 * dyVy + s13 * dzVz, n66 * dxVx + s13 * dzVz, s13 * (dxVx + dyVy)
    Rx = memvar!(memory(F, Val(:Rxx)), I, wx * pe * tp + ox * ts, vc, dt)
    Ry = memvar!(memory(F, Val(:Ryy)), I, wx * pe * tp + oy * ts, vc, dt)
    Rz = memvar!(memory(F, Val(:Rzz)), I, wz * pe * tp + oz * ts, vc, dt)
    gc = M.gx_c[i] * M.gy_c[j] * M.gz_c[k]
    Sxx[I] = (Sxx[I] + (wx * pe * tp1 + ox * ts1 + Rx) * dt) * gc
    Syy[I] = (Syy[I] + (wx * pe * tp1 + oy * ts1 + Ry) * dt) * gc
    Szz[I] = (Szz[I] + (wz * pe * tp1 + oz * ts1 + Rz) * dt) * gc
    muxz = havg4(mu[i, j, k], mu[i, j, k+1], mu[i+1, j, k], mu[i+1, j, k+1])               # C55
    muyz = havg4(mu[i, j, k], mu[i, j, k+1], mu[i, j+1, k], mu[i, j+1, k+1])               # C44
    muxy = havg4(m66[i, j, k], m66[i+1, j, k], m66[i, j+1, k], m66[i+1, j+1, k])           # C66
    Ryz = memvar!(memory(F, Val(:Ryz)), I, muyz * ts * eyz, vc, dt)
    Rxz = memvar!(memory(F, Val(:Rxz)), I, muxz * ts * exz, vc, dt)
    Rxy = memvar!(memory(F, Val(:Rxy)), I, muxy * ts * exy, vc, dt)
    Syz[I] = (Syz[I] + (muyz * ts1 * eyz + Ryz) * dt) * M.gx_c[i] * M.gy_b[j] * M.gz_b[k]
    Sxz[I] = (Sxz[I] + (muxz * ts1 * exz + Rxz) * dt) * M.gx_b[i] * M.gy_c[j] * M.gz_b[k]
    Sxy[I] = (Sxy[I] + (muxy * ts1 * exy + Rxy) * dt) * M.gx_b[i] * M.gy_b[j] * M.gz_c[k]
end

# ------------------------------------------------------------------ gradient kernels (as the isotropic ones in adjoint.jl):
# P and S are the adjoint weights of the parts with τp and τs of every stress component, T those of τ itself.

Base.@propagate_inbounds function grad_psv!(::Val{true}, G, E, Rb, Q, Δ, Vn, Vp, Z, M, c, vc, dt, i, k)
    I, ex, ez = CartesianIndex(i, k), CartesianIndex(1, 0), CartesianIndex(0, 1)
    two = M.two[I]
    η = adjoint_strain!(E, I, adjoint_strain_psv(AdjD(M, c), Q, Δ, I), dt)
    aU, aR = η .* dt, adjoint_memory!(Rb, I, η, vc, dt)
    f, Vx, Vz = FwdD((fd(c[1], two), fd(coefat(c[2], I), two), fd(coefat(c[3], I), two))), Vp[1], Vp[2]
    dxVx = fst(Z, Val(:axVx), Val(:gxc), I, i, f(Vx, I - ex, ex, 1))
    dzVz = fst(Z, Val(:azVz), Val(:gzc), I, k, f(Vz, I - ez, ez, 3))
    vxz = fst(Z, Val(:axVz), Val(:gxe), I, i, f(Vz, I, ex, 1)) + fst(Z, Val(:azVx), Val(:gze), I, k, f(Vx, I, ez, 2))
    tp, ts = M.taup[I], M.taus[I]
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    P = aU .* tp1 .+ aR .* tp
    S = aU .* ts1 .+ aR .* ts
    T = aU .* (1 + d2(vc)) .+ aR
    wx, wz, s13 = M.wx[I], M.wz[I], M.s13[I]
    pe = wx * dxVx + wz * dzVz
    pw = P[1] * wx + P[2] * wz
    G.wx[I] += P[1] * pe + pw * dxVx
    G.wz[I] += P[2] * pe + pw * dzVz
    G.s13[I] += S[1] * dzVz + S[2] * dxVx
    G.xz[I] += S[3] * vxz
    if Rb !== nothing
        G.tp[I] += (T[1] * wx + T[2] * wz) * pe
        muxz = havg4(M.mu[i, k], M.mu[i, k+1], M.mu[i+1, k], M.mu[i+1, k+1])
        G.ts[I] += T[3] * muxz * vxz + (T[1] * dzVz + T[2] * dxVx) * s13
    end
    G.bx[I] += Q.Vx[I] * (Vn[1][I] - Vp[1][I])
    G.bz[I] += Q.Vz[I] * (Vn[2][I] - Vp[2][I])
end

Base.@propagate_inbounds function grad_sh!(::Val{true}, G, E, Rb, Q, Δ, Vn, Vp, Z, M, c, vc, dt, i, k)
    I = CartesianIndex(i, k)
    two = M.two[I]
    η = adjoint_strain!(E, I, adjoint_strain_sh(AdjD(M, c), Q, Δ, I), dt)
    aU, aR = η .* dt, adjoint_memory!(Rb, I, η, vc, dt)
    f = FwdD((fd(c[1], two), fd(coefat(c[2], I), two)))
    vyz = fst(Z, Val(:azVy), Val(:gze), I, k, f(Vp[1], I, CartesianIndex(0, 1), 2))
    vxy = fst(Z, Val(:axVy), Val(:gxe), I, i, f(Vp[1], I, CartesianIndex(1, 0), 1))
    ts = M.taus[I]
    ts1 = unrelaxed(ts, vc)
    S = aU .* ts1 .+ aR .* ts
    T = aU .* (1 + d2(vc)) .+ aR
    G.yz[I] += S[1] * vyz
    G.xy[I] += S[2] * vxy
    Rb === nothing || (G.ts[I] += T[1] * havg2(M.mu[i, k], M.mu[i, k+1]) * vyz + T[2] * havg2(M.mu66[i, k], M.mu66[i+1, k]) * vxy)
    G.by[I] += Q.Vy[I] * (Vn[1][I] - Vp[1][I])
end

Base.@propagate_inbounds function grad_3d!(::Val{true}, G, E, Rb, Q, Δ, Vn, Vp, Z, M, c, vc, dt, i, j, k)
    I = CartesianIndex(i, j, k)
    ex, ey, ez = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    two = M.two[I]
    η = adjoint_strain!(E, I, adjoint_strain_3d(AdjD(M, c), Q, Δ, I), dt)
    aU, aR = η .* dt, adjoint_memory!(Rb, I, η, vc, dt)
    f, (Vx, Vy, Vz) = FwdD((fd(c[1], two), fd(c[2], two), fd(coefat(c[3], I), two), fd(coefat(c[4], I), two))), Vp
    dxVx = fst(Z, Val(:axVx), Val(:gxc), I, i, f(Vx, I - ex, ex, 1))
    dyVy = fst(Z, Val(:ayVy), Val(:gyc), I, j, f(Vy, I - ey, ey, 2))
    dzVz = fst(Z, Val(:azVz), Val(:gzc), I, k, f(Vz, I - ez, ez, 4))
    vyz = fst(Z, Val(:ayVz), Val(:gye), I, j, f(Vz, I, ey, 2)) + fst(Z, Val(:azVy), Val(:gze), I, k, f(Vy, I, ez, 3))
    vxz = fst(Z, Val(:axVz), Val(:gxe), I, i, f(Vz, I, ex, 1)) + fst(Z, Val(:azVx), Val(:gze), I, k, f(Vx, I, ez, 3))
    vxy = fst(Z, Val(:axVy), Val(:gxe), I, i, f(Vy, I, ex, 1)) + fst(Z, Val(:ayVx), Val(:gye), I, j, f(Vx, I, ey, 2))
    tp, ts = M.taup[I], M.taus[I]
    tp1, ts1 = unrelaxed(tp, vc), unrelaxed(ts, vc)
    P = aU .* tp1 .+ aR .* tp
    S = aU .* ts1 .+ aR .* ts
    T = aU .* (1 + d2(vc)) .+ aR
    mu, m66 = M.mu, M.mu66
    wx, wz, s13, n66 = M.wx[I], M.wz[I], M.s13[I], -2m66[I]
    pe = wx * (dxVx + dyVy) + wz * dzVz
    pw = (P[1] + P[2]) * wx + P[3] * wz
    G.wx[I] += (P[1] + P[2]) * pe + pw * (dxVx + dyVy)
    G.wz[I] += P[3] * pe + pw * dzVz
    G.s13[I] += (S[1] + S[2]) * dzVz + S[3] * (dxVx + dyVy)
    G.m66[I] -= 2(S[1] * dyVy + S[2] * dxVx)
    G.yz[I] += S[4] * vyz
    G.xz[I] += S[5] * vxz
    G.xy[I] += S[6] * vxy
    if Rb !== nothing
        muxz = havg4(mu[i, j, k], mu[i, j, k+1], mu[i+1, j, k], mu[i+1, j, k+1])
        muyz = havg4(mu[i, j, k], mu[i, j, k+1], mu[i, j+1, k], mu[i, j+1, k+1])
        muxy = havg4(m66[i, j, k], m66[i+1, j, k], m66[i, j+1, k], m66[i+1, j+1, k])
        G.tp[I] += ((T[1] + T[2]) * wx + T[3] * wz) * pe
        G.ts[I] += T[4] * muyz * vyz + T[5] * muxz * vxz + T[6] * muxy * vxy +
                   T[1] * (n66 * dyVy + s13 * dzVz) + T[2] * (n66 * dxVx + s13 * dzVz) + T[3] * s13 * (dxVx + dyVy)
    end
    G.bx[I] += Q.Vx[I] * (Vn[1][I] - Vp[1][I])
    G.by[I] += Q.Vy[I] * (Vn[2][I] - Vp[2][I])
    G.bz[I] += Q.Vz[I] * (Vn[3][I] - Vp[3][I])
end

"Accumulators of the adjoint gradient and the shear moduli averaged at the shear-stress points, VTI (as `adjoint_setup`)."
vti_gradient_setup(::Val{:psv}) = ((:wx, :wz, :s13, :xz, :tp, :ts, :bx, :bz), (xz=(:mu, (O2(0, 0), O2(0, 1), O2(1, 0), O2(1, 1))),))
vti_gradient_setup(::Val{:sh}) = ((:yz, :xy, :ts, :by), (yz=(:mu, (O2(0, 0), O2(0, 1))), xy=(:mu66, (O2(0, 0), O2(1, 0)))))
vti_gradient_setup(::Val{:p3d}) = ((:wx, :wz, :s13, :m66, :yz, :xz, :xy, :tp, :ts, :bx, :by, :bz),
                                   (yz=(:mu, (O3(0, 0, 0), O3(0, 0, 1), O3(0, 1, 0), O3(0, 1, 1))),
                                    xz=(:mu, (O3(0, 0, 0), O3(0, 0, 1), O3(1, 0, 0), O3(1, 0, 1))),
                                    xy=(:mu66, (O3(0, 0, 0), O3(1, 0, 0), O3(0, 1, 0), O3(1, 1, 0)))))

"Gradients with respect to the stored VTI moduli (internal arrays) from the accumulators `G` (as `assemble_gradient`)."
function assemble_gradient_vti(G, med, mode)
    shear = vti_gradient_setup(Val(mode))[2]
    buoy = adjoint_setup(Val(mode))[13]
    z = zero(med.mu)
    g = Dict(:mu => copy(z), :mu66 => haskey(G, :m66) ? G.m66 : copy(z), :rho => copy(z))
    R = CartesianIndices(z)
    for (name, (field, offs)) in pairs(shear), I in R
        all(o -> I + o in R, offs) || continue
        m = map(o -> getfield(med, field)[I+o], offs)
        for (a, o) in enumerate(offs)
            g[field][I+o] += G[name][I] * dhavg(m, a)
        end
    end
    for (name, e) in pairs(buoy), I in R
        if e === nothing
            g[:rho][I] += G[name][I]
        elseif I + e in R
            g[:rho][I] += G[name][I] / 2
            g[:rho][I+e] += G[name][I] / 2
        end
    end
    get0(k) = haskey(G, k) ? G[k] : z
    return (; rho=g[:rho], wx=get0(:wx), wz=get0(:wz), s13=get0(:s13), mu=g[:mu], mu66=g[:mu66], taup=get0(:tp), taus=G.ts)
end

# ------------------------------------------------------------------ chain rule (gradient, Born) with tangent numbers

"The padded model fields of the simulation `s` (internal units) and the attenuation of the medium."
function vti_fields(s)
    m, g, N = s.m, s.g, length(s.g.n)
    pad(A) = pad_absorber(internal(A, Val(N)), g.n, s.na)
    rho_in = pad(m.rho)
    return (; rho_in, rho=max.(rho_in, 0.001), vp=pad(m.vp), vs=pad(m.vs), ε=pad(m.epsilon), δ=pad(m.delta), γ=pad(m.gamma),
            qp=pad(m.Qp), qs=pad(m.Qs), taup=s.med.taup, taus=s.med.taus)
end

"""
Tangents of the stored moduli (wx, wz, s13, mu, mu66) of cell `I` for the tangents `d` of (ρ, vp, vs, ε, δ, γ, τp, τs).
"""
function vti_tangent(f, I, d, ts, fq)
    χp, χs = visco_chi(ts, f.taup[I], fq), visco_chi(ts, f.taus[I], fq)
    χpt = Tan(χp, visco_dchi(ts, f.taup[I], fq) * d[7])
    χst = Tan(χs, visco_dchi(ts, f.taus[I], fq) * d[8])
    x = (f.rho[I], f.vp[I], f.vs[I], f.ε[I], f.δ[I], f.γ[I])
    return map(tangent, vti_moduli(ntuple(q -> Tan(x[q], d[q]), 6)..., χpt, χst))
end

"Gradient with respect to the model fields (VTI) from the gradient `dM` with respect to the stored arrays (as `model_gradient`)."
function model_gradient_vti(s, dM)
    f, ts, fq = vti_fields(s), s.med.ts, s.fq_ref
    gm = map(k -> Float64.(Array(getfield(dM, k))), (:wx, :wz, :s13, :mu, :mu66))
    grho, gtp, gts = (Float64.(Array(getfield(dM, k))) for k in (:rho, :taup, :taus))
    G = map(CartesianIndices(f.rho)) do I
        dir(j) = (t = vti_tangent(f, I, ntuple(q -> Float64(q == j), 8), ts, fq); sum(gm[q][I] * t[q] for q in 1:5))
        drho = (grho[I] + dir(1)) * (f.rho_in[I] >= 0.001)
        dtp, dts = gtp[I] + dir(7), gts[I] + dir(8)
        (drho / s.rho_unit, dir(2), dir(3), dir(4), dir(5), dir(6),
         isfinite(f.qp[I]) ? -dtp * f.taup[I] / f.qp[I] : 0.0, isfinite(f.qs[I]) ? -dts * f.taus[I] / f.qs[I] : 0.0)
    end
    return unpad_gradient(s, G, (:rho, :vp, :vs, :epsilon, :delta, :gamma, :Qp, :Qs))
end

"Perturbation of the stored VTI arrays by the perturbation `δm` of the model fields (as `medium_perturbation`)."
function medium_perturbation_vti(s, δm)
    f, ts, fq, N, g = vti_fields(s), s.med.ts, s.fq_ref, length(s.g.n), s.g
    field(k) = haskey(δm.fields, k) ? pad_absorber(internal(convert(Array{Float64,3}, δm.fields[k]), Val(N)), g.n, s.na) : zeros(size(f.rho))
    δrho = field(:rho) ./ s.rho_unit .* (f.rho_in .>= 0.001)
    δqp, δqs = field(:Qp), field(:Qs)
    δtp = @. ifelse(isfinite(f.qp), -f.taup / f.qp * δqp, 0.0)
    δts = @. ifelse(isfinite(f.qs), -f.taus / f.qs * δqs, 0.0)
    δx = (δrho, field(:vp), field(:vs), field(:epsilon), field(:delta), field(:gamma), δtp, δts)
    T = map(I -> vti_tangent(f, I, map(A -> A[I], δx), ts, fq), CartesianIndices(f.rho))
    wx, wz, s13, mu, mu66 = ntuple(q -> getindex.(T, q), 5)
    return (; rho=δrho, wx, wz, s13, mu, mu66, taup=δtp, taus=δts)
end
