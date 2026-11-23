# Misfit gradients by the adjoint-state method, with the kernels of the forward simulation (CPU threads and GPUs).
#
# Forward step n: σⁿ = σⁿ⁻¹ + Δt C G vⁿ⁻¹, vⁿ = vⁿ⁻¹ + Δt B D σⁿ (C: moduli, B: buoyancy, G: strain rate, D: divergence);
# vⁿ is recorded at step n + 1. On the staggered grid Gᵀ = -D, so for the adjoints λ (of v) and η (of σ) the fields
# q = -B λ and p = C η obey the forward scheme run backwards in time, driven by the residuals r of the seismograms:
#     qⁿ = qⁿ⁺¹ + Δt B D pⁿ⁺¹ - B rⁿ,    pⁿ = pⁿ⁺¹ + Δt C G qⁿ,    and   ηⁿ = ηⁿ⁺¹ + Δt G qⁿ.
# The gradients are ∂J/∂C = Δt Σₙ ηⁿ ∂(C G vⁿ⁻¹)/∂C and ∂J/∂B = -Σₙ qⁿ (vⁿ - vⁿ⁻¹) / B². This is the exact gradient of the
# discrete scheme: the adjoint kernels use the transposed differences (coefficients of the stencil points, which differ
# next to free surfaces), and adjoint memory variables give the gradient of the attenuation terms. Only the PML is not
# transposed: the adjoint kernels stretch their differences as the forward ones (an approximation, as usual).

# ------------------------------------------------------------------ differences
# Forward differences use the coefficients of the point they are computed at (2nd order next to free surfaces); their
# transposes, in the adjoint kernels, the coefficients of the points of the stencil (`M.two`: 2nd-order flags of the cells).
struct FwdD{C}
    c::C
end
Base.@propagate_inbounds (d::FwdD)(A, I, e, a) = Dp(A, I, e, d.c[a])
struct AdjD{MM,C}
    M::MM
    c::C
end
Base.@propagate_inbounds function (d::AdjD)(A, I, e, a)
    c, two = d.c[a], d.M.two            # no closures here: they would not inherit @inbounds
    return (A[I+e] * ifelse(two[I+e], c[3], c[1]) - A[I] * ifelse(two[I], c[3], c[1])) -
           (A[I+2e] * ifelse(two[I+2e], zero(c[2]), c[2]) - A[I-e] * ifelse(two[I-e], zero(c[2]), c[2]))
end

# strain rates as in the stress kernels, with the differences `d`
Base.@propagate_inbounds strain_psv(d, Vx, Vz, I) = (ex = CartesianIndex(1, 0); ez = CartesianIndex(0, 1);
    (d(Vx, I - ex, ex, 1), d(Vz, I - ez, ez, 2), d(Vz, I, ex, 1) + d(Vx, I, ez, 2)))
Base.@propagate_inbounds strain_sh(d, Vy, I) = (d(Vy, I, CartesianIndex(0, 1), 2), d(Vy, I, CartesianIndex(1, 0), 1))
Base.@propagate_inbounds function strain_3d(d, Vx, Vy, Vz, I)
    ex, ey, ez = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    return (d(Vx, I - ex, ex, 1), d(Vy, I - ey, ey, 2), d(Vz, I - ez, ez, 3),
            d(Vz, I, ey, 2) + d(Vy, I, ez, 3), d(Vz, I, ex, 1) + d(Vx, I, ez, 3), d(Vy, I, ex, 1) + d(Vx, I, ey, 2))
end

# ------------------------------------------------------------------ adjoint kernels: the forward kernels with the transposed
# differences (in the PML zones with the same stretching, an approximation)
Base.@propagate_inbounds function vel_psv_adj!(F, M, c, dt, P, i, k)
    d, I, J = AdjD(M, c), CartesianIndex(i, k), zone_index(P, CartesianIndex(i, k))
    ex, ez = CartesianIndex(1, 0), CartesianIndex(0, 1)
    (; Sxx, Szz, Sxz) = F
    dxSxx = ade(P, :axSxx, :gxe, J, i, d(Sxx, I, ex, 1), dt)
    dzSzz = ade(P, :azSzz, :gze, J, k, d(Szz, I, ez, 2), dt)
    dxSxz = ade(P, :axSxz, :gxc, J, i, d(Sxz, I - ex, ex, 1), dt)
    dzSxz = ade(P, :azSxz, :gzc, J, k, d(Sxz, I - ez, ez, 2), dt)
    vel_update_psv!(F, M, dt, i, k, dxSxx + dzSxz, dxSxz + dzSzz)
end

Base.@propagate_inbounds function stress_psv_adj!(F, M, c, vc, dt, P, i, k)
    d, I, J = AdjD(M, c), CartesianIndex(i, k), zone_index(P, CartesianIndex(i, k))
    ex, ez = CartesianIndex(1, 0), CartesianIndex(0, 1)
    (; Vx, Vz) = F
    dxVx = ade(P, :axVx, :gxc, J, i, d(Vx, I - ex, ex, 1), dt)
    dzVz = ade(P, :azVz, :gzc, J, k, d(Vz, I - ez, ez, 2), dt)
    dxVz = ade(P, :axVz, :gxe, J, i, d(Vz, I, ex, 1), dt)
    dzVx = ade(P, :azVx, :gze, J, k, d(Vx, I, ez, 2), dt)
    stress_update_psv!(F, M, vc, dt, i, k, dxVx, dzVz, dxVz + dzVx)
end

Base.@propagate_inbounds function vel_sh_adj!(F, M, c, dt, P, i, k)
    d, I, J = AdjD(M, c), CartesianIndex(i, k), zone_index(P, CartesianIndex(i, k))
    dzSyz = ade(P, :azSyz, :gzc, J, k, d(F.Syz, I - CartesianIndex(0, 1), CartesianIndex(0, 1), 2), dt)
    dxSxy = ade(P, :axSxy, :gxc, J, i, d(F.Sxy, I - CartesianIndex(1, 0), CartesianIndex(1, 0), 1), dt)
    vel_update_sh!(F, M, dt, i, k, dxSxy + dzSyz)
end

Base.@propagate_inbounds function stress_sh_adj!(F, M, c, vc, dt, P, i, k)
    d, I, J = AdjD(M, c), CartesianIndex(i, k), zone_index(P, CartesianIndex(i, k))
    dxVy = ade(P, :axVy, :gxe, J, i, d(F.Vy, I, CartesianIndex(1, 0), 1), dt)
    dzVy = ade(P, :azVy, :gze, J, k, d(F.Vy, I, CartesianIndex(0, 1), 2), dt)
    stress_update_sh!(F, M, vc, dt, i, k, dzVy, dxVy)
end

Base.@propagate_inbounds function vel_3d_adj!(F, M, c, dt, P, i, j, k)
    d, I, J = AdjD(M, c), CartesianIndex(i, j, k), zone_index(P, CartesianIndex(i, j, k))
    ex, ey, ez = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    (; Sxx, Syy, Szz, Syz, Sxz, Sxy) = F
    dSx = ade(P, :axSxx, :gxe, J, i, d(Sxx, I, ex, 1), dt) + ade(P, :aySxy, :gyc, J, j, d(Sxy, I - ey, ey, 2), dt) +
          ade(P, :azSxz, :gzc, J, k, d(Sxz, I - ez, ez, 3), dt)
    dSy = ade(P, :axSxy, :gxc, J, i, d(Sxy, I - ex, ex, 1), dt) + ade(P, :aySyy, :gye, J, j, d(Syy, I, ey, 2), dt) +
          ade(P, :azSyz, :gzc, J, k, d(Syz, I - ez, ez, 3), dt)
    dSz = ade(P, :axSxz, :gxc, J, i, d(Sxz, I - ex, ex, 1), dt) + ade(P, :aySyz, :gyc, J, j, d(Syz, I - ey, ey, 2), dt) +
          ade(P, :azSzz, :gze, J, k, d(Szz, I, ez, 3), dt)
    vel_update_3d!(F, M, dt, i, j, k, dSx, dSy, dSz)
end

Base.@propagate_inbounds function stress_3d_adj!(F, M, c, vc, dt, P, i, j, k)
    d, I, J = AdjD(M, c), CartesianIndex(i, j, k), zone_index(P, CartesianIndex(i, j, k))
    ex, ey, ez = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    (; Vx, Vy, Vz) = F
    dxVx = ade(P, :axVx, :gxc, J, i, d(Vx, I - ex, ex, 1), dt)
    dyVy = ade(P, :ayVy, :gyc, J, j, d(Vy, I - ey, ey, 2), dt)
    dzVz = ade(P, :azVz, :gzc, J, k, d(Vz, I - ez, ez, 3), dt)
    exy = ade(P, :axVy, :gxe, J, i, d(Vy, I, ex, 1), dt) + ade(P, :ayVx, :gye, J, j, d(Vx, I, ey, 2), dt)
    exz = ade(P, :axVz, :gxe, J, i, d(Vz, I, ex, 1), dt) + ade(P, :azVx, :gze, J, k, d(Vx, I, ez, 3), dt)
    eyz = ade(P, :ayVz, :gye, J, j, d(Vz, I, ey, 2), dt) + ade(P, :azVy, :gze, J, k, d(Vy, I, ez, 3), dt)
    stress_update_3d!(F, M, vc, dt, i, j, k, dxVx, dyVy, dzVz, eyz, exz, exy)
end

# Unrolled (generated): `map` with a closure over the tuples of arrays would not vectorize.
"Add `dt * q` to the adjoint strains `E` at `I` and return them."
@generated function adjoint_strain!(E::NTuple{N,Any}, I, q, dt) where {N}
    quote
        Base.@_inline_meta
        @inbounds Base.Cartesian.@nexprs $N m -> (e_m = E[m][I] + q[m] * dt; E[m][I] = e_m)
        return Base.Cartesian.@ntuple $N e
    end
end

"""
Adjoint memory variables of attenuation `Rb` (one set per stress component, as `F.R??`): R̄ₘⁿ = c1ₘ R̄ₘⁿ⁺¹ + Δt d1ₘ ηⁿ.
Returns -Δt Σₘ c2ₘ R̄ₘⁿ per component, the adjoint weight of the input of the memory variables (zero without attenuation).
"""
adjoint_memory!(::Nothing, I, e, vc, dt) = map(zero, e)
@generated function adjoint_memory!(Rb::NTuple{N,Any}, I, e, vc, dt) where {N}
    quote
        Base.@_inline_meta
        return Base.Cartesian.@ntuple $N m -> adjoint_memory1!(Rb[m], I, e[m], vc, dt)
    end
end
@inline function adjoint_memory1!(R, I, x, vc, dt)
    c1, c2, d1, _ = vc
    s = zero(x)
    @inbounds Base.Cartesian.@nexprs 3 m -> begin
        r = c1[m] * R[m, I] + dt * d1[m] * x
        R[m, I] = r
        s -= dt * c2[m] * r
    end
    return s
end

# ------------------------------------------------------------------ gradient kernels: step n at a cell
# `G` accumulators, `E` adjoint strains η, `Rb` adjoint memory variables, `Q` adjoint field (qⁿ), `Vn`, `Vp` forward
# velocities vⁿ and vⁿ⁻¹ (tuples). A stress component increases by Δt (C_U ε + Σ d1 R) with R driven by -c2 Δt A ε (A: the
# moduli times τ); its adjoint weights are aU = Δt η (unrelaxed part) and aR (memory part).

Base.@propagate_inbounds function grad_psv!(G, E, Rb, Q, Vn, Vp, M, c, vc, dt, i, k)
    I = CartesianIndex(i, k)
    two = M.two[I]
    η = adjoint_strain!(E, I, strain_psv(AdjD(M, c), Q.Vx, Q.Vz, I), dt)
    aU, aR = η .* dt, adjoint_memory!(Rb, I, η, vc, dt)
    dxVx, dzVz, vxz = strain_psv(FwdD((fd(c[1], two), fd(c[2], two))), Vp[1], Vp[2], I)
    tp, ts = M.taup[I], M.taus[I]
    tp1, ts1 = 1 + tp * (1 + vc[4]), 1 + ts * (1 + vc[4])
    P = aU .* tp1 .+ aR .* tp                    # weights of λ + 2μ
    S = aU .* ts1 .+ aR .* ts                    # of 2μ (normal) and μ (shear)
    T = aU .* (1 + vc[4]) .+ aR                  # of τ
    mu2 = 2M.mu[I]
    div = dxVx + dzVz
    G.lam[I] += (P[1] + P[2]) * div
    G.mu[I] += 2(P[1] + P[2]) * div - 2(S[1] * dzVz + S[2] * dxVx)
    G.xz[I] += S[3] * vxz                                                  # μ at the Sxz point
    if Rb !== nothing                                                      # attenuation (resolved at compile time)
        G.tp[I] += (T[1] + T[2]) * (M.lam[I] + mu2) * div
        muxz = havg4(M.mu[i, k], M.mu[i, k+1], M.mu[i+1, k], M.mu[i+1, k+1])
        G.ts[I] += T[3] * muxz * vxz - (T[1] * dzVz + T[2] * dxVx) * mu2
    end
    G.bx[I] += Q.Vx[I] * (Vn[1][I] - Vp[1][I])                             # buoyancy at the velocity points
    G.bz[I] += Q.Vz[I] * (Vn[2][I] - Vp[2][I])
end

Base.@propagate_inbounds function grad_sh!(G, E, Rb, Q, Vn, Vp, M, c, vc, dt, i, k)
    I = CartesianIndex(i, k)
    two = M.two[I]
    η = adjoint_strain!(E, I, strain_sh(AdjD(M, c), Q.Vy, I), dt)
    aU, aR = η .* dt, adjoint_memory!(Rb, I, η, vc, dt)
    vyz, vxy = strain_sh(FwdD((fd(c[1], two), fd(c[2], two))), Vp[1], I)
    ts = M.taus[I]
    ts1 = 1 + ts * (1 + vc[4])
    S = aU .* ts1 .+ aR .* ts
    T = aU .* (1 + vc[4]) .+ aR
    G.yz[I] += S[1] * vyz
    G.xy[I] += S[2] * vxy
    Rb === nothing || (G.ts[I] += T[1] * havg2(M.mu[i, k], M.mu[i, k+1]) * vyz + T[2] * havg2(M.mu[i, k], M.mu[i+1, k]) * vxy)
    G.by[I] += Q.Vy[I] * (Vn[1][I] - Vp[1][I])
end

Base.@propagate_inbounds function grad_3d!(G, E, Rb, Q, Vn, Vp, M, c, vc, dt, i, j, k)
    I = CartesianIndex(i, j, k)
    two = M.two[I]
    η = adjoint_strain!(E, I, strain_3d(AdjD(M, c), Q.Vx, Q.Vy, Q.Vz, I), dt)
    aU, aR = η .* dt, adjoint_memory!(Rb, I, η, vc, dt)
    dxVx, dyVy, dzVz, vyz, vxz, vxy = strain_3d(FwdD((fd(c[1], two), fd(c[2], two), fd(c[3], two))), Vp[1], Vp[2], Vp[3], I)
    tp, ts = M.taup[I], M.taus[I]
    tp1, ts1 = 1 + tp * (1 + vc[4]), 1 + ts * (1 + vc[4])
    P = aU .* tp1 .+ aR .* tp
    S = aU .* ts1 .+ aR .* ts
    T = aU .* (1 + vc[4]) .+ aR
    mu = M.mu
    mu2 = 2mu[I]
    div = dxVx + dyVy + dzVz
    o = (dyVy + dzVz, dxVx + dzVz, dxVx + dyVy)
    G.lam[I] += (P[1] + P[2] + P[3]) * div
    G.mu[I] += 2(P[1] + P[2] + P[3]) * div - 2(S[1] * o[1] + S[2] * o[2] + S[3] * o[3])
    G.yz[I] += S[4] * vyz
    G.xz[I] += S[5] * vxz
    G.xy[I] += S[6] * vxy
    if Rb !== nothing
        muxz = havg4(mu[i, j, k], mu[i, j, k+1], mu[i+1, j, k], mu[i+1, j, k+1])
        muxy = havg4(mu[i, j, k], mu[i+1, j, k], mu[i, j+1, k], mu[i+1, j+1, k])
        muyz = havg4(mu[i, j, k], mu[i, j, k+1], mu[i, j+1, k], mu[i, j+1, k+1])
        G.tp[I] += (T[1] + T[2] + T[3]) * (M.lam[I] + mu2) * div
        G.ts[I] += T[4] * muyz * vyz + T[5] * muxz * vxz + T[6] * muxy * vxy - (T[1] * o[1] + T[2] * o[2] + T[3] * o[3]) * mu2
    end
    G.bx[I] += Q.Vx[I] * (Vn[1][I] - Vp[1][I])
    G.by[I] += Q.Vy[I] * (Vn[2][I] - Vp[2][I])
    G.bz[I] += Q.Vz[I] * (Vn[3][I] - Vp[3][I])
end

# ------------------------------------------------------------------ adjoint sources: q -= B r (transpose of the recording)
"Velocity component `V` staggered along `e` (between the cells `I` and `I + e`): `q -= B r` at the two points around `I`."
Base.@propagate_inbounds function residual!(V, ρ, I, e, r)
    V[I] -= 2 / (ρ[I] + ρ[I+e]) * r
    V[I-e] -= 2 / (ρ[I-e] + ρ[I]) * r
end

Base.@propagate_inbounds function residual_psv!(Q, M, St, res, itw, nown)
    for n in 1:nown
        I = CartesianIndex(St.i[n], St.k[n])
        residual!(Q.Vx, M.rho, I, CartesianIndex(1, 0), res[itw, 1, n] / 2)
        residual!(Q.Vz, M.rho, I, CartesianIndex(0, 1), -res[itw, 2, n] / 2)       # z up
    end
end

Base.@propagate_inbounds function residual_sh!(Q, M, St, res, itw, nown)
    for n in 1:nown
        I = CartesianIndex(St.i[n], St.k[n])
        Q.Vy[I] -= res[itw, 1, n] / M.rho[I]
    end
end

Base.@propagate_inbounds function residual_3d!(Q, M, St, res, itw, nown)
    for n in 1:nown
        I = CartesianIndex(St.i[n], St.j[n], St.k[n])
        residual!(Q.Vx, M.rho, I, CartesianIndex(1, 0, 0), res[itw, 1, n] / 2)
        residual!(Q.Vy, M.rho, I, CartesianIndex(0, 1, 0), res[itw, 2, n] / 2)
        residual!(Q.Vz, M.rho, I, CartesianIndex(0, 0, 1), -res[itw, 3, n] / 2)
    end
end

# kernels, accumulators and the staggering of shear moduli (offsets of the cells averaged) and buoyancies per mode
const O2, O3 = CartesianIndex{2}, CartesianIndex{3}
adjoint_setup(::Val{:psv}) = (vel_psv_adj!, stress_psv_adj!, grad_psv!, residual_psv!, (:lam, :mu, :xz, :tp, :ts, :bx, :bz), 3,
                              (xz=(O2(0, 0), O2(0, 1), O2(1, 0), O2(1, 1)),), (bx=O2(1, 0), bz=O2(0, 1)))
adjoint_setup(::Val{:sh}) = (vel_sh_adj!, stress_sh_adj!, grad_sh!, residual_sh!, (:yz, :xy, :ts, :by), 2, (yz=(O2(0, 0), O2(0, 1)), xy=(O2(0, 0), O2(1, 0))), (by=nothing,))
adjoint_setup(::Val{:p3d}) = (vel_3d_adj!, stress_3d_adj!, grad_3d!, residual_3d!, (:lam, :mu, :yz, :xz, :xy, :tp, :ts, :bx, :by, :bz), 6,
                              (yz=(O3(0, 0, 0), O3(0, 0, 1), O3(0, 1, 0), O3(0, 1, 1)), xz=(O3(0, 0, 0), O3(0, 0, 1), O3(1, 0, 0), O3(1, 0, 1)),
                               xy=(O3(0, 0, 0), O3(1, 0, 0), O3(0, 1, 0), O3(1, 1, 0))), (bx=O3(1, 0, 0), by=O3(0, 1, 0), bz=O3(0, 0, 1)))

"∂ havg(m...) / ∂ m[a] of the harmonic average of the moduli `m` (as `havg2`, `havg4`)."
function dhavg(m::NTuple{K}, a) where {K}
    p(j) = prod(m[l] for l in 1:K if l != j)
    S = sum(p, 1:K)
    return S == 0 ? 0.0 : K * p(a)^2 / S^2
end

"Gradients with respect to rho, lam, mu (internal arrays) from the accumulators `G` (host arrays)."
function assemble_gradient(G, M, mode)
    shear, buoy = adjoint_setup(Val(mode))[7:8]
    mu = Float64.(Array(M.mu))
    z = zero(mu)
    glam, gmu, grho = haskey(G, :lam) ? G.lam : copy(z), haskey(G, :mu) ? G.mu : copy(z), copy(z)
    R = CartesianIndices(mu)
    for (name, offs) in pairs(shear), I in R
        all(o -> I + o in R, offs) || continue
        m = map(o -> mu[I+o], offs)
        for (a, o) in enumerate(offs)
            gmu[I+o] += G[name][I] * dhavg(m, a)
        end
    end
    for (name, e) in pairs(buoy), I in R                                  # B = 2 / (ρ₁ + ρ₂), or 1 / ρ (SH)
        if e === nothing
            grho[I] += G[name][I]
        elseif I + e in R
            grho[I] += G[name][I] / 2
            grho[I+e] += G[name][I] / 2
        end
    end
    return (; rho=grho, lam=glam, mu=gmu, taup=haskey(G, :tp) ? G.tp : z, taus=G.ts)
end

# ------------------------------------------------------------------ driver
struct AdjointDriver
    observed::Seismograms
    checkpoint_steps::Union{Nothing,Int}
end

"Forward step `it` (as `timeloop!`, without output); `record`: also record the stations."
function forward_step!(F, M, Src, FSrc, St, rec, acc, regions, it, p, record=true)
    (; c, vc, dtT, backend) = p
    vel!, stress!, inject!, force!, record! = p.kernels
    if record && p.nown > 0
        itw = (it - 1) % p.record_every == 0 ? (it - 1) ÷ p.record_every + 1 : 0
        foreach_item!(backend, record!, (rec, acc, F, St, c, dtT, itw), p.nown)
    end
    p.pw_edges && extrapolate_sides!(velocities(F), p.edges)
    map(((lo, ndr, P),) -> foreach_cell!(backend, stress!, (F, M, c, vc, dtT, P), lo, ndr), regions)
    p.nosource || run_once!(backend, inject!, (F, Src, it))
    p.pw_edges && extrapolate_sides!(stresses(F), p.edges)
    map(((lo, ndr, P),) -> foreach_cell!(backend, vel!, (F, M, c, dtT, P), lo, ndr), regions)
    p.noforce || run_once!(backend, force!, (F, M, FSrc, it))
    return nothing
end

"Check the observed seismograms against the simulation `s`; their velocities at the stations of this rank."
function observed_velocities(obs, s)
    length(obs.t) == s.nrec && obs.components == s.components ||
        error("the observed seismograms do not match the simulation (samples $(length(obs.t)) vs $(s.nrec), components)")
    return obs.v[:, :, s.own]
end

function (d::AdjointDriver)(s)
    o = observed_velocities(d.observed, s)
    (; F, M, St, rec, acc, regions, backend, nt, nown, c, vc, dtT, record_every) = s
    vel!, stress!, grad!, residual!, names, nE = adjoint_setup(Val(s.mode))
    V = velocities(F)
    new(A) = fill!(similar(A), 0)
    attenuation = maximum(abs, M.taup) > 0 || maximum(abs, M.taus) > 0
    state = attenuation ? (values(F)..., pml_state(regions)...) : (V..., stresses(F)..., pml_state(regions)...)
    # checkpoints every K steps (host memory) and a buffer of the K + 1 velocity fields of a segment (device)
    K = something(d.checkpoint_steps, ceil(Int, sqrt(nt * sum(length, state) / sum(length, V))))
    step!(it, record) = forward_step!(F, M, s.Src, s.FSrc, St, rec, acc, regions, it, s, record)

    # forward pass: seismograms and checkpoints
    saved = typeof(map(Array, state))[]
    for it in 1:nt
        (it - 1) % K == 0 && push!(saved, map(Array, state))
        step!(it, true)
    end
    KernelAbstractions.synchronize(backend)
    r = Array(rec)[:, 1:s.nc, 1:nown] .* s.scale .- o
    J = sum(abs2, r) / 2
    res = KernelAbstractions.zeros(backend, s.T, size(rec, 1), s.nc, max(nown, 1))
    copyto!(res, s.T.(cat(r .* s.scale, zeros(size(res, 1), s.nc, size(res, 3) - nown); dims=3)))    # ∂J/∂rec

    # 2nd-order flags of all cells (next to free surfaces and at the outer edge of the PML), for the transposed differences
    fs = map(A -> Array(A), (M.fs_top, M.fs_bot, M.ob_top, M.ob_bot))
    P = regions[end][3]
    o = P === nothing ? () : ndims(M.lam) == 3 ? (Array(P.ox), Array(P.oy)) : (Array(P.ox),)
    two = [second_order(last(Tuple(I)), map(f -> f[Base.front(Tuple(I))...], fs)...) | any(a -> o[a][I[a]], eachindex(o))
           for I in CartesianIndices(M.lam)]
    Ma = merge(M, (; two=copyto!(similar(M.lam, Bool), two)))
    # adjoint fields, absorbing layers with their own memory variables, adjoint strains and accumulators
    Q = map(new, F)
    aregions = map(((lo, ndr, P),) -> (lo, ndr, P === nothing ? nothing : merge(P, (; mem=map(new, P.mem)))), regions)
    E = ntuple(_ -> new(M.lam), nE)
    Rb = attenuation ? map(new, values(F)[end-nE+1:end]) : nothing
    G = NamedTuple{names}(map(_ -> new(M.lam), names))
    buf = [map(new, V) for _ in 0:K]
    inner = (ntuple(_ -> H, ndims(M.lam)), size(M.lam) .- 2H)
    # backward, segment by segment: recompute the forward velocities, then the adjoint steps n = b … a
    for (seg, a) in Iterators.reverse(enumerate(1:K:nt))
        b = min(a + K - 1, nt - 1)
        map(copyto!, state, saved[seg])
        map(copyto!, buf[1], V)
        for it in a:b
            step!(it, false)
            map(copyto!, buf[it-a+2], V)
        end
        for n in b:-1:a
            map(((lo, ndr, P),) -> foreach_cell!(backend, vel!, (Q, Ma, c, dtT, P), lo, ndr), aregions)
            itw = n % record_every == 0 ? n ÷ record_every + 1 : 0
            itw > 0 && nown > 0 && run_once!(backend, residual!, (Q, Ma, St, res, itw, nown))
            foreach_cell!(backend, grad!, (G, E, Rb, Q, buf[n-a+2], buf[n-a+1], Ma, c, vc, dtT), inner...)
            map(((lo, ndr, P),) -> foreach_cell!(backend, stress!, (Q, Ma, c, vc, dtT, P), lo, ndr), aregions)
        end
    end
    KernelAbstractions.synchronize(backend)
    return J, model_gradient(s, assemble_gradient(map(A -> Float64.(Array(A)), G), M, s.mode))
end
