# Misfit gradients by the adjoint-state method, with the kernels of the forward simulation (CPU threads and GPUs).
#
# Forward step n: σⁿ = σⁿ⁻¹ + Δt C G vⁿ⁻¹, vⁿ = vⁿ⁻¹ + Δt B D σⁿ (C: moduli, B: buoyancy, G: strain rate, D: divergence);
# vⁿ is recorded at step n + 1. On the staggered grid Gᵀ = -D, so for the adjoints λ (of v) and η (of σ) the fields
# q = -B λ and p = C η obey the forward scheme run backwards in time, driven by the residuals r of the seismograms:
#     qⁿ = qⁿ⁺¹ + Δt B D pⁿ⁺¹ - B rⁿ,    pⁿ = pⁿ⁺¹ + Δt C G qⁿ,    and   ηⁿ = ηⁿ⁺¹ + Δt G qⁿ.
# The gradients are ∂J/∂C = Δt Σₙ ηⁿ ∂(C G vⁿ⁻¹)/∂C and ∂J/∂B = -Σₙ qⁿ (vⁿ - vⁿ⁻¹) / B². This is the exact gradient of the
# discrete scheme: the adjoint kernels use the transposed differences (coefficients of the stencil points, which differ
# next to free surfaces) and the transposed PML stretching, and adjoint memory variables give the gradient of the
# attenuation terms. (Only the Cerjan sponge is not transposed exactly.)

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
    ca, two = d.c[a], d.M.two           # no closures here: they would not inherit @inbounds
    c1, c0, c2, cm = coefat(ca, I + e), coefat(ca, I), coefat(ca, I + 2e), coefat(ca, I - e)   # per layer in depth
    return (A[I+e] * ifelse(two[I+e], c1[3], c1[1]) - A[I] * ifelse(two[I], c0[3], c0[1])) -
           (A[I+2e] * ifelse(two[I+2e], zero(c2[2]), c2[2]) - A[I-e] * ifelse(two[I-e], zero(cm[2]), cm[2]))
end

# ------------------------------------------------------------------ PML
# In the PML a derivative d is stretched, d̃ = γ₁ d + γ₂ m, with the memory variable m ← γ₃ m + γ₄ Δt d (`ade`). Transposed:
# the adjoint field x gathered by the transposed difference is replaced by W = γ₁ x + γ₄ Δt m̄ at every point of the PML, and
# the adjoint memory variable m̄ ← γ₃ m̄ + γ₂ x. `Δ` holds δ = W - x of every derivative (zero outside the PML), so the adjoint
# kernels gather x + δ; `stretch_*!` compute δ and update m̄ (`P.mem`) after every adjoint update of the velocities (`v`:
# the derivatives of the velocity kernel) and the stresses (`s`: those of the stress kernel).
Base.@propagate_inbounds function stretch1!(δ, mb, X, prof, I, J, ix, dt)
    γ = prof[ix]
    x, m = X[I], mb[J]
    δ[I] = (γ[1] - 1) * x + γ[4] * dt * m
    mb[J] = γ[3] * m + γ[2] * x
end

Base.@propagate_inbounds function stretch_v_psv!(Δ, Q, P, dt, i, k)
    I = CartesianIndex(i, k); J = I - P.lo; m = P.mem
    stretch1!(Δ.axSxx, m.axSxx, Q.Vx, P.gxe, I, J, i, dt); stretch1!(Δ.azSxz, m.azSxz, Q.Vx, P.gzc, I, J, k, dt)
    stretch1!(Δ.axSxz, m.axSxz, Q.Vz, P.gxc, I, J, i, dt); stretch1!(Δ.azSzz, m.azSzz, Q.Vz, P.gze, I, J, k, dt)
end
Base.@propagate_inbounds function stretch_s_psv!(Δ, Q, P, dt, i, k)
    I = CartesianIndex(i, k); J = I - P.lo; m = P.mem
    stretch1!(Δ.axVx, m.axVx, Q.Sxx, P.gxc, I, J, i, dt); stretch1!(Δ.azVz, m.azVz, Q.Szz, P.gzc, I, J, k, dt)
    stretch1!(Δ.axVz, m.axVz, Q.Sxz, P.gxe, I, J, i, dt); stretch1!(Δ.azVx, m.azVx, Q.Sxz, P.gze, I, J, k, dt)
end
Base.@propagate_inbounds function stretch_v_sh!(Δ, Q, P, dt, i, k)
    I = CartesianIndex(i, k); J = I - P.lo; m = P.mem
    stretch1!(Δ.azSyz, m.azSyz, Q.Vy, P.gzc, I, J, k, dt); stretch1!(Δ.axSxy, m.axSxy, Q.Vy, P.gxc, I, J, i, dt)
end
Base.@propagate_inbounds function stretch_s_sh!(Δ, Q, P, dt, i, k)
    I = CartesianIndex(i, k); J = I - P.lo; m = P.mem
    stretch1!(Δ.axVy, m.axVy, Q.Sxy, P.gxe, I, J, i, dt); stretch1!(Δ.azVy, m.azVy, Q.Syz, P.gze, I, J, k, dt)
end
Base.@propagate_inbounds function stretch_v_3d!(Δ, Q, P, dt, i, j, k)
    I = CartesianIndex(i, j, k); J = I - P.lo; m = P.mem
    stretch1!(Δ.axSxx, m.axSxx, Q.Vx, P.gxe, I, J, i, dt); stretch1!(Δ.aySxy, m.aySxy, Q.Vx, P.gyc, I, J, j, dt)
    stretch1!(Δ.azSxz, m.azSxz, Q.Vx, P.gzc, I, J, k, dt); stretch1!(Δ.axSxy, m.axSxy, Q.Vy, P.gxc, I, J, i, dt)
    stretch1!(Δ.aySyy, m.aySyy, Q.Vy, P.gye, I, J, j, dt); stretch1!(Δ.azSyz, m.azSyz, Q.Vy, P.gzc, I, J, k, dt)
    stretch1!(Δ.axSxz, m.axSxz, Q.Vz, P.gxc, I, J, i, dt); stretch1!(Δ.aySyz, m.aySyz, Q.Vz, P.gyc, I, J, j, dt)
    stretch1!(Δ.azSzz, m.azSzz, Q.Vz, P.gze, I, J, k, dt)
end
Base.@propagate_inbounds function stretch_s_3d!(Δ, Q, P, dt, i, j, k)
    I = CartesianIndex(i, j, k); J = I - P.lo; m = P.mem
    stretch1!(Δ.axVx, m.axVx, Q.Sxx, P.gxc, I, J, i, dt); stretch1!(Δ.ayVy, m.ayVy, Q.Syy, P.gyc, I, J, j, dt)
    stretch1!(Δ.azVz, m.azVz, Q.Szz, P.gzc, I, J, k, dt); stretch1!(Δ.axVy, m.axVy, Q.Sxy, P.gxe, I, J, i, dt)
    stretch1!(Δ.ayVx, m.ayVx, Q.Sxy, P.gye, I, J, j, dt); stretch1!(Δ.axVz, m.axVz, Q.Sxz, P.gxe, I, J, i, dt)
    stretch1!(Δ.azVx, m.azVx, Q.Sxz, P.gze, I, J, k, dt); stretch1!(Δ.ayVz, m.ayVz, Q.Syz, P.gye, I, J, j, dt)
    stretch1!(Δ.azVy, m.azVy, Q.Syz, P.gze, I, J, k, dt)
end

"Transposed difference of `A` (+ its PML correction `δ`, if any)."
Base.@propagate_inbounds dt_(d, A, ::Nothing, I, e, a) = d(A, I, e, a)
Base.@propagate_inbounds dt_(d, A, δ, I, e, a) = d(A, I, e, a) + d(δ, I, e, a)
"PML correction `name` of `Δ` (`nothing` without PML)."
@inline δof(::Nothing, ::Val) = nothing
@inline δof(Δ, ::Val{name}) where {name} = getfield(Δ, name)

"Forward difference `d` stretched as in the PML zone `Z` (`nothing`: interior), with the forward memory `Z.fm` of the step."
@inline fst(::Nothing, ::Val, ::Val, I, ix, d) = d
Base.@propagate_inbounds function fst(Z, ::Val{m}, ::Val{g}, I, ix, d) where {m,g}
    γ = getfield(Z.P, g)[ix]
    return γ[1] * d + γ[2] * getfield(Z.fm, m)[I-Z.P.lo]
end

# adjoint strains: transposed differences of the velocity kernel, applied to the adjoint velocities q
Base.@propagate_inbounds function adjoint_strain_psv(d, Q, Δ, I)
    ex, ez = CartesianIndex(1, 0), CartesianIndex(0, 1)
    return (dt_(d, Q.Vx, δof(Δ, Val(:axSxx)), I - ex, ex, 1), dt_(d, Q.Vz, δof(Δ, Val(:azSzz)), I - ez, ez, 2),
            dt_(d, Q.Vz, δof(Δ, Val(:axSxz)), I, ex, 1) + dt_(d, Q.Vx, δof(Δ, Val(:azSxz)), I, ez, 3))
end
Base.@propagate_inbounds adjoint_strain_sh(d, Q, Δ, I) =
    (dt_(d, Q.Vy, δof(Δ, Val(:azSyz)), I, CartesianIndex(0, 1), 3), dt_(d, Q.Vy, δof(Δ, Val(:axSxy)), I, CartesianIndex(1, 0), 1))
Base.@propagate_inbounds function adjoint_strain_3d(d, Q, Δ, I)
    ex, ey, ez = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    return (dt_(d, Q.Vx, δof(Δ, Val(:axSxx)), I - ex, ex, 1), dt_(d, Q.Vy, δof(Δ, Val(:aySyy)), I - ey, ey, 2),
            dt_(d, Q.Vz, δof(Δ, Val(:azSzz)), I - ez, ez, 3),
            dt_(d, Q.Vz, δof(Δ, Val(:aySyz)), I, ey, 2) + dt_(d, Q.Vy, δof(Δ, Val(:azSyz)), I, ez, 4),
            dt_(d, Q.Vz, δof(Δ, Val(:axSxz)), I, ex, 1) + dt_(d, Q.Vx, δof(Δ, Val(:azSxz)), I, ez, 4),
            dt_(d, Q.Vy, δof(Δ, Val(:axSxy)), I, ex, 1) + dt_(d, Q.Vx, δof(Δ, Val(:aySxy)), I, ey, 2))
end

# ------------------------------------------------------------------ adjoint kernels: the transposed velocity and stress
# kernels, on the whole model (`Δ`: PML corrections or `nothing`)
Base.@propagate_inbounds function vel_psv_adj!(F, M, c, dt, Δ, i, k)
    d, I = AdjD(M, c), CartesianIndex(i, k)
    ex, ez = CartesianIndex(1, 0), CartesianIndex(0, 1)
    vel_update_psv!(F, M, dt, i, k, dt_(d, F.Sxx, δof(Δ, Val(:axVx)), I, ex, 1) + dt_(d, F.Sxz, δof(Δ, Val(:azVx)), I - ez, ez, 2),
                    dt_(d, F.Sxz, δof(Δ, Val(:axVz)), I - ex, ex, 1) + dt_(d, F.Szz, δof(Δ, Val(:azVz)), I, ez, 3))
end
Base.@propagate_inbounds stress_psv_adj!(F, M, c, vc, dt, Δ, i, k) =
    stress_update_psv!(F, M, vc, dt, i, k, adjoint_strain_psv(AdjD(M, c), F, Δ, CartesianIndex(i, k))...)

Base.@propagate_inbounds function vel_sh_adj!(F, M, c, dt, Δ, i, k)
    d, I = AdjD(M, c), CartesianIndex(i, k)
    ex, ez = CartesianIndex(1, 0), CartesianIndex(0, 1)
    vel_update_sh!(F, M, dt, i, k, dt_(d, F.Sxy, δof(Δ, Val(:axVy)), I - ex, ex, 1) + dt_(d, F.Syz, δof(Δ, Val(:azVy)), I - ez, ez, 2))
end
Base.@propagate_inbounds stress_sh_adj!(F, M, c, vc, dt, Δ, i, k) =
    stress_update_sh!(F, M, vc, dt, i, k, adjoint_strain_sh(AdjD(M, c), F, Δ, CartesianIndex(i, k))...)

Base.@propagate_inbounds function vel_3d_adj!(F, M, c, dt, Δ, i, j, k)
    d, I = AdjD(M, c), CartesianIndex(i, j, k)
    ex, ey, ez = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    (; Sxx, Syy, Szz, Syz, Sxz, Sxy) = F
    dSx = dt_(d, Sxx, δof(Δ, Val(:axVx)), I, ex, 1) + dt_(d, Sxy, δof(Δ, Val(:ayVx)), I - ey, ey, 2) + dt_(d, Sxz, δof(Δ, Val(:azVx)), I - ez, ez, 3)
    dSy = dt_(d, Sxy, δof(Δ, Val(:axVy)), I - ex, ex, 1) + dt_(d, Syy, δof(Δ, Val(:ayVy)), I, ey, 2) + dt_(d, Syz, δof(Δ, Val(:azVy)), I - ez, ez, 3)
    dSz = dt_(d, Sxz, δof(Δ, Val(:axVz)), I - ex, ex, 1) + dt_(d, Syz, δof(Δ, Val(:ayVz)), I - ey, ey, 2) + dt_(d, Szz, δof(Δ, Val(:azVz)), I, ez, 4)
    vel_update_3d!(F, M, dt, i, j, k, dSx, dSy, dSz)
end
Base.@propagate_inbounds stress_3d_adj!(F, M, c, vc, dt, Δ, i, j, k) =
    stress_update_3d!(F, M, vc, dt, i, j, k, adjoint_strain_3d(AdjD(M, c), F, Δ, CartesianIndex(i, j, k))...)

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

Base.@propagate_inbounds function grad_psv!(G, E, Rb, Q, Δ, Vn, Vp, Z, M, c, vc, dt, i, k)
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
    P = aU .* tp1 .+ aR .* tp                    # weights of λ + 2μ
    S = aU .* ts1 .+ aR .* ts                    # of 2μ (normal) and μ (shear)
    T = aU .* (1 + d2(vc)) .+ aR                  # of τ
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

Base.@propagate_inbounds function grad_sh!(G, E, Rb, Q, Δ, Vn, Vp, Z, M, c, vc, dt, i, k)
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
    Rb === nothing || (G.ts[I] += T[1] * havg2(M.mu[i, k], M.mu[i, k+1]) * vyz + T[2] * havg2(M.mu[i, k], M.mu[i+1, k]) * vxy)
    G.by[I] += Q.Vy[I] * (Vn[1][I] - Vp[1][I])
end

Base.@propagate_inbounds function grad_3d!(G, E, Rb, Q, Δ, Vn, Vp, Z, M, c, vc, dt, i, j, k)
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

# ------------------------------------------------------------------ source sensitivities
# At adjoint step n: w = ∂J/∂(amplitude injected at step n) for the components of every source (moment tensor: the stress
# glut, -η sampled as `inject_*!`; force: the adjoint velocity λ = -q/B sampled as `force_*!`), stored in `W[l, j, n]`, and
# Σₙ s(n) w at the neighbour cells (`X[l, j, ±axis]`) for the location derivative. Only points of this rank count (MPI).
@inline owned(A, J) = all(ntuple(a -> H < J[a] <= size(A, a) - H, ndims(A)))
Base.@propagate_inbounds ov(A, J) = owned(A, J) ? A[J] : zero(eltype(A))       # a branch: J may lie outside A
Base.@propagate_inbounds avg4(A, I, e1, e2) = (ov(A, I) + ov(A, I - e1) + ov(A, I - e2) + ov(A, I - e1 - e2)) / 4
Base.@propagate_inbounds avg2(A, I, e) = (ov(A, I) + ov(A, I - e)) / 2

Base.@propagate_inbounds function mt_psv(E, Q, M, I)
    z, ex, ez = zero(eltype(E[1])), CartesianIndex(1, 0), CartesianIndex(0, 1)
    return (-ov(E[1], I), z, -ov(E[2], I), z, -avg4(E[3], I, ex, ez), z)
end
Base.@propagate_inbounds function mt_sh(E, Q, M, I)
    z = zero(eltype(E[1]))
    return (z, z, z, -avg2(E[1], I, CartesianIndex(0, 1)), z, -avg2(E[2], I, CartesianIndex(1, 0)))
end
Base.@propagate_inbounds function mt_3d(E, Q, M, I)
    ex, ey, ez = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    return (-ov(E[1], I), -ov(E[2], I), -ov(E[3], I), -avg4(E[4], I, ey, ez), -avg4(E[5], I, ex, ez), -avg4(E[6], I, ex, ey))
end
Base.@propagate_inbounds function f_psv(E, Q, M, I)
    ρ, ex, ez = M.rho, CartesianIndex(1, 0), CartesianIndex(0, 1)
    lam(V, J, e) = owned(V, J) ? V[J] * (ρ[J] + ρ[J+e]) / 2 : zero(eltype(V))      # q / B
    owned(ρ, I) || return ntuple(_ -> zero(eltype(ρ)), 3)
    return (-(lam(Q.Vx, I, ex) + lam(Q.Vx, I - ex, ex)) / 2ρ[I], zero(eltype(ρ)), -(lam(Q.Vz, I, ez) + lam(Q.Vz, I - ez, ez)) / 2ρ[I])
end
Base.@propagate_inbounds f_sh(E, Q, M, I) = (zero(eltype(M.rho)), -ov(Q.Vy, I), zero(eltype(M.rho)))
Base.@propagate_inbounds f_3d(E, Q, M, I) = (-avg2(Q.Vx, I, CartesianIndex(1, 0, 0)), -avg2(Q.Vy, I, CartesianIndex(0, 1, 0)),
                                             -avg2(Q.Vz, I, CartesianIndex(0, 0, 1)))

@inline unit(::CartesianIndex{N}, a) where {N} = CartesianIndex(ntuple(b -> Int(b == a), Val(N)))
@inline cellof(S, l) = haskey(S, :j) ? CartesianIndex(S.i[l], S.j[l], S.k[l]) : CartesianIndex(S.i[l], S.k[l])
Base.@propagate_inbounds function source_sens!(W, X, sample, E, Q, M, S, n)
    for l in 1:length(S.i)
        I = cellof(S, l)
        w = sample(E, Q, M, I)
        for j in eachindex(w)
            W[l, j, n] = w[j]
        end
        s = S.s[l, n]
        for a in 1:length(I), (q, o) in ((2a - 1, 1), (2a, -1))
            wo = sample(E, Q, M, I + o * unit(I, a))
            for j in eachindex(wo)
                X[l, j, q] += s * wo[j]
            end
        end
    end
end

# kernels, accumulators and the staggering of shear moduli (offsets of the cells averaged) and buoyancies per mode
const O2, O3 = CartesianIndex{2}, CartesianIndex{3}
adjoint_setup(::Val{:psv}) = (mt_psv, f_psv, vel_psv_adj!, stress_psv_adj!, grad_psv!, residual_psv!, stretch_v_psv!, stretch_s_psv!, (:axVx, :azVz, :axVz, :azVx), (:lam, :mu, :xz, :tp, :ts, :bx, :bz), 3,
                              (xz=(O2(0, 0), O2(0, 1), O2(1, 0), O2(1, 1)),), (bx=O2(1, 0), bz=O2(0, 1)))
adjoint_setup(::Val{:sh}) = (mt_sh, f_sh, vel_sh_adj!, stress_sh_adj!, grad_sh!, residual_sh!, stretch_v_sh!, stretch_s_sh!, (:axVy, :azVy), (:yz, :xy, :ts, :by), 2, (yz=(O2(0, 0), O2(0, 1)), xy=(O2(0, 0), O2(1, 0))), (by=nothing,))
adjoint_setup(::Val{:p3d}) = (mt_3d, f_3d, vel_3d_adj!, stress_3d_adj!, grad_3d!, residual_3d!, stretch_v_3d!, stretch_s_3d!,
                              (:axVx, :ayVy, :azVz, :axVy, :ayVx, :axVz, :azVx, :ayVz, :azVy), (:lam, :mu, :yz, :xz, :xy, :tp, :ts, :bx, :by, :bz), 6,
                              (yz=(O3(0, 0, 0), O3(0, 0, 1), O3(0, 1, 0), O3(0, 1, 1)), xz=(O3(0, 0, 0), O3(0, 0, 1), O3(1, 0, 0), O3(1, 0, 1)),
                               xy=(O3(0, 0, 0), O3(1, 0, 0), O3(0, 1, 0), O3(1, 1, 0))), (bx=O3(1, 0, 0), by=O3(0, 1, 0), bz=O3(0, 0, 1)))

"∂ havg(m...) / ∂ m[a] of the harmonic average of the moduli `m` (as `havg2`, `havg4`)."
function dhavg(m::NTuple{K}, a) where {K}
    p(j) = prod(m[l] for l in 1:K if l != j)
    S = sum(p, 1:K)
    return S == 0 ? 0.0 : K * p(a)^2 / S^2
end

"Gradients with respect to rho, lam, mu (internal arrays) from the accumulators `G` and the shear modulus `mu` (host arrays)."
function assemble_gradient(G, mu, mode)
    shear, buoy = adjoint_setup(Val(mode))[12:13]
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
struct AdjointDriver{F}
    observed::Seismograms
    checkpoint_steps::Union{Nothing,Int}
    misfit::F
    illumination::Bool                  # also Σₙ |vⁿ|² of the forward wavefield (preconditioning, imaging)
end
AdjointDriver(observed, K, misfit) = AdjointDriver(observed, K, misfit, false)

"Illumination: add |v|² (the velocity components at the cell) to `L`."
Base.@propagate_inbounds illuminate!(L, V, i, k) = (I = CartesianIndex(i, k); L[I] += energy(V, I))
Base.@propagate_inbounds illuminate!(L, V, i, j, k) = (I = CartesianIndex(i, j, k); L[I] += energy(V, I))
Base.@propagate_inbounds energy(V::NTuple{1,Any}, I) = V[1][I]^2
Base.@propagate_inbounds energy(V::NTuple{2,Any}, I) = V[1][I]^2 + V[2][I]^2
Base.@propagate_inbounds energy(V::NTuple{3,Any}, I) = V[1][I]^2 + V[2][I]^2 + V[3][I]^2

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
    p.parallel && (KernelAbstractions.synchronize(backend); exchange_halo!(backend, p.nb, stresses(F)...))
    p.pw_edges && extrapolate_sides!(stresses(F), p.edges)
    map(((lo, ndr, P),) -> foreach_cell!(backend, vel!, (F, M, c, dtT, P), lo, ndr), regions)
    p.noforce || run_once!(backend, force!, (F, M, FSrc, it))
    p.parallel && (KernelAbstractions.synchronize(backend); exchange_halo!(backend, p.nb, velocities(F)...))
    return nothing
end

"Check the observed seismograms against the simulation `s`; their velocities at the stations of this rank."
function observed_velocities(obs, s)
    length(obs.t) == s.nrec && obs.components == s.components ||
        error("the observed seismograms do not match the simulation (samples $(length(obs.t)) vs $(s.nrec), components)")
    return obs.v[:, :, s.own]
end

function (d::AdjointDriver)(s)
    observed_velocities(d.observed, s)
    (; F, M, St, rec, acc, regions, backend, nt, nown, c, vc, dtT, record_every) = s
    mt_sample, f_sample, vel!, stress!, grad!, residual!, stretch_v!, stretch_s!, fmnames, names, nE = adjoint_setup(Val(s.mode))
    s.sinc && (residual! = sinc_kernels(s.mode)[3])
    sg = stagger(Val(s.mode))
    V = velocities(F)
    new(A) = fill!(similar(A), 0)
    attenuation = maximum(abs, M.taup) > 0 || maximum(abs, M.taus) > 0
    state = attenuation ? (values(F)..., pml_state(regions)...) : (V..., stresses(F)..., pml_state(regions)...)
    zones = Base.tail(regions)                                  # the PML zones (none with the Cerjan absorber)
    # forward PML memory of the stress kernel, kept for every step of a segment (stretched strain rates in the gradient)
    fmem() = map(z -> map(new, NamedTuple{fmnames}(z[3].mem)), zones)
    copy_fmem!(dst) = map((m, z) -> map(copyto!, m, NamedTuple{fmnames}(z[3].mem)), dst, zones)
    # checkpoints every K steps (host memory) and a buffer of the K + 1 velocity fields of a segment (device)
    per_step = sum(length, V) + sum(m -> sum(length, m; init=0), fmem(); init=0)
    K = something(d.checkpoint_steps, ceil(Int, sqrt(nt * sum(length, state) / per_step)))
    s.parallel && (K = MPI.Allreduce(K, max, IGG.comm()))      # the same segments (and halo exchanges) on all ranks
    step!(it, record) = forward_step!(F, M, s.Src, s.FSrc, St, rec, acc, regions, it, s, record)

    # forward pass: seismograms and checkpoints
    saved = typeof(map(Array, state))[]
    for it in 1:nt
        (it - 1) % K == 0 && push!(saved, map(Array, state))
        step!(it, true)
    end
    KernelAbstractions.synchronize(backend)
    # misfit of the seismograms of all stations (on every rank), and its derivative at the stations of this rank
    syn = zeros(size(rec, 1), s.nc, length(s.stations))
    syn[:, :, s.own] .= Array(rec)[:, 1:s.nc, 1:nown] .* s.scale
    s.parallel && MPI.Allreduce!(syn, +, IGG.comm())
    J, adj = d.misfit(syn, d.observed.v, d.observed.t)
    size(adj) == size(syn) || error("the misfit function must return the misfit and its derivative (size $(size(syn)))")
    res = KernelAbstractions.zeros(backend, s.T, size(rec, 1), s.nc, max(nown, 1))
    copyto!(res, s.T.(cat(adj[:, :, s.own] .* s.scale, zeros(size(res, 1), s.nc, size(res, 3) - nown); dims=3)))    # ∂J/∂rec

    # 2nd-order flags of all cells (next to free surfaces and at the outer edge of the PML), for the transposed differences
    fs = map(A -> Array(A), (M.fs_top, M.fs_bot, M.ob_top, M.ob_bot))
    P = regions[end][3]
    o = P === nothing ? () : ndims(M.lam) == 3 ? (Array(P.ox), Array(P.oy)) : (Array(P.ox),)
    two = [second_order(last(Tuple(I)), map(f -> f[Base.front(Tuple(I))...], fs)...) | any(a -> o[a][I[a]], eachindex(o))
           for I in CartesianIndices(M.lam)]
    Ma = merge(M, (; two=copyto!(similar(M.lam, Bool), two)))
    # adjoint fields, PML corrections and adjoint PML memory, adjoint strains, adjoint attenuation memory, accumulators
    Q = map(new, F)
    Δ = isempty(zones) ? nothing : map(_ -> new(M.lam), zones[1][3].mem)
    azones = map(((lo, ndr, P),) -> (lo, ndr, merge(P, (; mem=map(new, P.mem)))), zones)
    # corrections of the derivatives of the velocity kernel (used by the adjoint stress kernel) and of the stress kernel;
    # kernels get only what they use (GPU kernel arguments are limited, e.g. to 4 kB on Metal)
    Δv, Δs = Δ === nothing ? (nothing, nothing) : (Δ[filter(∉(fmnames), keys(Δ))], Δ[fmnames])
    halo_v, halo_s = Δ === nothing ? ((), ()) : (values(Δv), values(Δs))
    Qv = NamedTuple{filter(k -> first(string(k)) == 'V', keys(Q))}(Q)
    Zp = map(z -> Base.structdiff(z[3], NamedTuple{(:mem,)}), zones)
    # cells: the deep interior (stencils do not reach the PML: no corrections) and the rest (a band of 2 cells, the PML)
    inner = (ntuple(_ -> H, ndims(M.lam)), size(M.lam) .- 2H)
    deep, band = Δ === nothing ? (inner, ()) : split_box(regions[1][1], regions[1][2], 2)
    rest = Δ === nothing ? () : (band..., map(z -> (z[1], z[2]), zones)...)
    E = ntuple(_ -> new(M.lam), nE)
    Rb = attenuation ? map(new, values(F)[end-nE+1:end]) : nothing
    G = NamedTuple{names}(map(_ -> new(M.lam), names))
    L = d.illumination ? new(M.lam) : nothing
    buf = [(map(new, V), fmem()) for _ in 0:K]
    N = ndims(M.lam)
    SW, SX = KernelAbstractions.zeros(backend, s.T, length(s.Src.i), 6, nt), KernelAbstractions.zeros(backend, s.T, length(s.Src.i), 6, 2N)
    FW, FX = KernelAbstractions.zeros(backend, s.T, length(s.FSrc.i), 3, nt), KernelAbstractions.zeros(backend, s.T, length(s.FSrc.i), 3, 2N)
    sync_halo!(A) = s.parallel && (KernelAbstractions.synchronize(backend); exchange_halo!(backend, s.nb, A...))
    # backward, segment by segment: recompute the forward velocities, then the adjoint steps n = b … a
    for (seg, a) in Iterators.reverse(enumerate(1:K:nt))
        b = min(a + K - 1, nt - 1)
        map(copyto!, state, saved[seg])
        map(copyto!, buf[1][1], V)
        copy_fmem!(buf[1][2])
        for it in a:b
            step!(it, false)
            map(copyto!, buf[it-a+2][1], V)
            copy_fmem!(buf[it-a+2][2])
        end
        for n in b:-1:a
            (Vn, _), (Vp, fm) = buf[n-a+2], buf[n-a+1]
            foreach_cell!(backend, vel!, (Q, Ma, c, dtT, nothing), deep...)
            map(((lo, ndr),) -> foreach_cell!(backend, vel!, (Q, Ma, c, dtT, Δs), lo, ndr), rest)
            itw = n % record_every == 0 ? n ÷ record_every + 1 : 0
            itw > 0 && nown > 0 && run_once!(backend, residual!, (Q, Ma, St, res, itw, nown))
            sync_halo!(velocities(Q))
            map(((lo, ndr, P),) -> foreach_cell!(backend, stretch_v!, (Δv, Qv, P, dtT), lo, ndr), azones)
            sync_halo!(halo_v)
            foreach_cell!(backend, grad!, (G, E, Rb, Qv, nothing, Vn, Vp, nothing, Ma, c, vc, dtT), deep...)
            L === nothing || foreach_cell!(backend, illuminate!, (L, Vn), inner...)
            map(((lo, ndr),) -> foreach_cell!(backend, grad!, (G, E, Rb, Qv, Δv, Vn, Vp, nothing, Ma, c, vc, dtT), lo, ndr), band)
            map((z, P, m) -> foreach_cell!(backend, grad!, (G, E, Rb, Qv, Δv, Vn, Vp, (; P, fm=m), Ma, c, vc, dtT), z[1], z[2]), zones, Zp, fm)
            foreach_cell!(backend, stress!, (Q, Ma, c, vc, dtT, nothing), deep...)
            map(((lo, ndr),) -> foreach_cell!(backend, stress!, (Q, Ma, c, vc, dtT, Δv), lo, ndr), rest)
            if s.sinc
                s.nosource || run_once!(backend, source_sens_sinc!, (SW, SX, E, -one(s.T), s.Src, sg.mcols, n))
                s.noforce || run_once!(backend, source_sens_sinc!, (FW, FX, velocities(Qv), -one(s.T), s.FSrc, sg.fcols, n))
            else
                s.nosource || run_once!(backend, source_sens!, (SW, SX, mt_sample, E, Qv, M, s.Src, n))
                s.noforce || run_once!(backend, source_sens!, (FW, FX, f_sample, E, Qv, M, s.FSrc, n))
            end
            sync_halo!(stresses(Q))
            map(((lo, ndr, P),) -> foreach_cell!(backend, stretch_s!, (Δs, Q, P, dtT), lo, ndr), azones)
            sync_halo!(halo_s)
        end
    end
    KernelAbstractions.synchronize(backend)
    # accumulators of all ranks on the global grid (rank 0), chain rule there, result to all ranks
    Gg = map(A -> gather_interior(A, s), G)
    g = s.me == 0 ? model_gradient(s, assemble_gradient(Gg, s.med.mu, s.mode)) : nothing
    if L !== nothing
        Lg = gather_interior(L, s)
        if s.me == 0                    # the cells of the model (z up), in the wavefield units squared -> (m/s)²
            Lm = reverse(reshape(Lg[ntuple(a -> H+1:H+s.g.n[a], length(s.g.n))...], size(s.m.rho)), dims=3) .* s.scale^2
            g = CartData(g.x.val, g.y.val, g.z.val, merge(g.fields, (; illumination=Lm)))
        end
    end
    s.parallel && (g = MPI.bcast(g, 0, IGG.comm()))
    return J, g, source_gradients(s, SW, SX, FW, FX)
end

"Sum of the per-rank source sensitivities `A[l, …]` of the sources `idx` (indices in the list of all sources) over the ranks."
function global_sources(A, idx, n, s)
    G = zeros(n, size(A)[2:end]...)
    G[idx, :, :] .= Array(A)
    s.parallel && MPI.Allreduce!(G, +, IGG.comm())
    return G
end

"""
Gradients with respect to the parameters of every source of `s.sources` (`nothing` for plane waves and ignored sources), from
the sensitivities to the injected amplitudes `W` (per step) and `X` (neighbour cells): see [`misfit_gradient`](@ref).
"""
function source_gradients(s, SW, SX, FW, FX)
    g, N, nt, dt = s.g, length(s.g.n), s.nt, s.dt
    W, X = global_sources(SW, s.iS, length(s.moments), s), global_sources(SX, s.iS, length(s.moments), s)
    FWg, FXg = global_sources(FW, s.iF, length(s.forces), s), global_sources(FX, s.iF, length(s.forces), s)
    # ∂J/∂(position) from the sensitivities at the neighbour cells (or of the interpolation weights, per nominal spacing);
    # internal axes (x, [y,] depth) -> (x, y, z up)
    function position(A, amp, l, src)
        span(a) = a < N || g.uniform || s.sinc ? 2g.d[a] : (k = cell_index(g, N, -src.z); g.zc[k+1] - g.zc[k-1])
        d = ntuple(a -> sum(amp[j] * (A[l, j, 2a-1] - A[l, j, 2a]) for j in eachindex(amp)) / span(a), N)
        return N == 3 ? (d[1], d[2], -d[3]) : (d[1], 0.0, -d[2])
    end
    # ∂J/∂t0, ∂J/∂trise from the gradient with respect to the moment rate (central differences of the source time function)
    function timing(gm, times, src)
        mr(t, t0, tr) = moment_rate(s.stf, t, t0, tr)
        h = 1e-6 * src.trise
        dt0 = sum(gm[n] * (mr(times[n], src.t0 + h, src.trise) - mr(times[n], src.t0 - h, src.trise)) / 2h for n in 1:nt)
        dtr = sum(gm[n] * (mr(times[n], src.t0, src.trise + h) - mr(times[n], src.t0, src.trise - h)) / 2h for n in 1:nt)
        return dt0, dtr
    end
    flip(j) = j in (4, 5) ? -1.0 : 1.0                          # z up -> z down (as in `simulate`)
    tm = [s.tbeg + (n - 0.5) * dt for n in 1:nt]                 # times of the moment-tensor injections
    tf = [s.tbeg + n * dt for n in 1:nt]                         # and of the forces
    mts = map(enumerate(s.moments)) do (l, src)
        Mi = ntuple(j -> src.M[j] * flip(j), 6)                    # internal moment tensor
        vol = s.volume[l]                                          # as injected (`simulate`)
        sl = [src.M0 / s.M0 * moment_rate(s.stf, tm[n], src.t0, src.trise) * dt / vol for n in 1:nt]
        dMi = ntuple(j -> sum(sl[n] * W[l, j, n] for n in 1:nt), 6)  # ∂J/∂Mi
        dP = ntuple(j -> flip(j) * dMi[j] / src.M0, 6)             # ∂J/∂(M0 M)
        gm = [sum(Mi[j] * W[l, j, n] for j in 1:6) * src.M0 / s.M0 * dt / vol for n in 1:nt]    # ∂J/∂(moment rate)
        x, y, z = position(X, Mi, l, src)
        t0, trise = timing(gm, tm, src)
        (; M=dP, M0=sum(src.M[j] * dP[j] for j in 1:6), x, y, z, t0, trise, stf=gm, t=tm)
    end
    fs = map(enumerate(s.forces)) do (l, src)
        Fi = ntuple(m -> src.F[m] * (m == 3 ? -1.0 : 1.0), 3)
        vol = s.fvolume[l]
        sl = [1e3 / s.M0 * moment_rate(s.stf, tf[n], src.t0, src.trise) * dt / vol for n in 1:nt]
        dF = ntuple(m -> (m == 3 ? -1.0 : 1.0) * sum(sl[n] * FWg[l, m, n] for n in 1:nt), 3)
        gm = [sum(Fi[m] * FWg[l, m, n] for m in 1:3) * 1e3 / s.M0 * dt / vol for n in 1:nt]
        x, y, z = position(FXg, Fi, l, src)
        t0, trise = timing(gm, tf, src)
        (; F=dF, x, y, z, t0, trise, stf=gm, t=tf)
    end
    # in the order of the sources given to `simulate`
    return map(s.sources) do src
        i = src isa MomentTensorSource ? findfirst(x -> x === src, s.moments) : src isa ForceSource ? findfirst(x -> x === src, s.forces) : nothing
        i === nothing ? nothing : src isa MomentTensorSource ? mts[i] : fs[i]
    end
end

"""
The box `lo .+ (1:ndr)` as the box shrunk by `w` cells on every side, and up to 2N slabs (tuple) covering the rest.
"""
function split_box(lo, ndr, w)
    N = length(lo)
    n1 = min.(w, ndr)
    ilo, indr = lo .+ n1, max.(ndr .- 2w, 0)
    slabs = ntuple(2N) do q
        a, hi = (q + 1) ÷ 2, iseven(q)
        l, n = hi ? (ilo[a] + indr[a], ndr[a] - n1[a] - indr[a]) : (lo[a], n1[a])
        (ntuple(b -> b < a ? ilo[b] : b == a ? l : lo[b], N), ntuple(b -> b < a ? indr[b] : b == a ? n : ndr[b], N))
    end
    return (ilo, indr), slabs
end

"The cells of `A` owned by this rank, gathered on rank 0 into an array of the global (padded) grid; `nothing` on other ranks."
function gather_interior(A, s)
    N = length(s.nl)
    in_ = ntuple(a -> H+1:s.nl[a]-H, N)
    B = Float64.(Array(A)[in_...])
    G = s.me == 0 ? zeros(size(B) .* s.dims) : nothing
    s.parallel ? IGG.gather!(B, G) : (G = B)
    s.me == 0 || return nothing
    P = zeros(size(G) .+ 2H)
    P[ntuple(a -> H+1:H+size(G, a), N)...] = G
    return P
end
