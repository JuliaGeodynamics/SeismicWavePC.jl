# Misfit gradients on curvilinear grids (2D P-SV, elastic): the exact discrete adjoint of the leapfrog scheme of
# curvilinear.jl. The scheme conserves energy, E = ½ Σ H (ρJ |v|² + J σ:C⁻¹σ), so its momentum operator A (divergence,
# all SAT terms: ρJ ∂t v = A σ) and gradient operator G (strains, with the interface penalty: ∂t σ = C G v) satisfy
#     H A = -(H J G)ᵀ   (in the pairing σ11 exx + σ22 ezz + σ12 exz),
# i.e. Aᵀ = -(H J) G H⁻¹ and Gᵀ = -H A (H J)⁻¹: the adjoint applies the forward kernels (on "operator blocks" with
# ρJ = 1 for A, and λ = 0, μ = ½ for G) and pointwise scalings. The implicit impedance damping is diagonal. The forward
# wavefield comes from checkpoints, recomputed segment by segment. Gradients with respect to λ, μ, 1/(ρJ) and the damping
# (vp, vs at the boundary nodes) are turned into ρ, vp, vs on the nodes of the model.

struct CurvilinearGradient{O,M}
    observed::O
    misfit::M
    checkpoint_steps::Union{Int,Nothing}
end

"Add the residual (∂J/∂v at the recorded samples) × `s` to the adjoint velocities at the stations (transposed interpolation)."
Base.@propagate_inbounds function residual_cv!(blocks, St, R, itw, s)
    for n in 1:length(St.b)
        B = blocks[St.b[n]]
        for c in 1:6, a in 1:6
            w = s * St.wx[a, n] * St.wz[c, n]
            i, k = St.i0[n] + a - 1, St.k0[n] + c - 1
            B.V1[i, k] += w * R[itw, 1, n]
            B.V2[i, k] += w * R[itw, 2, n]
        end
    end
end
"The forces depend on 1/(ρJ) at their nodes: their part of the gradient."
Base.@propagate_inbounds function force_grad_cv!(G, A, S, it, dt)
    for m in 1:length(S.i)
        b, i, k, n = S.b[m], S.i[m], S.k[m], S.n[m]
        s = dt * S.c[m] * S.s[n, it]
        G[b].iρJ[i, k] += s * (S.F[n, 1] * A[b].V1[i, k] + S.F[n, 2] * A[b].V2[i, k])
    end
end

function drive_curvilinear(d::CurvilinearGradient, c)
    (; backend, FT, blocks, hostblocks, nb, nx, nzt, nl, no, hx, ox, xr, ia, ib, par, me, dt, dtT, nt, nrec, record_every,
       momentum!, stress!, step!, exchange_v!, exchange_s!, St, nst, FS, forces, Hξ, tbeg, attenuation, dev, model,
       all_traces, stations, density_unit) = c
    attenuation && error("misfit gradients on curvilinear grids: elastic models only (for now)")
    zero_(H) = KernelAbstractions.zeros(backend, FT, nl, H.nz)
    fields(B) = (B.V1, B.V2, B.S11, B.S22, B.S12)
    # 1. forward, with checkpoints (host) every K steps; the misfit and its residual
    K = something(d.checkpoint_steps, max(1, ceil(Int, sqrt(nt))))
    ckpt = Dict{Int,Any}()
    for it in 1:nt
        (it - 1) % K == 0 && (ckpt[it] = [Array.(fields(B)) for B in blocks])
        step!(it, true)
    end
    tr = all_traces()
    J, adj = d.misfit(tr[:, 1:2, :], d.observed.v, d.observed.t)
    # the residuals go to the nodes of this process of every station whose stencil reaches them (a station near a
    # subdomain boundary feeds both sides; the forward run reads the neighbour's nodes from the halo instead)
    sel = [n for (n, s) in enumerate(c.st_all) if s[2] <= ib && s[2] + 5 >= ia]
    St = (; b=dev([c.st_all[n][1] for n in sel]), i0=dev([c.st_all[n][2] - ox for n in sel]), k0=dev([c.st_all[n][3] for n in sel]),
          wx=dev(FT[(ia <= c.st_all[n][2] + a - 1 <= ib) * c.st_all[n][4][a] for a in 1:6, n in sel]),
          wz=dev(FT[c.st_all[n][5][a] for a in 1:6, n in sel]))
    nst = length(sel)
    Rres = dev(FT.(adj[:, :, sel]))
    # 2. the adjoint fields, operator blocks and gradient accumulators
    ones_(H) = (A = zero_(H); fill!(A, 1); A)
    halves(H) = (A = zero_(H); fill!(A, FT(0.5)); A)
    loc(A) = dev(FT.(A[xr, :]))
    Hw = [loc([Hξ[i] * h for i in 1:nx, h in sbp_weights(H.nz, H.hη)]) for H in hostblocks]          # SBP norm per node
    HJ = [Hw[b] .* blocks[b].J for b in 1:nb]
    Adj = Tuple((; V1=zero_(H), V2=zero_(H), S11=zero_(H), S22=zero_(H), S12=zero_(H)) for H in hostblocks)
    Grad = Tuple((; lam=zero_(H), mu=zero_(H), iρJ=zero_(H), a1=zero_(H), a2=zero_(H)) for H in hostblocks)
    Win = Tuple((zero_(H), zero_(H)) for H in hostblocks)                  # inputs of G (velocities)
    Gout = Tuple((zero_(H), zero_(H), zero_(H)) for H in hostblocks)       # its strains (exx, ezz, exz/2)
    Yin = Tuple((zero_(H), zero_(H), zero_(H)) for H in hostblocks)        # inputs of A (stresses)
    Aout = Tuple((zero_(H), zero_(H)) for H in hostblocks)                 # A y
    Gblk = Tuple(merge(blocks[b], (; V1=Win[b][1], V2=Win[b][2], S11=Gout[b][1], S22=Gout[b][2], S12=Gout[b][3],
                                    lam=zero_(hostblocks[b]), mu=halves(hostblocks[b]), tp=nothing, ts=nothing)) for b in 1:nb)
    Ablk = Tuple(merge(blocks[b], (; S11=Yin[b][1], S22=Yin[b][2], S12=Yin[b][3], V1=Aout[b][1], V2=Aout[b][2],
                                    iρJ=ones_(hostblocks[b]))) for b in 1:nb)
    AblkF = Tuple(merge(Ablk[b], (; S11=blocks[b].S11, S22=blocks[b].S22, S12=blocks[b].S12)) for b in 1:nb)   # A σ of the forward
    "G w: the strains of the velocities in Win (exchanged first); exz in Gout[3] is halved"
    function Gop!()
        foreach(b -> cv_exchange!(par, Win[b]), 1:nb)
        foreach(g -> foreach(A -> fill!(A, 0), g), Gout)
        stress!(nothing, one(FT), Gblk)
    end
    function Aop!(blk)
        blk === Ablk && foreach(b -> cv_exchange!(par, Yin[b]), 1:nb)
        foreach(a -> foreach(A -> fill!(A, 0), a), Aout)
        momentum!(one(FT), false, blk)
    end
    # 3. backward, segment by segment
    store = [Tuple(Tuple(zero_(H) for _ in 1:6) for H in hostblocks) for _ in 1:K]   # per step: v old (2), v new (2), A σ (2)
    for s0 in sort!(collect(keys(ckpt)); rev=true)
        s1 = min(s0 + K - 1, nt)
        for (b, B) in enumerate(blocks), (A, h) in zip(fields(B), ckpt[s0][b])
            copyto!(A, h)
        end
        for it in s0:s1
            q = store[it-s0+1]
            Aop!(AblkF)
            for b in 1:nb
                copyto!(q[b][1], blocks[b].V1); copyto!(q[b][2], blocks[b].V2)
                copyto!(q[b][5], Aout[b][1]); copyto!(q[b][6], Aout[b][2])
            end
            step!(it, false)
            for b in 1:nb
                copyto!(q[b][3], blocks[b].V1); copyto!(q[b][4], blocks[b].V2)
            end
        end
        for it in s1:-1:s0
            n = it - 1
            itw = n % record_every == 0 ? n ÷ record_every + 1 : 0
            q = store[it-s0+1]
            # stress update σ^{n+1} = σⁿ + dt C G v^{n+1/2}: gradient (λ, μ) and v̄ += dt Gᵀ C σ̄
            for b in 1:nb
                copyto!(Win[b][1], q[b][3]); copyto!(Win[b][2], q[b][4])
            end
            Gop!()
            for b in 1:nb
                a, g, (ex, ez, ex2) = Adj[b], Grad[b], Gout[b]
                @. g.lam += dtT * (a.S11 + a.S22) * (ex + ez)
                @. g.mu += dtT * (2a.S11 * ex + 2a.S22 * ez + 2a.S12 * ex2)
                l, m = blocks[b].lam, blocks[b].mu
                Y1, Y2, Y3 = Yin[b]
                @. Y1 = ((l + 2m) * a.S11 + l * a.S22) / HJ[b]
                @. Y2 = (l * a.S11 + (l + 2m) * a.S22) / HJ[b]
                @. Y3 = m * a.S12 / HJ[b]
            end
            Aop!(Ablk)
            for b in 1:nb
                @. Adj[b].V1 -= dtT * Hw[b] * Aout[b][1]
                @. Adj[b].V2 -= dtT * Hw[b] * Aout[b][2]
            end
            # recording at step n: v(tₙ) = (v^{n-1/2} + v^{n+1/2}) / 2
            itw > 0 && nst > 0 && run_once!(backend, residual_cv!, (Adj, St, Rres, itw, FT(0.5)))
            # momentum v^{n+1/2} = P v^{n-1/2} + Q dt (1/ρJ) A σⁿ (+ forces): gradient (1/ρJ, damping) and σ̄ += Aᵀ(...)
            length(forces) > 0 && run_once!(backend, force_grad_cv!, (Grad, Adj, FS, it, dtT))
            for b in 1:nb
                a, g, B = Adj[b], Grad[b], blocks[b]
                b1, b2 = B.a1, B.a2
                @. g.iρJ += dtT * (a.V1 * q[b][5] / (1 + b1 * dtT / 2) + a.V2 * q[b][6] / (1 + b2 * dtT / 2))
                @. g.a1 -= dtT / 2 * a.V1 * (q[b][1] + q[b][3]) / (1 + b1 * dtT / 2)
                @. g.a2 -= dtT / 2 * a.V2 * (q[b][2] + q[b][4]) / (1 + b2 * dtT / 2)
                @. Win[b][1] = dtT * B.iρJ * a.V1 / ((1 + b1 * dtT / 2) * Hw[b])
                @. Win[b][2] = dtT * B.iρJ * a.V2 / ((1 + b2 * dtT / 2) * Hw[b])
            end
            Gop!()
            for b in 1:nb
                a, B, (ex, ez, ex2) = Adj[b], blocks[b], Gout[b]
                @. a.S11 -= HJ[b] * ex
                @. a.S22 -= HJ[b] * ez
                @. a.S12 -= HJ[b] * 2ex2
                @. a.V1 *= (1 - B.a1 * dtT / 2) / (1 + B.a1 * dtT / 2)
                @. a.V2 *= (1 - B.a2 * dtT / 2) / (1 + B.a2 * dtT / 2)
            end
            itw > 0 && nst > 0 && run_once!(backend, residual_cv!, (Adj, St, Rres, itw, FT(0.5)))
        end
    end
    KernelAbstractions.synchronize(backend)
    # 4. to ρ, vp, vs [per kg/m³, km/s] on the nodes of the model (owned nodes; summed over the processes)
    gρ, gvp, gvs = zeros(nx, nzt), zeros(nx, nzt), zeros(nx, nzt)
    for (b, H) in enumerate(hostblocks)
        g = map(A -> Float64.(Array(A)), Grad[b])
        for k in 1:H.nz, il in hx+1:hx+no
            i = il + ox
            ρ, vp, vs, Jn = H.ρ[i, k], H.vp[i, k], H.vs[i, k], H.J[i, k]
            gl, gm, gi, ga1, ga2 = g.lam[il, k], g.mu[il, k], g.iρJ[il, k], g.a1[il, k], g.a2[il, k]
            r = H.kmap[i, k]
            gρ[i, r] += gl * (vp^2 - 2vs^2) + gm * vs^2 - gi / (ρ^2 * Jn)
            gvp[i, r] += gl * 2ρ * vp + ga1 * H.da.a1vp[i, k] + ga2 * H.da.a2vp[i, k]
            gvs[i, r] += -4gl * ρ * vs + 2gm * ρ * vs + ga1 * H.da.a1vs[i, k] + ga2 * H.da.a2vs[i, k]
        end
    end
    if par !== nothing
        foreach(A -> MPI.Allreduce!(A, +, par.comm), (gρ, gvp, gvs))
    end
    rho_unit = density_unit === :g_cm3 ? 1e3 : 1.0
    r3(A) = reshape(A, nx, 1, nzt)
    g = CartData(model.x.val, model.y.val, model.z.val, (; rho=r3(gρ .* rho_unit), vp=r3(gvp .* 1e3), vs=r3(gvs .* 1e3),
                                                          Qp=r3(zeros(nx, nzt)), Qs=r3(zeros(nx, nzt))))
    return J, g, Any[nothing for _ in 1:c.nsrc]              # (no source gradients on curvilinear grids yet)
end
