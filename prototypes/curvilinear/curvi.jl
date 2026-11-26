# Prototype: 2D P-SV on a terrain-following (curvilinear) grid with SBP operators and a SAT free surface, on the topography
# cases of Nobuaki Fuji (tilted planes, hills; as examples/topography_benchmark.jl), with SeismicWavePC runs as references.
#
# Grid: x = ξ, z = η + h(ξ) w(η): the surface h(x) at η = 0; w = 1 (columns sheared with the surface everywhere) or, with
# a flattening depth D, w goes smoothly from 1 at the surface to 0 at η = -D (a Cartesian grid below). Collocated
# velocity-stress, dual-pairing upwind SBP operators of SummationByPartsOperators.jl (default: boundary-optimized closures,
# Mattsson, Niemelä & Winters 2026, order 7; their first grid points are not equidistant):
# H D₊ + (H D₋)ᵀ = B, D₋ for the divergence (momentum, conservative form), D₊ for the gradient (stress, non-conservative
# form). The two are adjoint in the J-weighted SBP norm, so the semi-discrete energy changes only through boundary terms;
# the SAT (the traction removed at every boundary) makes all boundaries traction-free: the energy is conserved. The
# metric terms are computed with the divergence operators (discrete metric identities hold exactly). Material parameters
# are point values at the nodes (averaged over the node's cell: ρ arithmetic, μ and λ + 2μ harmonic). RK4.
#
#   julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/curvi.jl <case> [dx m] [T s] [flattening depth m | -]
#   (environment: FAMILY = MattssonNiemeläWinters2026 | Mattsson2017 | WilliamsDuru2024, ORDER = 7)
using SummationByPartsOperators, LinearAlgebra, Printf
using SeismicWavePC, GeophysicalModelGenerator

const VP, VS, RHO = 6000.0, 3464.0, 2700.0                    # the half-space
const f0 = 1.5 * 3464.0 / (16 * 500.0)
ricker(t) = (a = π * f0 * (t - 1.5 / f0); (1 - 2a^2) * exp(-a^2))

# the cases (metres): tilt<θ> as geometryCases.jl; hill (3 km, σ = 5 km) and steephill (3 km, σ = 2 km)
istilt(case) = startswith(case, "tilt")
θ(case) = istilt(case) ? parse(Float64, replace(case, "tilt" => "")) : 0.0
frame(case) = (s = sind(θ(case)); c = cosd(θ(case)); (t=(c, s), n=(-s, c)))
to_grid(case, a, b) = (f = frame(case); (a * f.t[1] + b * f.n[1], a * f.t[2] + b * f.n[2]))
surface(case) = case == "hill" ? (x -> 3e3 * exp(-x^2 / (2 * 5e3^2))) : case == "steephill" ? (x -> 3e3 * exp(-x^2 / (2 * 2e3^2))) :
                (x -> x * tand(θ(case)))
const XR = (-30e3, -20e3, -10e3, 0.0, 10e3)
receivers(case) = istilt(case) ?
    [(line, i, to_grid(case, x, line === :deep ? -10e3 : -1e3)) for line in (:deep, :shallow) for (i, x) in enumerate(XR)] :
    [(line, i, (x, line === :deep ? -10e3 : surface(case)(x) - 1e3)) for line in (:deep, :shallow) for (i, x) in enumerate(XR)]
source(case) = istilt(case) ? to_grid(case, -10e3, -20e3) : (-10e3, -20e3)
"the forces of the benchmark: Gaussian weights (σ = 500 m, cut at 1 km) on the nodes of the 500 m grid"
function force_nodes(case)
    s = source(case)
    p = [(x, z) for x in (floor(s[1] / 500 - 3) * 500):500:(s[1] + 1500) for z in (floor(s[2] / 500 - 3) * 500):500:(s[2] + 1500)
         if hypot(x - s[1], z - s[2]) <= 1000]
    w = [exp(-hypot(q[1] - s[1], q[2] - s[2])^2 / (2 * 500.0^2)) for q in p]
    zip(p, w ./ sum(w))
end
"(vp, vs, ρ) at (x, z): the half-space, or a layer of the given thickness under the surface (normal depth for the tilted
planes, vertical depth for the hills)"
function material(case, layer, x, z)
    layer === nothing && return (VP, VS, RHO)
    d = (surface(case)(x) - z) * (istilt(case) ? cosd(θ(case)) : 1.0)
    d < layer.thickness ? (layer.vp, layer.vs, layer.rho) : (VP, VS, RHO)
end

# Kaiser-windowed sinc (band-limited delta, as SeismicWavePC's :sinc) and Lagrange weights
besseli0(x) = Float64(sum((x / 2)^(2k) / factorial(big(k))^2 for k in 0:30))
const I0B = besseli0(6.31)
kaiser_sinc(x) = abs(x) >= 4 ? 0.0 : sinc(x) * besseli0(6.31 * sqrt(1 - (x / 4)^2)) / I0B
function sincw(g, x0)                                  # (uniform spacing around x0: sources are far from the boundaries)
    i = clamp(searchsortedlast(g, x0), 1, length(g) - 1)
    c = i + (x0 - g[i]) / (g[i+1] - g[i])
    idx = max(1, ceil(Int, c - 4)):min(length(g), floor(Int, c + 4))
    idx, [kaiser_sinc(c - j) for j in idx]
end
function lagrange(g, x0; n=6)                          # (any grid, e.g. boundary-optimized)
    i0 = clamp(searchsortedlast(g, x0) - n ÷ 2 + 1, 1, length(g) - n + 1)
    idx = i0:i0+n-1
    idx, [prod((x0 - g[m]) / (g[j] - g[m]) for m in idx if m != j) for j in idx]
end

"the flattening weight: 1 down to η = -D0, then smoothly to 0 at η = -D (w = 1 everywhere without D)"
wflat(η, D, D0=0.0) = D === nothing ? 1.0 : (s = clamp((η + D) / (D - D0), 0.0, 1.0); s^2 * (3 - 2s))

"""
    run_case(case; dx, T, dt, order, family, flatten, flatten_from, interfaces, layer)

Blocks stacked in η: the block boundaries `interfaces` (η values, e.g. material interfaces that follow the surface) are
grid lines, the blocks are coupled by SAT (continuity of velocity and traction, energy-neutral: half of the traction
mismatch on the momentum, half of the velocity mismatch on the stress of each side). Each block has at least 13 points
in η (the boundary closures); the time step follows from the finest block. The material is sampled inside each block only (no averaging across interfaces).
"""
function run_case(case; dx=500.0, T=9.0, dt=nothing, order=7, family=MattssonNiemeläWinters2026, flatten=nothing,
                  flatten_from=0.0, interfaces=Float64[], layer=nothing)
    h = surface(case)
    s = source(case)
    ξ0, ξ1, η0 = s[1] - 62e3, s[1] + 62e3, -80e3
    Nx = round(Int, (ξ1 - ξ0) / dx) + 1
    Uξ = upwind_operators(family, derivative_order=1, accuracy_order=order, xmin=ξ0, xmax=ξ1, N=Nx)
    Ddξ, Dgξ = Uξ.minus, Uξ.plus                                        # divergence, gradient
    gξ, Hξ = grid(Ddξ), diag(mass_matrix(Ddξ))
    zmap(ξ, η) = η + h(ξ) * wflat(η, flatten, flatten_from)
    "η of the physical point (x, z) (bisection; z increases with η)"
    function eta(x, z)
        a, b = η0 - 50e3, 10e3
        for _ in 1:80
            m = (a + b) / 2
            zmap(x, m) < z ? (a = m) : (b = m)
        end
        (a + b) / 2
    end
    edges = [η0; sort(interfaces); 0.0]                                 # block boundaries, bottom to top
    off = (-0.375, -0.125, 0.125, 0.375)
    blocks = map(1:length(edges)-1) do b
        lo, hi = edges[b], edges[b+1]
        Nz = max(13, round(Int, (hi - lo) / dx) + 1)
        Uη = upwind_operators(family, derivative_order=1, accuracy_order=order, xmin=lo, xmax=hi, N=Nz)
        gη = grid(Uη.minus)
        dξ!(out, A, D) = Threads.@threads for k in 1:Nz
            mul!(view(out, :, k), D, view(A, :, k))
        end
        dη!(out, A, D) = Threads.@threads for i in 1:Nx
            mul!(view(out, i, :), D, view(A, i, :))
        end
        # metric: x = ξ, J = z_η, Jξ = (z_η, 0), Jη = (-z_ξ, 1), with the divergence operators
        Zm = [zmap(ξ, η) for ξ in gξ, η in gη]
        J, Jηx = similar(Zm), similar(Zm)
        dη!(J, Zm, Uη.minus)
        dξ!(Jηx, Zm, Ddξ)
        Jηx .*= -1
        minimum(J) > 0 || error("the mapping folds (min J = $(minimum(J)))")
        # material: averaged over the cell of each node (4 × 4 samples), inside the block only
        hη = (hi - lo) / (Nz - 1)
        ρ, lam, mu = (zeros(Nx, Nz) for _ in 1:3)
        Threads.@threads for k in 1:Nz
            for i in 1:Nx
                sρ, sμ, sM = 0.0, 0.0, 0.0
                for a in off, c in off
                    ξ = clamp(gξ[i] + a * dx, ξ0, ξ1)
                    η = clamp(gη[k] + c * hη, lo + 1e-6, hi - 1e-6)
                    vp_, vs_, r_ = material(case, layer, ξ, zmap(ξ, η))
                    sρ += r_
                    sμ += 1 / (r_ * vs_^2)
                    sM += 1 / (r_ * vp_^2)
                end
                ρ[i, k] = sρ / 16
                mu[i, k] = 16 / sμ
                lam[i, k] = 16 / sM - 2mu[i, k]
            end
        end
        (; lo, hi, Nz, gη, Hη=diag(mass_matrix(Uη.minus)), Ddη=Uη.minus, Dgη=Uη.plus, dξ!, dη!, J, Jηx, ρ, lam, mu,
           M=lam .+ 2 .* mu, ρJ=ρ .* J, W=Tuple(zeros(Nx, Nz) for _ in 1:5), F=Dict{Tuple{Int,Int},Float64}())
    end
    nb = length(blocks)
    # time step: 0.25 h / vp (RK4 limit ≈ 0.43 h / vp for these operators), h the finer of the spacings of a block
    dt = something(dt, minimum(B -> 0.25 * min(dx, (B.hi - B.lo) / (B.Nz - 1)) / maximum(sqrt.(B.M ./ B.ρ)), blocks))
    inblock(η) = clamp(searchsortedlast(edges, η), 1, nb)
    # the forces: band-limited deltas in (ξ, η) divided by the SBP norm (J f = F δ(ξ) δ(η)); sources are far from interfaces
    n = frame(case).n
    for (p, w) in force_nodes(case)
        ηp = eta(p[1], p[2])
        B = blocks[inblock(ηp)]
        iξ, wξ = sincw(gξ, p[1])
        iη, wη = sincw(B.gη, ηp)
        for (a, i) in enumerate(iξ), (b, k) in enumerate(iη)
            B.F[(i, k)] = get(B.F, (i, k), 0.0) + 1e10 * w * wξ[a] * wη[b] / (Hξ[i] * B.Hη[k])
        end
    end
    Fpts = [collect(B.F) for B in blocks]
    newstate() = [Tuple(zeros(Nx, B.Nz) for _ in 1:5) for B in blocks]
    U, Utmp = newstate(), newstate()
    K = [newstate() for _ in 1:4]
    function rhs!(dU, U, t)
        r = ricker(t)
        # momentum: ρ J ∂t v_i = Dξ(Jξ_j σ_ij) + Dη(Jη_j σ_ij) (+ SAT, force)
        for (b, B) in enumerate(blocks)
            v1, v2, s11, s22, s12 = U[b]
            dv1, dv2 = dU[b]
            A, Gη1, C, Gη2, G = B.W
            J, Jηx, ρJ, Hη, Nz = B.J, B.Jηx, B.ρJ, B.Hη, B.Nz
            @. A = J * s11                 # Jξ_j σ_1j
            @. Gη1 = Jηx * s11 + s12       # Jη_j σ_1j (the traction across η = const, scaled)
            B.dξ!(dv1, A, Ddξ); B.dη!(C, Gη1, B.Ddη)
            @. dv1 = (dv1 + C) / ρJ
            @. G = J * s12
            @. Gη2 = Jηx * s12 + s22
            B.dξ!(dv2, G, Ddξ); B.dη!(C, Gη2, B.Ddη)
            @. dv2 = (dv2 + C) / ρJ
            # SAT at the outer boundaries: the traction removed (free surface on top; free, far away, elsewhere)
            @views begin
                @. dv1[1, :] += A[1, :] / (Hξ[1] * ρJ[1, :]);     @. dv2[1, :] += G[1, :] / (Hξ[1] * ρJ[1, :])
                @. dv1[Nx, :] -= A[Nx, :] / (Hξ[Nx] * ρJ[Nx, :]); @. dv2[Nx, :] -= G[Nx, :] / (Hξ[Nx] * ρJ[Nx, :])
                if b == nb
                    @. dv1[:, Nz] -= Gη1[:, Nz] / (Hη[Nz] * ρJ[:, Nz]); @. dv2[:, Nz] -= Gη2[:, Nz] / (Hη[Nz] * ρJ[:, Nz])
                end
                if b == 1
                    @. dv1[:, 1] += Gη1[:, 1] / (Hη[1] * ρJ[:, 1]); @. dv2[:, 1] += Gη2[:, 1] / (Hη[1] * ρJ[:, 1])
                end
            end
            for ((i, k), a) in Fpts[b]
                dv1[i, k] += a * r * n[1] / ρJ[i, k]
                dv2[i, k] += a * r * n[2] / ρJ[i, k]
            end
        end
        # interfaces (L below, its top row N; R above, its bottom row 1): traction mismatch on the momentum
        for b in 1:nb-1
            L, R = blocks[b], blocks[b+1]
            N = L.Nz
            @views for (c, Gi) in ((1, 2), (2, 4))
                ΔG = L.W[Gi][:, N] .- R.W[Gi][:, 1]
                dU[b][c][:, N] .-= ΔG ./ (2 .* L.Hη[N] .* L.ρJ[:, N])
                dU[b+1][c][:, 1] .-= ΔG ./ (2 .* R.Hη[1] .* R.ρJ[:, 1])
            end
        end
        # the gradients Dξ v, Dη v; velocity mismatch at the interfaces added to Dη v
        for (b, B) in enumerate(blocks)
            v1, v2 = U[b]
            A, Bη, C, E = B.W
            B.dξ!(A, v1, Dgξ); B.dη!(Bη, v1, B.Dgη); B.dξ!(C, v2, Dgξ); B.dη!(E, v2, B.Dgη)
        end
        for b in 1:nb-1
            L, R = blocks[b], blocks[b+1]
            N = L.Nz
            @views for (c, Wi) in ((1, 2), (2, 4))
                Δv = U[b][c][:, N] .- U[b+1][c][:, 1]
                L.W[Wi][:, N] .-= Δv ./ (2 * L.Hη[N])
                R.W[Wi][:, 1] .-= Δv ./ (2 * R.Hη[1])
            end
        end
        # stress: ∂x = Dξ + (Jηx / J) Dη, ∂z = Dη / J
        for (b, B) in enumerate(blocks)
            _, _, ds11, ds22, ds12 = dU[b]
            A, Bη, C, E = B.W
            J, Jηx, M, lam, mu = B.J, B.Jηx, B.M, B.lam, B.mu
            @. ds11 = M * (A + Jηx / J * Bη) + lam * E / J
            @. ds22 = lam * (A + Jηx / J * Bη) + M * E / J
            @. ds12 = mu * (Bη / J + C + Jηx / J * E)
        end
        dU
    end
    # receivers: Lagrange interpolation in (ξ, η) in their block
    rec = receivers(case)
    RW = map(rec) do (_, _, p)
        ηp = eta(p[1], p[2])
        b = inblock(ηp)
        b, (lagrange(gξ, p[1]), lagrange(blocks[b].gη, ηp))
    end
    sample(A, ((iξ, wξ), (iη, wη))) = sum(wξ[a] * wη[c] * A[i, k] for (a, i) in enumerate(iξ), (c, k) in enumerate(iη))
    nt = round(Int, T / dt)
    vrec = zeros(nt + 1, 2, length(rec))
    energy = zeros(nt + 1)
    S = [(Hξ * B.Hη') .* B.J for B in blocks]
    axpy(X, Y, a, Z) = foreach((x, y, z) -> foreach((xx, yy, zz) -> (@. xx = yy + a * zz), x, y, z), X, Y, Z)
    t0 = time()
    for it in 1:nt
        t = (it - 1) * dt
        rhs!(K[1], U, t)
        axpy(Utmp, U, dt / 2, K[1]); rhs!(K[2], Utmp, t + dt / 2)
        axpy(Utmp, U, dt / 2, K[2]); rhs!(K[3], Utmp, t + dt / 2)
        axpy(Utmp, U, dt, K[3]); rhs!(K[4], Utmp, t + dt)
        for b in 1:nb, c in 1:5
            @. U[b][c] += dt / 6 * (K[1][b][c] + 2K[2][b][c] + 2K[3][b][c] + K[4][b][c])
        end
        for (m, (b, w)) in enumerate(RW)
            vrec[it+1, 1, m] = sample(U[b][1], w)
            vrec[it+1, 2, m] = sample(U[b][2], w)
        end
        energy[it+1] = sum(enumerate(blocks)) do (b, B)
            v1, v2, s11, s22, s12 = U[b]
            sum(S[b] .* (B.ρ ./ 2 .* (v1 .^ 2 .+ v2 .^ 2) .+ (B.M .* (s11 .^ 2 .+ s22 .^ 2) .- 2 .* B.lam .* s11 .* s22) ./
                                                        (8 .* B.mu .* (B.lam .+ B.mu)) .+ s12 .^ 2 ./ (2 .* B.mu)))
        end
        isfinite(energy[it+1]) || error("blew up at t = $(it * dt)")
    end
    wall = time() - t0
    # displacement (trapezoid) on the 0.01 s axis, rotated to (u_t, u_n)
    tt = (0:nt) .* dt
    f = frame(case)
    tax = 0.01:0.01:min(T, 9.0)
    at(u) = [(j = min(searchsortedlast(tt, τ), nt); u[j] + (u[j+1] - u[j]) * (τ - tt[j]) / dt) for τ in tax]
    traces = Dict{Tuple{Symbol,Int},NTuple{2,Vector{Float64}}}()
    for (m, (line, i, _)) in enumerate(rec)
        ux = [0.0; cumsum((vrec[1:end-1, 1, m] .+ vrec[2:end, 1, m]) ./ 2 .* dt)]
        uz = [0.0; cumsum((vrec[1:end-1, 2, m] .+ vrec[2:end, 2, m]) ./ 2 .* dt)]
        traces[(line, i)] = (at(ux .* f.t[1] .+ uz .* f.t[2]), at(ux .* f.n[1] .+ uz .* f.n[2]))
    end
    (; traces, energy, tt, wall, Nx, Nz=sum(B -> B.Nz, blocks), nt, minJ=minimum(B -> minimum(B.J), blocks),
       maxJ=maximum(B -> maximum(B.J), blocks))
end

# SeismicWavePC on the same case (staircase of cells; free_surface = :openswpc or :corners), on its 0.01 s axis
const TAX = 0.01:0.01:9.0
function fd_run(case, free_surface, dx; layer=nothing)
    h = surface(case)
    s0, W = (-10e3, -20e3), 60e3
    c = istilt(case) ? [to_grid(case, a, b) for a in s0[1] .+ (-W, W), b in (s0[2] - W, 3dx)] :
        [(a, b) for a in s0[1] .+ (-W, W), b in (s0[2] - W, 3e3 + 3dx)]
    ax(d) = (floor(minimum(p[d] for p in c) / dx) * dx + dx / 2):dx:(ceil(maximum(p[d] for p in c) / dx) * dx)
    X, Y, Z = xyz_grid(ax(1) ./ 1e3, 0, ax(2) ./ 1e3)
    P = [material(case, layer, 1e3X[I], 1e3Z[I]) for I in CartesianIndices(X)]
    air = 1e3 .* Z .> h.(1e3 .* X)
    model = CartData(X, Y, Z, (; rho=ifelse.(air, 1.0, getindex.(P, 3)), vp=ifelse.(air, 0.0, getindex.(P, 1) ./ 1e3),
                                 vs=ifelse.(air, 0.0, getindex.(P, 2) ./ 1e3)))
    n = frame(case).n
    sources = [ForceSource(p[1] / 1e3, 0.0, p[2] / 1e3; fx=1e10 * w * n[1] / f0, fz=1e10 * w * n[2] / f0, t0=0.5 / f0, trise=1 / f0)
               for (p, w) in force_nodes(case)]
    rec = receivers(case)
    stations = [Station("$(line)_$i", p[1] / 1e3, 0.0, p[2] / 1e3) for (line, i, p) in rec]
    dt = 0.04dx / 1e3
    r = simulate(model; dt, nt=round(Int, 9.0 / dt), sources, stations, stf=:ricker, interpolation=:sinc, free_surface,
                 T=Float64, verbose=false)
    t, f = r.t .+ dt, frame(case)
    at(u) = [τ >= t[end] ? u[end] : (j = searchsortedlast(t, τ); j == 0 ? 0.0 : u[j] + (u[j+1] - u[j]) * (τ - t[j]) / dt) for τ in TAX]
    Dict((line, i) => (at(r.u[:, 1, k] .* f.t[1] .+ r.u[:, 2, k] .* f.t[2]), at(r.u[:, 1, k] .* f.n[1] .+ r.u[:, 2, k] .* f.n[2]))
         for (k, (line, i, _)) in enumerate(rec))
end

"the references: tilts, the flat case on a 62.5 m grid (exact: surface and layer on cell faces); hills, :corners at 62.5 m"
const REFS = Dict{Any,Any}()
reference(case; layer=nothing) = get!(REFS, (istilt(case) ? "flat" : case, layer)) do
    istilt(case) ? fd_run("tilt0", :openswpc, 62.5; layer) : fd_run(case, :corners, 62.5; layer)
end
err(a, r) = sqrt(sum(abs2, a[1] .- r[1]) + sum(abs2, a[2] .- r[2])) / sqrt(sum(abs2, r[1]) + sum(abs2, r[2]))
errors(tr, ref, line) = (e = [err(tr[(line, i)], ref[(line, i)]) for i in 1:5]; (sum(e) / 5, maximum(e)))

if abspath(PROGRAM_FILE) == @__FILE__
    case = ARGS[1]
    dx = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 500.0
    T = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 9.0
    flatten = length(ARGS) > 3 && ARGS[4] != "-" ? parse(Float64, ARGS[4]) : nothing
    family = getfield(SummationByPartsOperators, Symbol(get(ENV, "FAMILY", "MattssonNiemeläWinters2026")))
    r = run_case(case; dx, T, flatten, family, order=parse(Int, get(ENV, "ORDER", "7")))
    E = r.energy
    k = findfirst(>(6.0), r.tt)                                  # after the source
    @printf("%-9s dx = %4.0f m: %d × %d points, %d steps, %.1f s, J %.2f–%.2f; energy after the source: rel. change %.2e\n",
            case, dx, r.Nx, r.Nz, r.nt, r.wall, r.minJ, r.maxJ, E[end] / E[k] - 1)
    if T >= 9.0
        ref = reference(case)
        for line in (:deep, :shallow)
            @printf("  %-8s mean %5.1f %%  max %5.1f %%\n", line, (100 .* errors(r.traces, ref, line))...)
        end
    end
end
