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

The cases of the benchmark on a mapping z = η + h(ξ) w(η) (see the top), material from `material` averaged over the cell
of each node. Blocks stacked in η: the block boundaries `interfaces` (η values, e.g. material interfaces that follow the
surface) are grid lines. Builds the meshes and runs them with `run_blocks`.
"""
function run_case(case; dx=500.0, T=9.0, order=7, family=MattssonNiemeläWinters2026, flatten=nothing,
                  flatten_from=0.0, interfaces=Float64[], layer=nothing, xlim=nothing, bottom=-80e3, kwargs...)
    h = surface(case)
    s = source(case)
    ξ0, ξ1 = something(xlim, (s[1] - 62e3, s[1] + 62e3))
    η0 = bottom
    Nx = round(Int, (ξ1 - ξ0) / dx) + 1
    gξ = collect(grid(upwind_operators(family, derivative_order=1, accuracy_order=order, xmin=ξ0, xmax=ξ1, N=Nx).minus))
    zmap(ξ, η) = η + h(ξ) * wflat(η, flatten, flatten_from)
    edges = [η0; sort(interfaces); 0.0]                                 # block boundaries, bottom to top
    off = (-0.375, -0.125, 0.125, 0.375)
    meshes = map(1:length(edges)-1) do b
        lo, hi = edges[b], edges[b+1]
        Nz = max(13, round(Int, (hi - lo) / dx) + 1)
        gη = collect(grid(upwind_operators(family, derivative_order=1, accuracy_order=order, xmin=lo, xmax=hi, N=Nz).minus))
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
        (; Z=[zmap(ξ, η) for ξ in gξ, η in gη], ρ, lam, mu)
    end
    run_blocks(gξ, meshes; case, T, order, family, kwargs...)
end

"""
    run_blocks(gξ, meshes; case, T, dt, order, family)

The solver. `meshes` (bottom to top): blocks of a deformed mesh with vertical columns at the horizontal nodes `gξ` (the
grid of the SBP operator in x): `Z` (Nx × Nz, physical depth of the nodes, increasing with the index; the nodes are those
of the SBP operator on η̂ ∈ [0, 1]) and the material `ρ`, `lam`, `mu` at the nodes (SI). Adjacent blocks share their
interface (the top row of one is the bottom row of the next); they are coupled by SAT (continuity of velocity and
traction, energy-neutral: half of the traction mismatch on the momentum, half of the velocity mismatch on the stress of
each side). Metric terms from `Z` with the divergence operators. Each block needs ≥ 13 points in η; the time step follows
from the finest block. The top of the top block is the free surface.
"""
function run_blocks(gξ, meshes; case, T=9.0, dt=nothing, cfl=:auto, order=7, family=MattssonNiemeläWinters2026,
                    integrator=:rk4, absorbing=:none, nsponge=20, FT=Float64, tmax=9.0)
    Nx = length(gξ)
    Uξ = upwind_operators(family, derivative_order=1, accuracy_order=order, xmin=gξ[1], xmax=gξ[end], N=Nx)   # (metric, Float64)
    maximum(abs, grid(Uξ.minus) .- gξ) < 1e-6 * (gξ[end] - gξ[1]) || error("x is not the grid of the SBP operator")
    UξT = upwind_operators(family, derivative_order=1, accuracy_order=order, xmin=FT(gξ[1]), xmax=FT(gξ[end]), N=Nx)
    Ddξ, Dgξ = UξT.minus, UξT.plus                                      # divergence, gradient (in FT)
    Hξ = FT.(diag(mass_matrix(Uξ.minus)))
    dx = gξ[Nx÷2+1] - gξ[Nx÷2]
    for b in 1:length(meshes)-1
        maximum(abs, meshes[b].Z[:, end] .- meshes[b+1].Z[:, 1]) < 1e-6 * dx || error("blocks $b and $(b+1) do not share their interface")
    end
    blocks = map(meshes) do m
        Nz = size(m.Z, 2)
        Uη = upwind_operators(family, derivative_order=1, accuracy_order=order, xmin=0.0, xmax=1.0, N=Nz)
        UηT = upwind_operators(family, derivative_order=1, accuracy_order=order, xmin=zero(FT), xmax=one(FT), N=Nz)
        gη = collect(grid(Uη.minus))
        dξ!(out, A, D) = Threads.@threads for k in 1:Nz
            mul!(view(out, :, k), D, view(A, :, k))
        end
        dη!(out, A, D) = Threads.@threads for i in 1:Nx
            mul!(view(out, i, :), D, view(A, i, :))
        end
        # metric: x = ξ, J = z_η, Jξ = (z_η, 0), Jη = (-z_ξ, 1), with the divergence operators (in Float64)
        J, Jηx = similar(m.Z), similar(m.Z)
        dη!(J, m.Z, Uη.minus)
        dξ!(Jηx, m.Z, Uξ.minus)
        Jηx .*= -1
        minimum(J) > 0 || error("the mesh folds (min J = $(minimum(J)))")
        ρ, lam, mu = FT.(m.ρ), FT.(m.lam), FT.(m.mu)
        (; Nz, gη, Z=m.Z, Hη=FT.(diag(mass_matrix(Uη.minus))), Ddη=UηT.minus, Dgη=UηT.plus, dξ!, dη!, J=FT.(J), Jηx=FT.(Jηx),
           ρ, lam, mu, M=lam .+ 2 .* mu, ρJ=ρ .* FT.(J), W=Tuple(zeros(FT, Nx, Nz) for _ in 1:5), F=Dict{Tuple{Int,Int},Float64}())
    end
    nb = length(blocks)
    # time step: cfl h / vp (h the finer of the spacings of a block), or (cfl = :auto) from the spectral radius below
    cfl isa Number && (dt = something(dt, minimum(B -> cfl * min(dx, minimum(B.J) / (B.Nz - 1)) / maximum(sqrt.(B.M ./ B.ρ)), blocks)))
    # absorbing boundaries (sides and bottom): impedance SAT (σ·n = -Z v, Z = ρ vp normal, ρ vs tangential; dissipates
    # energy) and/or a Cerjan sponge (fields × exp(-(0.3 (n - d) / n)²) within n points of the boundary, every step)
    sat_abs = absorbing in (:sat, :both)
    sponge = map(enumerate(blocks)) do (b, B)
        f = ones(FT, Nx, B.Nz)
        absorbing in (:sponge, :both) || return f
        g(d) = d < nsponge ? exp(-(0.3 * (nsponge - d) / nsponge)^2) : 1.0
        for k in 1:B.Nz, i in 1:Nx
            f[i, k] = g(i - 1) * g(Nx - i) * (b == 1 ? g(k - 1) : 1.0)
        end
        f
    end
    VP = [sqrt.(B.M ./ B.ρ) for B in blocks]
    VS = [sqrt.(B.mu ./ B.ρ) for B in blocks]
    # the impedance SAT is a damping -a v at the boundary nodes, a = |J·| Z / (H ρ J) (normal: vp, tangential: vs); it is
    # stiff (a dt > 2 at corners), so leapfrog treats it implicitly (trapezoidal: unconditionally stable for damping)
    AD = map(enumerate(blocks)) do (b, B)
        a1, a2 = zeros(FT, Nx, B.Nz), zeros(FT, Nx, B.Nz)
        if sat_abs
            for i in (1, Nx)
                @. a1[i, :] += VP[b][i, :] / Hξ[i]; @. a2[i, :] += VS[b][i, :] / Hξ[i]
            end
            if b == 1
                @. a1[:, 1] += sqrt(B.Jηx[:, 1]^2 + 1) * VS[b][:, 1] / (B.Hη[1] * B.J[:, 1])
                @. a2[:, 1] += sqrt(B.Jηx[:, 1]^2 + 1) * VP[b][:, 1] / (B.Hη[1] * B.J[:, 1])
            end
        end
        (a1, a2)
    end
    "block and η̂ of the physical point (x, z): Lagrange interpolation of the mesh columns, bisection in η̂"
    function locate(x, z)
        iξ, wξ = lagrange(gξ, x)
        for (b, B) in enumerate(blocks)
            col = [sum(wξ[a] * B.Z[i, k] for (a, i) in enumerate(iξ)) for k in 1:B.Nz]
            (col[1] <= z <= col[end] || (b == nb && z > col[end])) || continue
            zof(η) = let (ik, wk) = lagrange(B.gη, η); sum(wk[c] * col[k] for (c, k) in enumerate(ik)) end
            lo, hi = 0.0, 1.0
            for _ in 1:60
                mid = (lo + hi) / 2
                zof(mid) < z ? (lo = mid) : (hi = mid)
            end
            return b, (lo + hi) / 2
        end
        error("($x, $z) is outside the mesh")
    end
    # the forces: band-limited deltas in (ξ, η) divided by the SBP norm (J f = F δ(ξ) δ(η)); sources are far from interfaces
    n = frame(case).n
    for (p, w) in force_nodes(case)
        b, ηp = locate(p[1], p[2])
        B = blocks[b]
        iξ, wξ = sincw(gξ, p[1])
        iη, wη = sincw(B.gη, ηp)
        for (a, i) in enumerate(iξ), (c, k) in enumerate(iη)
            B.F[(i, k)] = get(B.F, (i, k), 0.0) + 1e10 * w * wξ[a] * wη[c] / (Hξ[i] * B.Hη[k])
        end
    end
    Fpts = [collect(B.F) for B in blocks]
    newstate() = [Tuple(zeros(FT, Nx, B.Nz) for _ in 1:5) for B in blocks]
    U, Utmp = newstate(), newstate()
    K = [newstate() for _ in 1:4]
    function momentum!(dU, U, t)
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
            if sat_abs && integrator === :rk4       # impedance damping (explicit; implicit for leapfrog, see the time loop)
                @. dv1 -= AD[b][1] * U[b][1]
                @. dv2 -= AD[b][2] * U[b][2]
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
        dU
    end
    function stress!(dU, U)
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
    rhs!(dU, U, t) = (momentum!(dU, U, t); stress!(dU, U))
    if dt === nothing
        # power iteration for the largest eigenvalue |λ| of v -> M(S(v)) (stress from velocity, then momentum; with SAT,
        # metric and interfaces): leapfrog is stable for dt < 2 / √|λ|, RK4 for dt < 2.83 / √|λ|; 90 % of that
        X, Y = newstate(), newstate()
        foreach(b -> (X[b][1] .= randn.(FT); X[b][2] .= randn.(FT)), 1:nb)
        λ = 0.0
        for _ in 1:200
            stress!(K[1], X)
            for b in 1:nb, c in 1:5
                Y[b][c] .= c <= 2 ? 0 : K[1][b][c]
            end
            momentum!(K[2], Y, -1e9)                            # (no source)
            nrm = sqrt(sum(b -> sum(abs2, K[2][b][1]) + sum(abs2, K[2][b][2]), 1:nb))
            λ = nrm / sqrt(sum(b -> sum(abs2, X[b][1]) + sum(abs2, X[b][2]), 1:nb))
            for b in 1:nb
                X[b][1] .= K[2][b][1] ./ nrm; X[b][2] .= K[2][b][2] ./ nrm
            end
        end
        dt = 0.9 * (integrator === :leapfrog ? 2.0 : 2.83) / sqrt(λ)
    end
    # receivers: Lagrange interpolation in (ξ, η) in their block
    rec = receivers(case)
    RW = map(rec) do (_, _, p)
        b, ηp = locate(p[1], p[2])
        b, (lagrange(gξ, p[1]), lagrange(blocks[b].gη, ηp))
    end
    sample(A, ((iξ, wξ), (iη, wη))) = sum(wξ[a] * wη[c] * A[i, k] for (a, i) in enumerate(iξ), (c, k) in enumerate(iη))
    nt = round(Int, T / dt)
    vrec = zeros(nt + 1, 2, length(rec))
    energy = zeros(nt + 1)
    S = [(Hξ * B.Hη') .* B.J for B in blocks]
    axpy(X, Y, a, Z) = foreach((x, y, z) -> foreach((xx, yy, zz) -> (@. xx = yy + a * zz), x, y, z), X, Y, Z)
    Esum(b, B, w1, w2, s11, s22, s12) = sum(S[b] .* (B.ρ ./ 2 .* w1 .+ B.ρ ./ 2 .* w2 .+ (B.M .* (s11 .^ 2 .+ s22 .^ 2) .-
                                     2 .* B.lam .* s11 .* s22) ./ (8 .* B.mu .* (B.lam .+ B.mu)) .+ s12 .^ 2 ./ (2 .* B.mu)))
    vold = [(zeros(FT, Nx, B.Nz), zeros(FT, Nx, B.Nz)) for B in blocks]
    t0 = time()
    for it in 1:nt
        t = (it - 1) * dt
        if integrator === :rk4
            rhs!(K[1], U, t)
            axpy(Utmp, U, dt / 2, K[1]); rhs!(K[2], Utmp, t + dt / 2)
            axpy(Utmp, U, dt / 2, K[2]); rhs!(K[3], Utmp, t + dt / 2)
            axpy(Utmp, U, dt, K[3]); rhs!(K[4], Utmp, t + dt)
            for b in 1:nb, c in 1:5
                @. U[b][c] += dt / 6 * (K[1][b][c] + 2K[2][b][c] + 2K[3][b][c] + K[4][b][c])
            end
        else                        # leapfrog: v^{n+1/2} = v^{n-1/2} + dt M(σ^n, t_n), σ^{n+1} = σ^n + dt S(v^{n+1/2})
            for b in 1:nb
                vold[b][1] .= U[b][1]; vold[b][2] .= U[b][2]
            end
            momentum!(K[1], U, t)
            for b in 1:nb, c in 1:2
                a = AD[b][c]
                @. U[b][c] = ((1 - a * dt / 2) * U[b][c] + dt * K[1][b][c]) / (1 + a * dt / 2)
            end
        end
        if integrator === :leapfrog  # the energy conserved by leapfrog: ½ρ v^{n-1/2}·v^{n+1/2} + strain energy of σ^n
            energy[it+1] = sum(b -> Esum(b, blocks[b], vold[b][1] .* U[b][1], vold[b][2] .* U[b][2], U[b][3], U[b][4], U[b][5]), 1:nb)
            stress!(K[1], U)
            for b in 1:nb, c in 3:5
                @. U[b][c] += dt * K[1][b][c]
            end
        else
            energy[it+1] = sum(b -> Esum(b, blocks[b], U[b][1] .^ 2, U[b][2] .^ 2, U[b][3], U[b][4], U[b][5]), 1:nb)
        end
        if absorbing in (:sponge, :both)
            for b in 1:nb, c in 1:5
                U[b][c] .*= sponge[b]
            end
        end
        for (m, (b, w)) in enumerate(RW)
            vrec[it+1, 1, m] = sample(U[b][1], w)
            vrec[it+1, 2, m] = sample(U[b][2], w)
        end
        isfinite(energy[it+1]) || error("blew up at t = $(it * dt)")
    end
    wall = time() - t0
    # displacement (trapezoid) on the 0.01 s axis, rotated to (u_t, u_n)
    tt = (0:nt) .* dt
    f = frame(case)
    tax = 0.01:0.01:min(T, tmax)
    at(u) = [(j = min(searchsortedlast(tt, τ), nt); u[j] + (u[j+1] - u[j]) * (τ - tt[j]) / dt) for τ in tax]
    traces = Dict{Tuple{Symbol,Int},NTuple{2,Vector{Float64}}}()
    for (m, (line, i, _)) in enumerate(rec)
        if integrator === :rk4
            ux = [0.0; cumsum((vrec[1:end-1, 1, m] .+ vrec[2:end, 1, m]) ./ 2 .* dt)]
            uz = [0.0; cumsum((vrec[1:end-1, 2, m] .+ vrec[2:end, 2, m]) ./ 2 .* dt)]
        else                                                    # v^{n+1/2} recorded at step n + 1: midpoint rule
            ux = [0.0; cumsum(vrec[2:end, 1, m] .* dt)]
            uz = [0.0; cumsum(vrec[2:end, 2, m] .* dt)]
        end
        traces[(line, i)] = (at(ux .* f.t[1] .+ uz .* f.t[2]), at(ux .* f.n[1] .+ uz .* f.n[2]))
    end
    (; traces, energy, tt, wall, dt, Nx, Nz=sum(B -> B.Nz, blocks), nt, minJ=minimum(B -> minimum(B.J), blocks),
       maxJ=maximum(B -> maximum(B.J), blocks))
end

"""
    mesh = curvilinear_mesh(x, surfaces; dz, bottom, order=7, family=MattssonNiemeläWinters2026)

A plain deformed 2D `CartData` (size nx × 1 × nz, km) for the curvilinear solver: one grid line follows each surface.
`surfaces` are functions z(x) [km], top first: the topography, then interfaces (e.g. a Moho); the bottom is flat at
`bottom` [km]. Between two surfaces z is blended linearly. Columns are vertical, at the nodes of the SBP operator in x over
`first(x)`…`last(x)` with `length(x)` points; between two surfaces the nodes in z are those of the operator on [0, 1],
at least 13, with about `dz` spacing at the thickest point. Fill the material with the usual GMG tools, e.g. layers with
`Z .< moho.(X)` and a magma chamber with `add_ellipsoid!`.
"""
function curvilinear_mesh(x, surfaces; dz, bottom, order=7, family=MattssonNiemeläWinters2026)
    Nx = length(x)
    gx = grid(upwind_operators(family, derivative_order=1, accuracy_order=order, xmin=first(x), xmax=last(x), N=Nx).minus)
    S = [[f.(gx) for f in surfaces]; [fill(Float64(bottom), Nx)]]       # top to bottom
    Zs = Matrix{Float64}[]
    for b in length(S)-1:-1:1                                           # bottom first (z increases with the index)
        top, bot = S[b], S[b+1]
        all(top .> bot) || error("surfaces $b and $(b+1) cross")
        Nz = max(13, ceil(Int, maximum(top .- bot) / dz) + 1)
        η = grid(upwind_operators(family, derivative_order=1, accuracy_order=order, xmin=0.0, xmax=1.0, N=Nz).minus)
        Zb = [bot[i] + (top[i] - bot[i]) * e for i in 1:Nx, e in η]
        Zb[:, end] .= top                                               # (exactly on the surface)
        push!(Zs, isempty(Zs) ? Zb : Zb[:, 2:end])                      # the interface row once
    end
    Z2 = reduce(hcat, Zs)
    nz = size(Z2, 2)
    X = [gx[i] for i in 1:Nx, _ in 1:1, _ in 1:nz]
    Z = reshape(Z2, Nx, 1, nz)
    CartData(X, zero(X), Z, (; z=copy(Z)))
end

"""
    blocks = find_blocks(Z, vp, vs; jump=0.05)

Interfaces of a deformed mesh (nx × nz, z increasing with the index) from its material: a row r is an interface if vp or
vs jumps by more than `jump` (relative) along the full width, in every column between r - 1 and r or between r and r + 1.
Jumps over part of the width (a magma chamber) and smooth gradients are no interfaces. If all columns jump between the
same two rows, the mesh decides: the grid line that follows the interface separates two differently blended regions, so
the shape of the vertical spacing (along x) changes there. If it does not (e.g. a layer parallel to the surface), the
interface is the upper row (GMG: a node exactly on a surface is not below it). Returns the interface rows, and for every
interface whether each column's interface node has the material below (true) or above (false).
"""
function find_blocks(Z, vp, vs; jump=0.05)
    rel(a, b) = abs(a - b) / max(abs(a), abs(b))
    nx, nz = size(vp)
    J = [rel(vp[i, k], vp[i, k+1]) > jump || rel(vs[i, k], vs[i, k+1]) > jump for i in 1:nx, k in 1:nz-1]   # jump between k, k+1
    cand = [r for r in 2:nz-1 if all(i -> J[i, r-1] || J[i, r], 1:nx)]
    # the change of the shape of the vertical spacing across row k
    shape(k) = (d = Z[:, k+1] .- Z[:, k]; d ./ sum(d))
    kink(k) = 2 <= k <= nz - 1 ? maximum(abs, shape(k) .- shape(k - 1)) * nx : 0.0
    rows = Int[]
    for r in cand
        (r in rows || r - 1 in rows) && continue
        if r + 1 in cand && all(i -> J[i, r], 1:nx)              # every column jumps between r and r + 1: which row?
            push!(rows, kink(r) > 3kink(r + 1) ? r : r + 1)
        else
            push!(rows, r)
        end
    end
    rows, [[J[i, r] for i in 1:nx] for r in rows]       # J[i, r]: the jump is above node r, i.e. the node is below
end

"""
    run_cartdata(model; case, jump=0.05, T, dt, order, family)

Runs the curvilinear solver on a deformed 2D `CartData` (as from `curvilinear_mesh`, km) with the fields `rho` [kg/m³],
`vp`, `vs` [km/s] at the nodes. The blocks are found from the material (`find_blocks`): at an interface row each block
takes the material of its own side (the node's own, or that of the next row of its side). Sources and receivers of `case`.
"""
function run_cartdata(model::CartData; case, jump=0.05, kwargs...)
    X, Z = model.x.val, model.z.val
    size(X, 2) == 1 || error("a 2D model (nx × 1 × nz) is needed")
    gξ = 1e3 .* X[:, 1, 1]
    all(k -> X[:, 1, k] == X[:, 1, 1], axes(X, 3)) || error("the columns of the mesh must be vertical")
    Zm = 1e3 .* Z[:, 1, :]
    ρ = model.fields.rho[:, 1, :]
    vp, vs = 1e3 .* model.fields.vp[:, 1, :], 1e3 .* model.fields.vs[:, 1, :]
    rows, below = find_blocks(Zm, vp, vs; jump)
    edges = [1; rows; size(Zm, 2)]
    meshes = map(1:length(edges)-1) do b
        r = edges[b]:edges[b+1]
        mρ, mvp, mvs = ρ[:, r], vp[:, r], vs[:, r]
        for (j, side, src) in ((b - 1, false, 2), (b, true, length(r) - 1))   # (interface, its node must be ..., take row)
            (1 <= j <= length(rows)) || continue
            k = side ? length(r) : 1                                            # the interface row in this block
            for i in axes(mρ, 1)
                below[j][i] == side && continue                                 # the node has this side's material
                mρ[i, k], mvp[i, k], mvs[i, k] = mρ[i, src], mvp[i, src], mvs[i, src]
            end
        end
        (; Z=Zm[:, r], ρ=mρ, mu=mρ .* mvs .^ 2, lam=mρ .* (mvp .^ 2 .- 2 .* mvs .^ 2))
    end
    run_blocks(gξ, meshes; case, kwargs...)
end

# SeismicWavePC on the same case (staircase of cells; free_surface = :openswpc or :corners), on its 0.01 s axis
const TAX = 0.01:0.01:9.0
function fd_run(case, free_surface, dx; layer=nothing, mat=(x, z) -> material(case, layer, x, z), xlim=nothing, bottom=nothing,
                absorber=:pml, T=9.0)
    h = surface(case)
    s0, W = (-10e3, -20e3), 60e3
    c = istilt(case) ? [to_grid(case, a, b) for a in s0[1] .+ (-W, W), b in (s0[2] - W, 3dx)] :
        [(a, b) for a in something(xlim, s0[1] .+ (-W, W)), b in (something(bottom, s0[2] - W), 3e3 + 3dx)]
    ax(d) = (floor(minimum(p[d] for p in c) / dx) * dx + dx / 2):dx:(ceil(maximum(p[d] for p in c) / dx) * dx)
    X, Y, Z = xyz_grid(ax(1) ./ 1e3, 0, ax(2) ./ 1e3)
    P = [mat(1e3X[I], 1e3Z[I]) for I in CartesianIndices(X)]
    air = 1e3 .* Z .> h.(1e3 .* X)
    model = CartData(X, Y, Z, (; rho=ifelse.(air, 1.0, getindex.(P, 3)), vp=ifelse.(air, 0.0, getindex.(P, 1) ./ 1e3),
                                 vs=ifelse.(air, 0.0, getindex.(P, 2) ./ 1e3)))
    n = frame(case).n
    sources = [ForceSource(p[1] / 1e3, 0.0, p[2] / 1e3; fx=1e10 * w * n[1] / f0, fz=1e10 * w * n[2] / f0, t0=0.5 / f0, trise=1 / f0)
               for (p, w) in force_nodes(case)]
    rec = receivers(case)
    stations = [Station("$(line)_$i", p[1] / 1e3, 0.0, p[2] / 1e3) for (line, i, p) in rec]
    dt = 0.04dx / 1e3
    r = simulate(model; dt, nt=round(Int, T / dt), sources, stations, stf=:ricker, interpolation=:sinc, free_surface,
                 absorber, T=Float64, verbose=false)
    t, f = r.t .+ dt, frame(case)
    at(u) = [τ >= t[end] ? u[end] : (j = searchsortedlast(t, τ); j == 0 ? 0.0 : u[j] + (u[j+1] - u[j]) * (τ - t[j]) / dt) for τ in 0.01:0.01:T]
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
