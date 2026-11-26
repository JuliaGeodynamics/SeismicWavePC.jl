# Stage-1 prototype: 2D P-SV on a terrain-following (curvilinear) grid with SBP operators and a SAT free surface, on the
# topography cases of Nobuaki Fuji (tilted planes, hill; as examples/topography_benchmark.jl).
#
# Grid: x = ξ, z = η + h(ξ) (columns sheared with the surface h(x); J = 1), collocated velocity-stress, SBP first-derivative
# operators of SummationByPartsOperators.jl: dual-pairing upwind operators (Mattsson 2017, order 7), RK4.
# Split form: momentum with the divergence in conservative form, stress with the gradient in non-conservative form; with
# SBP operators the two are adjoint in the J-weighted norm, so the semi-discrete energy changes only through boundary
# terms, and the SAT (the traction removed at every boundary) makes all boundaries traction-free: energy conserved.
#
#   julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/curvi.jl <case> [dx m] [T s]
#   (environment variables ORDER = 5 (order of the upwind operators) and SWAP = 1 (D− for the divergence))
using SummationByPartsOperators, LinearAlgebra, Printf

const vp, vs, ρ = 6000.0, 3464.0, 2700.0
const μ = ρ * vs^2
const λ = ρ * vp^2 - 2μ
const f0 = 1.5 * 3464.0 / (16 * 500.0)
ricker(t) = (a = π * f0 * (t - 1.5 / f0); (1 - 2a^2) * exp(-a^2))

# the cases (metres), as geometryCases.jl
θ(case) = case == "hill" ? 0.0 : parse(Float64, replace(case, "tilt" => ""))
frame(case) = (s = sind(θ(case)); c = cosd(θ(case)); (t=(c, s), n=(-s, c)))
to_grid(case, a, b) = (f = frame(case); (a * f.t[1] + b * f.n[1], a * f.t[2] + b * f.n[2]))
surface(case) = case == "hill" ? (x -> 3e3 * exp(-x^2 / (2 * 5e3^2))) : (x -> x * tand(θ(case)))
const XR = (-30e3, -20e3, -10e3, 0.0, 10e3)
receivers(case) = case == "hill" ?
    [(line, i, (x, line === :deep ? -10e3 : surface(case)(x) - 1e3)) for line in (:deep, :shallow) for (i, x) in enumerate(XR)] :
    [(line, i, to_grid(case, x, line === :deep ? -10e3 : -1e3)) for line in (:deep, :shallow) for (i, x) in enumerate(XR)]
source(case) = case == "hill" ? (-10e3, -20e3) : to_grid(case, -10e3, -20e3)

"Kaiser-windowed sinc weights (r = 4, b = 6.31, as SeismicWavePC's :sinc) of the grid points around x0: a band-limited delta"
besseli0(x) = sum((x / 2)^(2k) / factorial(big(k))^2 for k in 0:30) |> Float64
const I0B = besseli0(6.31)
kaiser_sinc(x) = abs(x) >= 4 ? 0.0 : sinc(x) * besseli0(6.31 * sqrt(1 - (x / 4)^2)) / I0B
function sincw(g, x0)
    h = step(g)
    c = (x0 - first(g)) / h + 1
    idx = max(1, ceil(Int, c - 4)):min(length(g), floor(Int, c + 4))
    idx, [kaiser_sinc(c - j) for j in idx]
end

"Lagrange weights of the `n` grid points around x0 (inside the grid)"
function lagrange(g, x0; n=6)
    h = step(g)
    i0 = clamp(floor(Int, (x0 - first(g)) / h) - n ÷ 2 + 2, 1, length(g) - n + 1)
    idx = i0:i0+n-1
    w = [prod((x0 - g[m]) / (g[j] - g[m]) for m in idx if m != j) for j in idx]
    idx, w
end

function run_case(case; dx=500.0, T=9.0, dt=0.04dx / 1e3, order=5)
    h = surface(case)
    s = source(case)
    ξ0, ξ1 = s[1] - 62e3, s[1] + 62e3
    Nx, Nz = round(Int, (ξ1 - ξ0) / dx) + 1, round(Int, 80e3 / dx) + 1
    Dξ = derivative_operator(MattssonNordström2004(), derivative_order=1, accuracy_order=6, xmin=ξ0, xmax=ξ1, N=Nx)
    # dual-pairing (upwind) SBP operators (Mattsson 2017): H D+ + (H D−)ᵀ = B; divergence with D+, gradient with D−
    Uξ = upwind_operators(Mattsson2017, derivative_order=1, accuracy_order=order, xmin=ξ0, xmax=ξ1, N=Nx)
    Uη = upwind_operators(Mattsson2017, derivative_order=1, accuracy_order=order, xmin=-80e3, xmax=0.0, N=Nz)
    gξ, gη = grid(Uξ.minus), grid(Uη.minus)
    Hξ, Hη = diag(mass_matrix(Uξ.minus)), diag(mass_matrix(Uη.minus))
    # metric of x = ξ, z = η + h(ξ): J = 1, Jξ = (1, 0), Jη = (−h', 1); h' with the same SBP operator
    hp = Dξ * h.(collect(gξ))
    Jηx = -hp
    # fields and work arrays
    v1, v2, s11, s22, s12 = (zeros(Nx, Nz) for _ in 1:5)
    U = (v1, v2, s11, s22, s12)
    K = [Tuple(zeros(Nx, Nz) for _ in 1:5) for _ in 1:4]
    Utmp = Tuple(zeros(Nx, Nz) for _ in 1:5)
    W = Tuple(zeros(Nx, Nz) for _ in 1:4)
    dξ!(out, A, D) = Threads.@threads for k in 1:Nz
        mul!(view(out, :, k), D, view(A, :, k))
    end
    dη!(out, A, D) = Threads.@threads for i in 1:Nx
        mul!(view(out, i, :), D, view(A, i, :))
    end
    Dpξ, Dpη, Dmξ, Dmη = get(ENV, "SWAP", "1") == "1" ? (Uξ.minus, Uη.minus, Uξ.plus, Uη.plus) : (Uξ.plus, Uη.plus, Uξ.minus, Uη.minus)
    # the forces: Gaussian weights on the 500 m nodes, band-limited deltas (sinc weights / SBP norm) in (ξ, η)
    n = frame(case).n
    nodes = [(x, z) for x in (floor(s[1] / 500 - 3) * 500):500:(s[1] + 1500) for z in (floor(s[2] / 500 - 3) * 500):500:(s[2] + 1500)
             if hypot(x - s[1], z - s[2]) <= 1000]
    wts = [exp(-hypot(p[1] - s[1], p[2] - s[2])^2 / (2 * 500.0^2)) for p in nodes]
    wts ./= sum(wts)
    F = Dict{Tuple{Int,Int},Float64}()
    for (p, w) in zip(nodes, wts)
        iξ, wξ = sincw(gξ, p[1])
        iη, wη = sincw(gη, p[2] - h(p[1]))
        for (a, i) in enumerate(iξ), (b, k) in enumerate(iη)
            F[(i, k)] = get(F, (i, k), 0.0) + 1e10 * w * wξ[a] * wη[b] / (Hξ[i] * Hη[k])
        end
    end
    Fpts = collect(F)
    function rhs!(dU, U, t)
        v1, v2, s11, s22, s12 = U
        dv1, dv2, ds11, ds22, ds12 = dU
        A, B, C, E = W
        # momentum: ρ J ∂t v_i = Dξ(Jξ_j σ_ij) + Dη(Jη_j σ_ij) (+ SAT, force)
        @. A = s11                     # Jξ_j σ_1j = σ11
        @. B = Jηx * s11 + s12         # Jη_j σ_1j
        dξ!(dv1, A, Dpξ); dη!(C, B, Dpη)
        @. dv1 = (dv1 + C) / ρ
        @. A = s12
        @. E = Jηx * s12 + s22
        dξ!(dv2, A, Dpξ); dη!(C, E, Dpη)
        @. dv2 = (dv2 + C) / ρ
        # SAT: remove the traction at every boundary (free surface on top; free elsewhere too, far away)
        @views begin
            @. dv1[:, Nz] -= B[:, Nz] / (Hη[Nz] * ρ); @. dv2[:, Nz] -= E[:, Nz] / (Hη[Nz] * ρ)
            @. dv1[:, 1] += B[:, 1] / (Hη[1] * ρ);    @. dv2[:, 1] += E[:, 1] / (Hη[1] * ρ)
            @. dv1[1, :] += s11[1, :] / (Hξ[1] * ρ);  @. dv2[1, :] += s12[1, :] / (Hξ[1] * ρ)
            @. dv1[Nx, :] -= s11[Nx, :] / (Hξ[Nx] * ρ); @. dv2[Nx, :] -= s12[Nx, :] / (Hξ[Nx] * ρ)
        end
        r = ricker(t)
        for ((i, k), a) in Fpts
            dv1[i, k] += a * r * n[1] / ρ
            dv2[i, k] += a * r * n[2] / ρ
        end
        # stress: the gradient g_ij = ∂_j v_i = (Jξ_j Dξ v_i + Jη_j Dη v_i) / J
        dξ!(A, v1, Dmξ); dη!(B, v1, Dmη); dξ!(C, v2, Dmξ); dη!(E, v2, Dmη)
        @. ds11 = (λ + 2μ) * (A + Jηx * B) + λ * E
        @. ds22 = λ * (A + Jηx * B) + (λ + 2μ) * E
        @. ds12 = μ * (B + C + Jηx * E)
        dU
    end
    # receivers
    rec = receivers(case)
    RW = map(rec) do (line, i, p)
        lagrange(gξ, p[1]), lagrange(gη, p[2] - h(p[1]))
    end
    sample(A, ((iξ, wξ), (iη, wη))) = sum(wξ[a] * wη[b] * A[i, k] for (a, i) in enumerate(iξ), (b, k) in enumerate(iη))
    nt = round(Int, T / dt)
    vrec = zeros(nt + 1, 2, length(rec))
    energy = zeros(nt + 1)
    S = Hξ * Hη'
    strain_energy(a, b, c) = ((λ + 2μ) * (a^2 + b^2) - 2λ * a * b) / (8μ * (λ + μ)) + c^2 / (2μ)
    t0 = time()
    for it in 1:nt
        t = (it - 1) * dt
        # RK4
        rhs!(K[1], U, t)
        foreach((x, u, k) -> (@. x = u + dt / 2 * k), Utmp, U, K[1]); rhs!(K[2], Utmp, t + dt / 2)
        foreach((x, u, k) -> (@. x = u + dt / 2 * k), Utmp, U, K[2]); rhs!(K[3], Utmp, t + dt / 2)
        foreach((x, u, k) -> (@. x = u + dt * k), Utmp, U, K[3]); rhs!(K[4], Utmp, t + dt)
        foreach((u, a, b, c, d) -> (@. u += dt / 6 * (a + 2b + 2c + d)), U, K[1], K[2], K[3], K[4])
        for (m, w) in enumerate(RW)
            vrec[it+1, 1, m] = sample(v1, w)
            vrec[it+1, 2, m] = sample(v2, w)
        end
        energy[it+1] = sum(S .* (ρ / 2 .* (v1 .^ 2 .+ v2 .^ 2) .+ strain_energy.(s11, s22, s12)))
        isfinite(energy[it+1]) || error("blew up at t = $(it * dt)")
    end
    wall = time() - t0
    # displacement (trapezoid) on the 0.01 s axis, rotated to (u_t, u_n)
    tt = (0:nt) .* dt
    f = frame(case)
    tax = 0.01:0.01:min(T, 9.0)
    at(u) = [(j = min(searchsortedlast(tt, τ), nt); u[j] + (u[j+1] - u[j]) * (τ - tt[j]) / dt) for τ in tax]
    out = Dict{Tuple{Symbol,Int},NTuple{2,Vector{Float64}}}()
    for (m, (line, i, _)) in enumerate(rec)
        ux = [0.0; cumsum((vrec[1:end-1, 1, m] .+ vrec[2:end, 1, m]) ./ 2 .* dt)]
        uz = [0.0; cumsum((vrec[1:end-1, 2, m] .+ vrec[2:end, 2, m]) ./ 2 .* dt)]
        out[(line, i)] = (at(ux .* f.t[1] .+ uz .* f.t[2]), at(ux .* f.n[1] .+ uz .* f.n[2]))
    end
    (; traces=out, energy, tt, wall, Nx, Nz, nt)
end

# the references, with SeismicWavePC (examples/topography_benchmark.jl, in a module: its constants are in km): tilts, the
# flat case on a 62.5 m grid (exact: its surface lies on cell faces); hill, free_surface = :corners on a 62.5 m grid
module TB
include(joinpath(@__DIR__, "..", "..", "examples", "topography_benchmark.jl"))
end
const REFS = Dict{String,Any}()
reference(case) = get!(REFS, case == "hill" ? "hill" : "flat") do
    case == "hill" ? TB.run("hill", :corners, 0.0625) : TB.run("tilt0", :openswpc, 0.0625)
end
err(a, r) = sqrt(sum(abs2, a[1] .- r[1]) + sum(abs2, a[2] .- r[2])) / sqrt(sum(abs2, r[1]) + sum(abs2, r[2]))

if abspath(PROGRAM_FILE) == @__FILE__
    case = ARGS[1]
    dx = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 500.0
    T = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 9.0
    r = run_case(case; dx, T, order=parse(Int, get(ENV, "ORDER", "5")))
    E = r.energy
    k = findfirst(>(6.0), r.tt)                                  # after the source (ricker ends ~4.6 s)
    @printf("%-7s dx = %4.0f m: %d × %d points, %d steps, %.1f s; energy after the source: %.3e -> %.3e (rel. change %.2e)\n",
            case, dx, r.Nx, r.Nz, r.nt, r.wall, E[k], E[end], E[end] / E[k] - 1)
    if T >= 9.0
        ref = reference(case)
        for line in (:deep, :shallow)
            e = [err(r.traces[(line, i)], ref[(line, i)]) for i in 1:5]
            @printf("  %-8s mean %5.1f %%  max %5.1f %%\n", line, 100sum(e) / 5, 100maximum(e))
        end
    end
end
