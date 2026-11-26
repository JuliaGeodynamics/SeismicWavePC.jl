# Rayleigh modes of a 1D profile (depth d down, SI units) with linear finite elements (as RAYLEE, Haney & Tsai 2017): for a
# given ω, (k² K2 + k K1 + K0 − ω² M) x = 0 with u_x = U(d) e^{i(kx−ωt)}, u_z = i W(d) e^{i(kx−ωt)} (z down), x = (U, W) at
# the nodes, rigid at the bottom node. Used by test_rayleigh.jl and examples/firn_rayleigh_benchmark.jl.
using LinearAlgebra

"trapped modes (c < cmax), slowest first: phase velocity c, and U, W at depth drec"
function rayleigh_modes(znodes, lam, mu, rho, f; cmax, drec=0.0)
    n = length(znodes)                       # nodes; the last one (bottom) is fixed
    N = n - 1
    K2, K1, K0, M = (zeros(2N, 2N) for _ in 1:4)
    U(i) = i                                 # dof of U at node i
    W(i) = N + i
    for e in 1:n-1
        h = znodes[e+1] - znodes[e]
        l, m, r = lam[e], mu[e], rho[e]      # constant per element
        NN = h / 6 .* [2 1; 1 2]
        DD = [1 -1; -1 1] ./ h
        C = [-1 1; -1 1] ./ 2                # ∫ N_i N_j'
        for (a, i) in enumerate((e, e + 1)), (b, j) in enumerate((e, e + 1))
            (i > N || j > N) && continue
            K2[U(i), U(j)] += (l + 2m) * NN[a, b]
            K2[W(i), W(j)] += m * NN[a, b]
            K1[U(i), W(j)] += l * C[a, b] - m * C[b, a]
            K1[W(i), U(j)] += l * C[b, a] - m * C[a, b]
            K0[U(i), U(j)] += m * DD[a, b]
            K0[W(i), W(j)] += (l + 2m) * DD[a, b]
            M[U(i), U(j)] += r * NN[a, b]
            M[W(i), W(j)] += r * NN[a, b]
        end
    end
    ω = 2π * f
    F = eigen([zeros(2N, 2N) I; -(K2 \ (K0 - ω^2 * M)) -(K2 \ K1)])
    modes = NamedTuple{(:c, :U, :W),NTuple{3,Float64}}[]
    for (q, k) in enumerate(F.values)
        (isfinite(k) && abs(imag(k)) < 1e-8 * abs(k) && real(k) > ω / cmax) || continue
        x = real.(F.vectors[1:2N, q] ./ F.vectors[argmax(abs.(F.vectors[1:2N, q])), q])
        e = searchsortedlast(znodes, drec)   # U, W at depth drec (linear, as the elements)
        s = (drec - znodes[e]) / (znodes[e+1] - znodes[e])
        at(D) = (1 - s) * x[D(e)] + s * (e + 1 > N ? 0.0 : x[D(e + 1)])
        push!(modes, (; c=ω / real(k), U=at(U), W=at(W)))
    end
    sort!(modes, by=m -> m.c)
end

"""
    rayleigh_modes(p, f; drec=0, hfine=Inf, dfine=0, dinf=1e4)

Trapped modes (c < 0.95 vs(dinf)) of the profile `p(d) -> (; vp, vs, rho)` at frequency `f`. The mesh has 30 elements per
local S wavelength, at most `hfine` down to `dfine` (to resolve fine layering), 5 % growth per element at most, and reaches
5 decay lengths of the slowest-decaying trapped mode. `U/W < 0` is retrograde motion.
"""
function rayleigh_modes(p, f; drec=0.0, hfine=Inf, dfine=0.0, dinf=1e4)
    vinf = p(dinf).vs
    vsmin = 0.8 * p(0.0).vs
    bottom = max(2dfine, 5 * 0.95vinf / (2π * f * sqrt(1 - 0.95^2)))
    z, h = [0.0], 0.0
    while z[end] < bottom
        h = min(max(0.8 * p(z[end]).vs, vsmin) / 30f, z[end] < dfine ? hfine : Inf, h == 0 ? Inf : 1.05h)
        push!(z, z[end] + h)
    end
    P = p.((z[1:end-1] .+ z[2:end]) ./ 2)
    vp, vs, rho = getfield.(P, :vp), getfield.(P, :vs), getfield.(P, :rho)
    rayleigh_modes(z, rho .* (vp .^ 2 .- 2vs .^ 2), rho .* vs .^ 2, rho, f; cmax=0.95vinf, drec)
end
