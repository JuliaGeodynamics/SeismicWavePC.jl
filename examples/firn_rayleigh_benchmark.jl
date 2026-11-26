# Rayleigh waves in firn: the multi-mode Rayleigh waves of Chaput et al. (2022, J. Glaciol. 68, 785–798, Figs. 2, 8, 9)
# with SeismicWavePC.jl against a 1D eigen-solver (test/rayleigh_modes.jl, linear finite elements as RAYLEE).
#
# Three firn profiles: a smooth base model (vs from 300 m/s at the surface to 2010 m/s, after the base model of their
# Fig. 8a), the same with low-pass filtered random layering (their Fig. 8b), and that layering 2 m deeper (their Fig. 9).
# A vertical force (Ricker, 20 Hz) 1.25 m below the surface, 501 surface stations at 100–1100 m offset, 0.5 m grid, 2D P-SV.
# From a slant stack of the stations at each frequency: the phase velocity, H/V and the sense of the particle motion (ZH =
# Z/(1.5 H), > 0 prograde) of every excited mode, at the wavenumber of the mode next to that of the eigen-solver.
#
#   julia -t auto --project=examples examples/firn_rayleigh_benchmark.jl          (about 10 minutes)
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "test", "rayleigh_modes.jl"))

# the profiles, depth d [m] down: v(d) = v∞ − (v∞ − v0) exp(−d/d0), times 1 + 0.15 exp(−d/25 m) n(d) with smooth noise n
expo(d, v0, vinf, d0) = vinf - (vinf - v0) * exp(-max(d, 0) / d0)
base(d) = (vp=expo(d, 500, 4130, 30), vs=expo(d, 300, 2010, 30), rho=expo(d, 400, 880, 20))
const NOISE = let rng = Xoshiro(7), dz = 0.25, s = 1.5 / dz          # 500 m of unit noise, smoothed over 1.5 m, every 0.25 m
    w = exp.(-((-4s:4s) ./ s) .^ 2 ./ 2)
    e = randn(rng, 2000 + length(w))
    x = [sum(w .* e[i:i+length(w)-1]) for i in 1:2000]
    x ./ std(x)
end
noise(d) = (i = clamp(4d + 1, 1, length(NOISE) - 1); j = floor(Int, i); NOISE[j] + (i - j) * (NOISE[j+1] - NOISE[j]))
function layered(d; shift=0.0)
    b, p = base(d), 1 + 0.15 * exp(-d / 25) * noise(max(d - shift, 0))
    (vp=b.vp * p, vs=b.vs * p, rho=b.rho * p)
end
models = (base=base, layered=layered, layered_2m_deeper=(d -> layered(d; shift=2.0)))

dx = 0.5                                                          # [m]
offsets = 100:2:1100
freqs = 5.0:1.0:45.0
"FD seismograms of a profile"
function seismograms(profile)
    x, z = range(-20 + dx / 2, 1150, step=dx), range(-250 + dx / 2, 3dx, step=dx)
    X, Y, Z = xyz_grid(x ./ 1e3, 0, z ./ 1e3)
    P = profile.(-1e3 .* Z)
    air = Z .> 0
    model = CartData(X, Y, Z, (; rho=ifelse.(air, 1.0, getfield.(P, :rho)), vp=ifelse.(air, 0.0, getfield.(P, :vp) ./ 1e3),
                                 vs=ifelse.(air, 0.0, getfield.(P, :vs) ./ 1e3)))
    f0 = 20.0
    source = ForceSource(dx / 2e3, 0.0, -1.25e-3; fz=1e10, trise=1 / f0, t0=0.5 / f0)
    stations = [Station("r$o", (o + dx / 2) / 1e3, 0.0, 0.0; surface=true) for o in offsets]
    simulate(model; dt=5e-5, nt=68000, sources=source, stations, stf=:ricker, record_every=10, na=40, T=Float32)
end

"modes of the eigen-solver (at the depth of the stations: the centre of the top cell) and FD picks"
function analyse(profile, r)
    eig = [rayleigh_modes(profile, f; drec=dx / 2, hfine=0.5, dfine=75.0) for f in freqs]
    xo = collect(offsets .+ 0.0)
    w = sin.(π .* (xo .- xo[1]) ./ (xo[end] - xo[1])) .^ 2                    # Hann taper over the offsets
    cs = 150.0:5:2100
    img = (zeros(length(freqs), length(cs)), zeros(length(freqs), length(cs)))
    picks = [fill((c=NaN, hv=NaN, zh=NaN), length(freqs)) for _ in 1:maximum(length, eig)]
    for (i, f) in enumerate(freqs)
        ω = 2π * f
        x, z = (vec(transpose(cis.(-ω .* r.t)) * r.v[:, a, :]) for a in 1:2)   # x, z (up) at frequency f
        stack(u, k) = sum(w .* u .* cis.(k .* xo))
        for (a, u) in enumerate((x, z)), (j, c) in enumerate(cs)
            img[a][i, j] = abs(stack(u ./ abs.(u), ω / c)) / sum(w)
        end
        power(k) = abs2(stack(x, k)) + abs2(stack(z, k))
        pmax = maximum(power(ω / c) for c in cs)
        for (m, e) in enumerate(eig[i])
            ks = range(0.97, 1.03, length=601) .* (ω / e.c)
            p = power.(ks)
            q = argmax(p)
            (1 < q < length(ks) && p[q] > 1e-3pmax) || continue              # a peak, of an excited mode
            sx, sz = stack(x, ks[q]), stack(z, ks[q])
            picks[m][i] = (c=ω / ks[q], hv=abs(sx / sz), zh=real(-im * sz / sx) / 1.5)   # W/U = -i Sz/Sx (z up)
        end
    end
    (; eig, cs, img, picks)
end

results = map(models) do profile
    analyse(profile, seismograms(profile))
end

column(a, m, q) = [m <= length(e) ? q(e[m]) : NaN for e in a.eig]
println("FD against the eigen-solver: mean (max) relative difference of phase velocity and H/V, ZH signs that agree")
for (name, a) in pairs(results), m in eachindex(a.picks)
    ok = findall(p -> !isnan(p.c), a.picks[m])
    length(ok) < 3 && continue
    dc = abs.(getfield.(a.picks[m][ok], :c) ./ column(a, m, e -> e.c)[ok] .- 1)
    dh = abs.(getfield.(a.picks[m][ok], :hv) ./ column(a, m, e -> abs(e.U / e.W))[ok] .- 1)
    @printf("%-18s R%d  %2d–%2d Hz  c %.2f %% (%.2f)  H/V %.1f %% (%.1f)  ZH sign %d/%d\n", name, m - 1, Int(freqs[ok[1]]), Int(freqs[ok[end]]),
            100mean(dc), 100maximum(dc), 100mean(dh), 100maximum(dh),
            count(sign.(getfield.(a.picks[m][ok], :zh)) .== sign.(column(a, m, e -> e.W / e.U)[ok])), length(ok))
end

colors = Makie.wong_colors()
fig = Figure(size=(1500, 1500))
for (col, (name, a)) in enumerate(pairs(results))
    for (row, (comp, label)) in enumerate(((2, "vertical"), (1, "radial")))
        ax = Axis(fig[row, col], title="$name: $label, slant stack", ylabel=col == 1 ? "phase velocity [m/s]" : "")
        heatmap!(ax, collect(freqs), collect(a.cs), a.img[comp], colormap=:viridis)
        for m in eachindex(a.picks)
            lines!(ax, collect(freqs), column(a, m, e -> e.c), color=:white, linewidth=1.5, linestyle=:dash)
        end
        limits!(ax, freqs[1], freqs[end], a.cs[1], a.cs[end])
    end
    axh = Axis(fig[3, col], yscale=log10, title="H/V: eigen-solver (lines), SeismicWavePC (dots)", ylabel=col == 1 ? "H/V" : "")
    axz = Axis(fig[4, col], title="ZH = Z/(1.5 H), > 0 prograde", xlabel="frequency [Hz]", ylabel=col == 1 ? "ZH" : "")
    for m in 1:min(length(a.picks), 5)
        lines!(axh, collect(freqs), column(a, m, e -> abs(e.U / e.W)), color=colors[m], label="R$(m-1)")
        scatter!(axh, collect(freqs), getfield.(a.picks[m], :hv), color=colors[m], markersize=7)
        lines!(axz, collect(freqs), column(a, m, e -> e.W / e.U / 1.5), color=colors[m])
        scatter!(axz, collect(freqs), getfield.(a.picks[m], :zh), color=colors[m], markersize=7)
    end
    ylims!(axh, 0.1, 30)
    ylims!(axz, -2, 2)
    hlines!(axz, 0, color=:gray, linewidth=0.5)
    col == 3 && axislegend(axh, position=:rb, nbanks=2)
end
ax = Axis(fig[1:2, 4], title="profiles (top 60 m)", xlabel="vs [m/s] (solid), ρ [kg/m³] (dotted)", ylabel="depth [m]", yreversed=true)
for (k, (name, p)) in enumerate(pairs(models))
    P = p.(0:0.1:60)
    lines!(ax, getfield.(P, :vs), 0:0.1:60, color=colors[k], label=string(name))
    lines!(ax, getfield.(P, :rho), 0:0.1:60, color=colors[k], linestyle=:dot)
end
axislegend(ax, position=:lb)
colsize!(fig.layout, 4, Relative(0.16))
save("firn_rayleigh_benchmark.png", fig)
println("saved firn_rayleigh_benchmark.png")
