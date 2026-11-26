# Free surface on topography: the benchmark of Nobuaki Fuji (flexOPT.jl, dev/cutCells/geometryCases.jl). The point-force
# benchmark (examples/point_force_benchmark.jl) rotated by θ = 0, 10, 20, 30° (free surface, source, force along the normal
# and receivers rotated with it), and a Gaussian hill (3 km high, σ = 5 km) between the source and the receivers. Two lines
# of 5 receivers: "deep" (10 km below the surface) and "shallow" (1 km); components (u_t, u_n) along the surface frame.
#
# The staircase free surface (default, `free_surface = :openswpc`) and `free_surface = :corners` on the 500 m grid of the benchmark, against
# references: for the tilted planes the exact answer is the flat case (whose surface lies on cell faces) rotated, computed
# on a 62.5 m grid; for the hill, `:corners` on a 62.5 m grid. Error: relative L2 of (u_t, u_n) per receiver over 0–9 s.
#
#   julia -t auto --project=examples examples/topography_benchmark.jl
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Printf

# the cases (lengths in km)
const vp, vs, ρ = 6.0, 3.464, 2700.0
const f0 = 1.5 * 3464.0 / (16 * 500.0)                 # 0.6495 Hz
const src_flat, half, duration = (-10.0, -20.0), 60.0, 9.0
const xr, lines = (-30.0, -20.0, -10.0, 0.0, 10.0), (deep=-10.0, shallow=-1.0)
θ(case) = case == "hill" ? 0.0 : parse(Float64, replace(case, "tilt" => ""))
frame(case) = (s = sind(θ(case)); c = cosd(θ(case)); (t=(c, s), n=(-s, c)))
to_grid(case, a, b) = (f = frame(case); (a * f.t[1] + b * f.n[1], a * f.t[2] + b * f.n[2]))
surface(case) = case == "hill" ? (x -> 3.0 * exp(-x^2 / (2 * 5.0^2))) : (x -> x * tand(θ(case)))
receivers(case) = case == "hill" ?
    [(line, i, (x, line === :deep ? lines.deep : surface(case)(x) + lines.shallow)) for line in (:deep, :shallow) for (i, x) in enumerate(xr)] :
    [(line, i, to_grid(case, x, lines[line])) for line in (:deep, :shallow) for (i, x) in enumerate(xr)]
"node axes (multiples of dx) of the domain: ±60 km around the source, rotated, and 3 cells above the surface"
function nodes(case, dx)
    top = case == "hill" ? 3.0 + 3dx : 3dx
    c = case == "hill" ? [(a, b) for a in src_flat[1] .+ (-half, half), b in (src_flat[2] - half, top)] :
        [to_grid(case, a, b) for a in src_flat[1] .+ (-half, half), b in (src_flat[2] - half, top)]
    ax(d) = (floor(minimum(p[d] for p in c) / dx) * dx):dx:(ceil(maximum(p[d] for p in c) / dx) * dx)
    ax(1), ax(2)
end

"(u_t, u_n) at the receivers on the 0.01 s axis: Dict((line, i) => (ut, un))"
function run(case, free_surface, dx)
    zs = surface(case)
    xn, zn = nodes(case, dx)
    x, z = range(first(xn) + dx / 2, last(xn), step=dx), range(first(zn) + dx / 2, last(zn), step=dx)
    X, Y, Z = xyz_grid(x, 0, z)
    air = Z .> zs.(X)
    model = CartData(X, Y, Z, (; rho=ifelse.(air, 1.0, ρ), vp=ifelse.(air, 0.0, vp), vs=ifelse.(air, 0.0, vs)))
    # a line force of 1e10 N/m along n, Gaussian weights (σ = 500 m, cut at 2σ) over the nodes of the 500 m grid;
    # stf = :ricker with trise = 1/f0, t0 = 0.5/f0 is the Ricker wavelet of the benchmark (delay 1.5/f0) divided by trise
    s, n = to_grid(case, src_flat...), frame(case).n
    x5, z5 = nodes(case, 0.5)
    w = [(p, exp(-hypot(p[1] - s[1], p[2] - s[2])^2 / (2 * 0.5^2))) for p in Iterators.product(x5, z5) if hypot(p[1] - s[1], p[2] - s[2]) <= 1.0]
    W = sum(last, w)
    sources = [ForceSource(p[1], 0.0, p[2]; fx=1e10 * wi / W * n[1] / f0, fz=1e10 * wi / W * n[2] / f0, t0=0.5 / f0, trise=1 / f0) for (p, wi) in w]
    rec = receivers(case)
    stations = [Station("$(line)_$i", p[1], 0.0, p[2]) for (line, i, p) in rec]
    dt = 0.04dx
    r = simulate(model; dt, nt=round(Int, duration / dt), sources, stations, stf=:ricker, interpolation=:sinc, free_surface,
                 T=Float64, verbose=false)
    t, f = r.t .+ dt, frame(case)                     # displacement at sample it: Σ v dt up to step it
    at(u) = [τ >= t[end] ? u[end] : (j = searchsortedlast(t, τ); j == 0 ? 0.0 : u[j] + (u[j+1] - u[j]) * (τ - t[j]) / dt) for τ in TAX]
    Dict((line, i) => (at(r.u[:, 1, k] .* f.t[1] .+ r.u[:, 2, k] .* f.t[2]), at(r.u[:, 1, k] .* f.n[1] .+ r.u[:, 2, k] .* f.n[2]))
         for (k, (line, i, _)) in enumerate(rec))
end
const TAX = 0.01:0.01:duration
err(a, r) = sqrt(sum(abs2, a[1] .- r[1]) + sum(abs2, a[2] .- r[2])) / sqrt(sum(abs2, r[1]) + sum(abs2, r[2]))

# the comparison (skipped when the file is included, e.g. by prototypes/curvilinear)
if abspath(PROGRAM_FILE) == @__FILE__
    cases = ["tilt0", "tilt10", "tilt20", "tilt30", "hill"]
    variants = (:openswpc, :corners)
    flat = run("tilt0", :openswpc, 0.0625)
    ref = Dict(c => (c == "hill" ? run("hill", :corners, 0.0625) : flat) for c in cases)
    res = Dict((c, v) => run(c, v, 0.5) for c in cases, v in variants)
    E = Dict((c, v, line) => [err(res[(c, v)][(line, i)], ref[c][(line, i)]) for i in 1:5] for c in cases, v in variants, line in (:deep, :shallow))
    println("relative L2 error of (u_t, u_n) on the 500 m grid, mean (max) over the 5 receivers of a line")
    for c in cases, line in (:deep, :shallow)
        @printf("%-7s %-8s %s\n", c, line, join((@sprintf("%-10s %5.1f %% (%4.1f)", v, 100sum(E[(c, v, line)]) / 5, 100maximum(E[(c, v, line)])) for v in variants), "   "))
    end

    fig = Figure(size=(1300, 900), fontsize=16)
    colors = (openswpc=:darkorange, corners=:dodgerblue)
    labels = ["flat", "10°", "20°", "30°", "hill"]
    for (j, line) in enumerate((:deep, :shallow))
        ax = Axis(fig[1, j], title="$line line (500 m grid)", xticks=(1:5, labels), ylabel="relative L2 error (%)")
        for (k, v) in enumerate(variants)
            x = (1:5) .+ (k - 1.5) * 0.3
            barplot!(ax, x, [100sum(E[(c, v, line)]) / 5 for c in cases], width=0.28, color=colors[v], label=string(v))
            scatter!(ax, x, [100maximum(E[(c, v, line)]) for c in cases], color=:black, markersize=7)
        end
        ylims!(ax, 0, nothing)
        j == 1 && axislegend(ax, position=:lt)
    end
    for (j, (comp, name)) in enumerate(((2, "u_n"), (1, "u_t")))
        ax = Axis(fig[2, j], title="tilt20, shallow line: $name (black: exact)", xlabel="time (s)",
                  yticks=(-10 .* (0:4), ["x' = $(Int(x)) km" for x in xr]))
        scale = maximum(i -> maximum(abs, ref["tilt20"][(:shallow, i)][comp]), 1:5) / 4
        for i in 1:5
            lines!(ax, TAX, ref["tilt20"][(:shallow, i)][comp] ./ scale .- 10(i - 1), color=:black, linewidth=2.5)
            for v in variants
                lines!(ax, TAX, res[("tilt20", v)][(:shallow, i)][comp] ./ scale .- 10(i - 1), color=colors[v], linewidth=1.3)
            end
        end
    end
    save("topography_benchmark.png", fig)
    println("saved topography_benchmark.png")
end
