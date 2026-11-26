# Two questions about the curvilinear prototype (curvi.jl):
# 1. Flattening: a terrain-following grid near the surface only (w → 0 at depth D, Cartesian below). Hill (3 km high,
#    σ = 5 km) and steep hill (σ = 2 km, slopes up to 42°), 500 m grid, homogeneous.
# 2. Heterogeneous near-surface model: a low-velocity layer (2 km thick, vp = 3.5, vs = 2.0 km/s, ρ = 2200 kg/m³) under
#    the tilted surfaces (parallel to them; exact reference: the flat layered case on a 62.5 m grid with the interface on
#    cell faces) and under the hill (with flattening; reference: :corners at 62.5 m), 250 m grid.
#
#   julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/experiments.jl          (~5 minutes)
using CairoMakie
include(joinpath(@__DIR__, "curvi.jl"))
# the SBP operators (environment FAMILY, ORDER; default: boundary-optimized closures, order 7)
const OPS = (family=getfield(SummationByPartsOperators, Symbol(get(ENV, "FAMILY", "MattssonNiemeläWinters2026"))),
             order=parse(Int, get(ENV, "ORDER", "7")))
println("operators: ", OPS.family, ", order ", OPS.order)

pc(e) = @sprintf("%4.1f / %4.1f %%", 100e[1], 100e[2])
row(name, tr, ref) = @printf("  %-24s deep %s   shallow %s\n", name, pc(errors(tr, ref, :deep)), pc(errors(tr, ref, :shallow)))

println("1. Flattening (500 m grid; mean / max relative L2 error; reference :corners at 62.5 m)")
for case in ("hill", "steephill")
    ref = reference(case)
    println(case)
    row("staircase", fd_run(case, :openswpc, 500.0), ref)
    row(":corners", fd_run(case, :corners, 500.0), ref)
    for D in (nothing, 20e3, 10e3, 6e3)
        r = run_case(case; flatten=D, OPS...)
        row("curvilinear, " * (D === nothing ? "no flattening" : "D = $(Int(D / 1e3)) km") * @sprintf(" (J ≤ %.2f)", r.maxJ), r.traces, ref)
    end
end

println("\n2. Low-velocity layer under the surface (250 m grid)")
layer = (thickness=2e3, vp=3500.0, vs=2000.0, rho=2200.0)
results = Dict()
for case in ("tilt0", "tilt10", "tilt20", "tilt30")
    ref = reference(case; layer)
    println(case, " (reference: the exact flat layered case)")
    for (name, tr) in (("staircase", fd_run(case, :openswpc, 250.0; layer)), (":corners", fd_run(case, :corners, 250.0; layer)),
                       ("curvilinear", run_case(case; dx=250.0, layer, OPS...).traces))
        results[(case, name)] = (errors(tr, ref, :deep), errors(tr, ref, :shallow))
        row(name, tr, ref)
    end
end
let case = "hill", ref = reference("hill"; layer)
    println("hill (reference: :corners at 62.5 m)")
    row("staircase", fd_run(case, :openswpc, 250.0; layer), ref)
    row(":corners", fd_run(case, :corners, 250.0; layer), ref)
    row("curvilinear, D = 10 km", run_case(case; dx=250.0, layer, flatten=10e3, OPS...).traces, ref)
    row("curvilinear 125 m, D = 10 km", run_case(case; dx=125.0, layer, flatten=10e3, OPS...).traces, ref)
end

fig = Figure(size=(1300, 500), fontsize=16)
cols = Dict("staircase" => :darkorange, ":corners" => :dodgerblue, "curvilinear" => :forestgreen)
for (j, line) in enumerate(("deep", "shallow"))
    ax = Axis(fig[1, j], title="low-velocity layer under the surface, $line line, 250 m grid", xticks=(1:4, ["flat", "10°", "20°", "30°"]),
              ylabel="relative L2 error (%)")
    for (k, name) in enumerate(("staircase", ":corners", "curvilinear"))
        x = (1:4) .+ (k - 2) * 0.27
        e = [results[(c, name)][j] for c in ("tilt0", "tilt10", "tilt20", "tilt30")]
        barplot!(ax, x, 100 .* first.(e), width=0.25, color=cols[name], label=name)
        scatter!(ax, x, 100 .* last.(e), color=:black, markersize=7)
    end
    ylims!(ax, 0, nothing)
    j == 2 && axislegend(ax, position=:rt)
end
save("layer.png", fig)
println("saved layer.png")
