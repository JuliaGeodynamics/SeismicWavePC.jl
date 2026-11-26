# The curvilinear SBP prototype against the staircase (default) and :corners of SeismicWavePC on the 500 m grid, all
# against the same references (a few minutes).
#
#   julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/compare.jl
using CairoMakie
include(joinpath(@__DIR__, "curvi.jl"))

cases = ["tilt0", "tilt10", "tilt20", "tilt30", "hill"]
labels = ["flat", "10°", "20°", "30°", "hill"]
runs = Dict(c => run_case(c) for c in cases)
fdruns = Dict((c, v) => fd_run(c, v, 500.0) for c in cases, v in (:openswpc, :corners))
cv(c, line) = errors(runs[c].traces, reference(c), Symbol(line))
fd(c, line, v) = errors(fdruns[(c, v)], reference(c), Symbol(line))

fig = Figure(size=(1300, 900), fontsize=16)
cols = (staircase=:darkorange, corners=:dodgerblue, curvilinear=:forestgreen)
for (j, line) in enumerate(("deep", "shallow"))
    ax = Axis(fig[1, j], title="$line line, 500 m grid", xticks=(1:5, labels), ylabel="relative L2 error (%)")
    for (k, v) in enumerate((:staircase, :corners, :curvilinear))
        x = (1:5) .+ (k - 2) * 0.27
        e = v === :curvilinear ? [cv(c, line) for c in cases] : [fd(c, line, v === :staircase ? :openswpc : :corners) for c in cases]
        barplot!(ax, x, 100 .* first.(e), width=0.25, color=cols[v], label=string(v))
        scatter!(ax, x, 100 .* last.(e), color=:black, markersize=7)
    end
    ylims!(ax, 0, nothing)
    j == 2 && axislegend(ax, position=:rt)
end
ref = reference("tilt20")
for (j, (comp, name)) in enumerate(((2, "u_n"), (1, "u_t")))
    ax = Axis(fig[2, j], title="tilt20, shallow line: $name (black: exact)", xlabel="time (s)",
              yticks=(-10 .* (0:4), ["x' = $(Int(x / 1e3)) km" for x in XR]))
    s = maximum(i -> maximum(abs, ref[(:shallow, i)][comp]), 1:5) / 4
    tax = 0.01:0.01:9.0
    for i in 1:5
        lines!(ax, tax, ref[(:shallow, i)][comp] ./ s .- 10(i - 1), color=:black, linewidth=2.5)
        for (v, name) in ((:openswpc, :staircase), (:corners, :corners))
            lines!(ax, tax, fdruns[("tilt20", v)][(:shallow, i)][comp] ./ s .- 10(i - 1), color=cols[name], linewidth=1.2)
        end
        lines!(ax, tax, runs["tilt20"].traces[(:shallow, i)][comp] ./ s .- 10(i - 1), color=cols.curvilinear, linewidth=1.5)
    end
end
for c in cases, line in ("deep", "shallow")
    println(rpad(c, 7), rpad(line, 8), join(("$v $(round(100first(e), digits=1)) %" for (v, e) in
        (("staircase", fd(c, line, :openswpc)), ("corners", fd(c, line, :corners)), ("curvilinear", cv(c, line)))), "   "))
end
save("curvilinear_vs_staircase.png", fig)
println("saved curvilinear_vs_staircase.png")
