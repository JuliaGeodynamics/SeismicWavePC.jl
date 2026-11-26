# Which SBP operators are most accurate near the free surface at 16 points per S wavelength? Dual-pairing upwind operators
# of SummationByPartsOperators.jl: Mattsson (2017), dispersion-relation preserving (Williams & Duru 2024), and with
# boundary-optimized closures (Mattsson, Niemelä & Winters 2026), on the flat case, the 20° tilt and the hill (500 m grid).
#
#   julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/operators.jl
include(joinpath(@__DIR__, "curvi.jl"))

cases = ("tilt0", "tilt20", "hill")
@printf("%-28s %-6s %s\n", "operators", "order", join((rpad("$c deep / shallow", 24) for c in cases)))
for (family, orders) in ((Mattsson2017, 3:7), (WilliamsDuru2024, 4:7), (MattssonNiemeläWinters2026, 3:9))
    for order in orders
        cells = map(cases) do case
            try
                r = run_case(case; family, order)
                e = (errors(r.traces, reference(case), :deep)[1], errors(r.traces, reference(case), :shallow)[1])
                @sprintf("%5.1f / %5.1f %%          ", 100e[1], 100e[2])
            catch err
                rpad(sprint(showerror, err)[1:min(end, 20)], 24)
            end
        end
        @printf("%-28s %-6d %s\n", family, order, join(cells))
    end
end
