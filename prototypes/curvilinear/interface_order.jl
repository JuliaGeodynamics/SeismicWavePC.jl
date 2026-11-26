# Convergence at a sharp material interface between grid points: a flat free surface over a layer 1937.5 m thick
# (vp = 3.5, vs = 2.0 km/s, ρ = 2200 kg/m³) over the half-space. The interface lies on cell faces of the 62.5 m reference
# (SeismicWavePC, staggered) but between them at 125, 250 and 500 m. Staggered staircase, curvilinear one block (interface
# between nodes) and two blocks (interface a grid line).
#
#   julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/interface_order.jl
include(joinpath(@__DIR__, "curvi.jl"))

layer = (thickness=1937.5, vp=3500.0, vs=2000.0, rho=2200.0)
ref = fd_run("tilt0", :openswpc, 62.5; layer)
println("relative L2 error, mean over 5 receivers, deep / shallow line (the shallow one is inside the layer)")
for (name, f) in (("staggered (SeismicWavePC)", dx -> fd_run("tilt0", :openswpc, dx; layer)),
                  ("curvilinear, 1 block", dx -> run_case("tilt0"; dx, layer).traces),
                  ("curvilinear, 2 blocks", dx -> run_case("tilt0"; dx, layer, interfaces=[-layer.thickness]).traces))
    e = [(tr = f(dx); (dx, errors(tr, ref, :deep)[1], errors(tr, ref, :shallow)[1])) for dx in (500.0, 250.0, 125.0)]
    println(name)
    for (k, (dx, d, s)) in enumerate(e)
        rate(j) = k == 1 ? "" : @sprintf("  (order %.1f)", log2(e[k-1][j] / e[k][j]))
        @printf("  %5.0f m: deep %6.3f %%%s   shallow %6.3f %%%s\n", dx, 100d, rate(2), 100s, rate(3))
    end
end
