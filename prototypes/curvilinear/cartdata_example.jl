# The curvilinear solver on a single deformed CartData: a hill (3 km, σ = 5 km), an undulating interface with a large
# velocity jump (Moho-like, 5.5 → 6.5 km/s) and a magma chamber added with GMG's add_ellipsoid! on the deformed mesh.
# A grid line of the mesh follows the interface; the solver finds it from the jump of the material and splits the mesh
# into blocks there.
# Source and receivers of the hill case (a vertical force 20 km deep; receivers 10 km deep and 1 km below the surface).
#
# Against SeismicWavePC (staggered, staircase surface, interface and chamber) on fine grids; no exact reference.
#
#   julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/cartdata_example.jl        (~10 minutes)
using CairoMakie
include(joinpath(@__DIR__, "curvi.jl"))

topo(x) = 3.0 * exp(-x^2 / (2 * 5.0^2))                        # [km]
moho(x) = -7.0 + 1.5 * sin(2π * x / 40)                        # [km]
const MAT = (crust=(vp=5.5, vs=3.2, rho=2600.0), mantle=(vp=6.5, vs=3.75, rho=2900.0), magma=(vp=3.8, vs=2.0, rho=2400.0))
const CHAMBER = (cen=(0.0, 0.0, -3.0), axes=(3.0, 3.0, 1.5))   # [km]

"the model on a deformed mesh with spacing dx [km]: the layers (GMG style), the chamber with add_ellipsoid!"
function model(dx; below=(z, s) -> z < s)
    mesh = curvilinear_mesh(range(-72, 52, length=round(Int, 124 / dx) + 1), [topo, moho]; dz=dx, bottom=-80)
    Phase = ifelse.(below.(mesh.z.val, moho.(mesh.x.val)), 2, 1)
    Temp = zeros(size(Phase))
    add_ellipsoid!(Phase, Temp, mesh; CHAMBER..., phase=ConstantPhase(3))
    m = (MAT.crust, MAT.mantle, MAT.magma)
    addfield(mesh, (; rho=[m[p].rho for p in Phase], vp=[m[p].vp for p in Phase], vs=[m[p].vs for p in Phase], Phase))
end

"the same model for SeismicWavePC (x, z in m)"
function mat(x, z)
    xk, zk = x / 1e3, z / 1e3
    m = (xk / CHAMBER.axes[1])^2 + ((zk - CHAMBER.cen[3]) / CHAMBER.axes[3])^2 <= 1 ? MAT.magma : zk > moho(xk) ? MAT.crust : MAT.mantle
    (1e3m.vp, 1e3m.vs, m.rho)
end

if abspath(PROGRAM_FILE) == @__FILE__
    curvi = Dict(dx => run_cartdata(model(dx / 1e3); case="hill").traces for dx in (500.0, 250.0, 125.0))
    stag = Dict(dx => fd_run("hill", :openswpc, dx; mat) for dx in (250.0, 125.0, 62.5))
    corners = fd_run("hill", :corners, 250.0; mat)

    pc(tr, ref) = @sprintf("%5.2f / %5.2f %%", 100errors(tr, ref, :deep)[1], 100errors(tr, ref, :shallow)[1])
    println("relative L2 differences, mean over 5 receivers, deep / shallow line")
    println(rpad("", 30), rpad("vs curvilinear 125 m", 24), "vs staggered 62.5 m")
    for (name, tr) in [("curvilinear 500 m", curvi[500.0]), ("curvilinear 250 m", curvi[250.0]), ("curvilinear 125 m", curvi[125.0]),
                       ("staggered 250 m", stag[250.0]), ("staggered :corners 250 m", corners), ("staggered 125 m", stag[125.0]),
                       ("staggered 62.5 m", stag[62.5])]
        println(rpad(name, 30), rpad(pc(tr, curvi[125.0]), 24), pc(tr, stag[62.5]))
    end

    # figure: the model on its deformed mesh (top 25 km), and the shallow line
    m = model(0.25)
    X, Z, vs = m.x.val[:, 1, :], m.z.val[:, 1, :], m.fields.vs[:, 1, :]
    fig = Figure(size=(1400, 900), fontsize=15)
    ax = Axis(fig[1, 1:2], title="vs on the deformed mesh (every 4th grid line; a grid line follows the interface)", xlabel="x [km]", ylabel="z [km]")
    keep = findall(z -> z > -25, Z[1, :])
    surface!(ax, X[:, keep], Z[:, keep], zeros(size(X[:, keep])); color=vs[:, keep], colormap=:viridis, shading=NoShading)
    for k in keep[1:4:end]
        lines!(ax, X[:, k], Z[:, k], color=(:white, 0.4), linewidth=0.5)
    end
    for i in 1:8:size(X, 1)
        lines!(ax, X[i, keep], Z[i, keep], color=(:white, 0.4), linewidth=0.5)
    end
    scatter!(ax, collect(XR) ./ 1e3, [topo(x / 1e3) - 1 for x in XR], color=:red, marker=:dtriangle, markersize=12)
    limits!(ax, -40, 25, -25, 4)
    for (j, (comp, name)) in enumerate(((2, "u_n"), (1, "u_t")))
        axt = Axis(fig[2, j], title="shallow line: $name (black: curvilinear 125 m)", xlabel="time (s)",
                  yticks=(-10 .* (0:4), ["x = $(Int(x / 1e3)) km" for x in XR]))
        s = maximum(i -> maximum(abs, curvi[125.0][(:shallow, i)][comp]), 1:5) / 4
        for i in 1:5
            lines!(axt, TAX, curvi[125.0][(:shallow, i)][comp] ./ s .- 10(i - 1), color=:black, linewidth=2.5)
            lines!(axt, TAX, stag[250.0][(:shallow, i)][comp] ./ s .- 10(i - 1), color=:darkorange, linewidth=1.2, label=i == 1 ? "staggered 250 m" : nothing)
            lines!(axt, TAX, stag[62.5][(:shallow, i)][comp] ./ s .- 10(i - 1), color=:dodgerblue, linewidth=1.2, label=i == 1 ? "staggered 62.5 m" : nothing)
            lines!(axt, TAX, curvi[250.0][(:shallow, i)][comp] ./ s .- 10(i - 1), color=:forestgreen, linewidth=1.5, linestyle=:dash, label=i == 1 ? "curvilinear 250 m" : nothing)
        end
        j == 1 && axislegend(axt, position=:lb)
    end
    save("cartdata_example.png", fig)
    println("saved cartdata_example.png")
end
