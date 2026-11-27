# 2D P-SV on a curvilinear (terrain-following) grid: a hill, an undulating interface with a large jump (Moho-like) and a
# magma chamber. The mesh follows the topography and the interface; the free surface lies exactly on the topography.
#
#   julia -t auto --project=examples examples/2D_curvilinear.jl
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie

# 1. The mesh: a grid line along the topography and along the interface [km]; columns are vertical
topo(x) = 3.0 * exp(-x^2 / 50)
moho(x) = -7.0 + 1.5 * sin(2π * x / 40)
mesh = curvilinear_mesh(range(-40, 40, length=321), [topo, moho]; dz=0.25, bottom=-40)

# 2. The material at the nodes, with the usual GMG tools: the layers, and a magma chamber
X, Z = mesh.x.val, mesh.z.val
Phase = ifelse.(Z .< moho.(X), 2, 1)
add_ellipsoid!(Phase, zeros(size(Phase)), mesh; cen=(0.0, 0.0, -3.0), axes=(3.0, 3.0, 1.5), phase=ConstantPhase(3))
#       crust  mantle magma
vp_  = (5.5,   6.5,   3.8)
vs_  = (3.2,   3.75,  2.0)
rho_ = (2600.0, 2900.0, 2400.0)
model = addfield(mesh, (; rho=getindex.(Ref(rho_), Phase), vp=getindex.(Ref(vp_), Phase), vs=getindex.(Ref(vs_), Phase), Phase))

# 3. A thrust fault 12 km deep, stations on the surface (exactly on the topography)
source = DoubleCoupleSource(-15.0, 0.0, -12.0; Mw=4.0, strike=0, dip=45, rake=90, trise=0.5)
stations = [Station("S$i", x, 0.0, 0.0; surface=true) for (i, x) in enumerate(-30:2.0:30)]

# 4. Run: the time step follows from the stability limit; give the duration
result = simulate(model; tmax=20.0, sources=source, stations)
display(result)

# 5. The model and a record section of the vertical velocity
fig = Figure(size=(1000, 900))
ax = Axis(fig[1, 1], title="vs [km/s] on the deformed mesh", xlabel="x [km]", ylabel="z [km]")
keep = axes(Z, 3)                                                  # (the axis limits clip)
s = surface!(ax, X[:, 1, keep], Z[:, 1, keep], zeros(size(X[:, 1, keep])); color=model.fields.vs[:, 1, keep], colormap=:viridis, shading=NoShading)
Colorbar(fig[1, 2], s)
scatter!(ax, [st.x for st in stations], topo.([st.x for st in stations]), color=:red, marker=:dtriangle, markersize=8)
scatter!(ax, [source.x], [source.z], color=:white, marker=:star5, markersize=16)
limits!(ax, -40, 40, -20, 4)
ax2 = Axis(fig[2, 1], xlabel="time [s]", ylabel="station x [km]", title="vertical velocity")
for (n, st) in enumerate(result.stations)
    lines!(ax2, result.t, st.x .+ result.v[:, 2, n] ./ maximum(abs, result.v[:, 2, :]) .* 3, color=:black, linewidth=0.7)
end
save("2D_curvilinear.png", fig)
println("saved 2D_curvilinear.png")
