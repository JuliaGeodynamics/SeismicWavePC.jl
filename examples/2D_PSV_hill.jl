# 2D P-SV waves from an earthquake below a hill, with a layered crust built with GeophysicalModelGenerator.
#
#   julia -t auto --project=examples examples/2D_PSV_hill.jl          # CPU (or add: cuda | amdgpu | metal)
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie
include("backend.jl")

# 1. Model: a GMG CartData of size (nx, 1, nz) is a 2D model in the x-z plane (coordinates in km, z up)
x = range(-15, 15, length=600)                        # dx = 0.05 km
z = range(-12, 2.0, length=281)
model = CartData(xyz_grid(x, 0, z))
Phase, Temp = zeros(Int, size(model)), zeros(size(model))
add_layer!(Phase, Temp, model; zlim=(-2.5, 2), phase=ConstantPhase(1))      # sediments
add_layer!(Phase, Temp, model; zlim=(-6, -2.5), phase=ConstantPhase(2))    # upper crust
add_layer!(Phase, Temp, model; zlim=(-20, -6), phase=ConstantPhase(3))     # lower crust
X, Y, Z = xyz_grid(x, 0, z)
Phase[Z .> 1.5 .* exp.(-(X ./ 3) .^ 2)] .= 0                                # air above a 1.5 km high hill

#            air     sediments  upper crust  lower crust
rho_ = Dict(0 => 0.001, 1 => 2.2, 2 => 2.6, 3 => 2.8)
vp_  = Dict(0 => 0.0,   1 => 2.5, 2 => 5.0, 3 => 6.2)
vs_  = Dict(0 => 0.0,   1 => 1.2, 2 => 2.9, 3 => 3.5)
Q_   = Dict(0 => 10.0,  1 => 50.0, 2 => 300.0, 3 => 500.0)
model = addfield(model, (; rho=getindex.(Ref(rho_), Phase), vp=getindex.(Ref(vp_), Phase),
                           vs=getindex.(Ref(vs_), Phase), Qp=2 .* getindex.(Ref(Q_), Phase), Qs=getindex.(Ref(Q_), Phase), Phase))

# 2. Source (thrust fault at 5 km depth) and a line of stations on the free surface
source = DoubleCoupleSource(-4.0, 0.0, -5.0; Mw=4.0, strike=0, dip=45, rake=90, trise=0.4)
stations = [Station("S$(i)", xs, 0.0, 0.0; surface=true) for (i, xs) in enumerate(-12:1.0:12)]

# 3. Run (2D P-SV); every 200 steps the wavefield is written to ParaView files
result = simulate(model; dt=0.003, nt=3000, sources=source, stations, backend,
                  snapshot_every=200, snapshot_dir="hill_snapshots")
display(result)

# 4. Record section of the vertical velocity
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], xlabel="time [s]", ylabel="station x [km]", title="vertical velocity, 2D P-SV")
for (n, st) in enumerate(result.stations)
    vz = result.v[:, 2, n]
    lines!(ax, result.t, st.x .+ vz ./ maximum(abs, result.v[:, 2, :]) .* 2, color=:black, linewidth=0.7)
end
save("2D_PSV_hill.png", fig)
println("saved 2D_PSV_hill.png and hill_snapshots/wavefield.pvd")
