# Point force under a flat free surface: the benchmark of Nobuaki Fuji (flexOPT.jl, application/forBoris), in its exact
# conventions: a vertical force of 1e10 N/m with a Ricker wavelet (peak frequency 0.65 Hz, delay 1.5/f0), smoothed with a
# Gaussian (σ = 500 m, cut off at 2σ) over the nodes of a 500 m grid, 20 km deep in a homogeneous half-space (vp = 6 km/s,
# vs = 3.464 km/s, ρ = 2700 kg/m³). Displacement at receivers 10 km deep; written to a CSV file to compare with other codes
# directly (no rescaling or time shifts needed).
#
#   julia -t auto --project=examples examples/point_force_benchmark.jl [dx in km, default 0.5]
#
using SeismicWavePC, GeophysicalModelGenerator, DelimitedFiles, Printf

dx = isempty(ARGS) ? 0.5 : parse(Float64, ARGS[1])  # grid spacing [km]
vp, vs, ρ = 6.0, 3.464, 2700.0
f0 = 1.5 * 3464.0 / (16 * 500.0)                    # 0.6495 Hz
src = (x=-10.0, z=-20.0)
half, duration = 60.0, 9.0
recx, recz = [-30.0, -20.0, -10.0, 0.0, 10.0], -10.0

# the model: free surface at a cell face (z = 0) with 2 cells of air; the PML around it
x = range(src.x - half + dx / 2, src.x + half, step=dx)
z = range(src.z - half + dx / 2, 2dx, step=dx)
X, Y, Z = xyz_grid(x, 0, z)
air = Z .> 0
model = CartData(X, Y, Z, (; rho=ifelse.(air, 1.0, ρ), vp=ifelse.(air, 0.0, vp), vs=ifelse.(air, 0.0, vs)))

# the source: Gaussian weights (normalized to 1) at the nodes of the 500 m grid within 2σ of the source, one force each.
# stf = :ricker is (1 - 2a²) exp(-a²) / trise with a = π (t - t0 - trise) / trise: trise = 1/f0 and t0 = 0.5/f0 give the
# wavelet of the benchmark (delay 1.5/f0), and the force fz = F · trise its amplitude F [N/m].
σ, node = 0.5, 0.5
nodes = [(src.x + i * node, src.z + j * node) for i in -4:4 for j in -4:4 if hypot(i, j) * node <= 2σ]
weights = [exp(-hypot(p[1] - src.x, p[2] - src.z)^2 / (2σ^2)) for p in nodes]
weights ./= sum(weights)
trise = 1 / f0
sources = [ForceSource(p[1], 0.0, p[2]; fz=1e10 * w * trise, t0=0.5 / f0, trise) for (p, w) in zip(nodes, weights)]
stations = [Station("x$xr", xr, 0.0, recz) for xr in recx]

dt = 0.008 * dx
nt = round(Int, duration / dt)
r = simulate(model; dt, nt, sources, stations, stf=:ricker, interpolation=:sinc)   # sources and stations at their positions
t = r.t .+ dt                                        # displacement sample it: Σ v dt up to the step it
file = joinpath(@__DIR__, "point_force_benchmark_dx$(dx).csv")
writedlm(file, vcat(permutedims(["time"; ["ux_x=$(xr)km" for xr in recx]; ["uz_x=$(xr)km" for xr in recx]]),
                    hcat(t, r.u[:, 1, :], r.u[:, 2, :])), ',')
@printf("peak |u_z| [m] at x = %s km: %s\nsaved %s\n", join(recx, ", "),
        join([@sprintf("%.3e", maximum(abs, r.u[:, 2, i])) for i in eachindex(recx)], ", "), file)
