# Seismic waves through the magmatic system below Unzen volcano (Japan), as computed by MagmaThermoKinematics.jl (MTK).
#
#   julia -t auto --project=examples/MagmaThermoKinematics examples/MagmaThermoKinematics/Unzen_2D.jl [res]
#
# 1. Runs MTK's 2D Unzen showcase (examples/HighRes_Unzen/HighRes_Unzen_2D.jl of MTK) with grid spacing `res` [m]
#    (default 100; the run is skipped if its result already exists). It gives temperature, melt fraction and rock type.
# 2. Turns these into density, P and S velocities and melt fraction with MAGEMin phase diagrams (including the reduction of
#    the velocities by partial melt), and into attenuation (low Q in partially molten rock).
# 3. Simulates an earthquake below the volcano and makes a movie of the wavefield while the simulation runs,
#    and a record section of the stations on the surface.
using SeismicWavePC, GeophysicalModelGenerator, MagmaThermoKinematics, MAGEMin_C, JLD2, CairoMakie, Printf

res = isempty(ARGS) ? 100 : parse(Int, ARGS[1])                  # grid spacing of the MTK run [m]
dir = joinpath(@__DIR__, "HighRes2D_$(res)m")
if !isfile(joinpath(dir, "final.jld2"))
    mtk = joinpath(pkgdir(MagmaThermoKinematics), "examples", "HighRes_Unzen", "HighRes_Unzen_2D.jl")
    run(Cmd(`$(Base.julia_cmd()) -t auto --project=$(dirname(Base.active_project())) $mtk $res`; dir=@__DIR__))
end
mtk = load(joinpath(dir, "final.jld2"))                          # x, z [m]; temperature T [°C] and Phases on the grid

# --- Rock physics with MAGEMin -------------------------------------------------------------------------------------------
# Tables of density, seismic velocities and melt fraction on a pressure–temperature grid, for two rock compositions.
Xoxides = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3", "K2O", "Na2O", "TiO2", "Cr2O3", "H2O"]
dacite = [64.5, 15.9, 5.0, 2.4, 3.2, 1.4, 2.3, 3.9, 0.7, 0.0, 2.0]          # wt%: Unzen-like dacite with 2 % water
basalt = [48.4, 15.2, 11.6, 10.1, 6.7, 1.6, 0.6, 1.9, 0.7, 0.0, 0.5]        # wt%: mafic lower crust
composition = (dacite, dacite, basalt, dacite)       # MTK phases 1 volcanic cover, 2 upper crust, 3 lower crust, 4 intrusions
Ps, Ts = 0.0:0.25:7.0, 200.0:25.0:1200.0             # pressure [kbar] and temperature [°C] of the tables

"Density [kg/m³], vp, vs [km/s] and melt fraction (by volume) of a MAGEMin result."
function properties(o)
    ϕ = o.frac_M_vol
    vp_melt = ϕ > 0 ? sqrt(o.bulkModulus_M * 1.0e9 / o.rho_M) / 1.0e3 : 0.0
    return (o.rho, max(o.Vp_cor, vp_melt), o.Vs_cor, ϕ)   # Vs_cor = 0: fully molten. The melt correction can give vp < vp_melt
end
data = Initialize_MAGEMin("ig", verbose=false)
tables = Dict(X => properties.(multi_point_minimization([P for P in Ps, T in Ts], [T for P in Ps, T in Ts], data;
                                                        X, Xoxides, sys_in="wt", seismic_cor=true, progressbar=false))
              for X in unique(composition))
Finalize_MAGEMin(data)

"Bilinear interpolation in the table `tab` at pressure `P` [kbar] and temperature `T` [°C] (clamped to the table)."
function lookup(tab, P, T)
    fi, fj = clamp((P - first(Ps)) / step(Ps) + 1, 1, length(Ps)), clamp((T - first(Ts)) / step(Ts) + 1, 1, length(Ts))
    i, j = min(floor(Int, fi), length(Ps) - 1), min(floor(Int, fj), length(Ts) - 1)
    u, v = fi - i, fj - j
    return (1 - u) * (1 - v) .* tab[i, j] .+ u * (1 - v) .* tab[i+1, j] .+ (1 - u) * v .* tab[i, j+1] .+ u * v .* tab[i+1, j+1]
end

# Partial melt attenuates seismic waves strongly: Qs falls from that of the rock to 10 for 10 % melt or more.
Q_rock = (50.0, 200.0, 300.0, 200.0)                 # Qs of solid rock for MTK phases 1–4; Qp = 2Qs

x, z = mtk["x"] ./ 1.0e3, mtk["z"] ./ 1.0e3          # [km]
dx = x[2] - x[1]
trise = 10dx                                         # rise time of the source [s]: frequencies up to about 2 / trise
vs_min = 5dx * 2 / trise                             # S waves slower than this have < 5 cells per wavelength: such mush is treated as molten
phase, T = mtk["Phases"], Float64.(mtk["T"])
top = [z[findlast(!=(0), phase[i, :])] for i in eachindex(x)]                 # height of the surface [km]
props = map(CartesianIndices(phase)) do I
    i, k = Tuple(I)
    phase[I] == 0 && return (1.0, 0.0, 0.0, 0.0, 10.0)                        # air
    P = 2700 * 9.81 * 1.0e3 * (top[i] - z[k]) / 1.0e8                          # lithostatic pressure [kbar]
    ρ, vp, vs, ϕ = lookup(tables[composition[phase[I]]], P, T[I])
    vs = vs < vs_min ? 0.0 : vs
    return (ρ, vp, vs, ϕ, max(Q_rock[phase[I]] / (1 + 100ϕ), 10.0))
end
field(n) = reshape(getindex.(props, n), length(x), 1, length(z))
model = CartData(xyz_grid(x, 0, z)..., (; rho=field(1), vp=field(2), vs=field(3), ϕ=field(4), Qs=field(5), Qp=2 .* field(5),
                                           T=reshape(T, length(x), 1, length(z))))

# --- Earthquake and seismometers -----------------------------------------------------------------------------------------
source = DoubleCoupleSource(-12.0, 0.0, -10.0; Mw=4.0, strike=0, dip=60, rake=90, trise)
stations = [Station(@sprintf("S%02d", i), xs, 0.0, 0.0; surface=true) for (i, xs) in enumerate(-17:1.0:17)]   # just below the surface
dt = 0.6 * dx / (maximum(model.fields.vp) * sqrt(2))             # Courant number 0.6
nt = round(Int, 14 / dt)                                         # 14 s

# --- Movie, made while the simulation runs -------------------------------------------------------------------------------
air = dropdims(model.fields.vp .== 0, dims=2)
masked(A) = (B = Float32.(dropdims(A, dims=2)); B[air] .= NaN; B)
ϕ = dropdims(model.fields.ϕ, dims=2)
surface_x = [s.x for s in stations]
surface_z = [z[findlast(.!air[argmin(abs.(x .- s.x)), :])] for s in stations]

set_theme!(fontsize=16)
fig = Figure(size=(1200, 900))
ax1 = Axis(fig[1, 1], title="S-wave velocity from the MTK model ($res m grid)", ylabel="height [km]")
hm = heatmap!(ax1, x, z, masked(model.fields.vs), colormap=:lajolla, colorrange=(0, 4))
Colorbar(fig[1, 2], hm, label="vs [km/s]")
vz = Observable(fill(NaN32, length(x), length(z)))
c = Observable(1.0f-12)
title = Observable("")
ax2 = Axis(fig[2, 1], title=title, xlabel="distance W–E [km]", ylabel="height [km]")
hw = heatmap!(ax2, x, z, vz, colormap=:balance, colorrange=@lift((-$c, $c)))
Colorbar(fig[2, 2], hw, label="vz [m/s]", ticklabelspace=55, tickformat=v -> [@sprintf("%.2g", x) for x in v])   # fixed width: frames do not shift
for ax in (ax1, ax2)
    contour!(ax, x, z, ϕ, levels=[0.1], color=:white, linewidth=1.5)      # partially molten rock (melt > 10 %, from MAGEMin)
    scatter!(ax, surface_x, surface_z, marker=:dtriangle, color=:black, markersize=10)
    scatter!(ax, [source.x], [source.z], marker=:star5, color=:yellow, strokecolor=:black, strokewidth=1, markersize=22)
    limits!(ax, extrema(x)..., z[1], 3)
end
colsize!(fig.layout, 1, Aspect(1, (x[end] - x[1]) / (3 - z[1])))       # true aspect ratio
resize_to_layout!(fig)
movie = VideoStream(fig; framerate=12)

function frame(wavefield, t)
    vz[] = masked(wavefield.fields.V[3])
    c[] = max(0.3f0 * maximum(v -> isnan(v) ? 0.0f0 : abs(v), vz[]), 1.0f-12)   # colour scale follows the amplitude
    title[] = @sprintf("vertical velocity, t = %.2f s", t)
    recordframe!(movie)
end

result = simulate(model; dt, nt, sources=source, stations, snapshot_every=round(Int, 0.1 / dt), callback=frame)
save(joinpath(@__DIR__, "Unzen_2D_waves.mp4"), movie)

# --- Seismograms -----------------------------------------------------------------------------------------------------------
molten = extrema(x[i] for i in eachindex(x) if any(>(0.1), ϕ[i, :]))      # horizontal extent of the partially molten region
peak = [maximum(abs, result.v[:, c, n]) for n in eachindex(result.stations), c in 1:2]
fig2 = Figure(size=(1100, 700))
ax = Axis(fig2[1, 1], xlabel="time [s]", ylabel="station x [km]", title="vertical velocity, each trace normalized (shaded: above partially molten rock)")
ax2 = Axis(fig2[1, 2], xlabel="peak velocity [m/s]", title="peak amplitude")
for a in (ax, ax2)
    hspan!(a, molten..., color=(:orange, 0.2))                               # stations above the partially molten region
end
for (n, s) in enumerate(result.stations)
    v = result.v[:, 2, n]
    lines!(ax, result.t, s.x .+ 0.8 .* v ./ maximum(abs, v), color=:black, linewidth=0.7)
end
scatterlines!(ax2, peak[:, 2], surface_x, label="vertical")
scatterlines!(ax2, peak[:, 1], surface_x, label="horizontal")
axislegend(ax2, position=:rt)
linkyaxes!(ax, ax2)
hideydecorations!(ax2, grid=false)
colsize!(fig2.layout, 2, Relative(0.25))
save(joinpath(@__DIR__, "Unzen_2D_seismograms.png"), fig2)
println("saved Unzen_2D_waves.mp4 and Unzen_2D_seismograms.png in ", @__DIR__)
