# Step 2 of the high-resolution Unzen example (README movie): waves through the 10 m model of Unzen_2D_10m_model.jl, from an
# Mw 3 volcano-tectonic earthquake with a Brune (ω⁻²) source time function; movie and record section. The corner frequency
# follows from Brune (1970) for a stress drop of 3 MPa (~7 Hz); the 10 m grid resolves up to ~20 Hz in rock with
# vs ≥ 1 km/s (5 cells per wavelength). 10 million cells and 23 400 time steps: 2.2 minutes on an Apple M-series GPU.
#
#   julia -t auto --project=examples examples/MagmaThermoKinematics/Unzen_2D_10m_waves.jl [cpu|cuda|amdgpu|metal] [dx in m, default 10]
#
# (the GPU package added to the environment). The GIF of the README:
#   ffmpeg -i Unzen_2D_waves_10m.mp4 -vf "fps=8,scale=560:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" Unzen_2D_waves_10m.gif
using SeismicWavePC, GeophysicalModelGenerator, Serialization, CairoMakie, Printf
include(joinpath(@__DIR__, "..", "backend.jl"))
dev = DEVICE
res = length(ARGS) > 1 ? ARGS[2] : "10"
m = deserialize(joinpath(@__DIR__, "HighRes2D_100m", "Unzen_$(res)m.jls"))
x, z = m.x, m.z
r3(A) = reshape(Float64.(A), length(x), 1, length(z))
model = CartData(xyz_grid(x, 0, z)..., (; rho=r3(m.rho), vp=r3(m.vp), vs=r3(m.vs), Qs=r3(m.Qs), Qp=r3(2 .* m.Qs)))
dx = x[2] - x[1]
Mw, Δσ, β = 3.0, 3.0e6, 3500.0                        # magnitude, stress drop [Pa], S velocity at the source [m/s]
M0 = 10^(1.5Mw + 9.1)                                 # [N m]
fc = 0.37β * (16Δσ / (7M0))^(1 / 3)                   # Brune corner frequency [Hz]
ωc = 2π * fc
brune(τ) = τ < 0 ? 0.0 : ωc^2 * τ * exp(-ωc * τ)      # moment rate / M0 (integral 1)
# High frequencies are removed near the surface in nature (κ); here the Brune function is smoothed over 0.1 s (a Küpper pulse),
# the shortest period the 10 m grid carries in rock with vs ≥ 1 km/s
ts = 0:1.0e-4:1.0
kup = [τ <= 0.1 ? sin(π * τ / 0.1)^3 : 0.0 for τ in ts]
stf = SampledSTF(ts, [sum(brune(τ - u) * w for (u, w) in zip(ts, kup)) for τ in ts] ./ sum(kup))
source = DoubleCoupleSource(-12.0, 0.0, -10.0; Mw, strike=0, dip=60, rake=90, trise=1 / fc)
stations = [Station(@sprintf("S%02d", i), xs, 0.0, 0.0; surface=true) for (i, xs) in enumerate(-17:1.0:17)]
dt = 0.6 * dx / (maximum(m.vp) * sqrt(2))
nt = round(Int, 14 / dt)

# --- movie (the fields are shown on every s-th cell: about the pixels of the figure) -------------------------------------
s = max(1, round(Int, 0.04 / dx))
xd, zd = x[1:s:end], z[1:s:end]
air = (m.vp .== 0)[1:s:end, 1:s:end]
masked(A) = (B = Float32.(A[1:s:end, 1:s:end]); B[air] .= NaN; B)
ϕ = m.ϕ[1:s:end, 1:s:end]
surface_x = [st.x for st in stations]
surface_z = [z[findlast(m.vp[argmin(abs.(x .- st.x)), :] .> 0)] for st in stations]
set_theme!(fontsize=16)
fig = Figure(size=(1200, 900))
ax1 = Axis(fig[1, 1], title="S-wave velocity ($(res) m grid; temperature from the 100 m MTK model)", ylabel="height [km]")
hm = heatmap!(ax1, xd, zd, masked(m.vs), colormap=:lajolla, colorrange=(0, 4))
Colorbar(fig[1, 2], hm, label="vs [km/s]")
vz = Observable(fill(NaN32, length(xd), length(zd)))
c = Observable(1.0f-12)
title = Observable("")
ax2 = Axis(fig[2, 1], title=title, xlabel="distance W–E [km]", ylabel="height [km]")
hw = heatmap!(ax2, xd, zd, vz, colormap=:balance, colorrange=@lift((-$c, $c)))
Colorbar(fig[2, 2], hw, label="vz [m/s]", ticklabelspace=55, tickformat=v -> [@sprintf("%.2g", x) for x in v])
for ax in (ax1, ax2)
    contour!(ax, xd, zd, ϕ, levels=[0.1], color=:white, linewidth=1.5)
    scatter!(ax, surface_x, surface_z, marker=:dtriangle, color=:black, markersize=10)
    scatter!(ax, [source.x], [source.z], marker=:star5, color=:yellow, strokecolor=:black, strokewidth=1, markersize=22)
    limits!(ax, extrema(x)..., z[1], 3)
end
colsize!(fig.layout, 1, Aspect(1, (x[end] - x[1]) / (3 - z[1])))
resize_to_layout!(fig)
movie = VideoStream(fig; framerate=12)
function frame(wavefield, t)
    vz[] = masked(dropdims(wavefield.fields.V[3], dims=2))
    c[] = max(0.3f0 * maximum(v -> isnan(v) ? 0.0f0 : abs(v), vz[]), 1.0f-12)
    title[] = @sprintf("vertical velocity, t = %.2f s", t)
    recordframe!(movie)
end

@printf("Mw %.1f, Brune corner frequency %.1f Hz\n", Mw, fc)
println("$(length(x)) × $(length(z)) cells, dt = $(round(dt, sigdigits=3)) s, $nt steps on $dev"); flush(stdout)
t = @elapsed result = simulate(model; dt, nt, sources=source, stations, backend, stf, snapshot_every=round(Int, 0.1 / dt),
                               callback=frame, verbose=false, na=round(Int, 2.0 / dx))   # PML 2 km thick, as in the 100 m example
@printf("simulation with movie frames: %.0f s\n", t)
save(joinpath(@__DIR__, "Unzen_2D_waves_$(res)m.mp4"), movie)

# --- record section ----------------------------------------------------------------------------------------------------
molten = extrema(x[i] for i in eachindex(x) if any(>(0.1), m.ϕ[i, :]))
peak = [maximum(abs, result.v[:, c, n]) for n in eachindex(result.stations), c in 1:2]
fig2 = Figure(size=(1100, 700))
ax = Axis(fig2[1, 1], xlabel="time [s]", ylabel="station x [km]", title="vertical velocity, each trace normalized (shaded: above partially molten rock)")
ax2 = Axis(fig2[1, 2], xlabel="peak velocity [m/s]", title="peak amplitude")
for a in (ax, ax2)
    hspan!(a, molten..., color=(:orange, 0.2))
end
for (n, st) in enumerate(result.stations)
    v = result.v[:, 2, n]
    lines!(ax, result.t, st.x .+ 0.8 .* v ./ maximum(abs, v), color=:black, linewidth=0.5)
end
scatterlines!(ax2, peak[:, 2], surface_x, label="vertical")
scatterlines!(ax2, peak[:, 1], surface_x, label="horizontal")
axislegend(ax2, position=:rt)
linkyaxes!(ax, ax2)
colsize!(fig2.layout, 2, Relative(0.25))
save(joinpath(@__DIR__, "Unzen_2D_seismograms_$(res)m.png"), fig2)
println("saved Unzen_2D_waves_$(res)m.mp4 and Unzen_2D_seismograms_$(res)m.png")
