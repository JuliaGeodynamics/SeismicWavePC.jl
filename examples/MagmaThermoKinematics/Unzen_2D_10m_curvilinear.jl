# The high-resolution Unzen example (README) on the curvilinear grid: the model of Unzen_2D_10m_model.jl (10 m) on a
# terrain-following mesh whose top row is the free surface of the 100 m MTK run (Zsurf, monotone cubic between its columns),
# simulated with the curvilinear SBP-SAT solver (summation-by-parts finite differences, simultaneous approximation terms for
# the free surface, PML at the sides and the bottom). The solver handles solids only: partially molten mush that the
# staggered example treats as fluid (vs = 0) has vs = 1 km/s here. Movie and record section as Unzen_2D_10m_waves.jl;
# 4001 × 2124 nodes, 32 926 time steps: 4.3 minutes on an Apple M-series GPU (the movie frames 22 s).
#
#   julia -t auto --project=examples/MagmaThermoKinematics examples/MagmaThermoKinematics/Unzen_2D_10m_curvilinear.jl [cpu|cuda|amdgpu|metal]
#
# (run Unzen_2D_10m_model.jl first; the GPU package added to the environment). The GIF of the README:
#   ffmpeg -i Unzen_2D_waves_10m_curvilinear.mp4 -vf "fps=8,scale=560:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" Unzen_2D_waves_10m_curvilinear.gif
using SeismicWavePC, GeophysicalModelGenerator, Serialization, JLD2, CairoMakie, Printf
include(joinpath(@__DIR__, "..", "backend.jl"))
E = joinpath(@__DIR__, "HighRes2D_100m")
m = deserialize(joinpath(E, "Unzen_10m.jls"))
x, z = m.x, m.z
h = x[2] - x[1]
vs_floor = 1.0
mtk = load(joinpath(E, "final.jld2"))
xc, zs = mtk["x"] ./ 1e3, mtk["Zsurf"] ./ 1e3
# monotone cubic (PCHIP, Fritsch–Carlson) through the 100 m surface heights: smooth, no overshoot
dc = xc[2] - xc[1]
δ = diff(zs) ./ dc
dk = [k == 1 ? δ[1] : k == length(zs) ? δ[end] : (δ[k-1] * δ[k] <= 0 ? 0.0 : 2 / (1 / δ[k-1] + 1 / δ[k])) for k in eachindex(zs)]
function surf(xv)
    f = clamp((xv - xc[1]) / dc + 1, 1, length(xc)); i = min(floor(Int, f), length(xc) - 1); t = f - i
    h00, h10, h01, h11 = 2t^3 - 3t^2 + 1, t^3 - 2t^2 + t, -2t^3 + 3t^2, t^3 - t^2
    return h00 * zs[i] + h10 * dc * dk[i] + h01 * zs[i+1] + h11 * dc * dk[i+1]
end
rock0 = m.vp .> 0
toprock = [findlast(view(rock0, i, :)) for i in eachindex(x)]
cellof(xv, zv) = (i = clamp(round(Int, (xv - x[1]) / h) + 1, 1, length(x)); (i, min(clamp(round(Int, (zv - z[1]) / h) + 1, 1, length(z)), toprock[i])))
mesh = curvilinear_mesh(range(x[1], x[end], length=length(x)), [surf]; dz=h, bottom=z[1])
X, Z = mesh.x.val, mesh.z.val
idc = [cellof(X[I], Z[I]) for I in CartesianIndices(X)]
pick(A) = [Float64(A[i, k]) for (i, k) in idc]
vsn, ϕn = max.(pick(m.vs), vs_floor), pick(m.ϕ)
cm = addfield(mesh, (; rho=pick(m.rho), vp=pick(m.vp), vs=vsn, Qs=pick(m.Qs), Qp=2 .* pick(m.Qs)))
# --- source and stations (as the README)
Mw, Δσ, β = 3.0, 3.0e6, 3500.0
M0 = 10^(1.5Mw + 9.1)
fc = 0.37β * (16Δσ / (7M0))^(1 / 3)
ωc = 2π * fc
brune(τ) = τ < 0 ? 0.0 : ωc^2 * τ * exp(-ωc * τ)
ts = 0:1.0e-4:1.0
kup = [τ <= 0.1 ? sin(π * τ / 0.1)^3 : 0.0 for τ in ts]
stf = SampledSTF(ts, [sum(brune(τ - u) * w for (u, w) in zip(ts, kup)) for τ in ts] ./ sum(kup))
source = DoubleCoupleSource(-12.0, 0.0, -10.0; Mw, strike=0, dip=60, rake=90, trise=1 / fc)
stations = [Station(@sprintf("S%02d", i), xs, 0.0, 0.0; surface=true) for (i, xs) in enumerate(-17:1.0:17)]
# --- the display grid (as the README movie: every s-th cell, here pixels of 40 m): every pixel takes the two nodes of its
# mesh column around it (row j, weight w; j = 0 in the air), found once: a frame is then one gather
s = 4
xd, zd = x[1:s:end], z[1:s:end]
jw = map(CartesianIndices((length(xd), length(zd)))) do I
    col, zv = view(Z, s * (I[1] - 1) + 1, 1, :), zd[I[2]]
    (col[1] <= zv <= col[end]) || return (0, 0.0f0)
    j = clamp(searchsortedlast(col, zv), 1, length(col) - 1)
    return (j, Float32((zv - col[j]) / (col[j+1] - col[j])))
end
todisplay(A) = [j == 0 ? NaN32 : (1 - w) * Float32(A[s*(I[1]-1)+1, j]) + w * Float32(A[s*(I[1]-1)+1, j+1])   # A: (nx, nz) on the mesh
                for (I, (j, w)) in pairs(jw)]
ztop = Z[:, 1, end]
surface_x = [st.x for st in stations]
surface_z = [surf(st.x) for st in stations]
set_theme!(fontsize=16)
fig = Figure(size=(1200, 900))
ax1 = Axis(fig[1, 1], title="S-wave velocity, curvilinear grid on the MTK surface (10 m; molten mush: vs = 1 km/s)", ylabel="height [km]")
hm = heatmap!(ax1, xd, zd, todisplay(vsn[:, 1, :]), colormap=:lajolla, colorrange=(0, 4))
Colorbar(fig[1, 2], hm, label="vs [km/s]")
ϕd = todisplay(ϕn[:, 1, :])
vz = Observable(fill(NaN32, length(xd), length(zd)))
c = Observable(1.0f-12)
title = Observable("")
ax2 = Axis(fig[2, 1], title=title, xlabel="distance W–E [km]", ylabel="height [km]")
hw = heatmap!(ax2, xd, zd, vz, colormap=:balance, colorrange=@lift((-$c, $c)))
Colorbar(fig[2, 2], hw, label="vz [m/s]", ticklabelspace=55, tickformat=v -> [@sprintf("%.2g", x) for x in v])
for ax in (ax1, ax2)
    contour!(ax, xd, zd, replace(ϕd, NaN32 => 0.0f0), levels=[0.1], color=:white, linewidth=1.5)
    lines!(ax, X[:, 1, end], ztop, color=:black, linewidth=1)          # the free surface (the top row of the mesh)
    scatter!(ax, surface_x, surface_z, marker=:dtriangle, color=:black, markersize=10)
    scatter!(ax, [source.x], [source.z], marker=:star5, color=:yellow, strokecolor=:black, strokewidth=1, markersize=22)
    limits!(ax, extrema(x)..., z[1], 3)
end
colsize!(fig.layout, 1, Aspect(1, (x[end] - x[1]) / (3 - z[1])))
resize_to_layout!(fig)
movie = VideoStream(fig; framerate=12)
tframes = Ref(0.0)
frame(wavefield, t) = tframes[] += @elapsed begin
    vz[] = todisplay(view(wavefield.fields.V[3], :, 1, :))
    c[] = max(0.3f0 * maximum(v -> isnan(v) ? 0.0f0 : abs(v), vz[]), 1.0f-12)
    title[] = @sprintf("vertical velocity, t = %.2f s (curvilinear grid)", t)
    recordframe!(movie)
end
t = @elapsed r = simulate(cm; tmax=14.0, sources=source, stations, backend, stf, verbose=false, absorber=:pml,
                          na=round(Int, 2.0 / h), snapshot_every=235, callback=frame)
@printf("curvilinear grid on %s with movie frames: %.0f s (frames %.0f s), %d steps (dt %.3g s)\n", DEVICE, t, tframes[], length(r.t), r.t[2] - r.t[1])
save(joinpath(@__DIR__, "Unzen_2D_waves_10m_curvilinear.mp4"), movie)

# --- record section ----------------------------------------------------------------------------------------------------
molten = extrema(m.x[i] for i in eachindex(m.x) if any(>(0.1), m.ϕ[i, :]))
peak = [maximum(abs, r.v[:, c, n]) for n in eachindex(surface_x), c in 1:2]
set_theme!(fontsize=16)
fig2 = Figure(size=(1100, 700))
ax = Axis(fig2[1, 1], xlabel="time [s]", ylabel="station x [km]", title="vertical velocity, curvilinear grid, each trace normalized (shaded: above partially molten rock)")
ax2 = Axis(fig2[1, 2], xlabel="peak velocity [m/s]", title="peak amplitude")
for a in (ax, ax2)
    hspan!(a, molten..., color=(:orange, 0.2))
end
for (n, xs) in enumerate(surface_x)
    v = r.v[:, 2, n]
    lines!(ax, r.t, xs .+ 0.8 .* v ./ maximum(abs, v), color=:black, linewidth=0.5)
end
scatterlines!(ax2, peak[:, 2], surface_x, label="vertical")
scatterlines!(ax2, peak[:, 1], surface_x, label="horizontal")
axislegend(ax2, position=:rt)
linkyaxes!(ax, ax2)
colsize!(fig2.layout, 2, Relative(0.25))
save(joinpath(@__DIR__, "Unzen_2D_seismograms_10m_curvilinear.png"), fig2)
println("saved Unzen_2D_waves_10m_curvilinear.mp4 and Unzen_2D_seismograms_10m_curvilinear.png")
