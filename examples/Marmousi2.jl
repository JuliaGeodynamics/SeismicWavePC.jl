# Marmousi2 (Martin et al., 2006), the classic benchmark model of seismic wave propagation: elastic 2D P-SV simulation of a marine
# shot (explosion 10 m below the sea surface, 5 Hz Ricker wavelet) recorded by a line of receivers 10 m below the sea surface.
#
#   julia -t auto --project=examples examples/Marmousi2.jl [dx in m (default 10)] [cuda|amdgpu|metal]
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Printf
include("backend.jl")
include("marmousi2_model.jl")

dx = something(tryparse(Float64, get(ARGS, 1, "")), 10.0)
model = marmousi2(dx)
println("Marmousi2: ", join(size(model), " × "), " cells of ", dx, " m")

source = MomentTensorSource(8.5, 0.0, -0.01, 1e13, (1, 0, 1, 0, 0, 0) ./ sqrt(2); trise=0.2)   # explosion, 5 Hz Ricker (stf = :ricker)
stations = [Station(@sprintf("R%04d", i), x, 0.0, -0.01) for (i, x) in enumerate(0.5:0.025:16.5)]
dt = 0.5 * dx / 1000 / (4.7 * sqrt(2))                          # Courant number 0.5 (vp ≤ 4.7 km/s)
nt = round(Int, 5.0 / dt)                                       # 5 s

# movie, made while the simulation runs: the waves (vertical velocity) over the velocity model, transparent where they are weak
x, z = model.x.val[:, 1, 1], model.z.val[1, 1, :]
air = model.fields.vp[:, 1, :] .== 0
masked(A) = (B = Float32.(A); B[air] .= NaN; B)
waves = [RGBAf(c.r, c.g, c.b, clamp(3abs(2t - 1), 0, 1)) for (c, t) in zip(Makie.to_colormap(:balance), range(0, 1, length=256))]
vz = Observable(fill(NaN32, size(air)))
clim = Observable(1.0f-12)
label = Observable("")
mfig = Figure(size=(1400, 400))
max_ = Axis(mfig[1, 1], title=label, xlabel="x [km]", ylabel="z [km]", aspect=DataAspect())
heatmap!(max_, x, z, masked(model.fields.vp[:, 1, :]), colormap=:grays, colorrange=(1.2, 4.7))
heatmap!(max_, x, z, vz, colormap=waves, colorrange=@lift((-$clim, $clim)))
scatter!(max_, [source.x], [source.z], marker=:star5, color=:yellow, markersize=14)
limits!(max_, extrema(x)..., extrema(z)...)
resize_to_layout!(mfig)
movie = VideoStream(mfig; framerate=15)
frames = Matrix{Float32}[]
function frame(w, t)
    v = masked(w.fields.V[3][:, 1, :])
    push!(frames, v)
    a = sort!(filter(!isnan, abs.(vec(v))))
    clim[] = max(a[round(Int, 0.995 * length(a))], 1.0f-12)          # colour scale: 99.5 % of the amplitudes
    vz[] = v
    label[] = @sprintf("Marmousi2: vertical velocity over the P-wave velocity, t = %.2f s", t)
    recordframe!(movie)
end

result = simulate(model; dt, nt, sources=source, stations, stf=:ricker, backend, snapshot_every=round(Int, 0.05 / dt), callback=frame)
save(joinpath(@__DIR__, "Marmousi2.mp4"), movie)

# model, wavefield snapshots and shot gather
water_air(A) = (B = copy(A); B[model.fields.vp[:, 1, :] .== 0] .= NaN; B)
fig = Figure(size=(1400, 1100))
ax = Axis(fig[1, 1], title="Marmousi2: P-wave velocity", ylabel="z [km]")
hm = heatmap!(ax, x, z, water_air(model.fields.vp[:, 1, :]), colormap=:turbo)
Colorbar(fig[1, 2], hm, label="vp [km/s]")
scatter!(ax, [source.x], [source.z], marker=:star5, color=:white, markersize=18)
ax2 = Axis(fig[2, 1], title=@sprintf("vertical velocity at t = %.1f s", 2.0), ylabel="z [km]")
f = frames[min(41, end)]
c = 0.2 * maximum(v -> isnan(v) ? 0.0f0 : abs(v), f)
heatmap!(ax2, x, z, water_air(f), colormap=:balance, colorrange=(-c, c))
ax3 = Axis(fig[3, 1], title="shot gather: vertical velocity 10 m below the sea surface (gain t²)", xlabel="x [km]", ylabel="t [s]", yreversed=true)
g = result.v[:, 2, :]' .* (result.t' .^ 2)
c = 0.05 * maximum(abs, g)
heatmap!(ax3, [s.x for s in result.stations], result.t, g, colormap=:grays, colorrange=(-c, c))
for a in (ax, ax2)
    a.aspect = DataAspect()
end
save(joinpath(@__DIR__, "Marmousi2.png"), fig; px_per_unit=1)
println("saved examples/Marmousi2.png and examples/Marmousi2.mp4")
