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

frames = Matrix{Float32}[]
result = simulate(model; dt, nt, sources=source, stations, stf=:ricker, backend, snapshot_every=round(Int, 1.0 / dt),
                  callback=(w, t) -> push!(frames, Float32.(w.fields.V[3][:, 1, :])))

# model, wavefield snapshots and shot gather
x, z = model.x.val[:, 1, 1], model.z.val[1, 1, :]
water_air(A) = (B = copy(A); B[model.fields.vp[:, 1, :] .== 0] .= NaN; B)
fig = Figure(size=(1400, 1100))
ax = Axis(fig[1, 1], title="Marmousi2: P-wave velocity", ylabel="z [km]")
hm = heatmap!(ax, x, z, water_air(model.fields.vp[:, 1, :]), colormap=:turbo)
Colorbar(fig[1, 2], hm, label="vp [km/s]")
scatter!(ax, [source.x], [source.z], marker=:star5, color=:white, markersize=18)
ax2 = Axis(fig[2, 1], title=@sprintf("vertical velocity at t = %.1f s", 2.0), ylabel="z [km]")
f = frames[min(3, end)]
c = 0.2 * maximum(abs, f)
heatmap!(ax2, x, z, water_air(f), colormap=:balance, colorrange=(-c, c))
ax3 = Axis(fig[3, 1], title="shot gather: vertical velocity 10 m below the sea surface (gain t²)", xlabel="x [km]", ylabel="t [s]", yreversed=true)
g = result.v[:, 2, :]' .* (result.t' .^ 2)
c = 0.05 * maximum(abs, g)
heatmap!(ax3, [s.x for s in result.stations], result.t, g, colormap=:grays, colorrange=(-c, c))
for a in (ax, ax2)
    a.aspect = DataAspect()
end
save(joinpath(@__DIR__, "Marmousi2.png"), fig; px_per_unit=1)
println("saved examples/Marmousi2.png")
