# Reverse-time migration (RTM) of Marmousi2: marine shots (explosions 10 m below the sea surface, 5 Hz Ricker wavelet) recorded
# by a line of receivers, migrated in a smooth background model. The image is the gradient of the least-squares misfit with
# respect to vp in the background model (cross-correlation of the source wavefield with the back-propagated reflections),
# divided by the illumination (the energy of the source wavefield) and filtered with a Laplacian.
#
#   julia -t auto --project=examples examples/Marmousi2_RTM.jl [cpu|cuda|amdgpu|metal] [number of shots (default 40)]
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Printf
include("backend.jl")
include("marmousi2_model.jl")

nshots = something(tryparse(Int, get(ARGS, 2, "")), 40)
dx = 10.0
model = marmousi2(dx)
x, z = model.x.val[:, 1, 1], model.z.val[1, 1, :]
rock = model.fields.vs[:, 1, :] .> 0

"Gaussian smoothing (σ in cells) of `A` inside `mask` (normalized: values outside the mask do not leak in)."
function smooth(A, mask, σ)
    k = exp.(-(-3σ:3σ) .^ 2 ./ (2σ^2))
    conv(B, d) = mapslices(v -> [sum(k[j+3σ+1] * v[clamp(i + j, 1, length(v))] for j in -3σ:3σ) for i in eachindex(v)], B; dims=d)
    num, den = conv(conv(A .* mask, 1), 2), conv(conv(Float64.(mask), 1), 2)
    return ifelse.(mask, num ./ max.(den, eps()), A)
end
# background model: 150 m smoothing in the rock; the water layer and the sea floor stay sharp
f = model.fields
background = CartData(model.x.val, model.y.val, model.z.val,
                      map(A -> reshape(smooth(A[:, 1, :], rock, 15), size(A)), (; rho=f.rho, vp=f.vp, vs=f.vs)))

# acquisition: shots along the line, each recorded by the receivers within 3 km (towed streamers), 3.5 s
shots = [MomentTensorSource(xs, 0.0, -0.01, 1e13, (1, 0, 1, 0, 0, 0) ./ sqrt(2); trise=0.2) for xs in range(0.5, 16.5, length=nshots)]
spread(s) = [Station(@sprintf("R%05.0f", 1000xr), xr, 0.0, -0.01) for xr in s.x-3:0.025:s.x+3 if 0.1 <= xr <= 16.9]
dt = 0.5 * dx / 1000 / (4.7 * sqrt(2))
kw = (; dt, nt=round(Int, 3.5 / dt), stf=:ricker, backend)

t_obs = @elapsed observed = [simulate(model; kw..., sources=s, stations=spread(s), verbose=false) for s in shots]
@printf("%d shots modelled in the true model: %.1f s\n", nshots, t_obs)
# migrate the reflections: per receiver only the part of the trace after the direct wave through the water (earlier: waves
# refracted in the rock, whose travel times the smooth model does not reproduce)
function migrate(s, obs)
    window = [(0.3 + abs(st.x - s.x) / 1.5, Inf) for st in obs.stations]
    return misfit_gradient(background, obs; kw..., sources=s, misfit=L2Misfit(; window), illumination=true, mute=(sources=0.3,))[2]
end
t_rtm = @elapsed g = reduce((a, b) -> CartData(a.x.val, a.y.val, a.z.val, map(+, a.fields, b.fields)), migrate.(shots, observed))
@printf("migration (adjoint-state gradients of all shots, with illumination): %.1f s\n", t_rtm)

# image: reflectivity ~ -∂J/∂vp; raw and divided by the illumination; Laplacian against the low-wavenumber RTM artefacts
lap(A) = [i in (1, size(A, 1)) || k in (1, size(A, 2)) ? 0.0 :
          4A[i, k] - A[i-1, k] - A[i+1, k] - A[i, k-1] - A[i, k+1] for i in axes(A, 1), k in axes(A, 2)]
inside = rock .& [24 < i <= length(x) - 24 && k > 24 for i in eachindex(x), k in eachindex(z)]   # without the absorbing layers
L = g.fields.illumination[:, 1, :]
Lmax = sort(L[inside])[round(Int, 0.99 * count(inside))]      # the illumination is singular at the sources: clip it
Lc = min.(L, Lmax)
raw = lap(-g.fields.vp[:, 1, :]) .* inside
compensated = lap(-g.fields.vp[:, 1, :] ./ (Lc .+ 0.01Lmax)) .* inside

fig = Figure(size=(1400, 1050))
masked(A) = (B = Float32.(A); B[.!rock] .= NaN; B)
clip(A) = (a = sort(abs.(A[inside])); a[round(Int, 0.99 * length(a))])
ax = [Axis(fig[i, 1], title=t, ylabel="z [km]", xlabel=i == 4 ? "x [km]" : "", aspect=DataAspect())
      for (i, t) in enumerate(("Marmousi2: vp [km/s]", "background model for the migration: vp [km/s]",
                              "RTM image ($nshots shots)", "RTM image, compensated for the illumination"))]
hm = heatmap!(ax[1], x, z, masked(f.vp[:, 1, :]), colormap=:viridis, colorrange=(1.5, 4.7))
heatmap!(ax[2], x, z, masked(background.fields.vp[:, 1, :]), colormap=:viridis, colorrange=(1.5, 4.7))
scatter!(ax[2], [s.x for s in shots], [s.z for s in shots], marker=:star5, color=:red, markersize=8)
Colorbar(fig[1:2, 2], hm)
heatmap!(ax[3], x, z, raw, colormap=:grays, colorrange=(-1, 1) .* clip(raw))
heatmap!(ax[4], x, z, compensated, colormap=:grays, colorrange=(-1, 1) .* clip(compensated))
foreach(a -> limits!(a, extrema(x)..., -3.5, 0.0), ax)
save(joinpath(@__DIR__, "Marmousi2_RTM.png"), fig)
println("saved examples/Marmousi2_RTM.png")
