# Movie of the 3D volcano example: the volcano with the quadrant x > 0, y < 0 cut out (faces through the summit and the
# magma chamber), coloured by the vertical velocity over grey terrain and velocity structure, rendered while the simulation
# runs (callback). Writes examples/3D_volcano.mp4.
#
#   julia -t auto --project=examples examples/3D_volcano_movie.jl [cuda|amdgpu|metal]
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Printf
include("backend.jl")

proj = ProjectionPoint(Lon=14.43, Lat=40.82)
x, y, z = range(-8, 8, length=160), range(-8, 8, length=160), range(-10, 2.5, length=126)
model = CartData(xyz_grid(x, y, z))
Phase, Temp = zeros(Int, size(model)), zeros(size(model))
add_layer!(Phase, Temp, model; zlim=(-20, 0), phase=ConstantPhase(1))
add_volcano!(Phase, Temp, model; volcanic_phase=2, center=(0, 0, 0), height=2.0, radius=6.0, crater=0.5, base=0.0)
add_sphere!(Phase, Temp, model; cen=(0, 0, -4), radius=1.5, phase=ConstantPhase(3))
rho_ = Dict(0 => 1.0, 1 => 2700.0, 2 => 2300.0, 3 => 2500.0)
vp_ = Dict(0 => 0.0, 1 => 5.5, 2 => 3.0, 3 => 3.5)
vs_ = Dict(0 => 0.0, 1 => 3.2, 2 => 1.6, 3 => 1.0)
Q_ = Dict(0 => 10.0, 1 => 300.0, 2 => 60.0, 3 => 20.0)
f(d) = getindex.(Ref(d), Phase)
model = addfield(model, (; rho=f(rho_), vp=f(vp_), vs=f(vs_), Qp=2 .* f(Q_), Qs=f(Q_)))
# rise time 1 s (3D_volcano.jl: 0.3 s): frequencies up to ~2 Hz, so that the slow volcanic rock and the magma chamber have
# at least 5 cells per wavelength (shorter rise times give grid noise there)
source = DoubleCoupleSource(14.45, 40.80, -3.0, proj; Mw=3.5, strike=120, dip=60, rake=-90, trise=1.0)

# geometry: the free surface (top solid cell) of every column; the quadrant x > 0, y < 0 is cut out
solid = model.fields.vs .> 0
ks = [findlast(view(solid, i, j, :)) for i in axes(solid, 1), j in axes(solid, 2)]
ix0, jy0 = findfirst(>=(0), x), findfirst(>=(0), y)
incut(i, j) = i > ix0 && j < jy0
topo = [incut(i, j) ? NaN : z[ks[i, j]] for i in eachindex(x), j in eachindex(y)]
faces = [[CartesianIndex(ix0, j, k) for j in 1:jy0, k in eachindex(z)],              # cut face x = 0
         [CartesianIndex(i, jy0, k) for i in ix0:length(x), k in eachindex(z)],      # cut face y = 0
         [CartesianIndex(i, 1, k) for i in 1:ix0, k in eachindex(z)],                # outer faces
         [CartesianIndex(length(x), j, k) for j in jy0:length(y), k in eachindex(z)]]

# colours: waves (balance colour map, transparent where weak) over grey terrain and grey structure (vs)
wavecol = Makie.to_colormap(:balance)
function blend(base, v, clim)
    t = clamp((v / clim + 1) / 2, 0, 1)
    c = wavecol[round(Int, 1 + t * (length(wavecol) - 1))]
    α = clamp(3abs(2t - 1), 0, 1)
    return RGBAf(α * c.r + (1 - α) * base.r, α * c.g + (1 - α) * base.g, α * c.b + (1 - α) * base.b, base.alpha)
end
h = [z[ks[i, j]] for i in eachindex(x), j in eachindex(y)]
shade = [RGBAf(g, g * 0.97, g * 0.9, 1) for g in 0.55 .+ 0.35 .* (h .- minimum(h)) ./ (maximum(h) - minimum(h))]
grey(vs) = vs == 0 ? RGBAf(0, 0, 0, 0) : (g = 0.35 + 0.5 * (vs - 1) / 2.2; RGBAf(g, g, g, 1))
structure = [grey.(model.fields.vs[F]) for F in faces]

topcol, facecol, label = Observable(shade), [Observable(c) for c in structure], Observable("")
fig = Figure(size=(900, 620), backgroundcolor=:white)
ax = Axis3(fig[1, 1], aspect=(1, 1, 0.65), azimuth=-0.3π, elevation=0.16π, title=label, viewmode=:fitzoom,
           xlabel="x [km]", ylabel="y [km]", zlabel="z [km]", protrusions=(30, 30, 30, 40))
surface!(ax, x, y, topo; color=topcol, shading=NoShading)
for (F, c) in zip(faces, facecol)
    surface!(ax, [x[I[1]] for I in F], [y[I[2]] for I in F], [z[I[3]] for I in F]; color=c, shading=NoShading)
end
scatter!(ax, [source.x], [source.y], [source.z], marker=:star5, color=:yellow, strokecolor=:black, strokewidth=1, markersize=22)
limits!(ax, extrema(x)..., extrema(y)..., -10, 2.5)
movie = VideoStream(fig; framerate=15)
function frame(w, t)
    vz = w.fields.V[3]
    top = [vz[i, j, ks[i, j]] for i in eachindex(x), j in eachindex(y)]
    a = sort!(abs.([vec(top); [vz[I] for F in faces for I in F]]))
    clim = max(a[round(Int, 0.995 * length(a))], 1e-12)
    topcol[] = blend.(shade, top, clim)
    for (F, s, c) in zip(faces, structure, facecol)
        c[] = [s[n].alpha == 0 ? s[n] : blend(s[n], vz[F[n]], clim) for n in CartesianIndices(F)]
    end
    label[] = @sprintf("Mw 3.5 earthquake below a volcano with a magma chamber: vertical velocity, t = %.2f s", t)
    recordframe!(movie)
end

simulate(model; dt=0.006, nt=1200, sources=source, backend, snapshot_every=10, callback=frame)
save(joinpath(@__DIR__, "3D_volcano.mp4"), movie)
println("saved examples/3D_volcano.mp4")
