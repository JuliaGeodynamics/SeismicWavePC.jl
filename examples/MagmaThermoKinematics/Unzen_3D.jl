# Seismic waves through a 3D magmatic system below an Unzen-like volcano, with a movie made while the simulation runs.
#
#   julia -t auto --project=examples/MagmaThermoKinematics examples/MagmaThermoKinematics/Unzen_3D.jl [cpu|cuda|metal]
#
# 1. Geometry: a 1.4 km high volcano on a low plain; 30 km × 30 km × 22 km, 200 m cells. (A shallow sea around it would
#    need a finer grid: a water layer of one or two cells gives artefacts.)
# 2. Temperature: a linear geotherm plus superposed Gaussian anomalies, a simplified version of the plumbing system inferred
#    below Unzen: a deep mush reservoir west of the volcano, a body rising eastwards from it, a reservoir below the summit.
# 3. Density, P and S velocities and melt fraction from MAGEMin phase diagrams (as in Unzen_2D.jl); low Q in partially
#    molten rock.
# 4. A Mw 4 earthquake; movie of the vertical velocity in a cut-away view (Unzen_3D.mp4).
using SeismicWavePC, GeophysicalModelGenerator, MAGEMin_C, CairoMakie, Printf
include(joinpath(@__DIR__, "..", "backend.jl"))

# --- Geometry -------------------------------------------------------------------------------------------------------------
dx = 0.2                                                         # km
x, y, z = range(-15, 15, step=dx), range(-15, 15, step=dx), range(-20, 2, step=dx)
X, Y, Z = xyz_grid(x, y, z)
r = hypot.(X, Y .* 1.3)                                          # slightly elongated edifice (E–W)
top = @. 0.1 + 1.4 * exp(-(r / 4.5)^2)                          # surface height [km]
air = Z .> top
water = falses(size(Z))                                          # (no sea)

# --- Temperature: linear geotherm + Gaussian anomalies [°C] --------------------------------------------------------------
gauss(x0, y0, z0, wx, wy, wz) = @. exp(-((X - x0) / wx)^2 - ((Y - y0) / wy)^2 - ((Z - z0) / wz)^2)
depth = max.(top .- Z, 0)
T = @. 15 + 30 * depth                                           # 30 °C/km
T .+= 720 .* gauss(-7, 0, -11, 4.0, 3.0, 2.0)                    # deep mush reservoir west of the volcano
T .+= 650 .* gauss(-3.5, 0, -8.5, 3.0, 1.5, 1.5)                 # body rising eastwards
T .+= 680 .* gauss(0, 0, -6, 1.8, 1.8, 1.2)                      # reservoir below the summit
T .+= 600 .* gauss(0.3, 0, -2.5, 0.6, 0.6, 1.0)                  # small shallow body (conduit)
T[air .|| water] .= 15

# --- Rock physics with MAGEMin (as in Unzen_2D.jl) ------------------------------------------------------------------------
Xoxides = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3", "K2O", "Na2O", "TiO2", "Cr2O3", "H2O"]
dacite = [64.5, 15.9, 5.0, 2.4, 3.2, 1.4, 2.3, 3.9, 0.7, 0.0, 2.0]   # wt%: Unzen-like dacite with 2 % water
basalt = [48.4, 15.2, 11.6, 10.1, 6.7, 1.6, 0.6, 1.9, 0.7, 0.0, 0.5] # wt%: mafic lower crust
Ps, Ts = 0.0:0.25:7.0, 0.0:25.0:1200.0                         # pressure [kbar] and temperature [°C] of the tables
function properties(o)
    ϕ = o.frac_M_vol
    vp_melt = ϕ > 0 ? sqrt(o.bulkModulus_M * 1.0e9 / o.rho_M) / 1.0e3 : 0.0
    return (o.rho, max(o.Vp_cor, vp_melt), o.Vs_cor, ϕ)
end
data = Initialize_MAGEMin("ig", verbose=false)
tables = Dict(c => properties.(multi_point_minimization([P for P in Ps, T in Ts], [T for P in Ps, T in Ts], data;
                                                        X=c, Xoxides, sys_in="wt", seismic_cor=true, progressbar=false))
              for c in (dacite, basalt))
Finalize_MAGEMin(data)
function lookup(tab, P, T)
    fi, fj = clamp((P - first(Ps)) / step(Ps) + 1, 1, length(Ps)), clamp((T - first(Ts)) / step(Ts) + 1, 1, length(Ts))
    i, j = min(floor(Int, fi), length(Ps) - 1), min(floor(Int, fj), length(Ts) - 1)
    u, v = fi - i, fj - j
    return (1 - u) * (1 - v) .* tab[i, j] .+ u * (1 - v) .* tab[i+1, j] .+ (1 - u) * v .* tab[i, j+1] .+ u * v .* tab[i+1, j+1]
end

trise = 10dx                                                     # rise time [s]: frequencies up to about 1 Hz
vs_min = 5dx * 2 / trise                                         # slower S waves are not resolved: treated as molten
props = map(CartesianIndices(T)) do I
    air[I] && return (1.0, 0.0, 0.0, 0.0, 10.0)
    water[I] && return (1030.0, 1.5, 0.0, 0.0, 1000.0)
    P = 2700 * 9.81 * 1.0e3 * depth[I] / 1.0e8                   # lithostatic pressure [kbar]
    ρ, vp, vs, ϕ = lookup(tables[Z[I] < -15 ? basalt : dacite], P, T[I])
    vs = vs < vs_min ? 0.0 : vs
    return (ρ, vp, vs, ϕ, max(200.0 / (1 + 100ϕ), 10.0))         # Qs falls to 10 for 10 % melt or more
end
field(n) = getindex.(props, n)
model = CartData(X, Y, Z, (; rho=field(1), vp=field(2), vs=field(3), Qs=field(5), Qp=2 .* field(5), ϕ=field(4), T))
@printf("model: %s cells, melt fraction up to %.2f, vs %.2f–%.2f km/s in solid rock\n", join(size(model), " × "),
        maximum(model.fields.ϕ), extrema(filter(>(0), model.fields.vs))...)

# --- Earthquake ------------------------------------------------------------------------------------------------------------
source = DoubleCoupleSource(-9.0, -4.0, -6.0; Mw=4.0, strike=90, dip=70, rake=-90, trise)
dt = 0.6 * dx / (maximum(model.fields.vp) * sqrt(3))
nt = round(Int, 9 / dt)

# --- Movie: the block with the quadrant x > -6.5 km, y < 0 cut out (faces through the magmatic system) -------------------
solid = .!(air .|| water)
ix0, jy0 = findfirst(>=(-6.5), x), findfirst(>=(0), y)
incut(i, j) = i > ix0 && j < jy0
ktop = [findlast(view(.!air, i, j, :)) for i in eachindex(x), j in eachindex(y)]   # top cell (rock or sea) of each column
topo = [incut(i, j) ? NaN : z[ktop[i, j]] for i in eachindex(x), j in eachindex(y)]
faces = [[CartesianIndex(ix0, j, k) for j in 1:jy0, k in eachindex(z)],              # cut face x = -6.5 km
         [CartesianIndex(i, jy0, k) for i in ix0:length(x), k in eachindex(z)],      # cut face y = 0
         [CartesianIndex(i, 1, k) for i in 1:ix0, k in eachindex(z)],                # outer face y = -15 km
         [CartesianIndex(length(x), j, k) for j in jy0:length(y), k in eachindex(z)]] # outer face x = 15 km
wavecol = Makie.to_colormap(:balance)
function blend(base, v, clim)
    t = clamp((v / clim + 1) / 2, 0, 1)
    c = wavecol[round(Int, 1 + t * (length(wavecol) - 1))]
    α = clamp(3abs(2t - 1), 0, 1)
    return RGBAf(α * c.r + (1 - α) * base.r, α * c.g + (1 - α) * base.g, α * c.b + (1 - α) * base.b, base.alpha)
end
land(h) = (g = 0.5 + 0.35 * h / 1.5; RGBAf(g, g * 0.97, g * 0.9, 1))
seacol = RGBAf(0.55, 0.7, 0.85, 1)
shade = [water[i, j, ktop[i, j]] ? seacol : land(z[ktop[i, j]]) for i in eachindex(x), j in eachindex(y)]
function rock(I)                                                 # faces: grey by vs, orange where partially molten
    air[I] && return RGBAf(0, 0, 0, 0)
    water[I] && return seacol
    model.fields.ϕ[I] > 0.1 && return RGBAf(0.95, 0.55, 0.2, 1)
    g = 0.3 + 0.55 * clamp((model.fields.vs[I] - 1) / 2.8, 0, 1)
    return RGBAf(g, g, g, 1)
end
structure = [rock.(F) for F in faces]
visible = [I for F in faces for I in F if solid[I] && model.fields.ϕ[I] <= 0.1]   # for the colour scale
topcol, facecol, label = Observable(shade), [Observable(c) for c in structure], Observable("")
set_theme!(fontsize=15)
fig = Figure(size=(820, 640), backgroundcolor=:white)
ax = Axis3(fig[1, 1], aspect=(1, 1, 0.55), azimuth=-0.3π, elevation=0.16π, title=label, viewmode=:fitzoom,
           xlabel="x (W–E) [km]", ylabel="y (S–N) [km]", zlabel="z [km]", protrusions=(40, 20, 20, 50))
surface!(ax, x, y, topo; color=topcol, shading=NoShading)
for (F, c) in zip(faces, facecol)
    surface!(ax, [x[I[1]] for I in F], [y[I[2]] for I in F], [z[I[3]] for I in F]; color=c, shading=NoShading)
end
scatter!(ax, [source.x], [source.y], [source.z], marker=:star5, color=:yellow, strokecolor=:black, strokewidth=1, markersize=22)
limits!(ax, extrema(x)..., extrema(y)..., z[1], 2)
Label(fig[2, 1], "orange: partially molten rock (> 10 % melt, MAGEMin); star: earthquake", fontsize=13)
movie = VideoStream(fig; framerate=15)
function frame(w, t)
    vz = w.fields.V[3]
    top = [vz[i, j, ktop[i, j]] for i in eachindex(x), j in eachindex(y)]
    a = sort!(abs.([vec(top); vz[visible]]))
    clim = max(a[round(Int, 0.995 * length(a))], 1e-12)         # colour scale: surface and solid rock
    topcol[] = blend.(shade, top, clim)
    for (F, s, c) in zip(faces, structure, facecol)
        c[] = [s[n].alpha == 0 ? s[n] : blend(s[n], vz[F[n]], clim) for n in CartesianIndices(F)]
    end
    label[] = @sprintf("Mw 4 earthquake below an Unzen-like volcano: vertical velocity, t = %.1f s", t)
    recordframe!(movie)
end

simulate(model; dt, nt, sources=source, backend, snapshot_every=round(Int, 0.1 / dt), callback=frame)
save(joinpath(@__DIR__, "Unzen_3D.mp4"), movie)
println("saved ", joinpath(@__DIR__, "Unzen_3D.mp4"))
