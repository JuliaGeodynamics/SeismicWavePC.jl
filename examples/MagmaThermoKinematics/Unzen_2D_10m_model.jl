# Step 1 of the high-resolution Unzen example (README movie): the model of Unzen_2D.jl on a 10 m grid, from its 100 m MTK
# run (run Unzen_2D.jl first, or this script runs MTK at 100 m): temperature interpolated bilinearly, the surface linearly,
# rock types from the nearest MTK cell; density, velocities and melt fraction from the MAGEMin tables of Unzen_2D.jl.
# Saves HighRes2D_100m/Unzen_10m.jls for Unzen_2D_10m_waves.jl (which can run in another environment, e.g. with Metal).
#
#   julia -t auto --project=examples/MagmaThermoKinematics examples/MagmaThermoKinematics/Unzen_2D_10m_model.jl [dx in m, default 10]
using Serialization
E = @__DIR__
src = read(joinpath(E, "Unzen_2D.jl"), String)     # MAGEMin tables, lookup, Q_rock and the 100 m MTK result of Unzen_2D.jl
src = replace(src[1:findfirst("x, z = mtk[", src)[1]-1], "@__DIR__" => repr(E), "isempty(ARGS) ? 100 : parse(Int, ARGS[1])" => "100")
include_string(Main, src)

h = (isempty(ARGS) ? 10 : parse(Int, ARGS[1])) / 1.0e3       # grid spacing [km]
xc, zc = mtk["x"] ./ 1.0e3, mtk["z"] ./ 1.0e3                 # the 100 m grid [km]
dc = xc[2] - xc[1]
phc, Tc = mtk["Phases"], Float64.(mtk["T"])
topc = [zc[findlast(!=(0), phc[i, :])] for i in eachindex(xc)]
x, z = xc[1]:h:xc[end], zc[1]:h:zc[end]
frac(v, c) = (f = clamp((v - c[1]) / dc + 1, 1, length(c)); i = min(floor(Int, f), length(c) - 1); (i, f - i))
function interp(A, xv, zv)
    (i, u), (k, w) = frac(xv, xc), frac(zv, zc)
    return (1 - u) * (1 - w) * A[i, k] + u * (1 - w) * A[i+1, k] + (1 - u) * w * A[i, k+1] + u * w * A[i+1, k+1]
end
top = [(((i, u) = frac(xv, xc)); (1 - u) * topc[i] + u * topc[i+1]) for xv in x]
trise = 10h                                           # as the example: frequencies up to about 2 / trise
vs_min = 5h * 2 / trise
nx, nz = length(x), length(z)
F = [zeros(Float32, nx, nz) for _ in 1:6]             # rho, vp, vs, ϕ, Qs, T
Threads.@threads for i in 1:nx
    ic = clamp(round(Int, (x[i] - xc[1]) / dc) + 1, 1, length(xc))
    for k in 1:nz
        if z[k] > top[i]                              # air
            F[1][i, k], F[5][i, k], F[6][i, k] = 1, 10, 0
            continue
        end
        kc = clamp(round(Int, (z[k] - zc[1]) / dc) + 1, 1, length(zc))
        kc = min(kc, findlast(!=(0), phc[ic, :]))     # below the interpolated surface: rock of the top cell
        p = phc[ic, kc]
        T = interp(Tc, x[i], z[k])
        P = 2700 * 9.81 * 1.0e3 * (top[i] - z[k]) / 1.0e8
        ρ, vp, vs, ϕ = lookup(tables[composition[p]], P, T)
        F[1][i, k], F[2][i, k], F[3][i, k], F[4][i, k] = ρ, vp, vs < vs_min ? 0 : vs, ϕ
        F[5][i, k], F[6][i, k] = max(Q_rock[p] / (1 + 100ϕ), 10), T
    end
end
serialize(joinpath(dir, "Unzen_$(round(Int, 1e3h))m.jls"),
          (; x=collect(x), z=collect(z), rho=F[1], vp=F[2], vs=F[3], ϕ=F[4], Qs=F[5], T=F[6], trise))
println("$(nx) × $(nz) cells, vp ", extrema(F[2]), ", molten cells (vs = 0 in rock) ", count((F[3] .== 0) .& (F[2] .> 0)))
