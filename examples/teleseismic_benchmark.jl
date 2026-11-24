# Teleseismic plane waves (TeleseismicWave, injected along a box) against the analytical solution: a plane P or SV wave from
# below in a homogeneous half-space with a free surface, and its reflections (P and SV) at the surface. The reflection
# coefficients follow from the traction-free surface (σ_xz = σ_zz = 0); the displacement at a point is a sum of shifted
# wavelets (no numerics). Displacement seismograms at several depths, as in the usual benchmarks of SPECFEM and other codes.
#
#   julia -t auto --project=examples examples/teleseismic_benchmark.jl
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Printf

α, β, ρ = 6.0, 6.0 / sqrt(3), 2700.0                      # km/s, kg/m³
dx = 0.1
x = range(dx / 2, step=dx, length=600)                     # 60 km
z = range(-40 + dx / 2, step=dx, length=402)               # 40 km deep, 2 cells of air
X, Y, Z = xyz_grid(x, 0, z)
air = Z .> 0
model = CartData(X, Y, Z, (; rho=ifelse.(air, 1.0, ρ), vp=ifelse.(air, 0.0, α), vs=ifelse.(air, 0.0, β)))

f0, dt, nt = 1.0, 0.005, 4000                              # Ricker wavelet (peak frequency f0), 20 s
depths = [0.05, 3.0, 5.0, 10.0, 15.0, 28.0]
xs = 30.0
stations = [Station("z$d", xs, 0.0, -d) for d in depths]
box = (5.0, 55.0, -32.0)

"""
Analytical displacement (ux, uz; z down) at (x, depth d) of a plane wave (`:p` at the angle `i`, or `:sv`) of the incident
particle velocity `stf(τ)`, arriving at the depth `zB` and the position `x0` at the time `t0`.
"""
function analytical(wave, i, x, d, t, x0, zB, t0, w)
    v = wave === :p ? α : β
    p = sind(i) / v
    ηa, ηb = sqrt(1 / α^2 - p^2), sqrt(1 / β^2 - p^2)
    λ, μ = ρ * (α^2 - 2β^2), ρ * β^2
    # waves (vertical slowness, polarization): incident (up-going), reflected P and SV (down-going); polarizations as in
    # TeleseismicWave: P along the propagation, SV of the up-going wave -(cos j, sin j)
    inc = wave === :p ? (-ηa, (α * p, -α * ηa)) : (-ηb, (-β * ηb, -β * p))
    rP, rS = (ηa, (α * p, α * ηa)), (ηb, (β * ηb, -β * p))
    trac((sz, d)) = (μ * (sz * d[1] + p * d[2]), λ * (p * d[1] + sz * d[2]) + 2μ * sz * d[2])    # σ_xz, σ_zz (× -f')
    A = [trac(rP)[1] trac(rS)[1]; trac(rP)[2] trac(rS)[2]]
    R = A \ -collect(trac(inc))                             # traction-free surface
    τ0 = t0 + p * (x - x0) + (wave === :p ? ηa : ηb) * zB   # the incident wave at the surface: τ = t - τ0
    u = zeros(length(t), 2)
    for ((sz, dv), a) in ((inc, 1.0), (rP, R[1]), (rS, R[2]))
        u .+= a .* [w(tt - τ0 - sz * d) * dv[c] for tt in t, c in 1:2]
    end
    return u
end

fig = Figure(size=(1300, 1400))
for (col, (wave, angle)) in enumerate(((:p, 20.0), (:sv, 15.0)))
    tw = TeleseismicWave(wave; angle, box, trise=1 / f0, t0=0.5, amplitude=1.0)
    r = simulate(model; dt, nt, sources=tw, stations, stf=:ricker, interpolation=:sinc, verbose=false)   # stations at their positions
    # the wavelet as displacement: integral of the incoming particle velocity (Ricker)
    τ = range(-2, 20, step=dt / 10)
    D = cumsum([moment_rate(:ricker, s, 0.0, 1 / f0) for s in τ]) .* (dt / 10)
    w(s) = s < τ[1] ? 0.0 : D[min(length(D), 1 + round(Int, (s - τ[1]) / (dt / 10)))]
    # the box boundaries of the injection (a quarter cell beyond the cell centres): where the wave arrives at t0
    g = SeismicWavePC.Grid(SeismicWavePC.model_arrays(model), Val(2))
    b = SeismicWavePC.injection_box(tw, g, 20)
    x0, zB = b.lo[1], b.zB
    t = r.t .+ dt                                           # displacement sample it: Σ v dt up to the step it (time it dt)
    for (row, d) in enumerate(depths)
        ua = analytical(wave, angle, xs, d, t, x0, zB, 0.5, w)
        ux, uz = r.u[:, 1, row], -r.u[:, 2, row]            # z down
        err = sqrt(sum(abs2, [ux uz] .- ua) / sum(abs2, ua))
        @printf("%s %4.1f°, depth %5.2f km: relative error of the displacement %.1e\n", wave, angle, d, err)
        ax = Axis(fig[row, col], title=@sprintf("%s wave, %g°, depth %g km (error %.1e)", uppercase(string(wave)), angle, d, err),
                  xlabel=row == length(depths) ? "time (s)" : "", ylabel="u (m)")
        lines!(ax, t, ua[:, 2], color=:black, label="analytical u_z")
        lines!(ax, t, uz, color=:orange, linestyle=:dash, label="SeismicWavePC u_z")
        lines!(ax, t, ua[:, 1], color=:gray50, label="analytical u_x")
        lines!(ax, t, ux, color=:dodgerblue, linestyle=:dash, label="SeismicWavePC u_x")
        row == 1 && col == 1 && axislegend(ax, position=:lt, framevisible=false)
        xlims!(ax, 0, nt * dt)
    end
end
save(joinpath(@__DIR__, "teleseismic_benchmark.png"), fig)
println("saved examples/teleseismic_benchmark.png")
