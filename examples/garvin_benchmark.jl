# Garvin's problem on tilted free surfaces, the test of de la Puente et al. (2014, Geophysics 79, T125): a line explosion
# 25 m below the free surface of a half-space (vp = 2 km/s, vs = 1 km/s, ρ = 1600 kg/m³, Ricker wavelet of 10 Hz), five
# receivers on the surface 100–500 m from the source, the surface tilted by 0–40°. The curvilinear solver against the
# exact solution (`garvin`, Cagniard–de Hoop), with the time-frequency envelope and phase misfits of Kristeková et al.
# (2006) (`tf_misfits`, 2–30 Hz) of the components along (v_t) and normal to (v_n) the surface. Grids of 6.67 m (6 points
# per minimum S wavelength, as in the paper), 5 m and 3.33 m.
#
#   julia -t auto --project=examples examples/garvin_benchmark.jl
#
using SeismicWavePC, GeophysicalModelGenerator, CairoMakie, Printf

const vp, vs, ρ, h, f0, M0 = 2.0, 1.0, 1600.0, 0.025, 10.0, 1e10
const offsets = [0.1, 0.2, 0.3, 0.4, 0.5]
const stf = (; stf=:ricker, t0=0.5 / f0, trise=1 / f0)      # moment rate: Ricker wavelet, peak at 10 Hz, < 5 % above 23 Hz
const band = (2.0, 30.0)

"(v_t, v_n) at the receivers, the sample times and the run time of the simulation, for the tilt θ [°] and grid spacing dx [km]."
function run(θ, dx)
    s, c = sind(θ), cosd(θ)
    flat = curvilinear_mesh(range(-1.0, 1.5, step=dx), [x -> 0.0]; dz=dx, bottom=-1.0)
    Z = flat.z.val .+ flat.x.val .* tand(θ)                  # vertical columns, bottom parallel to the surface
    o = ones(size(Z))
    model = CartData(flat.x.val, zero(Z), Z, (; z=copy(Z), rho=ρ .* o, vp=vp .* o, vs=vs .* o))
    source = ExplosionSource(h * s, 0.0, -h * c; M0, stf.t0, stf.trise)        # depth h along the normal (-s, c)
    stations = [Station("r$i", d * c, 0.0, d * s; surface=true) for (i, d) in enumerate(offsets)]
    time = @elapsed r = simulate(model; grid=:curvilinear, tmax=1.0, sources=source, stations, stf=stf.stf, T=Float64,
                                 verbose=false, absorber=:pml, na=round(Int, 0.15 / dx))
    vx, vz = r.v[:, 1, :], r.v[:, 2, :]
    return (t=vx .* c .+ vz .* s, n=.-vx .* s .+ vz .* c), r.t, time
end

tilts, grids = (0, 10, 20, 30, 40), (0.00667, 0.005, 0.00333)
run(0, 0.01)                                                   # (compile)
res = Dict((θ, dx) => run(θ, dx) for θ in tilts, dx in grids)
println("EM, PM (%) of (v_t, v_n), mean (max) over the 5 receivers; run time ($(Threads.nthreads()) threads)")
M = Dict()
for dx in grids, θ in tilts
    v, t, time = res[(θ, dx)]
    ex = garvin(offsets, h, t; vp, vs, rho=ρ, M0, stf...)
    M[(θ, dx)] = map((v.t, v.n), ex) do a, b
        tf_misfits(a, b, t; band)
    end
    f(x) = @sprintf("%5.2f (%5.2f)", 100sum(x) / 5, 100maximum(x))
    (et, pt), (en, pn) = M[(θ, dx)]
    @printf("dx = %4.2f m, tilt %2d°: v_t EM %s PM %s | v_n EM %s PM %s | %5.1f s\n", 1e3dx, θ, f(et), f(pt), f(en), f(pn), time)
end

fig = Figure(size=(1300, 950), fontsize=16)
for (j, θ) in enumerate((0, 40))                             # v_n; on the flat surface the 6.67 m grid rings at ~21 Hz
    v, t, _ = res[(θ, 0.00667)]
    e = garvin(offsets, h, t; vp, vs, rho=ρ, M0, stf...)[2]
    ax = Axis(fig[1, j], title="tilt $(θ)°, v_n (black: exact, red: 6.67 m, blue: 3.33 m)", xlabel="time (s)",
              yticks=(-(0:4), ["$(Int(1e3d)) m" for d in offsets]))
    scale = maximum(abs, e) * 1.1
    v3, t3, _ = res[(θ, 0.00333)]
    for i in 1:5
        lines!(ax, t, e[:, i] ./ scale .- (i - 1), color=:black, linewidth=3)
        lines!(ax, t, v.n[:, i] ./ scale .- (i - 1), color=:crimson, linewidth=1.3)
        lines!(ax, t3, v3.n[:, i] ./ scale .- (i - 1), color=:dodgerblue, linewidth=1.3, linestyle=:dash)
    end
end
for (j, (k, name)) in enumerate(((1, "envelope misfit EM"), (2, "phase misfit PM")))
    ax = Axis(fig[2, j], title="$name, mean over receivers and components", xlabel="tilt (°)", ylabel="%", yscale=log10,
              xticks=collect(tilts))
    for (dx, col) in zip(grids, (:crimson, :darkorange, :dodgerblue))
        y = [100 * (sum(M[(θ, dx)][1][k]) + sum(M[(θ, dx)][2][k])) / 10 for θ in tilts]
        scatterlines!(ax, collect(tilts), y, color=col, label=@sprintf("%.2f m", 1e3dx))
    end
    j == 1 && axislegend(ax, position=:rt)
end
save("garvin_benchmark.png", fig)
println("saved garvin_benchmark.png")
