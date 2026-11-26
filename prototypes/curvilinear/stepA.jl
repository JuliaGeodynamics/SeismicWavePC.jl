# Open questions before production code:
# 1. Leapfrog (v at half steps, as the staggered code) instead of RK4, with the time step from the spectral radius
#    (power iteration: dt = 0.9 × 2/√λmax for leapfrog, 0.9 × 2.83/√λmax for RK4); stability over 40 s and accuracy, in
#    Float64 and Float32.
# 2. Absorbing boundaries: a small domain (receivers 15 km from the sides, bottom 20 km below the source) against the
#    large one (no reflections within 9 s); impedance SAT, Cerjan sponge, both; the staggered code's PML for comparison.
#
#   julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/stepA.jl
include(joinpath(@__DIR__, "curvi.jl"))

pc(tr, ref) = @sprintf("%5.2f / %5.2f %%", 100errors(tr, ref, :deep)[1], 100errors(tr, ref, :shallow)[1])

println("1. Time stepping with dt from the spectral radius (500 m grid; error deep / shallow over 0-9 s; stability over 40 s)")
layer = (thickness=2e3, vp=3500.0, vs=2000.0, rho=2200.0)
for (name, f, ref) in (("hill, D = 10 km", kw -> run_case("hill"; flatten=10e3, kw...), reference("hill")),
                       ("tilt45 (sheared)", kw -> run_case("tilt45"; kw...), reference("tilt45")),
                       ("tilt20 + layer, 2 blocks", kw -> run_case("tilt20"; layer, interfaces=[-2e3 / cosd(20)], kw...), reference("tilt20"; layer)))
    println(name)
    for integrator in (:rk4, :leapfrog), FT in (Float64, Float32)
        r = f((; integrator, FT, T=40.0))
        E = r.energy
        k = findfirst(>(6.0), r.tt)
        @printf("  %-8s %-7s dt = %.4f s (cfl %.2f): %s   energy after the source max |E/E₀ - 1| %.1e  (%d steps, %.1f s)\n",
                integrator, FT, r.dt, r.dt * 6000 / 500, pc(r.traces, ref), maximum(abs, E[k:end] ./ E[k] .- 1), r.nt, r.wall)
    end
end

println("\n2. Absorbing boundaries: a small domain (sides 12 km from the outer receivers, bottom 12 km below the source) against")
println("   the large one, 0-12 s (reflections from 7 s on), hill, 500 m, deep / shallow")
small = (xlim=(-42e3, 22e3), bottom=-32e3)
large = run_case("hill"; flatten=10e3, integrator=:leapfrog, T=12.0, tmax=12.0).traces
for absorbing in (:none, :sat, :sponge, :both), nsponge in (absorbing in (:sponge, :both) ? (10, 20) : (20,))
    r = run_case("hill"; flatten=10e3, integrator=:leapfrog, absorbing, nsponge, T=12.0, tmax=12.0, small...)
    @printf("  curvilinear, %-7s %s: %s\n", absorbing, absorbing in (:sponge, :both) ? "($nsponge points)" : "           ",
            pc(r.traces, large))
end
largefd = fd_run("hill", :corners, 500.0; T=12.0)
for absorber in (:pml, :cerjan)
    @printf("  staggered,   %-7s (20 cells) : %s\n", absorber, pc(fd_run("hill", :corners, 500.0; absorber, T=12.0, small...), largefd))
end
