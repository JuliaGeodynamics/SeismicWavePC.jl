# Rayleigh waves along a free surface over a velocity gradient (2D P-SV) against the modes of the 1D eigen-solver
# (rayleigh_modes.jl): phase velocity, H/V and the sense of the particle motion of the fundamental mode, from a slant stack
# of a line of surface stations.
using SeismicWavePC, GeophysicalModelGenerator, LinearAlgebra, Test
include("rayleigh_modes.jl")

@testset "Rayleigh modes" begin
    # the eigen-solver: homogeneous half-space (ν = 0.25): c = 0.919402 vs, H/V = 0.6813, retrograde (U/W < 0)
    m = rayleigh_modes(d -> (vp=sqrt(3) * 1000.0, vs=1000.0, rho=1000.0), 20.0)
    @test m[1].c ≈ 919.402 rtol = 1e-3
    @test m[1].U / m[1].W ≈ -0.6813 rtol = 5e-3                                 # linear elements, 30 per wavelength

    profile(d) = (e = exp(-max(d, 0) / 150); (vp=1.8 * (2000 - 1200e), vs=2000 - 1200e, rho=2600 - 600e))   # [m], SI
    dx = 10.0                                                                     # [m]
    x, z = range(-100 + dx / 2, 4000, step=dx), range(-1200 + dx / 2, 2dx, step=dx)
    X, Y, Z = xyz_grid(x ./ 1e3, 0, z ./ 1e3)
    P = profile.(-1e3 .* Z)
    air = Z .> 0
    model = CartData(X, Y, Z, (; rho=ifelse.(air, 1.0, getfield.(P, :rho)), vp=ifelse.(air, 0.0, getfield.(P, :vp) ./ 1e3),
                                 vs=ifelse.(air, 0.0, getfield.(P, :vs) ./ 1e3)))
    offsets = 500:20:3500
    f0 = 4.0
    r = simulate(model; dt=1e-3, nt=4000, sources=ForceSource(dx / 2e3, 0.0, -1.5dx / 1e3; fz=1e10, trise=1 / f0, t0=0.5 / f0),
                 stations=[Station("r$o", (o + dx / 2) / 1e3, 0.0, 0.0; surface=true) for o in offsets], stf=:ricker, T=Float64,
                 record_every=4, verbose=false)
    xo = collect(offsets .+ 0.0)
    w = sin.(π .* (xo .- xo[1]) ./ (xo[end] - xo[1])) .^ 2                       # Hann taper over the offsets
    # the fundamental mode at ≥ 16 points per S wavelength at the surface (the error grows as f²: 1 % at 10 points)
    for f in 3.0:0.5:5.0
        ω = 2π * f
        u = [vec(transpose(cis.(-ω .* r.t)) * r.v[:, a, :]) for a in 1:2]         # x, z (up) at frequency f
        stack(v, k) = sum(w .* v .* cis.(k .* xo))
        e = rayleigh_modes(profile, f; drec=dx / 2)[1]                           # stations: centre of the top cell
        ks = range(0.97, 1.03, length=301) .* (ω / e.c)
        q = argmax(k -> abs2(stack(u[1], k)) + abs2(stack(u[2], k)), ks)
        @test ω / q ≈ e.c rtol = 0.005
        sx, sz = stack(u[1], q), stack(u[2], q)
        @test abs(sx / sz) ≈ abs(e.U / e.W) rtol = 0.03
        @test sign(real(-im * sz / sx)) == sign(e.W / e.U)                        # retrograde
    end
end
