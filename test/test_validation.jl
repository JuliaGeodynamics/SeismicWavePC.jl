# Validation helpers: the time-frequency misfits of Kristeková et al. (2006) and the exact solution of Garvin's problem,
# against the curvilinear solver on a flat and a tilted free surface
@testset "validation helpers" begin
    # tf_misfits: amplitude errors go into EM, time shifts into PM (≈ 2 f δt for a narrow-band signal)
    t = (0:999) .* 0.002
    w(t0) = @. sin(2π * 5 * (t - t0)) * exp(-((t - t0) / 0.3)^2)
    @test all(tf_misfits(w(1.0), w(1.0), t; band=(1.0, 20.0)) .< 1e-12)
    EM, PM = tf_misfits(1.1 .* w(1.0), w(1.0), t; band=(1.0, 20.0))
    @test EM ≈ 0.1 rtol = 1e-6
    @test PM < 1e-10
    EM, PM = tf_misfits(w(1.002), w(1.0), t; band=(1.0, 20.0))
    @test PM ≈ 2 * 5 * 0.002 rtol = 0.2
    @test EM < 0.5PM                                  # (the envelope moves too)
    E, P = tf_misfits(stack([w(1.0), 2w(1.0)]), stack([w(1.0), w(1.0)]), t; band=(1.0, 20.0))
    @test size(E) == (2,) && E[1] < 1e-12
    @test E[2] ≈ 1 rtol = 1e-6
    @test_throws "no signal" tf_misfits(w(1.0), zero(t), t; band=(1.0, 20.0))

    # Garvin's problem: nothing before the P wave; the curvilinear solver on a flat and a 30° tilted surface (8 points per
    # minimum S wavelength), within 1 % (envelope) and 0.5 % (phase) of the exact solution
    vp, vs, ρ, h, f0, dx = 2.0, 1.0, 1600.0, 0.025, 10.0, 0.005
    stf = (; stf=:ricker, t0=0.5 / f0, trise=1 / f0)
    offsets = [0.15, 0.3]
    for θ in (0, 30)
        s, c = sind(θ), cosd(θ)
        flat = curvilinear_mesh(range(-0.5, 0.9, step=dx), [x -> 0.0]; dz=dx, bottom=-0.5)
        Z = flat.z.val .+ flat.x.val .* tand(θ)
        o = ones(size(Z))
        model = CartData(flat.x.val, zero(Z), Z, (; z=copy(Z), rho=ρ .* o, vp=vp .* o, vs=vs .* o))
        r = simulate(model; grid=:curvilinear, tmax=0.5, sources=ExplosionSource(h * s, 0.0, -h * c; M0=1e10, stf.t0, stf.trise),
                     stations=[Station("r$i", d * c, 0.0, d * s; surface=true) for (i, d) in enumerate(offsets)],
                     stf=stf.stf, T=Float64, verbose=false, absorber=:pml, na=20)
        vt, vn = r.v[:, 1, :] .* c .+ r.v[:, 2, :] .* s, .-r.v[:, 1, :] .* s .+ r.v[:, 2, :] .* c
        ex = garvin(offsets, h, r.t; vp, vs, rho=ρ, M0=1e10, stf...)
        θ == 0 && @test all(iszero, ex[2][r.t .< hypot(offsets[1], h) / vp .- 1e-9, 1])
        for (v, e) in ((vt, ex[1]), (vn, ex[2]))
            EM, PM = tf_misfits(v, e, r.t; band=(2.0, 30.0))
            @test maximum(EM) < 0.01
            @test maximum(PM) < 0.005
        end
    end
end
