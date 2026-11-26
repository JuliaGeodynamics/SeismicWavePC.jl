# Curvilinear (terrain-following) grids: SBP operators, the mesh and its blocks, agreement with the staggered solver on a
# flat model (moment tensor source, attenuation), Rayleigh waves with stations exactly on the surface, and a model with
# topography and a curved interface (stability, Float32).
using SeismicWavePC, GeophysicalModelGenerator, LinearAlgebra, Test
const SW = SeismicWavePC
isdefined(@__MODULE__, :rayleigh_modes) || include("rayleigh_modes.jl")

@testset "curvilinear" begin
    @testset "SBP operators" begin
        n = 40
        x, h = SW.sbp_grid(0.0, 3.0, n)
        H = SW.sbp_weights(n, h)
        Dm = [SW.dminus(j -> Float64(j == c), r, n) / h for r in 1:n, c in 1:n]
        Dp = [SW.dplus(j -> Float64(j == c), r, n) / h for r in 1:n, c in 1:n]
        B = zeros(n, n); B[1, 1] = -1; B[n, n] = 1
        @test maximum(abs, Diagonal(H) * Dp + (Diagonal(H) * Dm)' - B) < 1e-12      # dual pairing: H D+ + (H D-)ᵀ = B
        @test sum(H) ≈ 3.0
        for p in 0:3                                                                   # exact for polynomials
            @test maximum(abs, Dm * x .^ p .- p .* x .^ max(p - 1, 0)) < 1e-9
            @test maximum(abs, Dp * x .^ p .- p .* x .^ max(p - 1, 0)) < 1e-9
        end
    end

    topo(x) = 1.5 * exp(-x^2 / (2 * 3.0^2))
    moho(x) = -6.0 + 1.0 * sin(2π * x / 20)
    @testset "mesh and blocks" begin
        mesh = curvilinear_mesh(range(-20, 20, length=81), [topo, moho]; dz=0.5, bottom=-30)
        @test SW.is_curvilinear(mesh)
        @test !SW.is_curvilinear(CartData(xyz_grid(-1:0.5:1, 0, -1:0.5:0)))
        X, Z = mesh.x.val[:, 1, :], mesh.z.val[:, 1, :]
        @test Z[:, end] ≈ topo.(X[:, 1])
        r = findfirst(k -> Z[:, k] ≈ moho.(X[:, 1]), axes(Z, 2))                      # a grid line follows the interface
        @test r !== nothing
        vp = ifelse.(Z .< moho.(X), 6.5, 5.5)
        for v in (vp, ifelse.(Z .<= moho.(X), 6.5, 5.5))                               # either side on the interface
            rows, _ = SW.find_blocks(Z, v, v ./ 1.73)
            @test rows == [r]
        end
        vc = copy(vp); vc[30:40, r+3:r+6] .= 3.0                                       # a chamber is no interface
        @test first(SW.find_blocks(Z, vc, vc ./ 1.73)) == [r]
    end

    @testset "flat model: against the staggered solver" begin
        # a double couple 5 km deep, Q = 50, receivers on the surface and at depth; staggered at half the spacing
        src = DoubleCoupleSource(-2.0, 0.0, -5.0; Mw=3.0, strike=0, dip=45, rake=90, trise=1.0)
        stations = [Station("s$x", Float64(x), 0.0, 0.0; surface=true) for x in (2, 5, 8)]
        push!(stations, Station("d", 4.0, 0.0, -3.0))
        mat = (; vp=4.0, vs=2.3, rho=2500.0, Q=50.0)
        mesh = curvilinear_mesh(range(-20, 20, length=401), [x -> 0.0]; dz=0.1, bottom=-30)
        o = ones(size(mesh.x.val))
        cm = addfield(mesh, (; rho=mat.rho .* o, vp=mat.vp .* o, vs=mat.vs .* o, Qp=2mat.Q .* o, Qs=mat.Q .* o))
        dx = 0.05
        # the staggered solver records "surface" stations at the centre of its top cell (dx/2 deep): the curvilinear one
        # there too (the horizontal motion of surface waves changes quickly with depth: ~15 % per 25 m here)
        cstations = [s.surface ? Station(s.name, s.x, 0.0, -dx / 2) : s for s in stations]
        rc = simulate(cm; grid=:curvilinear, tmax=8.0, sources=src, stations=cstations, T=Float64, verbose=false)
        X, Y, Z = xyz_grid(-20+dx/2:dx:20, 0, -30+dx/2:dx:2dx)
        air = Z .> 0
        sm = CartData(X, Y, Z, (; rho=ifelse.(air, 1.0, mat.rho), vp=ifelse.(air, 0.0, mat.vp), vs=ifelse.(air, 0.0, mat.vs),
                                  Qp=ifelse.(air, 10.0, 2mat.Q), Qs=ifelse.(air, 10.0, mat.Q)))
        rs = simulate(sm; dt=0.004, nt=2000, sources=src, stations, interpolation=:sinc, T=Float64, verbose=false)
        @test rc.components == rs.components && size(rc.v, 2) == 2
        # on the curvilinear time axis
        at(r, u) = [(j = clamp(searchsortedlast(r.t, τ), 1, length(r.t) - 1); u[j] + (u[j+1] - u[j]) * (τ - r.t[j]) / (r.t[j+1] - r.t[j])) for τ in 0.05:0.01:7.9]
        for n in eachindex(stations), c in 1:2
            a, b = at(rc, rc.v[:, c, n]), at(rs, rs.v[:, c, n])
            @test norm(a - b) / norm(b) < 0.05
        end
        # stress at depth (Pa, z up as the staggered solver)
        for c in 1:3
            a, b = at(rc, rc.stress[:, c, 4]), at(rs, rs.stress[:, c, 4])
            @test norm(a - b) / norm(b) < 0.06
        end
    end

    @testset "Rayleigh waves: stations exactly on the surface" begin
        profile(d) = (e = exp(-max(d, 0) / 150); (vp=1.8 * (2000 - 1200e), vs=2000 - 1200e, rho=2600 - 600e))   # SI
        mesh = curvilinear_mesh(range(-0.1, 4.0, length=411), [x -> 0.0]; dz=0.01, bottom=-1.2)
        P = profile.(-1e3 .* mesh.z.val)
        m = addfield(mesh, (; rho=getfield.(P, :rho), vp=getfield.(P, :vp) ./ 1e3, vs=getfield.(P, :vs) ./ 1e3))
        offsets = 500:20:3500
        f0 = 4.0
        r = simulate(m; grid=:curvilinear, tmax=4.0, sources=ForceSource(0.0, 0.0, -0.015; fz=1e10, trise=1 / f0, t0=0.5 / f0),
                     stations=[Station("r$o", o / 1e3, 0.0, 0.0; surface=true) for o in offsets], stf=:ricker, T=Float64,
                     record_every=2, verbose=false)
        xo = collect(offsets .+ 0.0)
        w = sin.(π .* (xo .- xo[1]) ./ (xo[end] - xo[1])) .^ 2
        for f in 3.0:0.5:5.0
            ω = 2π * f
            u = [vec(transpose(cis.(-ω .* r.t)) * r.v[:, a, :]) for a in 1:2]
            stack(v, k) = sum(w .* v .* cis.(k .* xo))
            e = rayleigh_modes(profile, f)[1]                                          # at the surface itself
            ks = range(0.97, 1.03, length=301) .* (ω / e.c)
            q = argmax(k -> abs2(stack(u[1], k)) + abs2(stack(u[2], k)), ks)
            @test ω / q ≈ e.c rtol = 0.005
            sx, sz = stack(u[1], q), stack(u[2], q)
            @test abs(sx / sz) ≈ abs(e.U / e.W) rtol = 0.02
            @test sign(real(-im * sz / sx)) == sign(e.W / e.U)
        end
    end

    @testset "topography and a curved interface: stable, Float32" begin
        mesh = curvilinear_mesh(range(-20, 20, length=81), [topo, moho]; dz=0.5, bottom=-30)
        Z, X = mesh.z.val, mesh.x.val
        lower = Z .< moho.(X)
        m = addfield(mesh, (; rho=ifelse.(lower, 2900.0, 2600.0), vp=ifelse.(lower, 6.5, 5.5), vs=ifelse.(lower, 3.75, 3.2)))
        src = ForceSource(0.0, 0.0, -10.0; fz=1e15, trise=2.0)
        st = [Station("a", 5.0, 0.0, 0.0; surface=true), Station("b", -8.0, 0.0, -8.0)]
        r64 = simulate(m; tmax=30.0, sources=src, stations=st, T=Float64, verbose=false)
        r32 = simulate(m; dt=r64.t[2] - r64.t[1], tmax=30.0, sources=src, stations=st, T=Float32, verbose=false)
        @test all(isfinite, r64.v) && maximum(abs, r64.v) > 0
        @test maximum(abs, r64.v[end-50:end, :, :]) < 0.05maximum(abs, r64.v)           # absorbed, not growing
        @test norm(r32.v - r64.v) / norm(r64.v) < 5e-3                                 # rounding over 30 s
        @test r64.wavefield isa CartData && size(r64.wavefield.fields.V[1]) == size(Z)
        # snapshots through a callback: the velocity on the deformed mesh
        times = Float64[]
        simulate(m; tmax=2.0, sources=src, snapshot_every=20, callback=(w, t) -> (push!(times, t); @assert size(w.fields.V[3]) == size(Z)),
                 verbose=false)
        @test length(times) > 2 && times[1] == 0
    end

    @testset "3D: mesh, blocks, against the staggered solver, stability" begin
        topo3(x, y) = 1.0 * exp(-(x^2 + (y - 1)^2) / (2 * 3.0^2))
        moho3(x, y) = -5.0 + 0.8 * sin(2π * x / 16) * cos(2π * y / 20)
        mesh = curvilinear_mesh(range(-10, 10, length=41), range(-8, 8, length=33), [topo3, moho3]; dz=0.5, bottom=-15)
        X, Y, Z = mesh.x.val, mesh.y.val, mesh.z.val
        @test SW.is_curvilinear(mesh) && size(Z)[1:2] == (41, 33)
        @test Z[:, :, end] ≈ topo3.(X[:, :, 1], Y[:, :, 1])
        r = findfirst(k -> Z[:, :, k] ≈ moho3.(X[:, :, 1], Y[:, :, 1]), axes(Z, 3))
        low = Z .< moho3.(X, Y)
        vp = ifelse.(low, 6.5, 5.5)
        flat(A) = reshape(A, :, size(A, 3))
        @test first(SW.find_blocks(flat(Z), flat(vp), flat(vp ./ 1.73))) == [r]
        # stable without absorbing boundaries (energy conserved), absorbed with them; Float32
        m = addfield(mesh, (; rho=ifelse.(low, 2900.0, 2600.0), vp, vs=ifelse.(low, 3.75, 3.2)))
        src = DoubleCoupleSource(1.0, -2.0, -8.0; Mw=4.0, strike=30, dip=60, rake=70, trise=1.5)
        st = [Station("a", 3.0, 2.0, 0.0; surface=true), Station("b", -4.0, -3.0, -7.0)]
        r0 = simulate(m; tmax=30.0, sources=src, stations=st, absorber=:none, T=Float64, verbose=false)
        v0 = vec(maximum(abs, r0.v; dims=(2, 3)))
        @test maximum(v0[end÷2:end]) < 3maximum(v0[1:end÷2])
        r64 = simulate(m; tmax=20.0, sources=src, stations=st, T=Float64, verbose=false)
        r32 = simulate(m; dt=r64.t[2] - r64.t[1], tmax=20.0, sources=src, stations=st, T=Float32, verbose=false)
        @test norm(r32.v - r64.v) / norm(r64.v) < 1e-3
        v = vec(maximum(abs, r64.v; dims=(2, 3)))
        @test maximum(v[end-50:end]) < 0.2maximum(v)
        @test size(r64.wavefield.fields.V[3]) == size(Z)

        # a flat 3D model (Q = 50, double couple) against the staggered solver at half the spacing
        L, D = 6.0, 7.0
        mat = (; vp=4.0, vs=2.3, rho=2500.0, Q=50.0)
        src = DoubleCoupleSource(-0.5, 0.3, -3.5; Mw=3.0, strike=30, dip=60, rake=70, trise=1.5)
        dxs, dxc = 0.1, 0.2
        stations = [Station("s1", 1.5, -1.0, 0.0; surface=true), Station("d", 1.0, -0.5, -2.0)]
        cst = [s.surface ? Station(s.name, s.x, s.y, -dxs / 2) : s for s in stations]
        mesh = curvilinear_mesh(range(-L, L, length=61), range(-L, L, length=61), [(x, y) -> 0.0]; dz=dxc, bottom=-D)
        o = ones(size(mesh.x.val))
        cm = addfield(mesh, (; rho=mat.rho .* o, vp=mat.vp .* o, vs=mat.vs .* o, Qp=2mat.Q .* o, Qs=mat.Q .* o))
        rc = simulate(cm; grid=:curvilinear, tmax=3.5, sources=src, stations=cst, T=Float32, verbose=false)
        X, Y, Z = xyz_grid(-L+dxs/2:dxs:L, -L+dxs/2:dxs:L, -D+dxs/2:dxs:2dxs)
        air = Z .> 0
        sm = CartData(X, Y, Z, (; rho=ifelse.(air, 1.0, mat.rho), vp=ifelse.(air, 0.0, mat.vp), vs=ifelse.(air, 0.0, mat.vs),
                                  Qp=ifelse.(air, 10.0, 2mat.Q), Qs=ifelse.(air, 10.0, mat.Q)))
        rs = simulate(sm; dt=0.008, nt=440, sources=src, stations, interpolation=:sinc, T=Float32, verbose=false)
        at(r, u) = [(j = clamp(searchsortedlast(r.t, τ), 1, length(r.t) - 1); u[j] + (u[j+1] - u[j]) * (τ - r.t[j]) / (r.t[j+1] - r.t[j])) for τ in 0.05:0.01:3.4]
        rel(a, b) = norm(a - b) / norm(b)
        for n in 1:2, c in 1:3
            @test rel(at(rc, rc.v[:, c, n]), at(rs, rs.v[:, c, n])) < 0.07
        end
        for c in (1, 3)
            @test rel(at(rc, rc.stress[:, c, 2]), at(rs, rs.stress[:, c, 2])) < 0.06
        end
    end

    @testset "misfit gradient (2D, attenuation) against finite differences" begin
        topog(x) = 1.2 * exp(-x^2 / 12)
        mohog(x) = -4.0 + 0.6 * sin(2π * x / 10)
        mesh = curvilinear_mesh(range(-8, 8, length=49), [topog, mohog]; dz=0.33, bottom=-10)
        X, Z = mesh.x.val, mesh.z.val
        low = Z .< mohog.(X)
        base = (; rho=ifelse.(low, 2900.0, 2600.0), vp=ifelse.(low, 6.5, 5.5), vs=ifelse.(low, 3.75, 3.2), Qp=ifelse.(low, 300.0, 60.0),
                Qs=ifelse.(low, 150.0, 30.0))
        srcs = [DoubleCoupleSource(-2.0, 0.0, -5.0; Mw=4.0, strike=30, dip=60, rake=70, trise=1.0), ForceSource(2.0, 0.0, -2.5; fz=1e14, trise=1.0)]
        sts = [[Station("s$x", Float64(x), 0.0, 0.0; surface=true) for x in -6:3:6]; Station("d", 0.3, 0.0, -6.0)]
        r0 = simulate(addfield(mesh, base); tmax=0.1, sources=srcs, T=Float64, verbose=false)
        kw = (; dt=r0.t[2] - r0.t[1], tmax=4.0, sources=srcs, T=Float64)
        pert = merge(base, (; vs=base.vs .* (1 .- 0.04 .* exp.(-((X .- 0.5) .^ 2 .+ (Z .+ 1) .^ 2) ./ 3)), Qs=base.Qs .* 1.3))
        obs = simulate(addfield(mesh, pert); kw..., stations=sts, verbose=false)
        J, g, _ = misfit_gradient(addfield(mesh, base), obs; kw..., checkpoint_steps=17)
        @test keys(g.fields) == (:rho, :vp, :vs, :Qp, :Qs)
        Jof(m) = L2Misfit()(simulate(m; kw..., stations=sts, verbose=false).v, obs.v, obs.t)[1]
        @test Jof(addfield(mesh, base)) ≈ J rtol = 1e-12
        for (f, δ) in ((:vs, 0.02 .* base.vs .* exp.(-((X .- 2) .^ 2 .+ (Z .- 0.2) .^ 2) ./ 3)),     # at the surface
                       (:vp, 0.02 .* base.vp .* exp.(-((X .+ 2) .^ 2 .+ (Z .+ 4) .^ 2) ./ 3)),     # across the interface
                       (:rho, 0.02 .* base.rho .* exp.(-((X .- 0.5) .^ 2 .+ (Z .+ 2) .^ 2) ./ 4)),
                       (:vs, 0.02 .* base.vs .* exp.(-((X .+ 7.5) .^ 2 .+ (Z .+ 5) .^ 2) ./ 2)),   # at an absorbing side
                       (:Qs, 5 .* exp.(-((X .- 1) .^ 2 .+ (Z .+ 1) .^ 2) ./ 4)), (:Qp, 10 .* exp.(-((X .+ 1) .^ 2 .+ (Z .+ 2) .^ 2) ./ 4)))
            ε = 1e-3
            fd = (Jof(addfield(mesh, merge(base, (; f => base[f] .+ ε .* δ)))) - Jof(addfield(mesh, merge(base, (; f => base[f] .- ε .* δ))))) / 2ε
            @test sum(g.fields[f] .* δ) ≈ fd rtol = 1e-6
        end
    end
    @testset "misfit gradient (3D) against finite differences" begin
        topog3(x, y) = 1.0 * exp(-(x^2 + y^2) / 10)
        mohog3(x, y) = -4.0 + 0.5 * sin(2π * x / 10) * cos(2π * y / 12)
        mesh = curvilinear_mesh(range(-6, 6, length=33), range(-5, 5, length=29), [topog3, mohog3]; dz=0.4, bottom=-8)
        X, Y, Z = mesh.x.val, mesh.y.val, mesh.z.val
        low = Z .< mohog3.(X, Y)
        base = (; rho=ifelse.(low, 2900.0, 2600.0), vp=ifelse.(low, 6.5, 5.5), vs=ifelse.(low, 3.75, 3.2))
        srcs = [DoubleCoupleSource(-1.0, 0.5, -5.0; Mw=4.0, strike=30, dip=60, rake=70, trise=1.0),
                ForceSource(1.5, -1.0, -2.0; fz=1e14, fx=5e13, trise=1.0)]
        sts = [[Station("s$x$y", Float64(x) + 0.1, Float64(y) - 0.1, 0.0; surface=true) for x in -4:2:4 for y in (-3, 0, 3)]; Station("d", 0.3, 0.2, -5.0)]
        r0 = simulate(addfield(mesh, base); tmax=0.05, sources=srcs, T=Float64, verbose=false)
        kw = (; dt=r0.t[2] - r0.t[1], tmax=3.0, sources=srcs, T=Float64)
        pert = merge(base, (; vs=base.vs .* (1 .- 0.04 .* exp.(-((X .- 0.5) .^ 2 .+ Y .^ 2 .+ (Z .+ 1) .^ 2) ./ 3))))
        obs = simulate(addfield(mesh, pert); kw..., stations=sts, verbose=false)
        J, g, _ = misfit_gradient(addfield(mesh, base), obs; kw..., checkpoint_steps=13)
        Jof(m) = L2Misfit()(simulate(m; kw..., stations=sts, verbose=false).v, obs.v, obs.t)[1]
        @test Jof(addfield(mesh, base)) ≈ J rtol = 1e-12
        for (f, δ) in ((:vs, 0.02 .* base.vs .* exp.(-((X .- 2) .^ 2 .+ (Y .+ 1) .^ 2 .+ (Z .- 0.2) .^ 2) ./ 3)),   # at the surface
                       (:vp, 0.02 .* base.vp .* exp.(-((X .+ 1) .^ 2 .+ (Y .- 1) .^ 2 .+ (Z .+ 4) .^ 2) ./ 3)),    # across the interface
                       (:rho, 0.02 .* base.rho .* exp.(-((X .- 0.5) .^ 2 .+ Y .^ 2 .+ (Z .+ 2) .^ 2) ./ 4)),
                       (:vs, 0.02 .* base.vs .* exp.(-((X .- 5.5) .^ 2 .+ (Y .+ 4.5) .^ 2 .+ (Z .+ 3) .^ 2) ./ 2)))  # at an absorbing corner
            ε = 1e-3
            fd = (Jof(addfield(mesh, merge(base, (; f => base[f] .+ ε .* δ)))) - Jof(addfield(mesh, merge(base, (; f => base[f] .- ε .* δ))))) / 2ε
            @test sum(g.fields[f] .* δ) ≈ fd rtol = 1e-6
        end
    end
    @testset "Born modelling: 2D with attenuation (finite differences), 3D (dot test with the gradient)" begin
        topob(x) = 1.2 * exp(-x^2 / 12)
        mohob(x) = -4.0 + 0.6 * sin(2π * x / 10)
        mesh = curvilinear_mesh(range(-8, 8, length=49), [topob, mohob]; dz=0.33, bottom=-10)
        X, Z = mesh.x.val, mesh.z.val
        low = Z .< mohob.(X); o = ones(size(X))
        base = (; rho=ifelse.(low, 2900.0, 2600.0), vp=ifelse.(low, 6.5, 5.5), vs=ifelse.(low, 3.75, 3.2), Qp=150 .* o, Qs=80 .* o)
        srcs = [DoubleCoupleSource(-2.0, 0.0, -5.0; Mw=4.0, strike=30, dip=60, rake=70, trise=1.0), ForceSource(2.0, 0.0, -2.5; fz=1e14, trise=1.0)]
        sts = [[Station("s$x", Float64(x), 0.0, 0.0; surface=true) for x in -6:3:6]; Station("d", 0.3, 0.0, -6.0)]
        r0 = simulate(addfield(mesh, base); tmax=0.1, sources=srcs, T=Float64, verbose=false)
        kw = (; dt=r0.t[2] - r0.t[1], tmax=3.0, sources=srcs, stations=sts, T=Float64)
        bump(c, s) = @. exp(-((X - c[1])^2 + (Z - c[2])^2) / s)
        δ = (; rho=50 .* bump((0.5, -2), 4), vp=0.1 .* bump((-2, -4), 3), vs=0.08 .* bump((2, 0.2), 3) .+ 0.05 .* bump((-7.5, -5), 2),
             Qp=20 .* bump((0, -1), 4), Qs=10 .* bump((1, -3), 4))
        b0, δd = born(addfield(mesh, base), addfield(mesh, δ); kw...)
        @test b0.v == simulate(addfield(mesh, base); kw..., verbose=false).v
        ε = 1e-3
        sp(s) = simulate(addfield(mesh, map((a, b) -> a .+ s .* b, base, δ)); kw..., verbose=false)
        P, M = sp(ε), sp(-ε)
        @test maximum(abs, δd.v .- (P.v .- M.v) ./ 2ε) < 1e-6 * maximum(abs, δd.v)
        @test maximum(abs, δd.stress .- (P.stress .- M.stress) ./ 2ε) < 1e-6 * maximum(abs, δd.stress)
        # 3D
        topob3(x, y) = 1.0 * exp(-(x^2 + y^2) / 10)
        mohob3(x, y) = -4.0 + 0.5 * sin(2π * x / 10) * cos(2π * y / 12)
        mesh = curvilinear_mesh(range(-6, 6, length=33), range(-5, 5, length=29), [topob3, mohob3]; dz=0.4, bottom=-8)
        X, Y, Z = mesh.x.val, mesh.y.val, mesh.z.val
        low = Z .< mohob3.(X, Y)
        base = (; rho=ifelse.(low, 2900.0, 2600.0), vp=ifelse.(low, 6.5, 5.5), vs=ifelse.(low, 3.75, 3.2))
        srcs = [DoubleCoupleSource(-1.0, 0.5, -5.0; Mw=4.0, strike=30, dip=60, rake=70, trise=1.0)]
        sts = [Station("s$x$y", Float64(x) + 0.1, Float64(y) - 0.1, 0.0; surface=true) for x in -4:2:4 for y in (-3, 0, 3)]
        r0 = simulate(addfield(mesh, base); tmax=0.05, sources=srcs, T=Float64, verbose=false)
        kw = (; dt=r0.t[2] - r0.t[1], tmax=2.5, sources=srcs, T=Float64)
        bump3(c, s) = @. exp(-((X - c[1])^2 + (Y - c[2])^2 + (Z - c[3])^2) / s)
        δ = (; rho=50 .* bump3((0.5, 0, -2), 4), vp=0.1 .* bump3((-1, 1, -4), 3), vs=0.08 .* bump3((2, -1, 0.2), 3))
        obs = simulate(addfield(mesh, merge(base, (; vs=base.vs .* (1 .- 0.04 .* bump3((0.5, 0, -1), 3))))); kw..., stations=sts, verbose=false)
        b0, δd = born(addfield(mesh, base), addfield(mesh, δ); kw..., stations=sts)
        _, gr, _ = misfit_gradient(addfield(mesh, base), obs; kw...)
        _, adj = L2Misfit()(b0.v, obs.v, obs.t)
        @test sum(adj .* δd.v) ≈ sum(k -> sum(gr.fields[k] .* δ[k]), keys(δ)) rtol = 1e-10
    end
end
