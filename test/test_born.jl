# Born (linearized) modelling: the background equals `simulate`, the scattered seismograms are the derivative (central
# differences converge as h²), and the Born operator is the transpose of the adjoint gradient (dot test).
using LinearAlgebra, Random

"A smooth perturbation of all model fields, zero in the air."
function test_perturbation(m)
    X, Y, Z = m.x.val, m.y.val, m.z.val
    rock = m.fields.vp .> 0
    bump(x0, z0, w) = @. exp(-((X - x0)^2 + Y^2 + (Z - z0)^2) / w^2) * rock
    return CartData(X, Y, Z, (; rho=200 .* bump(0.3, -1.0, 0.4), vp=0.3 .* bump(-0.5, -1.6, 0.5), vs=0.2 .* bump(0.8, -0.8, 0.4),
                              Qp=10 .* bump(0.0, -1.2, 0.5), Qs=8 .* bump(0.0, -1.2, 0.5)))
end
shifted(m, δm, h) = CartData(m.x.val, m.y.val, m.z.val, map(k -> m.fields[k] .+ h .* δm.fields[k], NamedTuple{keys(m.fields)}(keys(m.fields))))

"⟨J δm, r⟩ and ⟨δm, Jᵀ r⟩ for random residuals r."
function dot_test(m, δm, kw)
    d0, δd = born(m, δm; kw...)
    r = randn(MersenneTwister(1), size(d0.v))
    kwg = Base.structdiff(kw, NamedTuple{(:stations,)})
    g = misfit_gradient(m, d0; kwg..., misfit=(syn, obs, t) -> (sum(r .* syn), r))[2]
    return dot(δd.v, r), sum(dot(g.fields[k], δm.fields[k]) for k in keys(δm.fields))
end

@testset "Born modelling" begin
    m = test_model(:psv; step=true)
    δm = test_perturbation(m)
    kw = (; dt=DT, nt=400, sources=test_source(), stations=test_stations(:psv), T=Float64)
    d0, δd = born(m, δm; kw...)
    @test d0.v == simulate(m; kw..., verbose=false).v
    @test maximum(abs, δd.v) > 0
    err(h) = (fd = (simulate(shifted(m, δm, h); kw..., verbose=false).v .- simulate(shifted(m, δm, -h); kw..., verbose=false).v) ./ 2h;
              norm(δd.v .- fd) / norm(δd.v))
    e1, e2 = err(1e-2), err(1e-3)
    @test e2 < 1e-5 && e1 / e2 > 50                          # second order: the error of the central difference only
    @test all(iszero, born(m, CartData(m.x.val, m.y.val, m.z.val, (; vs=zero(m.fields.vs))); kw...)[2].v)
    # dot tests: P-SV with forces and sinc interpolation, Cerjan sponge; SH; 3D
    a, b = dot_test(m, δm, kw)
    @test a ≈ b rtol = 1e-10
    force = ForceSource(0.2, 0.0, -1.0; fx=1e12, fz=2e12, t0=0.05, trise=0.3)
    a, b = dot_test(m, δm, (; kw..., sources=force, interpolation=:sinc))
    @test a ≈ b rtol = 1e-10
    a, b = dot_test(m, δm, (; kw..., absorber=:cerjan))
    @test a ≈ b rtol = 1e-6                                   # the adjoint of the sponge is not exact
    msh = test_model(:sh; step=true)
    a, b = dot_test(msh, test_perturbation(msh), (; kw..., wave=:sh, stations=test_stations(:sh)))
    @test a ≈ b rtol = 1e-10
    m3 = test_model(:p3d)
    a, b = dot_test(m3, test_perturbation(m3), (; kw..., nt=150, stations=test_stations(:p3d)))
    @test a ≈ b rtol = 1e-10
    @test_throws "callback" born(m, δm; kw..., callback=(w, t) -> nothing)   # MPI: test_mpi.jl
end
