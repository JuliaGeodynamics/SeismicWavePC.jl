# Immersed free surface (option `topography`, experimental): on a surface along cell faces close to the staircase, stable
# on a slope over a long run, and the cases it does not cover yet give errors.
using LinearAlgebra

@testset "immersed free surface" begin
    h = 0.05
    x, z = range(-5 + h / 2, 5 - h / 2, step=h), range(-5 + h / 2, 1.5 - h / 2, step=h)
    X, Y, Z = xyz_grid(x, 0, z)
    halfspace(zs) = (air = Z .> zs.(X); CartData(X, Y, Z, (; rho=ifelse.(air, 1.0, 2500.0), vp=ifelse.(air, 0.0, 3.0), vs=ifelse.(air, 0.0, 1.7))))
    src(zs) = MomentTensorSource(-1.0, 0.0, zs(-1.0) - 1.0, 1e15, (1, 1, 1, 0, 0, 0); trise=0.3)
    st(zs) = [Station("S$k", xs, 0.0, zs(xs) - 0.4) for (k, xs) in enumerate(-2.0:1.0:2.0)]
    kw = (; dt=0.5h / 3.0, interpolation=:sinc, T=Float64, verbose=false)

    # a flat surface along cell faces: the staircase is exact there, the immersed surface close to it
    flat(x) = 0.0
    a = simulate(halfspace(flat); kw..., nt=500, sources=src(flat), stations=st(flat))
    b = simulate(halfspace(flat); kw..., nt=500, sources=src(flat), stations=st(flat), topography=flat)
    @test norm(b.v .- a.v) / norm(a.v) < 0.1

    # a slope of 20°: stable (the wavefield at the end of a long run is small)
    slope(x) = x * tand(20)
    r = simulate(halfspace(slope); kw..., nt=round(Int, 20 / kw.dt), sources=src(slope), stations=st(slope), topography=slope)
    @test all(isfinite, r.v)
    @test maximum(A -> maximum(abs, A), r.wavefield.fields.V) < 0.05maximum(abs, r.v)

    # not covered (yet)
    m = halfspace(flat)
    @test_throws "topography" simulate(m; kw..., nt=10, wave=:sh, sources=src(flat), topography=flat)
    @test_throws "surface = true" simulate(m; kw..., nt=10, sources=src(flat), stations=[Station("S", 0.0, 0.0, 0.0; surface=true)], topography=flat)
    @test_throws "above the topography" simulate(m; kw..., nt=10, sources=MomentTensorSource(0.0, 0.0, 0.2, 1e15, (1, 1, 1, 0, 0, 0)), topography=x -> 0.1)
end
