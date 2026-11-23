# Checks on an NVIDIA GPU: forward simulations and misfit gradients on CUDA must agree with the CPU (P-SV, SH, 3D).
#
#   julia -t auto --project=examples examples/check_cuda.jl
#
using SeismicWavePC, GeophysicalModelGenerator, CUDA, Printf
include(joinpath(@__DIR__, "..", "test", "setup.jl"))          # small test models (as in the test suite)

println("GPU: ", CUDA.name(CUDA.device()), ", CUDA ", CUDA.runtime_version(), ", ", Threads.nthreads(), " CPU threads")
reldiff(a, b) = maximum(abs, a .- b) / maximum(abs, a)
ok = true
for mode in (:psv, :sh, :p3d)
    m = test_model(mode)
    kw = (; dt=DT, nt=mode === :p3d ? 300 : 600, sources=test_source(), stations=test_stations(mode),
          wave=mode === :sh ? :sh : :psv, T=Float64, verbose=false)
    simulate(m; kw..., nt=10, backend=CUDABackend())                                # compile
    tc = @elapsed cpu = simulate(m; kw...)
    tg = @elapsed gpu = simulate(m; kw..., backend=CUDABackend())
    e = reldiff(cpu.v, gpu.v)
    @printf("forward  %-4s: CPU %.2f s, CUDA %.2f s, difference %.1e\n", mode, tc, tg, e)
    e < 1e-10 || (global ok = false)
    # misfit gradient against a model with 3 % higher vs
    obs = simulate(CartData(m.x.val, m.y.val, m.z.val, merge(m.fields, (; vs=m.fields.vs .* 1.03))); kw...)
    kg = (; dt=DT, nt=kw.nt, sources=test_source(), wave=kw.wave, T=Float64)
    misfit_gradient(m, obs; kg..., nt=kw.nt, backend=CUDABackend())                   # compile
    tc = @elapsed (Jc, gc) = misfit_gradient(m, obs; kg...)
    tg = @elapsed (Jg, gg) = misfit_gradient(m, obs; kg..., backend=CUDABackend())
    e = maximum(f -> reldiff(gc.fields[f], gg.fields[f]), mode === :sh ? (:rho, :vs, :Qs) : (:rho, :vp, :vs, :Qp, :Qs))
    @printf("gradient %-4s: CPU %.2f s, CUDA %.2f s, misfit %.1e, gradient difference %.1e\n", mode, tc, tg, abs(Jg / Jc - 1), e)
    e < 1e-8 || (global ok = false)
end
println(ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED")
