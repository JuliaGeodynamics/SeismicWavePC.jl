# Compare with seismograms computed by the Fortran OpenSWPC code (swpc_psv, swpc_sh, swpc_3d; Cerjan absorber, na = 20,
# fq_min = 0.05, fq_max = 5, fq_ref = 1, kupper source time function) for the same models, stored in test/reference.
using Test, DelimitedFiles

@testset "agrees with OpenSWPC: $mode" for (mode, file) in ((:psv, "psv"), (:sh, "sh"), (:p3d, "3d"))
    ref = readdlm(joinpath(@__DIR__, "reference", "$file.txt"), comments=true)
    model = test_model(mode)
    r = simulate(model; dt=DT, nt=NT, sources=test_source(), stations=test_stations(mode), wave=mode === :p3d ? :psv : mode, verbose=false)
    v = reshape(r.v, NT, :)                        # columns: station-major, components fastest (as in the reference)
    @test size(v) == size(ref)
    @test maximum(abs, v .- ref) / maximum(abs, ref) < 1e-3
end
