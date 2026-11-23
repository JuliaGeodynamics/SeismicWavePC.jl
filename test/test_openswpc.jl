# Compare with seismograms computed by the Fortran OpenSWPC code (swpc_psv, swpc_sh, swpc_3d; Cerjan or PML absorber, na = 20,
# fq_min = 0.05, fq_max = 5, fq_ref = 1, kupper source time function) for the same models, stored in test/reference.
using Test, DelimitedFiles

@testset "agrees with OpenSWPC: $mode" for (mode, file) in ((:psv, "psv"), (:sh, "sh"), (:p3d, "3d"))
    ref = readdlm(joinpath(@__DIR__, "reference", "$file.txt"), comments=true)
    model = test_model(mode)
    r = simulate(model; dt=DT, nt=NT, sources=test_source(), stations=test_stations(mode), wave=mode === :p3d ? :psv : mode, verbose=false, absorber=:cerjan)
    v = reshape(r.v, NT, :)                        # columns: station-major, components fastest (as in the reference)
    @test size(v) == size(ref)
    @test maximum(abs, v .- ref) / maximum(abs, ref) < 1e-3
    # displacement, stress and strain (every 4th sample), each compared relative to its own maximum
    ref2 = readdlm(joinpath(@__DIR__, "reference", "$(file)_u_stress_strain.txt"), comments=true)
    nc, ns = length(r.components), length(r.tensor_components)
    q = cat(r.u, r.stress, r.strain, dims=2)[1:4:end, :, :]
    @test size(reshape(q, size(q, 1), :)) == size(ref2)
    for (range, name) in ((1:nc, "displacement"), (nc+1:nc+ns, "stress"), (nc+ns+1:nc+2ns, "strain"))
        a, b = q[:, range, :], reshape(ref2, size(q))[:, range, :]
        @test maximum(abs, a .- b) / maximum(abs, b) < 1e-3
    end
end

@testset "agrees with OpenSWPC, PML: $mode" for (mode, file) in ((:psv, "psv"), (:sh, "sh"), (:p3d, "3d"))
    ref = readdlm(joinpath(@__DIR__, "reference", "$(file)_pml.txt"), comments=true)
    r = simulate(test_model(mode); dt=DT, nt=NT, sources=test_source(), stations=test_stations(mode), wave=mode === :p3d ? :psv : mode,
                 verbose=false, absorber=:pml)
    @test maximum(abs, reshape(r.v, NT, :) .- ref) / maximum(abs, ref) < 1e-3
end
