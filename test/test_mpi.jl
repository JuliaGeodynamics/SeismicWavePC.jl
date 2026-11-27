# Run test/mpi_script.jl on 4 MPI processes: domain-decomposed runs must equal serial runs.
using Test, MPI

@testset "MPI: 4 processes give the same result as 1" begin
    script = joinpath(@__DIR__, "mpi_script.jl")
    # each part in fresh processes: the code compiled for one part (~3 GB per process for 3D gradients) is freed before the next
    for part in ("forward", "gradient", "born", "vti", "mpml", "other", "curvilinear")
        cmd = `$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script $part`
        @test success(pipeline(cmd, stdout=stdout, stderr=stderr))
    end
end
