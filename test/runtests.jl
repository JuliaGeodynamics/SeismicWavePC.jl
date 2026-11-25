using Test

include("setup.jl")

# Pkg.test("SeismicWavePC"; test_args=["types", "gradient"]) runs only those files
const TESTS = ["basics", "openswpc", "files", "types", "gradient", "inversion", "interpolation", "distributed", "mpi"]

@testset "SeismicWavePC" begin
    for t in (isempty(ARGS) ? TESTS : ARGS)
        # MPI tests need a working mpiexec; MicrosoftMPI on Windows CI runners is not set up for that
        t == "mpi" && Sys.iswindows() && continue
        include("test_$t.jl")
    end
end
