using Test

include("setup.jl")

# Pkg.test("SeismicWavePC"; test_args=["types", "gradient"]) runs only those files. MPI first: its 4 processes start while
# this process is still small (memory of the CI runners)
const TESTS = ["mpi", "basics", "openswpc", "files", "types", "gradient", "born", "anisotropy", "inversion", "interpolation", "distributed"]

@testset "SeismicWavePC" begin
    for t in (isempty(ARGS) ? TESTS : ARGS)
        # MPI tests need a working mpiexec; MicrosoftMPI on Windows CI runners is not set up for that
        t == "mpi" && Sys.iswindows() && continue
        println("running test_$t (peak memory so far $(round(Sys.maxrss() / 2^30, digits=1)) GB)"); flush(stdout)
        include("test_$t.jl")
    end
end
