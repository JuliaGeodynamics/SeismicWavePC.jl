using Test

include("setup.jl")

@testset "SeismicWavePC" begin
    include("test_basics.jl")
    include("test_openswpc.jl")
    include("test_files.jl")
    include("test_types.jl")
    include("test_gradient.jl")
    include("test_inversion.jl")
    include("test_interpolation.jl")
    # MPI tests need a working mpiexec; MicrosoftMPI on Windows CI runners is not set up for that
    Sys.iswindows() || include("test_mpi.jl")
end
