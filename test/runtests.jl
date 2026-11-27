using Test

include("setup.jl")

# Pkg.test("SeismicWavePC"; test_args=["types", "gradient"]) runs only those files; "mpi:forward,born" only those parts of
# the MPI test (mpi_script.jl). MPI first: its 4 processes start while this process is still small (memory of the CI runners)
const TESTS = ["mpi", "basics", "openswpc", "files", "types", "gradient", "born", "anisotropy", "inversion", "interpolation", "distributed", "rayleigh", "curvilinear"]
# CI runs the groups as parallel jobs (environment variable SWPC_TEST_GROUP; unset: all tests)
const GROUPS = Dict("mpi1" => ["mpi:forward,gradient,born,vti"],
                    "mpi2" => ["mpi:mpml,other", "basics", "openswpc", "files", "types", "inversion", "interpolation", "rayleigh"],
                    "mpi3" => ["mpi:curvilinear"],
                    "gradient" => ["gradient"],
                    "curvilinear" => ["curvilinear"],
                    "other" => ["born", "anisotropy", "distributed"])
const MPI_PARTS = Ref(["forward", "gradient", "born", "vti", "mpml", "other", "curvilinear"])        # (read by test_mpi.jl)
group = get(ENV, "SWPC_TEST_GROUP", "")
isempty(group) || haskey(GROUPS, group) || error("SWPC_TEST_GROUP: one of $(join(sort(collect(keys(GROUPS))), ", ")), got $group")

@testset "SeismicWavePC" begin
    for entry in (!isempty(ARGS) ? ARGS : isempty(group) ? TESTS : GROUPS[group])
        t, parts = occursin(':', entry) ? split(entry, ':') : (entry, "")
        isempty(parts) || (MPI_PARTS[] = split(parts, ','))
        # MPI tests need a working mpiexec; MicrosoftMPI on Windows CI runners is not set up for that
        t == "mpi" && Sys.iswindows() && continue
        println("running test_$entry (peak memory so far $(round(Sys.maxrss() / 2^30, digits=1)) GB)"); flush(stdout)
        # the time of every testset, printed when the file is done (CI logs; a job that times out shows the files before)
        t0 = time()
        ts = @testset verbose = true "$entry" begin
            include("test_$t.jl")
        end
        println("test_$entry: $(round((time() - t0) / 60, digits=1)) min")
        isdefined(Test, :print_test_results) && Test.print_test_results(ts)
        flush(stdout)
    end
end
