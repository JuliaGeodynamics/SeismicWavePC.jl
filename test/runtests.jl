using Test

include("setup.jl")

# Pkg.test("SeismicWavePC"; test_args=["types", "gradient"]) runs only those files; "file:a,b" only those parts of a file
# ("mpi:forward,born": parts of mpi_script.jl, "curvilinear:1": the first half of test_curvilinear.jl). MPI first: its 4 processes start while this process is still small (memory of the CI runners)
const TESTS = ["mpi", "basics", "openswpc", "files", "types", "gradient", "born", "anisotropy", "inversion", "interpolation", "distributed", "rayleigh", "sources", "validation", "curvilinear"]
# CI runs the groups as parallel jobs (environment variable SWPC_TEST_GROUP; unset: all tests)
const GROUPS = Dict("mpi1" => ["mpi:forward,gradient,born"],
                    "mpi2" => ["mpi:vti,mpml,other"],
                    "mpi3" => ["mpi:curvilinear"],
                    "mpi4" => ["mpi:curvilinear3d"],
                    "gradient" => ["gradient", "basics", "openswpc", "files", "types", "inversion", "interpolation", "rayleigh"],
                    "curvilinear1" => ["curvilinear:1"],
                    "curvilinear2" => ["curvilinear:2"],
                    "other" => ["born", "anisotropy", "distributed", "sources", "validation"])
const TEST_PARTS = Ref(String[])            # the parts of the file being run (empty: all; read by the test files)
group = get(ENV, "SWPC_TEST_GROUP", "")
isempty(group) || haskey(GROUPS, group) || error("SWPC_TEST_GROUP: one of $(join(sort(collect(keys(GROUPS))), ", ")), got $group")

@testset "SeismicWavePC" begin
    for entry in (!isempty(ARGS) ? ARGS : isempty(group) ? TESTS : GROUPS[group])
        t, parts = occursin(':', entry) ? split(entry, ':') : (entry, "")
        TEST_PARTS[] = isempty(parts) ? String[] : String.(split(parts, ','))
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
