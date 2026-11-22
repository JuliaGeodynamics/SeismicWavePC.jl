# The 3D volcano example on 4 MPI processes (2 x 2 subdomains in x and y), on CPUs or one GPU per process:
#
#   julia --project=examples -e 'using MPI; MPI.install_mpiexecjl()'     # once
#   mpiexecjl -n 4 julia --project=examples examples/3D_mpi.jl          # CPU   (or add: cuda | amdgpu)
#
# The number of cells in x and y must be divisible by the number of processes in that direction.
# With CUDA/AMDGPU, ImplicitGlobalGrid selects one GPU per process on each node.
using MPI
MPI.Init()
const NPROC = (2, 2)
include("3D_volcano.jl")       # `simulate` in there picks up NPROC through the keyword below
