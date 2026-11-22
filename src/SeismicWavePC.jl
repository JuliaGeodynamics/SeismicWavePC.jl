"""
    SeismicWavePC

Julia version of the OpenSWPC finite-difference seismic wave propagation code (Maeda et al., 2017),
for 2D (P-SV, SH) and 3D visco-elastic media given as GeophysicalModelGenerator `CartData`.
Runs on CPUs and GPUs (KernelAbstractions) and on multiple MPI processes (ImplicitGlobalGrid).
"""
module SeismicWavePC

using KernelAbstractions
using GeophysicalModelGenerator: CartData, write_paraview, movie_paraview
import ImplicitGlobalGrid as IGG
import MPI

include("viscoelastic.jl")
include("sources.jl")
include("medium.jl")
include("kernels.jl")
include("simulation.jl")

export MomentTensorSource, DoubleCoupleSource, Station, simulate, moment_rate, CPU

end
