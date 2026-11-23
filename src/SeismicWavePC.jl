"""
    SeismicWavePC

Julia version of the OpenSWPC finite-difference seismic wave propagation code (Maeda et al., 2017),
for 2D (P-SV, SH) and 3D visco-elastic media given as GeophysicalModelGenerator `CartData`.
Runs on CPUs and GPUs (KernelAbstractions) and on multiple MPI processes (ImplicitGlobalGrid).
"""
module SeismicWavePC

using KernelAbstractions
using GeophysicalModelGenerator: CartData, GeoData, ProjectionPoint, convert2CartData, write_paraview, movie_paraview, xyz_grid
import ImplicitGlobalGrid as IGG
import MPI
using Printf

include("viscoelastic.jl")
include("sources.jl")
include("medium.jl")
include("kernels.jl")
include("pml.jl")
include("simulation.jl")
include("files.jl")
include("layered.jl")

export MomentTensorSource, DoubleCoupleSource, Station, Seismograms, simulate, moment_rate, CPU
export read_netcdf, write_sac, read_sac, read_lhm, read_lgm, read_grd

end
