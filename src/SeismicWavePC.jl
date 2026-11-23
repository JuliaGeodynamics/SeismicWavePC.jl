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
using Serialization: serialize, deserialize
using FFTW: fft, ifft, fftfreq
import Random

include("viscoelastic.jl")
include("sources.jl")
include("medium.jl")
include("kernels.jl")
include("pml.jl")
include("planewave.jl")
include("snapshots.jl")
include("simulation.jl")
include("files.jl")
include("layered.jl")
include("green.jl")
include("randommedium.jl")

export MomentTensorSource, DoubleCoupleSource, ForceSource, PlaneWave, Station, Seismograms, simulate, moment_rate, CPU
export ReciprocalGreen, reciprocal_green, seismogram, Snapshots, random_medium, perturb, read_random_medium
export read_netcdf, write_sac, read_sac, read_lhm, read_lgm, read_grd, flatten_earth, flatten_depth

end
