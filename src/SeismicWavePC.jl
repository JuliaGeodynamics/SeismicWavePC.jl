"""
    SeismicWavePC

Julia version of the OpenSWPC finite-difference seismic wave propagation code (Maeda et al., 2017),
for 2D (P-SV, SH) and 3D visco-elastic media given as GeophysicalModelGenerator `CartData`, or built per MPI process
(`DistributedModel`).
Runs on CPUs and GPUs (KernelAbstractions) and on multiple MPI processes (ImplicitGlobalGrid).
"""
module SeismicWavePC

using KernelAbstractions
using GeophysicalModelGenerator: CartData, GeoData, ProjectionPoint, convert2CartData, write_paraview, movie_paraview, xyz_grid
import ImplicitGlobalGrid as IGG
import MPI
using Printf
using Serialization: serialize, deserialize
using FFTW: fft, ifft, fftfreq, rfft, irfft
using SparseArrays: sparse
using LinearAlgebra: LinearAlgebra
import Random

include("viscoelastic.jl")
include("sources.jl")
include("medium.jl")
include("kernels.jl")
include("interpolation.jl")
include("pml.jl")
include("planewave.jl")
include("injection.jl")
include("snapshots.jl")
include("simulation.jl")
include("files.jl")
include("layered.jl")
include("green.jl")
include("randommedium.jl")
include("misfits.jl")
include("gradient.jl")
include("adjoint.jl")
include("born.jl")
include("anisotropy.jl")
include("inversion.jl")

export MomentTensorSource, DoubleCoupleSource, ForceSource, PlaneWave, TeleseismicWave, Station, Seismograms, simulate, moment_rate, SampledSTF, CPU, DistributedModel
export ReciprocalGreen, reciprocal_green, seismogram, Snapshots, random_medium, perturb, read_random_medium, misfit_gradient, born,
       L2Misfit, TravelTimeMisfit, ModelParameters, initial_parameters, to_model, WaveformProblem, objective,
       objective_and_gradient, fg!
export read_netcdf, write_sac, read_sac, read_lhm, read_lgm, read_grd, flatten_earth, flatten_depth, read_snapshots, SnapshotSeries

end
