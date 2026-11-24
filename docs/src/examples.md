# Examples

The examples are in the [`examples`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/tree/main/examples) folder of the
repository (see [Installation](@ref) for their environments). Most take `cuda`, `amdgpu` or `metal` as last argument to run
on a GPU.

## 3D: waves through the magmatic system below an Unzen-like volcano

[`MagmaThermoKinematics/Unzen_3D.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/MagmaThermoKinematics/Unzen_3D.jl): a volcano on a peninsula, with a temperature
field from a linear geotherm and superposed Gaussian anomalies (a deep mush reservoir, a body rising from it, a reservoir
below the summit). MAGEMin phase diagrams give density, velocities and melt fraction; partially molten rock has a low Q.
The movie of a Mw 4 earthquake is made while the simulation runs.

![Waves below an Unzen-like volcano](assets/Unzen_3D.gif)

## 3D: a volcano with a magma chamber

[`3D_volcano.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/3D_volcano.jl) builds a volcano with GMG (`add_volcano!`), a magma chamber and stations given in
longitude and latitude; [`3D_mpi.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/3D_mpi.jl) runs it on 4 MPI processes, and
[`3D_volcano_movie.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/3D_volcano_movie.jl) makes a movie of it:

![Waves in a volcano](assets/3D_volcano.gif)

## 2D: Unzen from MagmaThermoKinematics.jl and MAGEMin

[`MagmaThermoKinematics/Unzen_2D.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/MagmaThermoKinematics/Unzen_2D.jl) runs the 2D Unzen model of
[MagmaThermoKinematics.jl](https://github.com/boriskaus/MagmaThermoKinematics.jl) (about 6 minutes at 100 m resolution),
computes density, velocities and melt fraction with [MAGEMin](https://github.com/ComputationalThermodynamics/MAGEMin_C.jl),
and simulates an earthquake recorded by seismometers just below the surface:

![Waves through the magmatic system below Unzen](assets/Unzen_2D_waves.gif)

*Top: S-wave velocity from MAGEMin for the temperature field of MTK (black: molten, white lines: more than 10 % melt).
Bottom: vertical velocity of a Mw 4 earthquake (star), recorded by seismometers just below the surface (triangles).*

![Seismograms of the Unzen example](assets/Unzen_2D_seismograms.png)

*Vertical velocity at the seismometers (left) and peak velocities (right). Stations above the partially molten region (shaded)
record weaker S waves and a longer coda.*

## Marmousi2

The elastic Marmousi2 model (Martin et al., 2006; 17 km × 3.5 km, 451 m of water) is the classic benchmark of seismic wave
propagation codes. [`Marmousi2.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/Marmousi2.jl) downloads
it on first use, simulates a marine shot (explosion 10 m below the sea surface, 5 Hz Ricker wavelet) and makes a movie of the
waves over the P-wave velocity while the simulation runs (10 m grid, 5 s, 12 s on a laptop):

![Waves in Marmousi2](assets/Marmousi2.gif)

![Marmousi2](assets/Marmousi2.png)

*P-wave velocity, a snapshot of the vertical velocity, and the shot gather recorded 10 m below the sea surface.* The same
model is the benchmark in [Performance](@ref), and the comparison with OpenSWPC and SeismicWaves.jl is in
[Validation](@ref).

## Reverse-time migration of Marmousi2

[`Marmousi2_RTM.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/Marmousi2_RTM.jl) images Marmousi2
by reverse-time migration (RTM): 40 marine shots (explosions 10 m below the sea surface, 5 Hz Ricker wavelet), each recorded by
the receivers within 3 km, are migrated in a smooth background model (the rock smoothed over 150 m; the water layer and the
sea floor stay sharp). The image is the gradient of the least-squares misfit with respect to vp in the background model,
`misfit_gradient`, which correlates the source wavefield with the back-propagated reflections:
- only the part of each trace after the direct wave through the water enters the misfit (`L2Misfit(; window)` per receiver;
  earlier arrivals are waves refracted in the rock, whose travel times the smooth model does not reproduce);
- `illumination = true` also returns the energy of the source wavefield, Σₜ |v|², per cell; dividing by it compensates the
  weaker illumination of the deep parts;
- a Laplacian filter removes the low-wavenumber artefacts of RTM.

![RTM image of Marmousi2](assets/Marmousi2_RTM.png)

*The flat layers on the left, the dipping, faulted layers in the centre and the anticline on the right are imaged; with the
illumination compensation down to about 3 km.* 8 minutes on the laptop GPU (Metal; 10 m grid, 3.5 s): 3 minutes to model the
40 shots in the true model, 5 minutes for the migration (`julia --project=examples examples/Marmousi2_RTM.jl metal`).

## Waveform inversion and Hamiltonian Monte Carlo

In [`examples/Inversion`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/tree/main/examples/Inversion) (own environment),
noisy seismograms of one earthquake at 11 stations are inverted for the shear velocity in three depth bands:
- [`inversion_turing.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/Inversion/inversion_turing.jl): L-BFGS with Optim.jl, then Hamiltonian Monte Carlo (NUTS) with
  Turing.jl;
- [`inversion_hmc.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/Inversion/inversion_hmc.jl): NUTS with AdvancedHMC.jl directly, without Turing and without AD.

![Posterior of the inversion](assets/inversion_turing.png)

*Posterior distributions (NUTS, Turing) of the shear-velocity changes, with the true values (black) and the L-BFGS result (red).*

## Others

- [`2D_PSV_hill.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/2D_PSV_hill.jl): layered crust under a hill, with a record section ([Getting started](@ref)).
- [`benchmark.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/benchmark.jl), [`benchmark_marmousi2.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/benchmark_marmousi2.jl): the timings in
  [Performance](@ref).
- [`check_cuda.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/check_cuda.jl), [`check_cuda_mpi.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/check_cuda_mpi.jl): checks of NVIDIA GPUs and of MPI on GPUs.
