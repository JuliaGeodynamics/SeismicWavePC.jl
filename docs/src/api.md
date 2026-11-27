# API reference

```@docs
SeismicWavePC
```

## Simulation

```@docs
simulate
Seismograms
Station
MomentTensorSource
DoubleCoupleSource
ForceSource
ExplosionSource
finite_fault
read_sources
read_srf
read_cmtsolution
PlaneWave
TeleseismicWave
moment_rate
SampledSTF
BerlageSTF
KlauderSTF
Snapshots
read_snapshots
SeismicWavePC.SnapshotSeries
```

## Models

```@docs
curvilinear_mesh
SeismicWavePC.simulate_curvilinear
SeismicWavePC.simulate_curvilinear3d
DistributedModel
random_medium
perturb
read_random_medium
read_netcdf
read_lhm
read_lgm
read_grd
flatten_earth
flatten_depth
```

## Output and Green's functions

```@docs
write_sac
read_sac
write_su
read_su
ReciprocalGreen
reciprocal_green
seismogram
```

## Gradients and inversion

```@docs
misfit_gradient
born
L2Misfit
TravelTimeMisfit
ModelParameters
initial_parameters
to_model
WaveformProblem
objective
objective_and_gradient
fg!
```
