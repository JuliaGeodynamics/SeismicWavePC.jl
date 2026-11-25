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
PlaneWave
TeleseismicWave
moment_rate
SampledSTF
Snapshots
```

## Models

```@docs
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
