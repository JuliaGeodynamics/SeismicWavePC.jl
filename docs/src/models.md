# Models

## The model

A model is a regular GMG `CartData` (coordinates in km, z up) with the fields
- `rho` [kg/m³] (or g/cm³ with `density_unit = :g_cm3`, as in OpenSWPC),
- `vp`, `vs` [km/s],
- optionally `Qp`, `Qs` (without them the model is elastic). Q is constant between `fq_min` and `fq_max`, and the velocities
  apply at `fq_ref` (keywords of `simulate`; defaults 0.05, 5 and 1 Hz).

Cells with `vp = vs = 0` are air: the free surface and topography follow from them. Cells with `vs = 0` and `vp > 0` are
water (oceans, lakes). A `CartData` of size `(nx, 1, nz)` is a 2D model (P-SV by default, `wave = :sh` for SH waves).
Any GMG tool can be used to build the model: layers, volcanoes, spheres, slabs, geographic data and projections.

## Geographic coordinates

Models built from geographic data with a GMG `ProjectionPoint` keep their projection: sources and stations can be given in
longitude and latitude with the same `ProjectionPoint` ([Sources and stations](@ref)).

## Models from thermomechanical and petrological models

The temperature of a [MagmaThermoKinematics.jl](https://github.com/boriskaus/MagmaThermoKinematics.jl) model, or any other
temperature field, can be turned into density, seismic velocities and melt fraction with
[MAGEMin](https://github.com/ComputationalThermodynamics/MAGEMin_C.jl) phase diagrams, including the reduction of the
velocities by partial melt and a low Q in partially molten rock. See the Unzen examples in [Examples](examples.md).

## Random media

```julia
xi = random_medium(x, y, z; spectrum=:vonkarman, a=0.5, epsilon=0.05, kappa=0.3)
model = perturb(model, xi)
```
makes fluctuations with a Gaussian, exponential or von Kármán spectrum (anisotropic with `a=(ax, ay, az)`) and applies them as
OpenSWPC does (vp, vs × (1 + ξ), ρ × (1 + 0.8 ξ) in solid rock). `read_random_medium(file)` reads OpenSWPC's random media.

## Layered models and OpenSWPC files

`read_lhm`, `read_lgm` and `read_grd` build models from layered descriptions (with OpenSWPC's options `vcut`, `munk` and
`earth_flattening`), and `read_netcdf` reads OpenSWPC's netCDF velocity models; see [Coming from OpenSWPC](@ref).
`flatten_earth(model)` applies the Earth-flattening transformation to any model.
