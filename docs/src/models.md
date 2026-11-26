# Models

## The model

A model is a GMG `CartData` (coordinates in km, z up; constant spacing in x and y, see below for z) with the fields
- `rho` [kg/m³] (or g/cm³ with `density_unit = :g_cm3`, as in OpenSWPC),
- `vp`, `vs` [km/s],
- optionally `Qp`, `Qs` (without them the model is elastic). Q is constant between `fq_min` and `fq_max`, and the velocities
  apply at `fq_ref` (keywords of `simulate`; defaults 0.05, 5 and 1 Hz),
- optionally `epsilon`, `delta`, `gamma`: VTI anisotropy (see below).

Cells with `vp = vs = 0` are air: the free surface and topography follow from them. Cells with `vs = 0` and `vp > 0` are
water (oceans, lakes). A `CartData` of size `(nx, 1, nz)` is a 2D model (P-SV by default, `wave = :sh` for SH waves).
Any GMG tool can be used to build the model: layers, volcanoes, spheres, slabs, geographic data and projections.

Topography is a staircase of cells. As in OpenSWPC, the shear modulus at a shear-stress point is the harmonic mean of the
four cells around it, so it is zero next to air, also at the inner corner at the foot of every step. A top cell one cell wide
in x (or y) then has no shear coupling to the ground in that direction and does not move in it. With
`simulate(...; free_surface = :corners)` these inner corners (one cell of the four with μ = 0) take the harmonic mean of the
three solid cells; flat surfaces and outer corners are unchanged. In 2D tests on stair-stepped slopes of 10–30° against
the exact flat surface, it makes the surface seismograms 2–4 times more accurate (misfit 17 instead of 34–52 % at 20 cells
per S wavelength, 3–4 instead of 6–12 % at 80). See [Validation](@ref) for a benchmark with tilted planes and a hill.

## Anisotropy (VTI)

With any of the fields `epsilon`, `delta`, `gamma` (Thomsen parameters; missing ones are 0) the model is transversely
isotropic with a vertical symmetry axis (VTI: layered sediments, horizontally aligned cracks or melt lenses). `vp` and `vs`
are then the vertical velocities:

| | |
|---|---|
| qP | `vp` vertically, `vp √(1 + 2ε)` horizontally; `δ` shapes the wavefront near the vertical |
| SH | `vs` vertically, `vs √(1 + 2γ)` horizontally |
| qSV | `vs` vertically and horizontally, its anisotropy follows from ε − δ |

```julia
model = CartData(X, Y, Z, (; rho, vp, vs, epsilon=fill(0.1, size(X)), delta=fill(0.05, size(X)), gamma=fill(0.08, size(X))))
```

- The symmetry axis is the z axis of the grid: every stress needs only the strain rates at its own point, as in isotropic
  media, so the accuracy and the stability (time step: the fastest qP direction) are those of the isotropic code. Tilted
  axes (TTI) are not supported.
- **Attenuation:** the normal stiffness is split as in the isotropic scheme into a P part (`Qp`) and the rest (`Qs`), and
  stays symmetric: qP waves along the axes attenuate with `Qp`, S waves with `Qs`; isotropic media give exactly the
  isotropic scheme.
- Everything else works as for isotropic models: free surface and topography, oceans, PML, sources and stations between
  grid points, GPUs, MPI (also a [`DistributedModel`](@ref) created with `anisotropy = true`), gradients (`misfit_gradient` also returns `epsilon`, `delta`, `gamma`; exact, as checked against
  automatic differentiation) and Born modelling. Not yet: plane waves and teleseismic waves (their incoming field assumes
  isotropic layers).
- Models without these fields run the isotropic kernels: anisotropy costs nothing when it is not used.

## Finer cells near the surface

The spacing in z may vary: give the `CartData` the depths you want, for example fine cells in slow near-surface layers
(sediments, volcanic deposits) and coarser ones at depth. The solver then works on a grid stretched in depth (the depth
derivatives are taken along the cell index and divided by the local spacing, with 4th-order consistent factors), so that
the waves see the same physics with far fewer cells. The spacing must change smoothly, by at most 25 % from cell to cell (a
few percent is best). The spacing in x and y stays constant.

```julia
depths, h = [0.0025], 0.005                         # cell centres [km]: 5 m cells down to 250 m,
while depths[end] < 3.0                              # then 3 % larger per cell up to 40 m
    hn = depths[end] < 0.25 ? 0.005 : min(1.03h, 0.04)
    push!(depths, depths[end] + (h + hn) / 2)
    h = hn
end
z = [reverse(-depths); 0.0025; 0.0075]              # z up, with 2 cells of air on top
X, Y, Z = xyz_grid(range(0, 4, step=0.005), 0, z)
```

Everything else works as on a uniform grid: sources and stations (also between grid points), the PML (its profile follows
the actual depths), MPI, GPUs, and the exact adjoint gradients. The time step is limited by the smallest cells, as before:
dt < 6/7 / (vp √(1/dx² + 1/dz²)) layer by layer, so fine cells in slow rock cost little extra. Plane waves need a uniform
grid.

Example: a 250 m thick layer with vs = 0.4 km/s over faster rock (4 Hz Ricker, P-SV, stations 10 m deep), compared with a
uniform 5 m grid:

| Grid in depth | Cells in depth | Time | Difference (vx, vz) |
|---|---:|---:|---:|
| uniform 5 m (reference) | 602 | 12.0 s | — |
| stretched, 5 m to 250 m, then +3 % per cell to 40 m | 163 | 1.7 s | 1.2 %, 1.8 % |
| uniform 10 m | 302 | 1.7 s | 10 %, 5.3 % |
| uniform 20 m | 152 | 1.1 s | 67 %, 29 % |

## Models built per MPI process

A `CartData` holds the whole model, and in a run on several MPI processes every process sets up the whole medium before it
keeps the part of its subdomain. A model that does not fit into the memory of one process is given as a
[`DistributedModel`](@ref) instead: the coordinates of the whole grid and a function that builds the model of a range of
columns. Every process calls it once, for its own subdomain:

```julia
x, y, z = range(0, 100, 2001), range(0, 80, 1601), range(-30, 2, 641)     # cell centres [km], z up
model = DistributedModel(x, y, z) do ix, iy
    X, Y, Z = xyz_grid(x[ix], y[iy], z)                                   # this part of the grid: use any GMG tool on it
    air = Z .> 0
    (rho=ifelse.(air, 1.0, 2700.0), vp=ifelse.(air, 0.0, 5.0), vs=ifelse.(air, 0.0, 2.9), Qp=fill(200.0, size(X)), Qs=fill(100.0, size(X)))
end
result = simulate(model; nproc=(2, 2), dt, nt, sources, stations)
```

The function returns a `NamedTuple` of arrays of size `(length(ix), length(iy), length(z))` or a `CartData`, with the fields of
a `CartData` model (with `DistributedModel(...; anisotropy = true)` also `epsilon`, `delta`, `gamma`). The results are those of the same model given as a `CartData`. The memory a process needs for the setup
then grows with its subdomain, not with the model. Moment tensor and force sources at their cells (`interpolation = :cell`),
stations, snapshots, checkpoints, GPUs and stretched grids work as usual; plane and teleseismic waves, sources between grid
points, misfit gradients and Born modelling need a `CartData`. The wavefield at the end is not collected (`wavefield = true` does).

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
