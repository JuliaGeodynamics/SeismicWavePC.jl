# Sources and stations

## Sources

- `DoubleCoupleSource(x, y, z; Mw, strike, dip, rake, t0, trise)`: a double couple from fault angles (strike clockwise from
  north = +y).
- `MomentTensorSource(x, y, z, M0, M; t0, trise)`: a general moment tensor `M = (xx, yy, zz, yz, xz, xy)` in the model frame.
- `ForceSource(x, y, z; fx, fy, fz, t0, trise)`: a point force [N] (OpenSWPC's body force).
- `TeleseismicWave(:p, :sv or :sh; angle, backazimuth, box, t0, trise, amplitude)`: a plane wave from below, injected along
  a box (see [Teleseismic waves](@ref)).
- `PlaneWave(:p or :s; depth, length, strike, dip, rake)`: an incoming plane wave as an initial wavefield (OpenSWPC's
  `pw_mode`; for teleseismic modelling use `TeleseismicWave`).

All can be combined in a vector, and all accept `(lon, lat, z, proj::ProjectionPoint)` instead of `x, y, z`. Sources outside
the model or in the air are ignored with a warning (as in OpenSWPC, e.g. parts of a finite fault).

## Teleseismic waves

`TeleseismicWave` models a plane wave arriving from below at an angle of incidence and a backazimuth, as from a distant
earthquake: receiver functions, teleseismic tomography and full-waveform inversion of teleseismic body waves.

```julia
w = TeleseismicWave(:p; angle=20, backazimuth=240, box=(5, 35, 5, 35, -30), trise=1.0, t0=0.5)   # 3D; 2D: box=(xmin, xmax, zmin)
r = simulate(model; dt, nt, sources=w, stations)
```

The wave is injected along the sides and the bottom of `box` (total-field / scattered-field method): inside the box the
simulation computes the total field (the incoming wave, its reflections at the free surface and at the layer interfaces,
and the waves scattered by the structure in the box), outside only the scattered waves, which leave through the absorbing
boundaries. The incoming field is the exact solution for the horizontally layered background in which the box sits
(computed in the frequency domain for every depth of the grid, with all reverberations), so
- the wave enters the box continuously from the bottom and the sides, at any angle and backazimuth, without artefacts from
  the edges of the model;
- P, SV and SH waves, in 2D (P-SV, SH) and 3D, on CPUs, GPUs and with MPI, also on grids stretched in depth;
- `misfit_gradient` gives exact gradients with it (the injection acts as sources independent of the model inside the box).

Requirements: the model must be horizontally layered, elastic and without water along the box boundary and below it (the
background; it is read from the model), and the box must keep 4 cells from the absorbing layers. Inside the box the model
can be anything (also attenuating). The particle velocity of the incoming wave is `amplitude * stf(t)` (for source time
functions with integral 1, such as the default `:kupper`, it carries the displacement `amplitude` [m]), it reaches the box
first at `t0`, and `backazimuth` is the direction towards the source (clockwise from north, +y); in 2D 270° (from −x, the
default) or 90°. SH waves are polarized 90° counterclockwise from the direction of propagation (seen from above).

Accuracy (100 and 200 m grids, see [Validation](@ref)): seismograms agree with the analytical solution of a half-space
(P and SV, the wave and its reflections at the free surface) to 0.1–0.5 %, with the exact solution for a crust over a mantle
(also oblique to the grid in 3D) to 0.1–0.6 %; the ratio of horizontal and vertical motion at the surface is exact to
2·10⁻⁴ (P at 0–40°), and waves leaking out of the box stay below 10⁻³ of the incoming wave. An example:
[`teleseismic_benchmark.jl`](https://github.com/JuliaGeodynamics/SeismicWavePC.jl/blob/main/examples/teleseismic_benchmark.jl).

`PlaneWave` (OpenSWPC) instead starts the wave as an initial wavefield that has to fit into the model: at oblique incidence
it is cut where it reaches the sides, and the edges of the model send spurious surface waves of a few percent of its
amplitude.

## Between grid points

By default (`interpolation = :cell`), a source or station is placed at the cell that contains it, as in OpenSWPC: the
position is rounded to the grid. With `simulate(...; interpolation=:sinc)` sources and stations sit at their exact
position: every quantity is spread onto (sources) or interpolated from (stations) the points of its own staggered grid
around it with a Kaiser-windowed sinc function (Hicks, 2002; 8 points per axis, as in SeismicWaves.jl).
- Shifting sources and stations by a fraction of a cell then shifts the seismograms smoothly; the source-location gradient
  of [`misfit_gradient`](@ref) is exact.
- Points in the air (and, for shear stresses, in water) are left out and the other weights scaled up, so stations on
  the free surface (`surface = true`) and sources in water work.
- Velocity, displacement and stress are interpolated; the strain is still taken at the cell.
- It works with MPI (the points may lie on several processes) and on GPUs.

## Source time functions

The keyword `stf` of [`simulate`](@ref) sets the moment-rate function (or force history) of all sources, with their origin
time `t0` and rise time `trise` (see [`moment_rate`](@ref)):
- `:kupper` (default), `:boxcar`, `:triangle`, `:herrmann`, `:cosine`, `:texp` (as in OpenSWPC), `:ricker`, `:gaussian`,
  `:gaussian_derivative`;
- any function `stf(t, t0, trise)`;
- sampled values: `stf = SampledSTF(t, values)` (times after `t0`, linearly interpolated), e.g. a source time function
  from a source inversion, or one that is being inverted for: [`misfit_gradient`](@ref) returns the gradient with respect
  to it.

## Stations

`Station(name, x, y, z)` or `Station(name, lon, lat, z, proj)`; with `surface = true` the station is placed in the first
solid cell below the free surface. Velocity, displacement, stress and strain are recorded every `record_every` steps.

## Reciprocal Green's functions

`reciprocal_green(model, receiver, points; component=:z)` puts a unit force at the receiver and records the displacement
gradients at many `points` (OpenSWPC's `green_mode`). `seismogram(green, n, source)` then gives the receiver seismogram for a
source at any of the points, without a simulation per source.
