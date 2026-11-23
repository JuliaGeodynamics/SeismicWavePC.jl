# Sources and stations

## Sources

- `DoubleCoupleSource(x, y, z; Mw, strike, dip, rake, t0, trise)`: a double couple from fault angles (strike clockwise from
  north = +y).
- `MomentTensorSource(x, y, z, M0, M; t0, trise)`: a general moment tensor `M = (xx, yy, zz, yz, xz, xy)` in the model frame.
- `ForceSource(x, y, z; fx, fy, fz, t0, trise)`: a point force [N] (OpenSWPC's body force).
- `PlaneWave(:p or :s; depth, length, strike, dip, rake)`: an incoming plane wave (OpenSWPC's `pw_mode`).

All can be combined in a vector, and all accept `(lon, lat, z, proj::ProjectionPoint)` instead of `x, y, z`. Sources outside
the model or in the air are ignored with a warning (as in OpenSWPC, e.g. parts of a finite fault).

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
