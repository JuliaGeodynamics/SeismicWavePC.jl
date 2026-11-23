# Sources and stations

## Sources

- `DoubleCoupleSource(x, y, z; Mw, strike, dip, rake, t0, trise)`: a double couple from fault angles (strike clockwise from
  north = +y).
- `MomentTensorSource(x, y, z, M0, M; t0, trise)`: a general moment tensor `M = (xx, yy, zz, yz, xz, xy)` in the model frame.
- `ForceSource(x, y, z; fx, fy, fz, t0, trise)`: a point force [N] (OpenSWPC's body force).
- `PlaneWave(:p or :s; depth, length, strike, dip, rake)`: an incoming plane wave (OpenSWPC's `pw_mode`).

All can be combined in a vector, and all accept `(lon, lat, z, proj::ProjectionPoint)` instead of `x, y, z`. Sources outside
the model or in the air are ignored with a warning (as in OpenSWPC, e.g. parts of a finite fault). The source time function
is set with `stf = :kupper` (default), `:boxcar`, `:triangle`, `:herrmann`, `:cosine`, `:texp` or `:ricker` (see
[`moment_rate`](@ref)).

## Stations

`Station(name, x, y, z)` or `Station(name, lon, lat, z, proj)`; with `surface = true` the station is placed in the first
solid cell below the free surface. Velocity, displacement, stress and strain are recorded every `record_every` steps.

## Reciprocal Green's functions

`reciprocal_green(model, receiver, points; component=:z)` puts a unit force at the receiver and records the displacement
gradients at many `points` (OpenSWPC's `green_mode`). `seismogram(green, n, source)` then gives the receiver seismogram for a
source at any of the points, without a simulation per source.
