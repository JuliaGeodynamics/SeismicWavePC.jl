# Output

## Seismograms

`simulate` returns [`Seismograms`](@ref): `result.t` (times), `result.v` (velocity [m/s]), `result.u` (displacement [m]),
`result.stress` [Pa] and `result.strain`, each as `[it, component, station]`, with z positive up; `result["name"]` gives the
velocities of one station. `write_sac(result, dir)` writes SAC files as OpenSWPC does.

**Seismic Unix.** [`write_su`](@ref)`(result, dir; quantities)` writes SU files as SOFI does: one file per quantity and
component (`swpc.psv.vx.su`, …), one trace per station, with the station (and source) positions, offset, sample interval
and delay in the trace headers. Quantities: `:v`, `:u`, `:stress`, `:strain`, `:p` (pressure, minus the mean normal
stress) and `:div`, `:curl`. [`read_su`](@ref) reads SU files (of either byte order), e.g. a recorded source wavelet:
`su = read_su("wavelet.su"); stf = SampledSTF(su.t, su.data[:, 1])`.

**Divergence and curl.** `simulate(...; derivatives = true)` also returns the divergence and the curl of the velocity at
every station (`result.div[it, station]`, `result.curl[it, component, station]` with components x, y, z [1/s]): P and S
energy, rotational motion. They are 4th-order differences of the velocity at auxiliary stations ±`dh`, ±2`dh` around each
station (`dh` the grid spacing by default); at stations on the surface the vertical derivatives are one-sided from below.
For an explosion in a homogeneous medium the curl is zero to round-off and the divergence the rate of the volumetric strain
(3 %); at a free surface they satisfy the free-surface conditions to 1–3 %.

## Wavefield and movies

- `result.wavefield`: the velocity field at the end, as a `CartData` (collected on one process; `wavefield = false` leaves it
  out).
- `snapshot_every = 100`: writes the velocity field to `snapshot_dir` as a ParaView movie (`wavefield.pvd`).
- `callback = (wavefield, t) -> …`: called every `snapshot_every` steps with the wavefield as a `CartData` (field `V`),
  instead of writing files, e.g. to plot the waves while the simulation runs (see [Getting started](@ref) and the movies in
  [Examples](examples.md)).
- `snapshots = Snapshots(xy=-5.0, xz=0.0, fs=true, quantities=(:v, :u, :ps))`: planes (horizontal, vertical, free surface,
  ocean bottom) in OpenSWPC's netCDF format, including peak ground velocity maps (needs `using NCDatasets`).
  On several MPI processes, process 0 collects every plane and writes the files. With `pieces = true` every process writes
  the points of its own subdomain to files of its own (`swpc.3d.fs.v.0000.nc`, …), without communication, so that no process
  holds a whole plane: for runs on many processes or with large planes.
- `s = read_snapshots("snapshots/swpc.3d.fs.v.nc")` reads the snapshots of a plane, whether in one file or in pieces:
  `s.axes`, `s.t`, `s["Vx", k]` (snapshot `k`), `s["max-H"]` (a field without time).
