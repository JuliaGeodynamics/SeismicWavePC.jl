# Coming from OpenSWPC

Velocity models and seismograms in the file formats of OpenSWPC can still be used:

```julia
using SeismicWavePC, NCDatasets
model  = read_netcdf("velocity_model.nc")      # netCDF model of OpenSWPC (vmodel_type = "velm", OpenSWPC.write_netcdf)
model  = read_lhm("layers.dat", x, y, z)       # or a layered model (lhm; read_lgm for lgm) on the grid x, y, z [km];
                                               # options vcut, munk (sea water) and earth_flattening, as in OpenSWPC
model  = read_grd("grd.lst", x, y, z, proj)    # or layer boundaries from GMT grids (grd), placed with a GMG ProjectionPoint
result = simulate(model; dt, nt, sources=source, stations)
write_sac(result, "out/wav"; title="swpc", source, quantities=(:v, :u, :stress, :strain))   # swpc.psv.<station>.Vx.sac, …
t, v, header = read_sac("old_run/wav/swpc.psv.ST01.Vz.sac")   # e.g. to compare with an earlier OpenSWPC run
```

Compared with the Fortran code:

| | OpenSWPC (Fortran, via OpenSWPC.jl) | SeismicWavePC.jl |
|---|---|---|
| Code | Fortran binaries, input files | pure Julia: no binaries, no input files |
| Operating systems | Linux, macOS (Windows via WSL) | Linux, macOS, Windows |
| GPUs | NVIDIA only (OpenACC build) | NVIDIA, AMD, Apple (and more) via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) |
| Parallel CPU | MPI processes, single-threaded each | Julia threads, plus MPI |
| Multi-node / multi-GPU | MPI | MPI via [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) (CUDA- and ROCm-aware MPI) |
| Model input | netCDF velocity file written from GMG | the GMG `CartData` itself |
| Output | SAC and netCDF files | seismograms in memory, final wavefield as a `CartData`, ParaView snapshot movies |
| Geographic coordinates | own projection | the same GMG `ProjectionPoint` used to build the model |
| Safety | | stability (Courant) check, sources and stations outside the model are rejected |
| Waveform inversion | | exact misfit gradients (adjoint-state method; CPU, GPU, MPI) |

How OpenSWPC's options map to SeismicWavePC.jl:

| OpenSWPC | SeismicWavePC.jl |
|---|---|
| input file (`dt`, `nt`, `na`, `fq_min`, `stftype`, …) | keyword arguments of `simulate` (`stf` for `stftype`) |
| velocity model (`vmodel_type`) | any GMG `CartData`; `read_netcdf` (`velm`), `read_lhm` (`lhm`), `read_lgm` (`lgm`), `read_grd` (`grd`); `velm` and `grd` need `using NCDatasets` |
| source and station files | `read_sources(file; format)` (every `stf_format`), `DoubleCoupleSource`, `MomentTensorSource`, `ForceSource` (`bf_mode`), `PlaneWave` (`pw_mode`), `Station` (in x, y or longitude, latitude) |
| `green_mode` (reciprocal Green's functions) | `reciprocal_green(model, receiver, points; component)`, `seismogram(green, n, source)`, `write_sac(green, dir)` |
| netCDF in general | optional: NCDatasets is only loaded with `using NCDatasets` (package extension) |
| random media (`gen_rmed3d`, `gen_rmed2d`, `*_rmed` models) | `perturb(model, random_medium(x, y, z; spectrum, a, epsilon, kappa))`; `read_random_medium(file)` for existing files |
| `vcut`, `munk_profile`, `earth_flattening` | keywords of `read_lhm`, `read_lgm`, `read_grd`; `flatten_earth(model)` for any model, `flatten_depth` for sources |
| seismograms (`sw_wav_v`, `_u`, `_stress`, `_strain`) | always recorded: `result.v` [m/s], `result.u` [m], `result.stress` [Pa], `result.strain` |
| SAC files in `odir/wav` | `write_sac(result, dir; quantities)`: same file names and units (nm/s, nm, Pa), z positive up |
| snapshots (netCDF, `xy_v_sw`, `fs_ps_sw`, …) | `snapshots=Snapshots(xy, xz, yz, fs, ob; quantities=(:v, :u, :ps))`: same files (needs `using NCDatasets`); or a ParaView movie, or your own plots (`callback`) |
| x north, y east, z down | x east, y north, z up (as in GMG) |
| density in g/cm³ | kg/m³ (`density_unit=:g_cm3` for g/cm³) |

## Differences in the results

All features of OpenSWPC are available. Differences in the results:

- Plane waves agree with `swpc_3d` to 10⁻⁵. `swpc_psv` and `swpc_sh` give 1–5% different plane waves (even at vertical incidence),
  while the 2D initial wavefield here equals the analytic formula and uses the same code as 3D; the difference appears to lie in the
  2D Fortran codes. OpenSWPC's 3D P-wave formula for `sxy` has `sd·cd·sf` where `sd²·sf·cf` is correct (only for inclined waves);
  the correct form is used here.
- Near vertical topography steps, the 2D results differ by 7–28% from the `swpc_psv`/`swpc_sh` binaries. They agree to 10⁻⁶ for flat
  surfaces, and for the same step in 3D, so the difference appears to lie in the 2D Fortran codes.

