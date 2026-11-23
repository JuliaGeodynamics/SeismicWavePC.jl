# File formats of OpenSWPC: SAC seismograms (written and read here) and netCDF velocity models (read by the NCDatasets extension).

"""
    model = read_netcdf(file)

Read a 2D or 3D velocity model in the netCDF format of OpenSWPC (`vmodel_type = "velm"`, as written by
`OpenSWPC.write_netcdf`) and return it as a `CartData` that can be passed to [`simulate`](@ref).

The file has the coordinates `x`, (`y`,) `z` [km, z is depth] and the fields `rho` [g/cm³], `Qp`, `Qs` and either `vp`, `vs`
[km/s] or `mu`, `lambda` [g/cm³ (km/s)²]. In the result, `z` points up, `rho` is in kg/m³ and `vp`, `vs` are present; other
fields are kept. Every grid point is a cell centre (OpenSWPC places cell `i` half a cell further, at `x[i] + dx/2`).

Requires `using NCDatasets`.
"""
function read_netcdf end

const SAC_UNDEF = -12345

"""
    write_sac(result::Seismograms, dir="."; title="swpc", source=nothing)

Write the seismograms as SAC files, one per station and component, as OpenSWPC does:
`dir/<title>.<solver>.<station>.<component>.sac` with solver `psv`, `sh` or `3d` and component `Vx`, `Vy` or `Vz`.
As in OpenSWPC the velocities are in nm/s and `Vz` is positive up. Unlike OpenSWPC, `x` points east (`cmpaz = 90`) and
`y` north. With a `source` (`MomentTensorSource`), its depth, the distance and azimuths, and the moment tensor
(`user0`–`user5`: xx, yy, zz, yz, xz, xy [N m]) are added to the headers. Returns the file names.
"""
function write_sac(r::Seismograms, dir="."; title="swpc", source=nothing)
    solver = r.components == (:x, :z) ? "psv" : r.components == (:y,) ? "sh" : "3d"
    mkpath(dir)
    files = String[]
    for (n, st) in enumerate(r.stations), (c, comp) in enumerate(r.components)
        data = Float32.(r.v[:, c, n] .* 1.0e9)
        f = zeros(Float32, 70) .+ SAC_UNDEF
        i = zeros(Int32, 40) .+ Int32(SAC_UNDEF)
        k = fill("-12345", 23)
        delta = length(r.t) > 1 ? r.t[2] - r.t[1] : 1.0
        f[1], f[6], f[7] = delta, r.t[1], r.t[end]                                      # delta, b, e
        f[2], f[3], f[57] = minimum(data), maximum(data), sum(data) / length(data)     # depmin, depmax, depmen
        f[35] = st.surface ? 0.0 : -st.z * 1.0e3                                       # stdp [m]
        f[58], f[59] = comp === :x ? (90, 90) : comp === :y ? (0, 90) : (0, 0)          # cmpaz, cmpinc
        if source !== nothing
            f[39] = -source.z                                                          # evdp [km]
            f[51] = hypot(st.x - source.x, st.y - source.y)                           # dist [km]
            f[52] = mod(atand(st.x - source.x, st.y - source.y), 360)                  # az (from north)
            f[53] = mod(f[52] + 180, 360)                                              # baz
            f[41:46] .= source.M0 .* collect(source.M)                                 # user0-5
        end
        i[7], i[10], i[16], i[17], i[18] = 6, length(data), 1, 7, 9    # nvhdr, npts, iftype = time series, idep = velocity, iztype = b
        i[31], i[34] = 1, 0                                            # leven, lcalda
        k[1], k[2], k[20] = st.name, title, "V" * string(comp)         # kstnm, kevnm (16 characters), kcmpnm
        file = joinpath(dir, "$title.$solver.$(st.name).V$comp.sac")
        open(file, "w") do io
            write(io, htol.(f), htol.(i))
            for (m, s) in enumerate(k)
                w = m == 2 ? 16 : 8
                write(io, rpad(first(s, w), w))
            end
            write(io, htol.(data))
        end
        push!(files, file)
    end
    return files
end

"""
    t, v, header = read_sac(file)

Read a SAC file (for example an OpenSWPC seismogram): times `t` [s], data `v` and a `NamedTuple` with some header values
(`delta`, `b`, `kstnm`, `kcmpnm`, `kevnm`, `stdp`, `evdp`, `dist`, `az`).
"""
function read_sac(file)
    open(file) do io
        f = ltoh.(read!(io, Vector{Float32}(undef, 70)))
        i = ltoh.(read!(io, Vector{Int32}(undef, 40)))
        k = String(read(io, 192))
        v = ltoh.(read!(io, Vector{Float32}(undef, i[10])))
        str(a, b) = strip(k[a:b])
        t = f[6] .+ (0:i[10]-1) .* f[1]
        return t, v, (; delta=f[1], b=f[6], kstnm=str(1, 8), kevnm=str(9, 24), kcmpnm=str(161, 168), stdp=f[35],
                        evdp=f[39], dist=f[51], az=f[52])
    end
end
