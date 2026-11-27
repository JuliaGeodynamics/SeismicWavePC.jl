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

# NCDatasets is an optional dependency (package extension): a clear error without it
const NETCDF_HINT = "netCDF files need the optional package NCDatasets: add it and load it with `using NCDatasets`"
read_netcdf(file::Any...) = error(NETCDF_HINT)

const SAC_UNDEF = -12345

"""
    write_sac(result::Seismograms, dir="."; title="swpc", source=nothing, quantities=(:v,))

Write the seismograms as SAC files, one per station and component, as OpenSWPC does:
`dir/<title>.<solver>.<station>.<component>.sac` with solver `psv`, `sh` or `3d`. `quantities` are any of
- `:v`: velocity `Vx`, `Vy`, `Vz` [nm/s]
- `:u`: displacement `Ux`, `Uy`, `Uz` [nm]
- `:stress`: `Sxx`, `Syy`, `Szz`, `Syz`, `Sxz`, `Sxy` [Pa]
- `:strain`: `Exx`, …, `Exy`
with the units of OpenSWPC. Unlike OpenSWPC, `x` points east (`cmpaz = 90`), `y` north and `z` up, also for the stress and
strain components. With a `source` (`MomentTensorSource`), its depth, the distance and azimuths, and the moment tensor
(`user0`–`user5`: xx, yy, zz, yz, xz, xy [N m]) are added to the headers. Returns the file names.
"""
function write_sac(r::Seismograms, dir="."; title="swpc", source=nothing, quantities=(:v,))
    solver = r.components == (:x, :z) ? "psv" : r.components == (:y,) ? "sh" : "3d"
    all(in((:v, :u, :stress, :strain)), quantities) || error("quantities must be :v, :u, :stress or :strain, got $quantities")
    mkpath(dir)
    files = String[]
    for q in quantities, (n, st) in enumerate(r.stations)
        data3, comps, prefix, unit, idep = q === :v ? (r.v, r.components, "V", 1.0e9, 7) : q === :u ? (r.u, r.components, "U", 1.0e9, 6) :
                                           q === :stress ? (r.stress, r.tensor_components, "S", 1.0, 5) : (r.strain, r.tensor_components, "E", 1.0, 5)
        for (c, comp) in enumerate(comps)
            name = prefix * string(comp)
            file = joinpath(dir, "$title.$solver.$(st.name).$name.sac")
            write_sac_file(file, Float32.(data3[:, c, n] .* unit), r.t, st, name, title, idep,
                           comp === :x ? (90, 90) : comp === :y ? (0, 90) : comp === :z ? (0, 0) : (SAC_UNDEF, SAC_UNDEF), source)
            push!(files, file)
        end
    end
    return files
end

function write_sac_file(file, data, t, st, kcmpnm, kevnm, idep, (cmpaz, cmpinc), source)
    f = zeros(Float32, 70) .+ SAC_UNDEF
    i = zeros(Int32, 40) .+ Int32(SAC_UNDEF)
    k = fill("-12345", 23)
    f[1], f[6], f[7] = length(t) > 1 ? t[2] - t[1] : 1.0, t[1], t[end]           # delta, b, e
    f[2], f[3], f[57] = minimum(data), maximum(data), sum(data) / length(data)   # depmin, depmax, depmen
    f[35] = st.surface ? 0.0 : -st.z * 1.0e3                                     # stdp [m]
    f[58], f[59] = cmpaz, cmpinc
    if source !== nothing
        f[39] = -source.z                                                        # evdp [km]
        f[51] = hypot(st.x - source.x, st.y - source.y)                         # dist [km]
        f[52] = mod(atand(st.x - source.x, st.y - source.y), 360)                # az (from north)
        f[53] = mod(f[52] + 180, 360)                                            # baz
        source isa MomentTensorSource && (f[41:46] .= source.M0 .* collect(source.M))  # user0-5
    end
    i[7], i[10], i[16], i[17], i[18] = 6, length(data), 1, idep, 9   # nvhdr, npts, iftype = time series, idep, iztype = b
    i[31], i[34] = 1, 0                                              # leven, lcalda
    k[1], k[2], k[20] = st.name, kevnm, kcmpnm                       # kstnm, kevnm (16 characters), kcmpnm
    open(file, "w") do io
        write(io, htol.(f), htol.(i))
        for (m, s) in enumerate(k)
            w = m == 2 ? 16 : 8
            write(io, rpad(first(s, w), w))
        end
        write(io, htol.(data))
    end
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

# ------------------------------------------------------------------ Seismic Unix (SU)

"""
    write_su(result, dir="."; title="swpc", source=nothing, quantities=(:v,))

Write the seismograms of [`Seismograms`](@ref) `result` as Seismic Unix (SU) files (as SOFI): one file per quantity and
component with one trace per station, `dir/title.solver.<quantity><component>.su` (e.g. `swpc.psv.vx.su`). `quantities`:
`:v`, `:u` (m/s, m), `:stress` (Pa), `:strain`, `:p` (pressure: minus the mean normal stress, Pa), `:div` and `:curl` (of the
velocity, 1/s: needs `simulate(...; derivatives = true)`). Trace headers: station and (with `source`) source positions in m
(scaler -100: cm), receiver and source elevation, offset, sample interval (µs) and delay (ms). Samples: 32-bit floats,
little endian (as Seismic Unix on x86 and ARM). Returns the file names.
"""
function write_su(r::Seismograms, dir="."; title="swpc", source=nothing, quantities=(:v,))
    solver = r.components == (:x, :z) ? "psv" : r.components == (:y,) ? "sh" : "3d"
    all(in((:v, :u, :stress, :strain, :p, :div, :curl)), quantities) ||
        error("quantities must be :v, :u, :stress, :strain, :p, :div or :curl, got $quantities")
    dt = length(r.t) > 1 ? r.t[2] - r.t[1] : 1.0
    dtus = round(Int, 1e6dt)
    abs(dtus - 1e6dt) > 1e-3 * 1e6dt && @warn "write_su: the sample interval $dt s is rounded to $dtus µs (SU headers)"
    (dtus <= typemax(UInt16) && length(r.t) <= typemax(UInt16)) || error("write_su: SU allows at most 65535 samples and a sample interval of 65.535 ms")
    mkpath(dir)
    files = String[]
    for q in quantities
        traces = if q in (:v, :u)
            data = q === :v ? r.v : r.u
            [(string(q) * string(c), data[:, k, :]) for (k, c) in enumerate(r.components)]
        elseif q in (:stress, :strain)
            data = q === :stress ? r.stress : r.strain
            [((q === :stress ? "s" : "e") * string(c), data[:, k, :]) for (k, c) in enumerate(r.tensor_components)]
        elseif q === :p
            normal = [k for (k, c) in enumerate(r.tensor_components) if c in (:xx, :yy, :zz)]
            isempty(normal) && error("write_su: no pressure for SH waves")
            [("p", -dropdims(sum(r.stress[:, normal, :]; dims=2); dims=2) ./ length(normal))]
        else
            r.div === nothing && error("write_su: :$q needs simulate(...; derivatives = true)")
            q === :div ? [("div", r.div)] : [("curl" * string(c), r.curl[:, k, :]) for (k, c) in enumerate((:x, :y, :z))]
        end
        for (name, data) in traces
            file = joinpath(dir, "$title.$solver.$name.su")
            open(file, "w") do io
                for (n, st) in enumerate(r.stations)
                    write(io, su_header(n, st, source, length(r.t), dtus, r.t[1]))
                    write(io, htol.(Float32.(data[:, n])))
                end
            end
            push!(files, file)
        end
    end
    return files
end

function su_header(n, st, source, ns, dtus, t0)
    h = zeros(UInt8, 240)
    i32!(o, v) = (h[o+1:o+4] .= reinterpret(UInt8, [htol(Int32(v))]))
    i16!(o, v) = (h[o+1:o+2] .= reinterpret(UInt8, [htol(Int16(v))]))
    u16!(o, v) = (h[o+1:o+2] .= reinterpret(UInt8, [htol(UInt16(v))]))
    cm(x) = round(Int32, 1e5x)                              # [km] -> [cm] (scaler -100)
    i32!(0, n); i32!(4, n); i32!(12, n)                     # tracl, tracr, tracf
    i16!(28, 1)                                             # trid: seismic data
    i32!(40, cm(st.z)); i16!(68, -100); i16!(70, -100)      # gelev; scalel, scalco
    i32!(80, cm(st.x)); i32!(84, cm(st.y))                  # gx, gy
    if source !== nothing
        i32!(36, round(Int32, 1e3hypot(st.x - source.x, st.y - source.y)))    # offset [m]
        i32!(44, cm(source.z)); i32!(48, cm(-source.z))     # selev, sdepth
        i32!(72, cm(source.x)); i32!(76, cm(source.y))      # sx, sy
    end
    i16!(108, round(Int, 1e3t0))                            # delrt [ms]
    u16!(114, ns); u16!(116, dtus)                          # ns, dt [µs]
    return h
end

"""
    su = read_su(file)

An SU file (little or big endian): `su.t` the times [s], `su.data` the traces (samples × traces), and per trace the
positions `su.sx, su.sy, su.gx, su.gy` [m] (with the coordinate scaler applied) and `su.offset`. E.g. a source wavelet:
`SampledSTF(su.t, su.data[:, 1])`.
"""
function read_su(file)
    bytes = read(file)
    length(bytes) >= 240 || error("read_su: $file is too short for an SU trace")
    for conv in (ltoh, ntoh)                                 # (little, then big endian)
        u16(b, o) = conv(reinterpret(UInt16, b[o+1:o+2])[1])
        ns = Int(u16(bytes, 114))
        len = 240 + 4ns
        ns > 0 && length(bytes) % len == 0 || continue
        ntr = length(bytes) ÷ len
        i32(b, o) = conv(reinterpret(Int32, b[o+1:o+4])[1])
        i16(b, o) = conv(reinterpret(Int16, b[o+1:o+2])[1])
        data = zeros(Float32, ns, ntr)
        sx, sy, gx, gy, offset = (zeros(ntr) for _ in 1:5)
        dt, delrt = Int(u16(bytes, 116)) * 1e-6, Int(i16(bytes, 108)) * 1e-3
        for k in 1:ntr
            b = view(bytes, (k-1)*len+1:k*len)
            sc = i16(b, 70)
            f = sc == 0 ? 1.0 : sc > 0 ? Float64(sc) : 1 / -sc
            sx[k], sy[k], gx[k], gy[k] = (i32(b, o) * f for o in (72, 76, 80, 84))
            offset[k] = i32(b, 36)
            data[:, k] .= conv.(reinterpret(Float32, b[241:end]))
        end
        return (; t=delrt .+ dt .* (0:ns-1), data, sx, sy, gx, gy, offset)
    end
    error("read_su: $file is not an SU file (trace lengths do not match the number of samples)")
end
