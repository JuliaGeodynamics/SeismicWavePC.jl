module SeismicWavePCNCDatasetsExt

using SeismicWavePC, NCDatasets
using GeophysicalModelGenerator: CartData, GeoData, ProjectionPoint, xyz_grid, convert2UTMzone

# Coordinates are stored in single precision: rebuild the regular axis from the shortest decimals of its end points
function axis(v)
    a = parse.(Float64, string.(v))
    length(a) > 1 && all(d -> isapprox(d, a[2] - a[1]; rtol=1e-4), diff(a)) || error("netCDF model must be on a regular grid")
    return collect(range(a[1], a[end], length=length(a)))
end

function SeismicWavePC.read_netcdf(file)
    NCDataset(file) do ds
        is2d = !haskey(ds.dim, "y")
        x, z = axis(ds["x"][:]), axis(ds["z"][:])
        z = issorted(z) ? -z : z                 # the first z index is the top: z is depth (increasing) or height (as OpenSWPC.jl writes)
        y = is2d ? [0.0] : axis(ds["y"][:])
        up(name) = (A = Float64.(Array(ds[name])); A = is2d ? reshape(A, length(x), 1, length(z)) : A; reverse(A, dims=3))
        names = setdiff(keys(ds), ("x", "y", "z"))
        fields = Dict{Symbol,Array{Float64,3}}(Symbol(n) => up(n) for n in names)
        haskey(fields, :rho) || error("$file has no field rho")
        rho = fields[:rho]
        if !haskey(fields, :vp)
            haskey(fields, :mu) && haskey(fields, :lambda) || error("$file needs either vp and vs, or mu and lambda")
            fields[:vp] = @. sqrt(max(fields[:lambda] + 2fields[:mu], 0) / rho)
            fields[:vs] = @. sqrt(max(fields[:mu], 0) / rho)
        end
        fields[:rho] = 1000 .* rho                                                 # g/cm³ -> kg/m³
        return CartData(xyz_grid(x, is2d ? 0.0 : y, reverse(z))..., NamedTuple(fields))
    end
end

# OpenSWPC "grd" model: layer boundaries [m depth] on longitude-latitude grids (GMT netCDF: x, y, z variables)
function SeismicWavePC.read_grd(listfile, x, y, z, proj::ProjectionPoint; dir=dirname(listfile), ocean=true, vcut=0.0, munk=false,
                                earth_flattening=false)
    rows = [split(l) for l in eachline(listfile) if !isempty(strip(l)) && !startswith(strip(l), '#')]
    isempty(rows) && error("no layers in $listfile")
    X, Y, Z = xyz_grid(x, y, z)
    G = convert(GeoData, convert2UTMzone(CartData(X[:, :, 1:1], Y[:, :, 1:1], Z[:, :, 1:1], (; Z=Z[:, :, 1:1])), proj))
    lon, lat = G.lon.val[:, :, 1], G.lat.val[:, :, 1]
    top = map(rows) do r                     # depth [km] of every layer top below every model column
        NCDataset(joinpath(dir, r[1])) do ds
            v = collect(keys(ds))
            glon, glat, d = Float64.(ds[v[1]][:]), Float64.(ds[v[2]][:]), Float64.(Array(ds[v[3]])) ./ 1000
            t = bilinear.(Ref(glon), Ref(glat), Ref(d), lon, lat)
            earth_flattening ? SeismicWavePC.flatten_depth.(t) : t
        end
    end
    top = accumulate((a, b) -> max.(a, b), top)       # a layer starts not above the previous one
    props = SeismicWavePC.apply_vcut!([parse.(Float64, r[2:6]) for r in rows], vcut, 1)
    P = map(CartesianIndices(X)) do I
        depth = -Z[I]
        cv = earth_flattening ? exp(depth / SeismicWavePC.R_EARTH) : 1.0
        l = findlast(t -> t[I[1], I[2]] <= depth, top)
        p = l !== nothing ? props[l] : depth < 0 || !ocean ? SeismicWavePC.AIR : SeismicWavePC.seawater(depth, munk)
        (p[1], cv * p[2], cv * p[3], p[4], p[5])
    end
    f(n) = getindex.(P, n)
    return CartData(X, Y, Z, (; rho=1000 .* f(1), vp=f(2), vs=f(3), Qp=f(4), Qs=f(5)))
end

"Bilinear interpolation of `d` on the regular grid `gx, gy` at `(px, py)`, clamped to the grid."
function bilinear(gx, gy, d, px, py)
    fi = clamp((px - gx[1]) / (gx[2] - gx[1]) + 1, 1, length(gx))
    fj = clamp((py - gy[1]) / (gy[2] - gy[1]) + 1, 1, length(gy))
    i, j = min(floor(Int, fi), length(gx) - 1), min(floor(Int, fj), length(gy) - 1)
    u, v = fi - i, fj - j
    return (1 - u) * (1 - v) * d[i, j] + u * (1 - v) * d[i+1, j] + (1 - u) * v * d[i, j+1] + u * v * d[i+1, j+1]
end

# Snapshot files in OpenSWPC's netCDF layout: coordinates, time (unlimited), medium, and one variable per field.
# `piece = (whole, offset)`: the file holds the points of one MPI process, a piece of a plane of `whole` points that starts
# after `offset` points along the two axes.
function SeismicWavePC.snapshot_create(file, axes, vars, unit, medium; piece=nothing)
    NCDataset(file, "c") do ds
        ds.attrib["title"] = "SeismicWavePC.jl snapshot (OpenSWPC format)"
        ds.attrib["convention"] = "components at cell centres, x east, y north, z up; coordinate z is the depth"
        if piece !== nothing
            ds.attrib["whole_size"] = Int32[piece[1]...]
            ds.attrib["piece_offset"] = Int32[piece[2]...]
        end
        for (name, c) in axes
            defDim(ds, name, length(c))
            defVar(ds, name, Float32.(c), (name,), attrib=["long_name" => name, "units" => "km"])
        end
        dims = (first(axes[1]), first(axes[2]))
        defDim(ds, "t", Inf)
        defVar(ds, "t", Float32, ("t",), attrib=["long_name" => "time", "units" => "s"])
        units = (rho="g/cm^3", lambda="GPa", mu="GPa")
        for (name, A) in pairs(medium)
            defVar(ds, String(name), Float32.(A), dims, attrib=["long_name" => String(name), "units" => units[name]])
        end
        for v in vars
            defVar(ds, v, Float32, (dims..., "t"), attrib=["long_name" => v, "units" => unit])
        end
    end
    return nothing
end

function SeismicWavePC.snapshot_append(file, n, t, data)
    NCDataset(file, "a") do ds
        ds["t"][n] = t
        for (v, A) in data
            ds[v][:, :, n] = Float32.(A)
        end
    end
    return nothing
end

function SeismicWavePC.snapshot_peaks(file, (V, Hz, A))
    NCDataset(file, "a") do ds
        dims = dimnames(ds["rho"])
        for (name, X, long) in (("max-V", V, "Maximum amplitude of the vertical component"),
                                ("max-H", Hz, "Maximum amplitude of the horizontal components"),
                                ("max-A", A, "Maximum amplitude of the vector motion"))
            defVar(ds, name, Float32.(X), dims, attrib=["long_name" => long, "units" => "m/s"])
        end
    end
    return nothing
end

# Snapshots of one plane and quantity: one file, or the pieces `<name>.<rank>.nc` of the MPI processes
function SeismicWavePC.read_snapshots(file::AbstractString)
    dir, base = isempty(dirname(file)) ? "." : dirname(file), replace(basename(file), r"\.nc$" => "")
    pieces = isdir(dir) ? sort!([joinpath(dir, f) for f in readdir(dir) if startswith(f, base * ".") && occursin(r"^\.\d{4}\.nc$", f[length(base)+1:end])]) : String[]
    isfile(file) && !isempty(pieces) && error("both $file and pieces of it ($(basename(pieces[1])), …) exist: remove the files of the other run")
    files = isfile(file) ? [String(file)] : pieces
    isempty(files) && error("no snapshot file $file, nor pieces of it ($(base).0000.nc, …)")
    info = map(files) do f
        NCDataset(f) do ds
            dims = dimnames(ds["rho"])                                    # the two axes of the plane
            sz = (Int(ds.dim[dims[1]]), Int(ds.dim[dims[2]]))
            get2(name, default) = haskey(ds.attrib, name) ? (Int(ds.attrib[name][1]), Int(ds.attrib[name][2])) : default
            (; dims=(String(dims[1]), String(dims[2])), sz, offset=get2("piece_offset", (0, 0)), whole=get2("whole_size", sz),
               coords=(Float64.(ds[dims[1]][:]), Float64.(ds[dims[2]][:])), t=Float64.(ds["t"][:]),
               variables=String[v for v in keys(ds) if !(v in dims) && v != "t"])
        end
    end
    first_ = info[1]
    all(i -> i.dims == first_.dims && i.whole == first_.whole && i.variables == first_.variables && i.t == first_.t &&
             all(i.offset .+ i.sz .<= i.whole), info) || error("the pieces of $file do not belong together")
    if !isfile(file)                                                     # pieces: every point of the plane once
        covered = falses(first_.whole)
        for i in info
            r = (i.offset[1]+1:i.offset[1]+i.sz[1], i.offset[2]+1:i.offset[2]+i.sz[2])
            any(view(covered, r...)) && error("the pieces of $file overlap (pieces of another run?)")
            covered[r...] .= true
        end
        all(covered) || error("the pieces of $file leave points of the plane out (a piece is missing)")
    end
    coords = (fill(NaN, first_.whole[1]), fill(NaN, first_.whole[2]))
    for i in info, a in 1:2
        coords[a][i.offset[a]+1:i.offset[a]+i.sz[a]] .= i.coords[a]
    end
    return SeismicWavePC.SnapshotSeries(files, [i.offset for i in info], [i.sz for i in info],
                                        (first_.dims[1] => coords[1], first_.dims[2] => coords[2]), first_.t, first_.variables)
end

# One snapshot of a field, put together from the files (NaN where no piece has the point)
function Base.getindex(s::SeismicWavePC.SnapshotSeries, var::AbstractString, k::Integer)
    A = fill(NaN32, length(s.axes[1][2]), length(s.axes[2][2]))
    for (f, o, sz) in zip(s.files, s.offsets, s.sizes)
        NCDataset(f) do ds
            ndims(ds[var]) == 3 || error("the field $var has no time: read it with s[\"$var\"]")
            A[o[1]+1:o[1]+sz[1], o[2]+1:o[2]+sz[2]] = ds[var].var[:, :, k]
        end
    end
    return A
end
# A field without time, or all snapshots of a field
function Base.getindex(s::SeismicWavePC.SnapshotSeries, var::AbstractString)
    nd = NCDataset(ds -> ndims(ds[var]), s.files[1])
    nd == 3 && return cat((s[var, k] for k in eachindex(s.t))...; dims=3)
    A = fill(NaN32, length(s.axes[1][2]), length(s.axes[2][2]))
    for (f, o, sz) in zip(s.files, s.offsets, s.sizes)
        NCDataset(f) do ds
            A[o[1]+1:o[1]+sz[1], o[2]+1:o[2]+sz[2]] = ds[var].var[:, :]
        end
    end
    return A
end

# Random medium of OpenSWPC's gen_rmed2d / gen_rmed3d: dimensions x, (y,) z (z = depth), variable "random media"
function SeismicWavePC.read_random_medium(file)
    NCDataset(file) do ds
        A = Float64.(Array(ds["random media"]))
        A = ndims(A) == 2 ? reshape(A, size(A, 1), 1, size(A, 2)) : A
        return reverse(A, dims=3)                                  # z up
    end
end

end
