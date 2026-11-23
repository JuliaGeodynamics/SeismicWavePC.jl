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

# Snapshot files in OpenSWPC's netCDF layout: coordinates, time (unlimited), medium, and one variable per field
function SeismicWavePC.snapshot_create(file, axes, vars, unit, medium)
    NCDataset(file, "c") do ds
        ds.attrib["title"] = "SeismicWavePC.jl snapshot (OpenSWPC format)"
        ds.attrib["convention"] = "components at cell centres, x east, y north, z up; coordinate z is the depth"
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

# Random medium of OpenSWPC's gen_rmed2d / gen_rmed3d: dimensions x, (y,) z (z = depth), variable "random media"
function SeismicWavePC.read_random_medium(file)
    NCDataset(file) do ds
        A = Float64.(Array(ds["random media"]))
        A = ndims(A) == 2 ? reshape(A, size(A, 1), 1, size(A, 2)) : A
        return reverse(A, dims=3)                                  # z up
    end
end

end
