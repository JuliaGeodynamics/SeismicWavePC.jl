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
function SeismicWavePC.read_grd(listfile, x, y, z, proj::ProjectionPoint; dir=dirname(listfile), ocean=true)
    rows = [split(l) for l in eachline(listfile) if !isempty(strip(l)) && !startswith(strip(l), '#')]
    isempty(rows) && error("no layers in $listfile")
    X, Y, Z = xyz_grid(x, y, z)
    G = convert(GeoData, convert2UTMzone(CartData(X[:, :, 1:1], Y[:, :, 1:1], Z[:, :, 1:1], (; Z=Z[:, :, 1:1])), proj))
    lon, lat = G.lon.val[:, :, 1], G.lat.val[:, :, 1]
    top = map(rows) do r                     # depth [km] of every layer top below every model column
        NCDataset(joinpath(dir, r[1])) do ds
            v = collect(keys(ds))
            glon, glat, d = Float64.(ds[v[1]][:]), Float64.(ds[v[2]][:]), Float64.(Array(ds[v[3]])) ./ 1000
            bilinear.(Ref(glon), Ref(glat), Ref(d), lon, lat)
        end
    end
    top = accumulate((a, b) -> max.(a, b), top)       # a layer starts not above the previous one
    props = [Tuple(parse.(Float64, r[2:6])) for r in rows]
    P = map(CartesianIndices(X)) do I
        depth = -Z[I]
        l = findlast(t -> t[I[1], I[2]] <= depth, top)
        l === nothing ? (depth < 0 || !ocean ? SeismicWavePC.AIR : SeismicWavePC.SEAWATER) : props[l]
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

end
