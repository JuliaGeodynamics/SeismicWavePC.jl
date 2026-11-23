module SeismicWavePCNCDatasetsExt

using SeismicWavePC, NCDatasets
using GeophysicalModelGenerator: CartData, xyz_grid

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

end
