# Layered velocity models in the file formats of OpenSWPC (m_vmodel_lhm, m_vmodel_lgm, m_vmodel_grd), as GMG CartData,
# with the options of OpenSWPC: minimum velocity (vcut), Munk sea-water profile and Earth flattening.

"""
    model = read_lhm(file, x, y, z; vcut=0.0, munk=false, earth_flattening=false)
    model = read_lgm(file, x, y, z; vcut=0.0, munk=false, earth_flattening=false)

Velocity model from an OpenSWPC layer file (`vmodel_type = "lhm"` or `"lgm"`) on the grid `x, y, z` [km, z up] (vectors or
ranges as for `xyz_grid`; `y = 0.0` for a 2D model). Every line of the file gives the depth of the top of a layer [km] and its
`rho` [g/cm³], `vp`, `vs` [km/s], `Qp` and `Qs`; lines starting with `#` are comments.
- `read_lhm`: homogeneous layers.
- `read_lgm`: the properties change linearly with depth between the tops of two layers (constant in the last layer).
Above the first layer the model is air (z > 0) or sea water (vp = 1.5 km/s, between sea level and the first layer), as in
OpenSWPC. Returns a `CartData` with `rho` [kg/m³], `vp`, `vs`, `Qp`, `Qs`.

Options, as in OpenSWPC:
- `vcut` [km/s]: layers with a slower P or S velocity get the properties of the layer below.
- `munk = true`: depth-dependent sound speed of sea water (Munk profile) instead of 1.5 km/s.
- `earth_flattening = true`: the layers are given at spherical depths; the model is the flattened Earth (see
  [`flatten_earth`](@ref)). Convert source depths with [`flatten_depth`](@ref).
"""
read_lhm(file, x, y, z; kwargs...) = layered_model(read_layers(file; kwargs...), x, y, z, false; kwargs...)
read_lgm(file, x, y, z; kwargs...) = layered_model(read_layers(file; kwargs...), x, y, z, true; kwargs...)
@doc (@doc read_lhm) read_lgm

"Rows of an OpenSWPC layer file: depth, rho, vp, vs, Qp, Qs."
function read_layers(file; vcut=0.0, kwargs...)
    rows = [parse.(Float64, split(l)[1:6]) for l in eachline(file) if !isempty(strip(l)) && !startswith(strip(l), '#')]
    isempty(rows) && error("no layers in $file")
    issorted(first.(rows)) || error("the layers in $file must be sorted by depth")
    return apply_vcut!(rows, vcut, 2)
end

"OpenSWPC `vcut`: from the bottom up, a layer slower than `vcut` (but not fluid) takes the properties of the layer below."
function apply_vcut!(rows, vcut, first_property)
    p = first_property                     # index of rho in a row: rho, vp, vs, Qp, Qs follow
    for l in length(rows)-1:-1:1
        vp, vs = rows[l][p+1], rows[l][p+2]
        if (vp < vcut || vs < vcut) && vp > 0 && vs > 0
            rows[l][p:p+4] .= rows[l+1][p:p+4]
        end
    end
    return rows
end

const AIR = (0.001, 0.0, 0.0, 10.0, 10.0)               # rho [g/cm³], vp, vs, Qp, Qs (OpenSWPC)
const R_EARTH = 6371.0                                 # km

"Sea water at `depth` [km]: 1.5 km/s, or the Munk profile."
function seawater(depth, munk)
    zb = 2 * (depth * 1000 - 1300) / 1300
    return (1.0, 1.5 * (1 + (munk ? 0.00737 : 0.0) * (zb - 1 + exp(-zb))), 0.0, 1.0e6, 1.0e6)
end

"""
    flatten_depth(depth)

Depth [km] in the flattened Earth of a point at spherical `depth` [km] (Earth flattening transformation, as OpenSWPC applies
to source depths). Use it for the source depths of a model made with `earth_flattening = true` or [`flatten_earth`](@ref).
"""
flatten_depth(depth) = -R_EARTH * log((R_EARTH - depth) / R_EARTH)

"Spherical depth of the flat depth `z`, and the velocity scale factor there."
spherical(z) = (R_EARTH * (1 - exp(-z / R_EARTH)), exp(z / R_EARTH))

"Properties at flat `depth` [km] above the first layer, or in the medium (`lgm`: linear between the layer tops)."
function layer_properties(L, depth, gradient; munk=false, earth_flattening=false, kwargs...)
    zs, cv = earth_flattening ? spherical(depth) : (depth, 1.0)
    if zs < L[1][1]
        depth < 0 && return AIR
        w = seawater(depth, munk)
        return (w[1], cv * w[2], w[3], w[4], w[5])
    end
    l = findlast(r -> r[1] <= zs, L)
    p = (!gradient || l == length(L)) ? Tuple(L[l][2:6]) : Tuple(L[l][2:6] .+ (zs - L[l][1]) / (L[l+1][1] - L[l][1]) .* (L[l+1][2:6] .- L[l][2:6]))
    return (p[1], cv * p[2], cv * p[3], p[4], p[5])
end

function layered_model(L, x, y, z, gradient; kwargs...)
    X, Y, Z = xyz_grid(x, y, z)
    P = layer_properties.(Ref(L), -Z, gradient; kwargs...)
    f(n) = getindex.(P, n)
    return CartData(X, Y, Z, (; rho=1000 .* f(1), vp=f(2), vs=f(3), Qp=f(4), Qs=f(5)))
end

"""
    flat = flatten_earth(model::CartData)

Earth flattening transformation (as in OpenSWPC) of a model whose z axis is the spherical depth: at every point of the grid,
which is now the flattened Earth, the fields are taken from the spherical depth of that point (linear interpolation in z) and
`vp`, `vs` are scaled by `exp(depth / R)`. Points above z = 0 are not changed. Convert source depths with
[`flatten_depth`](@ref).
"""
function flatten_earth(model::CartData)
    z = model.z.val[1, 1, :]
    issorted(z) || error("the z axis of the model must increase")
    fields = map(keys(model.fields)) do name
        A = model.fields[name]
        B = similar(A, Float64)
        for k in eachindex(z)
            zs, cv = z[k] < 0 ? spherical(-z[k]) : (-z[k], 1.0)
            kk = clamp(searchsortedlast(z, -zs), 1, length(z) - 1)                # interpolate between z[kk] and z[kk + 1]
            w = clamp((-zs - z[kk]) / (z[kk+1] - z[kk]), 0, 1)
            scale = name in (:vp, :vs) ? cv : 1.0
            @views B[:, :, k] .= scale .* ((1 - w) .* A[:, :, kk] .+ w .* A[:, :, kk+1])
        end
        B
    end
    return CartData(model.x.val, model.y.val, model.z.val, NamedTuple{keys(model.fields)}(Tuple(fields)))
end

"""
    model = read_grd(listfile, x, y, z, proj::ProjectionPoint; dir=dirname(listfile), ocean=true, vcut=0.0, munk=false,
                     earth_flattening=false)

Velocity model from OpenSWPC layer boundaries given as GMT netCDF grids (`vmodel_type = "grd"`). Every line of `listfile`
names a grid file (in `dir`) with the depth of the top of a layer [m, positive down, on a longitude–latitude grid], and the
`rho` [g/cm³], `vp`, `vs` [km/s], `Qp`, `Qs` of that layer (and a boundary id, which is ignored). The model grid `x, y, z`
[km, z up] is placed with `proj`, the GMG `ProjectionPoint` of the model; the grids are interpolated bilinearly (OpenSWPC:
bicubic). Above the first boundary the model is air (z > 0) or sea water (`ocean = true`). `vcut`, `munk` and
`earth_flattening` as for [`read_lhm`](@ref). Returns a `CartData` with `rho` [kg/m³], `vp`, `vs`, `Qp`, `Qs`.

Requires `using NCDatasets`.
"""
function read_grd end
