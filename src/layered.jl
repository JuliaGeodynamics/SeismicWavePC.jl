# Layered velocity models in the file formats of OpenSWPC (m_vmodel_lhm, m_vmodel_lgm, m_vmodel_grd), as GMG CartData.

"""
    model = read_lhm(file, x, y, z)
    model = read_lgm(file, x, y, z)

Velocity model from an OpenSWPC layer file (`vmodel_type = "lhm"` or `"lgm"`) on the grid `x, y, z` [km, z up] (vectors or
ranges as for `xyz_grid`; `y = 0.0` for a 2D model). Every line of the file gives the depth of the top of a layer [km] and its
`rho` [g/cm³], `vp`, `vs` [km/s], `Qp` and `Qs`; lines starting with `#` are comments.
- `read_lhm`: homogeneous layers.
- `read_lgm`: the properties change linearly with depth between the tops of two layers (constant in the last layer).
Above the first layer the model is air (z > 0) or sea water (vp = 1.5 km/s, between sea level and the first layer), as in
OpenSWPC. Returns a `CartData` with `rho` [kg/m³], `vp`, `vs`, `Qp`, `Qs`.
"""
read_lhm(file, x, y, z) = layered_model(read_layers(file), x, y, z, false)
read_lgm(file, x, y, z) = layered_model(read_layers(file), x, y, z, true)
@doc (@doc read_lhm) read_lgm

"Rows of an OpenSWPC layer file: depth, rho, vp, vs, Qp, Qs."
function read_layers(file)
    rows = [parse.(Float64, split(l)[1:6]) for l in eachline(file) if !isempty(strip(l)) && !startswith(strip(l), '#')]
    isempty(rows) && error("no layers in $file")
    issorted(first.(rows)) || error("the layers in $file must be sorted by depth")
    return rows
end

const AIR = (0.001, 0.0, 0.0, 10.0, 10.0)               # rho [g/cm³], vp, vs, Qp, Qs (OpenSWPC)
const SEAWATER = (1.0, 1.5, 0.0, 1.0e6, 1.0e6)

"Properties at `depth` [km] above the first layer, or in the medium (`lgm`: linear between the layer tops)."
function layer_properties(L, depth, gradient)
    depth < L[1][1] && return depth < 0 ? AIR : SEAWATER
    l = findlast(r -> r[1] <= depth, L)
    (!gradient || l == length(L)) && return Tuple(L[l][2:6])
    w = (depth - L[l][1]) / (L[l+1][1] - L[l][1])
    return Tuple((1 - w) .* L[l][2:6] .+ w .* L[l+1][2:6])
end

function layered_model(L, x, y, z, gradient)
    X, Y, Z = xyz_grid(x, y, z)
    P = layer_properties.(Ref(L), -Z, gradient)
    f(n) = getindex.(P, n)
    return CartData(X, Y, Z, (; rho=1000 .* f(1), vp=f(2), vs=f(3), Qp=f(4), Qs=f(5)))
end

"""
    model = read_grd(listfile, x, y, z, proj::ProjectionPoint; dir=dirname(listfile), ocean=true)

Velocity model from OpenSWPC layer boundaries given as GMT netCDF grids (`vmodel_type = "grd"`). Every line of `listfile`
names a grid file (in `dir`) with the depth of the top of a layer [m, positive down, on a longitude–latitude grid], and the
`rho` [g/cm³], `vp`, `vs` [km/s], `Qp`, `Qs` of that layer (and a boundary id, which is ignored). The model grid `x, y, z`
[km, z up] is placed with `proj`, the GMG `ProjectionPoint` of the model; the grids are interpolated bilinearly (OpenSWPC:
bicubic). Above the first boundary the model is air (z > 0) or sea water (`ocean = true`). Returns a `CartData` with
`rho` [kg/m³], `vp`, `vs`, `Qp`, `Qs`.

Requires `using NCDatasets`.
"""
function read_grd end
