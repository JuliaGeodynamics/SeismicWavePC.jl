# Kinematic finite faults: a planar fault as subfaults, each a double-couple point source with its own slip, rake,
# rupture time, rise time and source time function.

"""
    finite_fault(; top, strike, dip, length, width, nl, nw, slip, rake, mu=nothing, model=nothing, density_unit=:kg_m3,
                 hypocenter=(0.0, width / 2), vr=nothing, t0=nothing, trise=1.0, stf=nothing, dims=3)

A planar finite fault as `nl × nw` subfaults (along strike × down dip), each a [`DoubleCoupleSource`](@ref) at its centre
with the scalar moment μ A slip. Returns a vector of [`MomentTensorSource`](@ref)s for the `sources` of [`simulate`](@ref) and
[`misfit_gradient`](@ref).

- `top = (x, y, z)` [km]: the centre of the upper edge (model coordinates, z up); `strike`, `dip` [°] as for
  `DoubleCoupleSource` (strike clockwise from north = +y, the fault dips to the right of the strike direction);
  `length`, `width` [km] along strike and down dip.
- `slip` [m], `rake` [°], `trise` [s]: numbers, functions `f(a, b)` of the position on the fault (`a` [km] along strike from
  the centre, `b` [km] down dip from the top edge), or `nl × nw` matrices.
- `mu` [Pa]: a number or a function `mu(x, y, z)`; or `model`: the shear modulus ρ vs² at the node nearest to each subfault
  (also of curvilinear meshes).
- Rupture times: from the `hypocenter = (a, b)` on the fault and the rupture velocity `vr` [km/s] (`t0` [s] the start
  there), or `t0` alone as a number, function `f(a, b)` or matrix.
- `stf`: the source time function of the subfaults (default: the `stf` of the simulation), e.g. a [`SampledSTF`](@ref), or a
  function `f(a, b)` returning one per subfault.
- `dims = 2` (or a 2D `model`): a fault in the x–z plane of a 2D model, striking along y (`strike` 0 or 180); `nl = 1`, and
  the moments are per unit length [N m/m].
"""
function finite_fault(; top, strike, dip, length=nothing, width, nl=1, nw, slip, rake, mu=nothing, model=nothing,
                      density_unit=:kg_m3, hypocenter=(0.0, width / 2), vr=nothing, t0=nothing, trise=1.0, stf=nothing,
                      dims=model === nothing ? 3 : (size(model)[2] == 1 ? 2 : 3))
    (mu === nothing) == (model === nothing) && error("finite_fault: give the shear modulus `mu` or the `model`")
    vr === nothing && t0 === nothing && error("finite_fault: give the rupture velocity `vr` (and `hypocenter`) or the rupture times `t0`")
    if dims == 2
        nl == 1 || error("finite_fault: a 2D fault has one subfault along strike (nl = 1)")
        mod(strike, 180) ≈ 0 || error("finite_fault: a 2D fault strikes along y (strike 0 or 180), got $strike")
        length = something(length, 1.0)
    else
        length === nothing && error("finite_fault: give the `length` of the fault")
    end
    dl, dw = length / nl, width / nw
    # unit vectors (east, north, up): along strike, and down dip
    es = (sind(strike), cosd(strike), 0.0)
    ed = (sind(strike + 90) * cosd(dip), cosd(strike + 90) * cosd(dip), -sind(dip))
    value(f, a, b, i, j) = f isa Number ? Float64(f) : f isa AbstractMatrix ? (size(f) == (nl, nw) ||
        error("finite_fault: a matrix needs size ($nl, $nw), got $(size(f))"); Float64(f[i, j])) : f(a, b)
    μat = model === nothing ? (mu isa Number ? (x, y, z) -> Float64(mu) : mu) : shear_modulus_at(model, density_unit)
    area = dims == 2 ? 1e3dw : 1e6 * dl * dw                      # [m²] (2D: [m] per unit length)
    sources = MomentTensorSource[]
    for j in 1:nw, i in 1:nl
        a, b = (i - 0.5) * dl - length / 2, (j - 0.5) * dw
        x, y, z = top .+ a .* es .+ b .* ed
        dims == 2 && (y = top[2])
        tr = value(trise, a, b, i, j)
        start = vr === nothing ? value(t0, a, b, i, j) :
                something(t0 isa Number ? t0 : nothing, 0.0) + hypot(a - hypocenter[1], b - hypocenter[2]) / vr
        M0 = μat(x, y, z) * area * value(slip, a, b, i, j)
        M0 > 0 || continue                                          # (no slip: no source)
        st = stf isa Function && !(stf isa SampledSTF) && applicable(stf, a, b) && !applicable(stf, 0.0, 0.0, 1.0) ? stf(a, b) : stf
        push!(sources, DoubleCoupleSource(x, y, z; M0, strike, dip, rake=value(rake, a, b, i, j), t0=start, trise=tr, stf=st))
    end
    return sources
end

"μ = ρ vs² [Pa] at the node of `model` nearest to (x, y, z) [km] (vertical columns: the nearest column, then the nearest node)."
function shear_modulus_at(model, density_unit)
    X, Y, Z = model.x.val, model.y.val, model.z.val
    xs, ys = X[:, 1, 1], Y[1, :, 1]
    ρf = density_unit === :g_cm3 ? 1e3 : 1.0
    return function (x, y, z)
        i, j = argmin(abs.(xs .- x)), argmin(abs.(ys .- y))
        k = argmin(abs.(view(Z, i, j, :) .- z))
        return model.fields.rho[i, j, k] * ρf * (1e3 * model.fields.vs[i, j, k])^2
    end
end
