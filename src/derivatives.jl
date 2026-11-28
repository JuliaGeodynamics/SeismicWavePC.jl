# Divergence and curl of the velocity at the stations (`simulate(...; derivatives = true)`): from auxiliary stations at
# ±dh, ±2dh around each station (4th-order centred differences; at surface stations the vertical derivatives are one-sided
# from below, 4th order).

"The mean grid spacing in x of a model [km] (the default step of the derivatives; curvilinear grids are finer at their sides)."
function model_dx(model)
    x = model isa CartData ? model.x.val[:, 1, 1] : model.x
    return (x[end] - x[1]) / (length(x) - 1)
end

"The height [km] of the surface of `model` at (x, y): the highest node (cell) with vp > 0 in the nearest column."
function surface_height(model::CartData, x, y)
    i, j = argmin(abs.(model.x.val[:, 1, 1] .- x)), argmin(abs.(model.y.val[1, :, 1] .- y))
    zc, vp = view(model.z.val, i, j, :), view(model.fields.vp, i, j, :)
    k = findall(>(0), vp)
    isempty(k) && error("derivatives: no rock or water below the station at ($x, $y)")
    return maximum(zc[k])
end
surface_height(::DistributedModel, x, y) = error("derivatives = true: stations at the surface need a CartData model (its surface height)")

function simulate_derivatives(model, dh; wave, kwargs...)
    stations = collect(Station, get(kwargs, :stations, Station[]))
    isempty(stations) && error("derivatives = true needs stations")
    h = something(dh, model_dx(model))
    three = size(model)[2] > 1
    # per station and axis: the points of the difference (0: the station; else the index of an auxiliary station) and weights
    central, onesided = (1, -8, 8, -1) ./ 12h, (25, -48, 36, -16, 3) ./ 12h          # (at -2h, -h, h, 2h; at 0, -h, …, -4h)
    aux = Station[]
    plan = map(stations) do st
        z0 = st.surface ? surface_height(model, st.x, st.y) : st.z
        add(dx, dy, dz) = (push!(aux, Station("$(st.name)#$(length(aux))", st.x + dx, st.y + dy, z0 + dz;
                                               surface=st.surface && dz == 0)); length(aux))
        line(e) = (Tuple(add((k .* h .* e)...) for k in (-2, -1, 1, 2)), central)
        (; x=line((1, 0, 0)), y=three ? line((0, 1, 0)) : nothing,
           z=st.surface ? ((0, (add(0, 0, -k * h) for k in 1:4)...), onesided) : line((0, 0, 1)))
    end
    kw = Base.structdiff(values(kwargs), NamedTuple{(:stations,)})
    r = simulate(model; wave, kw..., stations=[stations; aux])
    ns, nt = length(stations), length(r.t)
    comp(c) = findfirst(==(c), r.components)
    # d v_c / d axis at station n [1/s] (points: the station itself is index n, auxiliary ones ns + index)
    function d(n, c, a)
        q = comp(c)
        q === nothing && return zeros(nt)
        p = getfield(plan[n], a)
        p === nothing && return zeros(nt)
        pts, w = p
        return sum(w[k] .* r.v[:, q, pts[k] == 0 ? n : ns + pts[k]] for k in eachindex(pts)) ./ 1e3
    end
    div = zeros(nt, ns)
    curl = zeros(nt, 3, ns)
    for n in 1:ns
        div[:, n] = d(n, :x, :x) .+ d(n, :y, :y) .+ d(n, :z, :z)
        curl[:, 1, n] = d(n, :z, :y) .- d(n, :y, :z)
        curl[:, 2, n] = d(n, :x, :z) .- d(n, :z, :x)
        curl[:, 3, n] = d(n, :y, :x) .- d(n, :x, :y)
    end
    keep = 1:ns
    return Seismograms(r.t, r.v[:, :, keep], r.u[:, :, keep], r.stress[:, :, keep], r.strain[:, :, keep], stations, r.components,
                       r.tensor_components, r.wavefield, div, curl)
end
