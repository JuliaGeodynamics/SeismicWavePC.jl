# Reciprocal Green's functions (OpenSWPC green_mode, m_green): a unit force at the receiver, strains recorded at many points.

"""
    ReciprocalGreen

Result of [`reciprocal_green`](@ref). Fields:
- `t`: times [s]
- `G`: `G[it, m, n]`, Green's function `m` at point `n`; `components` are `(:xx, :yy, :zz, :yz, :xz, :xy)` in 3D (only the
  ones of the 2D solver in 2D) and, with `displacement = true`, also `(:fx, [:fy,] :fz)`
- `points`, `receiver` (x, y, z [km]) and `component` (`:x`, `:y` or `:z`) of the receiver

By reciprocity, the velocity [m/s] of the receiver `component` caused by a moment tensor source `M` [N m] (with the source time
function and rise time of the simulation) at point `n` is `Σ_m M[m] G[:, m, n]`, and by a force `F` [N] it is
`Σ_i F[i] G[:, fi, n]` (the force history then being the derivative of the source time function); see [`seismogram`](@ref).
`G` is the displacement gradient (twice the strain for the shear components) and the displacement for a unit force at the
receiver, as written by OpenSWPC (there in nm/s and nm; [`write_sac`](@ref) writes the same files).
"""
struct ReciprocalGreen
    t::Vector{Float64}
    G::Array{Float64,3}
    components::Tuple{Vararg{Symbol}}
    points::Vector{Station}
    receiver::NTuple{3,Float64}
    component::Symbol
end

function Base.show(io::IO, ::MIME"text/plain", g::ReciprocalGreen)
    print(io, "ReciprocalGreen: unit force along ", g.component, " at (", join(g.receiver, ", "), ") km, ", length(g.points),
          " point(s), ", length(g.t), " samples, components ", join(g.components, ", "))
end

"""
    green = reciprocal_green(model, receiver, points; component=:z, trise=1.0, displacement=false, dt, nt, kwargs...)

Reciprocal Green's functions, as OpenSWPC's `green_mode`: a unit force [1 N] along `component` (`:x`, `:y` or `:z`, up) at the
`receiver` `(x, y, z)` [km] with the source time function `stf` (keyword of [`simulate`](@ref)) and rise time `trise`, and the
displacement gradients (and with `displacement = true` the displacements) at the `points` (`Station`s, e.g. a grid of possible
source locations). One simulation then gives the seismograms at the receiver for sources at all points
([`seismogram`](@ref)). Other keywords are passed to [`simulate`](@ref). Returns a [`ReciprocalGreen`](@ref).
"""
function reciprocal_green(model::CartData, receiver, points; component=:z, trise=1.0, displacement=false, kwargs...)
    component in (:x, :y, :z) || error("component must be :x, :y or :z, got :$component")
    x, y, z = receiver
    force = ForceSource(x, y, z; fx=Float64(component === :x), fy=Float64(component === :y), fz=Float64(component === :z), trise)
    r = simulate(model; sources=force, stations=collect(Station, points), kwargs...)
    shear = [c in (:xx, :yy, :zz) ? 1.0 : 2.0 for c in r.tensor_components]
    G = r.strain .* reshape(shear, 1, :, 1)
    components = r.tensor_components
    if displacement
        G = cat(G, r.u; dims=2)
        components = (components..., Symbol.(:f, r.components)...)
    end
    return ReciprocalGreen(r.t, G, components, r.stations, (Float64(x), Float64(y), Float64(z)), component)
end

"""
    v = seismogram(green::ReciprocalGreen, n, source::Union{MomentTensorSource,ForceSource})

Velocity [m/s] of the receiver component of `green` for `source` at point `n`, by reciprocity (the source position is that of
point `n`; its time function is the one of the Green's functions).
"""
function seismogram(g::ReciprocalGreen, n::Integer, s::MomentTensorSource)
    names = (:xx, :yy, :zz, :yz, :xz, :xy)
    return sum(s.M0 * s.M[findfirst(==(c), names)] .* g.G[:, m, n] for (m, c) in enumerate(g.components) if c in names)
end
function seismogram(g::ReciprocalGreen, n::Integer, s::ForceSource)
    any(in((:fx, :fy, :fz)), g.components) || error("Green's functions without displacements: use displacement = true")
    return sum(s.F[i] .* g.G[:, findfirst(==(c), g.components), n] for (i, c) in enumerate((:fx, :fy, :fz)) if c in g.components)
end

"""
    write_sac(green::ReciprocalGreen, dir="."; title="swpc", station="RCV")

Write the Green's functions as SAC files with OpenSWPC's names and units:
`dir/<title>__<point>__<station>__<component>__<m>__.sac` (`m` = `mxx`, …, `fx_`, …) in nm/s (nm for the forces); the
point number is its index in `points`. Returns the file names.
"""
function write_sac(g::ReciprocalGreen, dir="."; title="swpc", station="RCV")
    mkpath(dir)
    files = String[]
    for (n, p) in enumerate(g.points), (m, c) in enumerate(g.components)
        name = startswith(string(c), "f") ? rpad(string(c), 3, '_') : "m" * string(c)
        file = joinpath(dir, "$(title)__$(lpad(n, 8, '0'))__$(station)__$(g.component)__$(name)__.sac")
        write_sac_file(file, Float32.(g.G[:, m, n] .* 1e9), g.t, p, "G_V$(g.component)_$name", lpad(n, 8, '0'),
                       startswith(name, "f") ? 6 : 7, (SAC_UNDEF, SAC_UNDEF), nothing)
        push!(files, file)
    end
    return files
end
