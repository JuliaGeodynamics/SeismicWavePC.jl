# Snapshots on planes in OpenSWPC's netCDF format (m_snap): xy, xz, yz sections, free surface (fs) and ocean bottom (ob).
# A kernel runs over the points of every plane at every step (displacement, peak values) and stores the snapshot fields at
# the snapshot steps; rank 0 collects them and writes the files, or every rank writes its own points in pieces (NCDatasets
# extension).

"""
    Snapshots(; xy=nothing, xz=nothing, yz=nothing, fs=false, ob=false, quantities=(:v,), decimate=1, title="swpc", pieces=false)

Snapshots of planes written as netCDF files in the format of OpenSWPC, every `snapshot_every` steps into `snapshot_dir`
(pass to [`simulate`](@ref) as `snapshots`; needs `using NCDatasets`):
- `xy = z` [km]: horizontal plane at height `z`; `xz = y`, `yz = x` [km]: vertical sections (in 2D: `xz = true` for the model)
- `fs = true`: the free surface (first solid cell below the air), `ob = true`: the ocean bottom (3D)
- `quantities`: `:v` (velocity `Vx`, `Vy`, `Vz` [m/s]), `:u` (displacement `Ux`, … [m]) and `:ps` (divergence `div` and
  rotation `rot_x`, `rot_y`, `rot_z` [1/s]; in 2D P-SV `divergence` and `rotation`)
- `decimate`: every `decimate`-th cell per direction (an integer or a tuple)
- `pieces = true`: every MPI process writes the points of its own subdomain to files of its own, named with its rank
  (`swpc.3d.fs.v.0000.nc`, …), instead of process 0 collecting every plane: then no process holds a whole plane.
  [`read_snapshots`](@ref) reads either kind.
Files are named `<title>.<solver>.<plane>.<quantity>.nc` (e.g. `swpc.3d.fs.v.nc`) with the coordinates `x`, `y`, `z` [km,
z positive down as in OpenSWPC], time `t` and the medium `rho`, `lambda`, `mu`; velocity files of horizontal planes also get
the peak values `max-V` (vertical), `max-H` (horizontal) and `max-A` (vector) over all time steps. Unlike OpenSWPC, the
components are at the cell centres with x east, y north and z up, and `t` is the time of the snapshot.

On curvilinear grids the points are the nodes of the mesh columns: `fs` is the topography, `xy` planes and the sections are
interpolated at regular heights (zero, and medium NaN, above the topography); the divergence and rotation (`:ps`) from the
SBP operators at the nodes.
"""
struct Snapshots
    xy::Union{Nothing,Float64}
    xz::Union{Nothing,Float64}
    yz::Union{Nothing,Float64}
    fs::Bool
    ob::Bool
    quantities::Tuple{Vararg{Symbol}}
    decimate::NTuple{3,Int}
    title::String
    pieces::Bool
end
function Snapshots(; xy=nothing, xz=nothing, yz=nothing, fs=false, ob=false, quantities=(:v,), decimate=1, title="swpc", pieces=false)
    all(in((:v, :u, :ps)), quantities) || error("snapshot quantities must be :v, :u or :ps, got $quantities")
    f(v) = v === nothing ? nothing : v === true ? 0.0 : Float64(v)
    dec = decimate isa Integer ? (decimate, decimate, decimate) : Tuple(Int.(decimate))
    return Snapshots(f(xy), f(xz), f(yz), fs, ob, Tuple(quantities), dec, title, pieces)
end

"Write the netCDF snapshot files (implemented in the NCDatasets extension)."
function snapshot_create end
snapshot_create(args::Any...; kwargs...) = error(NETCDF_HINT)
function snapshot_append end
function snapshot_peaks end

"Field names and units of the snapshot quantities, and their columns in the plane buffer."
function snapshot_fields(mode)
    v = mode === :psv ? ["Vx", "Vz"] : mode === :sh ? ["Vy"] : ["Vx", "Vy", "Vz"]
    nc = length(v)
    ps = mode === :psv ? ["divergence", "rotation"] : mode === :sh ? String[] : ["div", "rot_x", "rot_y", "rot_z"]
    return (v=(v, "m/s", 1:nc), u=(replace.(v, "V" => "U"), "m", nc+1:2nc), ps=(ps, "1/s", 2nc+1:2nc+length(ps)))
end

"""
Set up the planes of `s` on this rank: the points (global and local cells), device buffers and files (written by rank 0, or
in pieces by every rank that has points of the plane). Returns a vector of named tuples, one per plane.
"""
function setup_planes(s::Snapshots, ::Val{mode}, g::Grid{N}, med, off, nl, dev, backend, ::Type{T}, dir, me, parallel) where {mode,N,T}
    mode !== :p3d && (s.xy !== nothing || s.yz !== nothing || s.ob || s.fs) && error("2D snapshots: only xz")
    mode === :sh && :ps in s.quantities && error("SH snapshots have no :ps")
    solver = mode === :p3d ? "3d" : string(mode)
    nc = mode === :psv ? 2 : mode === :sh ? 1 : 3
    nq = 2nc + (mode === :psv ? 2 : mode === :sh ? 0 : 4)
    # decimated physical indices and cell-centre coordinates (z: depth) along each axis
    idx(a) = (d = s.decimate[a == N ? 3 : a]; [ii * d - d ÷ 2 for ii in 1:(g.n[a] + d ÷ 2) ÷ d])
    coord(a, i) = a == length(g.n) ? g.zc[i+H] : g.beg[a] + (i - 0.5) * g.d[a]
    ax(a, ii) = string((N == 3 ? "xyz" : "xz")[a]) => coord.(a, ii)
    # in pieces a rank has the points of its own subdomain: `part` of the indices of an axis, after those of the ranks `before`
    onrank(a, i) = a == N || off[a] < i <= off[a] + nl[a] - 2H
    part(a, ii) = s.pieces ? filter(i -> onrank(a, i), ii) : ii
    before(a, ii) = s.pieces && a < N ? count(<=(off[a]), ii) : 0
    valid(name, a, i) = 1 <= i <= g.n[a] || error("snapshot plane $name is outside the model")
    planes = @NamedTuple{name::Symbol, cells::Vector{NTuple{N,Int}}, axes::NTuple{2,Pair{String,Vector{Float64}}}, whole::NTuple{2,Int}, offset::NTuple{2,Int}}[]
    # a plane along the axes `a`: its cells on this rank, their indices `I` along the axes, and the indices `W` of the whole plane
    plane(name, cells, a, I, W) = push!(planes, (; name, cells=vec(cells), axes=(ax(a[1], I[1]), ax(a[2], I[2])), whole=length.(W),
                                                  offset=(before(a[1], W[1]), before(a[2], W[2]))))
    if N == 3
        xi, yi, zi = idx(1), idx(2), idx(3)
        xs, ys = part(1, xi), part(2, yi)
        if s.xy !== nothing
            k = cell_index(g, 3, -s.xy) - H
            valid(:xy, 3, k)
            plane(:xy, [(i, j, k) for i in xs, j in ys], (1, 2), (xs, ys), (xi, yi))
        end
        if s.xz !== nothing                                   # (in pieces: on the ranks the section runs through)
            j = cell_index(g, 2, s.xz) - H
            valid(:xz, 2, j)
            xr = !s.pieces || onrank(2, j) ? xs : Int[]
            plane(:xz, [(i, j, k) for i in xr, k in zi], (1, 3), (xr, zi), (xi, zi))
        end
        if s.yz !== nothing
            i = cell_index(g, 1, s.yz) - H
            valid(:yz, 1, i)
            yr = !s.pieces || onrank(1, i) ? ys : Int[]
            plane(:yz, [(i, j, k) for j in yr, k in zi], (2, 3), (yr, zi), (yi, zi))
        end
        # (of a DistributedModel a rank knows the surface of the columns around it only: it owns no points of the others)
        surf(kk) = [(i, j, held(kk, i + H, j + H) ? kk[i+H, j+H] + 1 : 1) for i in xs, j in ys]
        s.fs && plane(:fs, surf(med.kfs), (1, 2), (xs, ys), (xi, yi))
        s.ob && plane(:ob, surf(med.kob), (1, 2), (xs, ys), (xi, yi))
    else
        xi, zi = idx(1), idx(2)
        xs = part(1, xi)
        s.xz === nothing || plane(:xz, [(i, k) for i in xs, k in zi], (1, 2), (xs, zi), (xi, zi))
    end
    isempty(planes) && error("Snapshots without planes")
    fields = snapshot_fields(mode)
    map(planes) do (; name, cells, axes, whole, offset)
        G = [c .+ H for c in cells]                                                # global padded indices
        inside(c) = all(a -> 1 <= c[a] - H <= g.n[a], 1:N)
        all(inside, G) || error("snapshot plane $name is outside the model")
        own = [n for n in eachindex(G) if all(H + 1 .<= Base.front(G[n]) .- Base.front(off) .<= Base.front(nl) .- H)]
        names = N == 3 ? (:i, :j, :k) : (:i, :k)
        pts = NamedTuple{names}(ntuple(a -> dev(Int32[G[n][a] - off[a] for n in own]), N))
        m = max(length(own), 1)
        shape = (length(axes[1][2]), length(axes[2][2]))
        files = Dict{Symbol,String}()
        # the files: written by rank 0, which gets the medium of the whole plane; in pieces by every rank with points of it
        medium = s.pieces ? (rho=[med.rho[c...] for c in G], lambda=[med.lam[c...] for c in G], mu=[med.mu[c...] for c in G]) :
                 plane_medium(med.rho, med, G, own, parallel, me)
        if s.pieces ? !isempty(G) : me == 0
            for q in s.quantities
                vars, unit, _ = fields[q]
                file = joinpath(mkpath(dir), "$(s.title).$solver.$name.$q" * (s.pieces ? "." * lpad(me, 4, '0') : "") * ".nc")
                snapshot_create(file, axes, vars, unit, map(A -> reshape(A, shape), medium); piece=s.pieces ? (whole, offset) : nothing)
                files[q] = file
            end
        end
        (; name, own, npoints=length(G), shape, pts, files, pieces=s.pieces,
           B=KernelAbstractions.zeros(backend, T, m, nq), U=KernelAbstractions.zeros(backend, T, m, nc),
           PK=KernelAbstractions.zeros(backend, T, m, 3))
    end
end

"Medium `(rho, lambda, mu)` at the plane points `G`, on rank 0 (`nothing` on the others)."
plane_medium(::Array, med, G, own, parallel, me) =
    me == 0 ? (rho=[med.rho[c...] for c in G], lambda=[med.lam[c...] for c in G], mu=[med.mu[c...] for c in G]) : nothing
function plane_medium(::Window, med, G, own, parallel, me)      # a DistributedModel: collected from the ranks that own the points
    fields = map((med.rho, med.lam, med.mu)) do A
        v = zeros(length(G))
        for n in own
            v[n] = A[G[n]...]
        end
        parallel && MPI.Allreduce!(v, +, MPI.COMM_WORLD)
        v
    end
    return me == 0 ? NamedTuple{(:rho, :lambda, :mu)}(fields) : nothing
end

"""
Append the snapshot of every plane to its files (number `isnap`, time `t`): collected on rank 0, or in pieces the points of
this rank to its own files.
"""
function write_planes(planes, mode, isnap, t, scale, parallel, me)
    fields = snapshot_fields(mode)
    for p in planes
        if p.pieces
            isempty(p.files) && continue
            G = Float64.(Array(p.B))
        else
            G = zeros(p.npoints, size(p.B, 2))
            G[p.own, :] .= Array(p.B)[1:length(p.own), :]
            parallel && MPI.Allreduce!(G, +, MPI.COMM_WORLD)
            me == 0 || continue
        end
        for (q, file) in p.files
            vars, _, cols = fields[q]
            f = q === :ps ? scale * 1e-3 : scale
            snapshot_append(file, isnap, t, Dict(v => reshape(G[:, c] .* f, p.shape) for (v, c) in zip(vars, cols)))
        end
    end
end

"Peak values of the horizontal planes, written to their velocity files at the end (all ranks take part)."
function write_peaks(planes, quantities, mode, scale, parallel, me)
    (mode === :p3d && :v in quantities) || return
    for p in planes
        p.name in (:xy, :fs, :ob) || continue
        if p.pieces
            isempty(p.files) && continue
            P = Float64.(Array(p.PK))
        else
            P = zeros(p.npoints, 3)
            P[p.own, :] .= Array(p.PK)[1:length(p.own), :]
            parallel && MPI.Allreduce!(P, max, MPI.COMM_WORLD)
            me == 0 || continue
        end
        snapshot_peaks(p.files[:v], map(c -> reshape(P[:, c] .* scale, p.shape), (1, 2, 3)))
    end
end

"""
    SnapshotSeries

The snapshots of one plane and quantity, as returned by [`read_snapshots`](@ref): in one file, or in the pieces written by the
MPI processes. Fields: `axes` (the two coordinates of the plane as `name => values` [km]), `t` (times [s]), `variables` (the
names of the fields) and `files`.
"""
struct SnapshotSeries
    files::Vector{String}
    offsets::Vector{NTuple{2,Int}}        # where the piece of every file starts in the plane, and its size
    sizes::Vector{NTuple{2,Int}}
    axes::NTuple{2,Pair{String,Vector{Float64}}}
    t::Vector{Float64}
    variables::Vector{String}
end

function Base.show(io::IO, s::SnapshotSeries)
    print(io, "SnapshotSeries: ", join((string(length(c), " ", n) for (n, c) in s.axes), " × "), " points, ", length(s.t), " snapshot(s), fields ",
          join(s.variables, ", "), " (", length(s.files), " file", length(s.files) == 1 ? "" : "s", ")")
end

"""
    s = read_snapshots(file)

Read the snapshots of one plane and quantity written by [`simulate`](@ref) (see [`Snapshots`](@ref); needs `using NCDatasets`),
e.g. `read_snapshots("snapshots/swpc.3d.fs.v.nc")`. If they were written in pieces (`pieces = true`: `swpc.3d.fs.v.0000.nc`,
…), the pieces are put together, so that both kinds read the same; pieces that overlap or leave points out, or a single file
next to pieces of it (files of another run), are an error. Returns a [`SnapshotSeries`](@ref):
- `s.axes`: the two coordinates of the plane as `name => values` [km]; `s.t`: the times [s]; `s.variables`: the names of the fields
- `s["Vx", k]`: the field `Vx` of snapshot `k` as a matrix
- `s["max-H"]`: a field without time (medium, peak values) as a matrix; for a field with time all snapshots, `(n1, n2, nt)`
"""
function read_snapshots end
read_snapshots(args::Any...) = error(NETCDF_HINT)
