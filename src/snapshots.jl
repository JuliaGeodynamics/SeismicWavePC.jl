# Snapshots on planes in OpenSWPC's netCDF format (m_snap): xy, xz, yz sections, free surface (fs) and ocean bottom (ob).
# A kernel runs over the points of every plane at every step (displacement, peak values) and stores the snapshot fields at
# the snapshot steps; rank 0 collects them and writes the files (NCDatasets extension).

"""
    Snapshots(; xy=nothing, xz=nothing, yz=nothing, fs=false, ob=false, quantities=(:v,), decimate=1, title="swpc")

Snapshots of planes written as netCDF files in the format of OpenSWPC, every `snapshot_every` steps into `snapshot_dir`
(pass to [`simulate`](@ref) as `snapshots`; needs `using NCDatasets`):
- `xy = z` [km]: horizontal plane at height `z`; `xz = y`, `yz = x` [km]: vertical sections (in 2D: `xz = true` for the model)
- `fs = true`: the free surface (first solid cell below the air), `ob = true`: the ocean bottom (3D)
- `quantities`: `:v` (velocity `Vx`, `Vy`, `Vz` [m/s]), `:u` (displacement `Ux`, … [m]) and `:ps` (divergence `div` and
  rotation `rot_x`, `rot_y`, `rot_z` [1/s]; in 2D P-SV `divergence` and `rotation`)
- `decimate`: every `decimate`-th cell per direction (an integer or a tuple)
Files are named `<title>.<solver>.<plane>.<quantity>.nc` (e.g. `swpc.3d.fs.v.nc`) with the coordinates `x`, `y`, `z` [km,
z positive down as in OpenSWPC], time `t` and the medium `rho`, `lambda`, `mu`; velocity files of horizontal planes also get
the peak values `max-V` (vertical), `max-H` (horizontal) and `max-A` (vector) over all time steps. Unlike OpenSWPC, the
components are at the cell centres with x east, y north and z up, and `t` is the time of the snapshot.
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
end
function Snapshots(; xy=nothing, xz=nothing, yz=nothing, fs=false, ob=false, quantities=(:v,), decimate=1, title="swpc")
    all(in((:v, :u, :ps)), quantities) || error("snapshot quantities must be :v, :u or :ps, got $quantities")
    f(v) = v === nothing ? nothing : v === true ? 0.0 : Float64(v)
    dec = decimate isa Integer ? (decimate, decimate, decimate) : Tuple(Int.(decimate))
    return Snapshots(f(xy), f(xz), f(yz), fs, ob, Tuple(quantities), dec, title)
end

"Write the netCDF snapshot files (implemented in the NCDatasets extension)."
function snapshot_create end
snapshot_create(args::Any...) = error(NETCDF_HINT)
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
Set up the planes of `s` on this rank: the points (global and local cells), device buffers and files (rank 0).
Returns a vector of named tuples, one per plane.
"""
function setup_planes(s::Snapshots, ::Val{mode}, g::Grid{N}, med, off, nl, dev, backend, ::Type{T}, dir, me) where {mode,N,T}
    mode !== :p3d && (s.xy !== nothing || s.yz !== nothing || s.ob || s.fs) && error("2D snapshots: only xz")
    mode === :sh && :ps in s.quantities && error("SH snapshots have no :ps")
    solver = mode === :p3d ? "3d" : string(mode)
    nc = mode === :psv ? 2 : mode === :sh ? 1 : 3
    nq = 2nc + (mode === :psv ? 2 : mode === :sh ? 0 : 4)
    # decimated physical indices and cell-centre coordinates (z: depth) along each axis
    idx(a) = (d = s.decimate[a == N ? 3 : a]; [ii * d - d ÷ 2 for ii in 1:(g.n[a] + d ÷ 2) ÷ d])
    coord(a, i) = g.beg[a] + (i - 0.5) * g.d[a]
    planes = Tuple{Symbol,Vector{NTuple{N,Int}},NTuple{2,Pair{String,Vector{Float64}}}}[]
    if N == 3
        xi, yi, zi = idx(1), idx(2), idx(3)
        ax(a, ii) = string("xyz"[a]) => coord.(a, ii)
        s.xy === nothing || (k = cell_index(g, 3, -s.xy) - H; push!(planes, (:xy, [(i, j, k) for i in xi, j in yi][:], (ax(1, xi), ax(2, yi)))))
        s.xz === nothing || (j = cell_index(g, 2, s.xz) - H; push!(planes, (:xz, [(i, j, k) for i in xi, k in zi][:], (ax(1, xi), ax(3, zi)))))
        s.yz === nothing || (i = cell_index(g, 1, s.yz) - H; push!(planes, (:yz, [(i, j, k) for j in yi, k in zi][:], (ax(2, yi), ax(3, zi)))))
        surf(kk) = [(i, j, kk[i+H, j+H] + 1) for i in xi, j in yi][:]
        s.fs && push!(planes, (:fs, surf(med.kfs), (ax(1, xi), ax(2, yi))))
        s.ob && push!(planes, (:ob, surf(med.kob), (ax(1, xi), ax(2, yi))))
    else
        xi, zi = idx(1), idx(2)
        s.xz === nothing || push!(planes, (:xz, [(i, k) for i in xi, k in zi][:], ("x" => coord.(1, xi), "z" => coord.(2, zi))))
    end
    isempty(planes) && error("Snapshots without planes")
    fields = snapshot_fields(mode)
    map(planes) do (name, cells, axes)
        G = [c .+ H for c in cells]                                                # global padded indices
        inside(c) = all(a -> 1 <= c[a] - H <= g.n[a], 1:N)
        all(inside, G) || error("snapshot plane $name is outside the model")
        own = [n for n in eachindex(G) if all(H + 1 .<= Base.front(G[n]) .- Base.front(off) .<= Base.front(nl) .- H)]
        names = N == 3 ? (:i, :j, :k) : (:i, :k)
        pts = NamedTuple{names}(ntuple(a -> dev(Int32[G[n][a] - off[a] for n in own]), N))
        m = max(length(own), 1)
        files = Dict{Symbol,String}()
        if me == 0
            medium = (rho=[med.rho[c...] for c in G], lambda=[med.lam[c...] for c in G], mu=[med.mu[c...] for c in G])
            shape = (length(axes[1][2]), length(axes[2][2]))
            for q in s.quantities
                vars, unit, _ = fields[q]
                file = joinpath(mkpath(dir), "$(s.title).$solver.$name.$q.nc")
                snapshot_create(file, axes, vars, unit, map(A -> reshape(A, shape), medium))
                files[q] = file
            end
        end
        (; name, own, npoints=length(G), shape=(length(axes[1][2]), length(axes[2][2])), pts, files,
           B=KernelAbstractions.zeros(backend, T, m, nq), U=KernelAbstractions.zeros(backend, T, m, nc),
           PK=KernelAbstractions.zeros(backend, T, m, 3))
    end
end

"Collect the snapshot of every plane on rank 0 and append it to the files (number `isnap`, time `t`)."
function write_planes(planes, mode, isnap, t, scale, parallel, me)
    fields = snapshot_fields(mode)
    for p in planes
        G = zeros(p.npoints, size(p.B, 2))
        G[p.own, :] .= Array(p.B)[1:length(p.own), :]
        parallel && MPI.Allreduce!(G, +, MPI.COMM_WORLD)
        me == 0 || continue
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
        P = zeros(p.npoints, 3)
        P[p.own, :] .= Array(p.PK)[1:length(p.own), :]
        parallel && MPI.Allreduce!(P, max, MPI.COMM_WORLD)
        me == 0 && snapshot_peaks(p.files[:v], map(c -> reshape(P[:, c] .* scale, p.shape), (1, 2, 3)))
    end
end
