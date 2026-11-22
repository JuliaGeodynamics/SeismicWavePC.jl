# Driver: domain decomposition (ImplicitGlobalGrid), sources, stations, time loop and output.

"""
    Station(name, x, y, z; surface=false)
    Station(name, lon, lat, z, proj::ProjectionPoint; surface=false)

Receiver at model coordinates `x, y, z` [km] (z up, as in the `CartData`), or at a geographic location that is projected
with the same `ProjectionPoint` that was used to create the model (see `GeophysicalModelGenerator.convert2CartData`).
With `surface=true` the station is placed in the first solid cell below the free surface of its column and `z` is ignored.
In 2D models `y` is ignored.
"""
struct Station
    name::String
    x::Float64
    y::Float64
    z::Float64
    surface::Bool
end
Station(name, x, y, z; surface=false) = Station(name, x, y, z, surface)
Station(name, lon, lat, z, proj::ProjectionPoint; surface=false) = Station(name, project(lon, lat, z, proj)...; surface)

function Base.show(io::IO, s::Station)
    print(io, "Station(\"", s.name, "\", x=", s.x, ", y=", s.y, s.surface ? ", at surface)" : ", z=$(s.z))")
end

"""
    Seismograms

Result of [`simulate`](@ref). Fields:
- `t`: times [s] of the samples
- `v`: velocities `v[it, component, station]` [m/s], components along `components` (z positive up)
- `stations`: the `Station`s
- `components`: `(:x, :z)` (P-SV), `(:y,)` (SH) or `(:x, :y, :z)` (3D)
- `wavefield`: velocity field `V` [m/s] at the end of the simulation as `CartData` (on MPI rank 0; `nothing` elsewhere)

`r["name"]` returns the `(nt, ncomponents)` matrix of the station called `"name"`.
"""
struct Seismograms
    t::Vector{Float64}
    v::Array{Float64,3}
    stations::Vector{Station}
    components::Tuple{Vararg{Symbol}}
    wavefield::Union{CartData,Nothing}
end

function Base.getindex(r::Seismograms, name::AbstractString)
    n = findfirst(s -> s.name == name, r.stations)
    n === nothing && error("no station \"$name\"")
    return r.v[:, :, n]
end

function Base.show(io::IO, ::MIME"text/plain", r::Seismograms)
    println(io, "Seismograms: ", length(r.stations), " station(s), ", length(r.t), " samples, t = ",
            first(r.t), " … ", round(last(r.t), sigdigits=6), " s, components ", join(r.components, ", "))
    for (n, s) in enumerate(r.stations)
        println(io, "  ", rpad(s.name, 10), " max |v| = ", join((@sprintf("%.3e", maximum(abs, view(r.v, :, c, n))) for c in eachindex(r.components)), ", "), " m/s")
    end
    r.wavefield === nothing || print(io, "  final wavefield: CartData of size ", size(r.wavefield))
end

function igg_device(backend)
    name = string(nameof(typeof(backend)))
    name == "CUDABackend" && return "CUDA"
    name == "ROCBackend" && return "AMDGPU"
    backend isa CPU && return "none"
    error("MPI runs are supported on CPU, CUDA and AMDGPU backends, not $name")
end

"""
    result = simulate(model::CartData; dt, nt, sources, kwargs...)

Run a visco-elastic wave propagation simulation (OpenSWPC algorithm) on the velocity model `model` and return
[`Seismograms`](@ref).

`model` is a regular `CartData` (coordinates in km, z up) with fields `rho` [g/cm³], `vp`, `vs` [km/s] and optionally
`Qp`, `Qs` (default: no attenuation). Cells with `vp = vs = 0` are air: the free surface and topography follow from that.
Cells with `vs = 0` and `vp > 0` are water. A `CartData` of size `(nx, 1, nz)` gives a 2D simulation in the x-z plane.

Keyword arguments
- `dt`, `nt`: time step [s] and number of steps; `tbeg = 0.0` is the start time.
- `sources`: `MomentTensorSource` or vector of them (see `DoubleCoupleSource`); `stf = :kupper` source time function (see `moment_rate`).
- `stations = Station[]`; seismograms are recorded every `record_every = 1` steps.
- `wave = :psv`: `:psv` or `:sh`, for 2D models.
- `na = 20`: thickness of the Cerjan absorbing layer (sides and bottom) in cells.
- `fq_min = 0.05`, `fq_max = 5.0`, `fq_ref = 1.0`: frequency band [Hz] of constant Q, and reference frequency of the velocities.
- `snapshot_every = 0`: if > 0, write the velocity field every so many steps to ParaView files in `snapshot_dir = "snapshots"`.
- `backend = CPU()`: KernelAbstractions backend, e.g. `CUDABackend()`, `ROCBackend()`, `MetalBackend()`.
- `T = Float32`: floating point type of the wavefield.
- `nproc = 1`: MPI processes in x (2D), or `(npx, npy)` (3D); start julia with `mpiexec -n npx*npy`.
- `verbose = true`: print a summary and the progress (on rank 0).
"""
function simulate(model::CartData; wave=:psv, kwargs...)
    mode = size(model)[2] == 1 ? wave : :p3d
    mode in (:psv, :sh, :p3d) || error("wave must be :psv or :sh, got :$wave")
    return simulate(model, Val(mode); kwargs...)
end

# Function barrier on the solver type: from here on the dimension N is known to the compiler.
function simulate(model::CartData, ::Val{mode}; dt, nt, sources, stations=Station[], stf=:kupper, tbeg=0.0,
                  na=20, fq_min=0.05, fq_max=5.0, fq_ref=1.0, record_every=1,
                  snapshot_every=0, snapshot_dir="snapshots", backend=CPU(), T::Type{FT}=Float32, nproc=1, verbose=true) where {mode,FT}
    N = mode === :p3d ? 3 : 2
    sources = sources isa MomentTensorSource ? [sources] : collect(MomentTensorSource, sources)
    stations = collect(Station, stations)
    isempty(sources) && error("at least one source is required")
    m = model_arrays(model)
    g = Grid(m, Val(N))
    med = setup_medium(m, g, na, fq_min, fq_max, fq_ref)
    courant = med.vmax * dt * sqrt(sum(1 ./ g.d .^ 2))
    courant < 6 / 7 || error("unstable time step: Courant number $courant ≥ 6/7, use dt < $(dt * 6 / 7 / courant)")

    # domain decomposition in x (and y); every rank owns n/dims cells plus H ghost cells on each side
    dims = ntuple(a -> a < N && a <= length(nproc) ? nproc[a] : 1, N)
    all(g.n .% dims .== 0) || error("cells per direction $(g.n) must be divisible by nproc $dims")
    nl = g.n .÷ dims .+ 2H
    parallel = prod(dims) > 1
    me, coords = 0, ntuple(_ -> 0, N)
    if parallel
        me, _, _, c3 = IGG.init_global_grid(nl..., ntuple(_ -> 1, 3 - N)...; dimx=dims[1], dimy=N == 3 ? dims[2] : 1, dimz=1,
                                            overlaps=(2H, 2H, 2H), halowidths=(H, H, H), init_MPI=!MPI.Initialized(),
                                            device_type=igg_device(backend), quiet=true)
        coords = ntuple(a -> Int(c3[a]), N)
    end
    off = coords .* (nl .- 2H)
    R = ntuple(a -> off[a]+1:off[a]+nl[a], N)
    dev(A) = (B = KernelAbstractions.allocate(backend, eltype(A), size(A)); copyto!(B, A); B)

    # medium on this rank
    loc(A) = dev(FT.(A[R...]))
    col(A) = dev(A[Base.front(R)...])
    sponge(a) = map(v -> dev(FT.(v[R[a]])), cerjan(g.n[a], g.d[a], na; top=a < N))
    (gx_c, gx_b), (gz_c, gz_b) = sponge(1), sponge(N)
    G = N == 3 ? (; gx_c, gx_b, gy_c=sponge(2)[1], gy_b=sponge(2)[2], gz_c, gz_b) : (; gx_c, gx_b, gz_c, gz_b)
    M = merge((; rho=loc(med.rho), lam=loc(med.lam), mu=loc(med.mu), taup=loc(med.taup), taus=loc(med.taus),
                 fs_top=col(med.fs_top), fs_bot=col(med.fs_bot), ob_top=col(med.ob_top), ob_bot=col(med.ob_bot)), G)
    c = ntuple(a -> (FT(9 / 8 / g.d[a]), FT(1 / 24 / g.d[a]), FT(1 / g.d[a])), N)
    vc = memory_coefficients(med.ts, dt, FT)
    ijk = N == 3 ? (:i, :j, :k) : (:i, :k)

    # sources: stress glut at the cell containing the source (normalized by the total moment, as in OpenSWPC)
    M0 = sum(s.M0 for s in sources)
    position(x, y, z) = N == 3 ? (cell_index(g, 1, x), cell_index(g, 2, y), cell_index(g, 3, -z)) : (cell_index(g, 1, x), cell_index(g, 2, -z))
    inside(I) = all(H + 1 .<= I .<= g.n .+ H)
    S = filter(sources) do s
        I = position(s.x, s.y, s.z)
        inside(I) || error("source at ($(s.x), $(s.y), $(s.z)) is outside the model")
        med.mu[I...] == 0 && med.lam[I...] == 0 && error("source at ($(s.x), $(s.y), $(s.z)) is in the air (vp = vs = 0)")
        all(2 .<= I .- off .<= nl)       # the stencil reaches i-1, k-1; ghost copies are overwritten by the halo update
    end
    Is = [position(s.x, s.y, s.z) .- off for s in S]
    Src = merge(NamedTuple{ijk}(ntuple(a -> dev(Int32[p[a] for p in Is]), N)),
                (; M=dev(FT[s.M[m] * (m in (4, 5) ? -1 : 1) for s in S, m in 1:6]),     # z up -> z down
                   s=dev(FT[s.M0 / M0 * moment_rate(stf, tbeg + (it - 0.5) * dt, s.t0, s.trise) * dt / prod(g.d) for s in S, it in 1:nt])))

    # stations owned by this rank
    function station_cell(st)
        I = position(st.x, st.y, st.z)
        st.surface && (I = (Base.front(I)..., med.kfs[Base.front(I)...] + 1 + H))
        inside(I) || error("station $(st.name) is outside the model")
        return I
    end
    SI = map(station_cell, stations)
    own = [n for n in eachindex(SI) if all(H + 1 .<= Base.front(SI[n]) .- Base.front(off) .<= Base.front(nl) .- H)]
    St = NamedTuple{ijk}(ntuple(a -> dev(Int32[SI[n][a] - off[a] for n in own]), N))
    components = mode === :psv ? (:x, :z) : mode === :sh ? (:y,) : (:x, :y, :z)
    nrec = (nt - 1) ÷ record_every + 1
    rec = KernelAbstractions.zeros(backend, FT, nrec, length(components), max(length(own), 1))

    # wavefields
    z() = KernelAbstractions.zeros(backend, FT, nl...)
    zr() = KernelAbstractions.zeros(backend, FT, NM, nl...)
    F = mode === :psv ? (; Vx=z(), Vz=z(), Sxx=z(), Szz=z(), Sxz=z(), Rxx=zr(), Rzz=zr(), Rxz=zr()) :
        mode === :sh ? (; Vy=z(), Syz=z(), Sxy=z(), Ryz=zr(), Rxy=zr()) :
        (; Vx=z(), Vy=z(), Vz=z(), Sxx=z(), Syy=z(), Szz=z(), Syz=z(), Sxz=z(), Sxy=z(),
           Rxx=zr(), Ryy=zr(), Rzz=zr(), Ryz=zr(), Rxz=zr(), Rxy=zr())
    scale = M0 * (N == 3 ? 1e-15 : 1e-12)     # wavefield units -> m/s (OpenSWPC convention)
    out = Output{N,mode}(model, size(m.rho), nl, dims, parallel, me, scale)

    if verbose && me == 0
        @printf("SeismicWavePC: %s, %s cells (dx = %s km), %d steps of %g s, Courant number %.3f\n", mode === :p3d ? "3D" : uppercase(string(mode)),
                join(g.n, "×"), join(round.(g.d, sigdigits=4), ", "), nt, dt, courant)
        @printf("  %d source(s) (M0 = %.3e N m), %d station(s), backend %s, %s, %d MPI rank(s)\n", length(sources), M0,
                length(stations), backend isa CPU ? "CPU ($(Threads.nthreads()) threads)" : string(nameof(typeof(backend))), T, prod(dims))
    end
    movie = (snapshot_every > 0 && me == 0) ? movie_paraview(name=joinpath(mkpath(snapshot_dir), "wavefield")) : nothing
    wall = time()
    movie = timeloop!(F, M, Src, St, rec, c, vc, FT(dt), kernels(mode), nl, nt, isempty(S), length(own), record_every,
                      snapshot_every, snapshot_dir, movie, out, tbeg, dt, verbose && me == 0, parallel, backend)
    movie === nothing || movie_paraview(pvd=movie, Finalize=true)
    verbose && me == 0 && @printf("  finished in %.2f s\n", time() - wall)

    v = zeros(nrec, length(components), length(stations))
    v[:, :, own] .= Array(rec)[:, :, 1:length(own)] .* scale
    wavefield = velocity(out, F)
    parallel && (MPI.Allreduce!(v, +, MPI.COMM_WORLD); IGG.finalize_global_grid(finalize_MPI=false))
    return Seismograms(collect(tbeg .+ (0:nrec-1) .* (record_every * dt)), v, stations, components, wavefield)
end

kernels(mode) = mode === :psv ? (vel_psv!, stress_psv!, inject_psv!, record_psv!) :
                mode === :sh ? (vel_sh!, stress_sh!, inject_sh!, record_sh!) : (vel_3d!, stress_3d!, inject_3d!, record_3d!)

velocities(F) = haskey(F, :Vx) ? (haskey(F, :Vy) ? (F.Vx, F.Vy, F.Vz) : (F.Vx, F.Vz)) : (F.Vy,)
stresses(F) = haskey(F, :Sxx) ? (haskey(F, :Syy) ? (F.Sxx, F.Syy, F.Szz, F.Syz, F.Sxz, F.Sxy) : (F.Sxx, F.Szz, F.Sxz)) : (F.Syz, F.Sxy)

# Function barrier: everything below is concretely typed.
function timeloop!(F, M, Src, St, rec, c, vc, dtT, (vel!, stress!, inject!, record!), nl, nt, nosource, nown, record_every,
                   snapshot_every, snapshot_dir, movie, out, tbeg, dt, verbose, parallel, backend)
    ndr = nl .- 2H
    V, Σ = velocities(F), stresses(F)
    report = max(nt ÷ 10, 1)
    for it in 1:nt
        if (it - 1) % record_every == 0 && nown > 0
            foreach_item!(backend, record!, (rec, F, St, (it - 1) ÷ record_every + 1), nown)
        end
        if snapshot_every > 0 && (it - 1) % snapshot_every == 0
            KernelAbstractions.synchronize(backend)
            data = velocity(out, F)
            data === nothing || (movie = write_paraview(data, "wavefield_$(lpad(it - 1, 6, '0'))"; directory=snapshot_dir,
                                                        pvd=movie, time=tbeg + (it - 1) * dt, verbose=false))
        end
        foreach_cell!(backend, stress!, (F, M, c, vc, dtT), ndr)
        nosource || run_once!(backend, inject!, (F, Src, it))
        parallel && (KernelAbstractions.synchronize(backend); IGG.update_halo!(Σ...))
        foreach_cell!(backend, vel!, (F, M, c, dtT), ndr)
        parallel && (KernelAbstractions.synchronize(backend); IGG.update_halo!(V...))
        if verbose && it % report == 0
            KernelAbstractions.synchronize(backend)
            vmax = maximum(A -> maximum(abs, A), V) * out.scale
            @printf("  step %*d/%d  t = %8.4f s  max|v| = %.3e m/s\n", ndigits(nt), it, nt, tbeg + it * dt, vmax)
        end
    end
    KernelAbstractions.synchronize(backend)
    return movie
end

"How to turn the local wavefields into a `CartData` of the whole model."
struct Output{N,mode}
    model::CartData
    size::NTuple{3,Int}
    nl::NTuple{N,Int}
    dims::NTuple{N,Int}
    parallel::Bool
    me::Int
    scale::Float64
end

"Cell-centred velocity field `V` [m/s, z up] as `CartData` on rank 0 (`nothing` on other ranks)."
function velocity(o::Output{N,mode}, F) where {N,mode}
    in_ = ntuple(a -> H+1:o.nl[a]-H, N)
    centred(A, a) = (A[in_...] .+ A[ntuple(b -> b == a ? in_[b] .- 1 : in_[b], N)...]) ./ 2
    loc = mode === :psv ? (centred(F.Vx, 1), -centred(F.Vz, 2)) :
          mode === :sh ? (F.Vy[in_...],) : (centred(F.Vx, 1), centred(F.Vy, 2), -centred(F.Vz, 3))
    glob = map(loc) do A
        A = Array(A)
        o.parallel || return A
        G = o.me == 0 ? similar(A, size(A) .* o.dims) : nothing
        IGG.gather!(A, G)
        G
    end
    o.me == 0 || return nothing
    f(A) = reshape(reverse(Float64.(A) .* o.scale, dims=N), o.size)
    zero_ = zeros(o.size)
    V = mode === :psv ? (f(glob[1]), zero_, f(glob[2])) : mode === :sh ? (zero_, f(glob[1]), zero_) : f.(glob)
    return CartData(o.model.x.val, o.model.y.val, o.model.z.val, (; V))
end
