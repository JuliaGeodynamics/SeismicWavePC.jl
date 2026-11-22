# Driver: domain decomposition (ImplicitGlobalGrid), sources, stations, time loop and output.

"""
    Station(name, x, y, z; surface=false)

Receiver at model coordinates [km] (z up, as in the `CartData`). With `surface=true` the station is placed in the
first solid cell below the free surface of its column, ignoring `z`. In 2D models `y` is ignored.
"""
struct Station
    name::String
    x::Float64
    y::Float64
    z::Float64
    surface::Bool
end
Station(name, x, y, z; surface=false) = Station(name, x, y, z, surface)

function igg_device(backend)
    name = string(nameof(typeof(backend)))
    name == "CUDABackend" && return "CUDA"
    name == "ROCBackend" && return "AMDGPU"
    backend isa CPU && return "none"
    error("MPI runs are supported on CPU, CUDA and AMDGPU backends, not $name")
end

"""
    result = simulate(model::CartData; dt, nt, sources, kwargs...)

Run a visco-elastic wave propagation simulation (OpenSWPC algorithm) on the velocity model `model`.

`model` is a regular `CartData` (coordinates in km, z up) with fields `rho` [g/cm³], `vp`, `vs` [km/s] and optionally
`Qp`, `Qs` (default: no attenuation). Cells with `vp = vs = 0` are air: the free surface (and topography) follows from that.
Cells with `vs = 0` and `vp > 0` are water. A `CartData` of size `(nx, 1, nz)` gives a 2D simulation in the x-z plane.

Keyword arguments
- `dt`, `nt`: time step [s] and number of steps; `tbeg = 0.0` is the start time.
- `sources`: `MomentTensorSource` or vector of them (see `DoubleCoupleSource`); `stf = :kupper` source time function.
- `stations = Station[]`; seismograms are recorded every `record_every = 1` steps.
- `wave = :psv`: `:psv` or `:sh`, for 2D models.
- `na = 20`: thickness of the Cerjan absorbing layer (sides and bottom) in cells.
- `fq_min = 0.05`, `fq_max = 5.0`, `fq_ref = 1.0`: frequency band [Hz] of constant Q, and reference frequency of the velocities.
- `snapshot_every = 0`: if > 0, write the velocity field every so many steps to ParaView files in `snapshot_dir = "snapshots"`.
- `backend = CPU()`: KernelAbstractions backend, e.g. `CUDABackend()`, `ROCBackend()`, `MetalBackend()`.
- `T = Float32`: floating point type of the wavefield.
- `nproc = 1`: MPI processes in x (2D), or `(npx, npy)` (3D); start julia with `mpiexec -n npx*npy`.

Returns a NamedTuple `(; t, v, stations, components)` with velocity seismograms `v[it, component, station]` in m/s
(components `:x, :y, :z`, z positive up). With MPI every rank receives the seismograms.
"""
function simulate(model::CartData; dt, nt, sources, stations=Station[], wave=:psv, stf=:kupper, tbeg=0.0,
                  na=20, fq_min=0.05, fq_max=5.0, fq_ref=1.0, record_every=1,
                  snapshot_every=0, snapshot_dir="snapshots", backend=CPU(), T=Float32, nproc=1)
    mode = size(model)[2] == 1 ? wave : :p3d
    mode in (:psv, :sh, :p3d) || error("wave must be :psv or :sh, got :$wave")
    sources = sources isa MomentTensorSource ? [sources] : sources
    isempty(sources) && error("at least one source is required")
    g = Grid(model, mode)
    N = length(g.n)
    med = setup_medium(model, g, mode, na, fq_min, fq_max, fq_ref)
    courant = med.vmax * dt * sqrt(sum(1 ./ g.d .^ 2))
    courant < 6 / 7 || error("unstable time step: courant number $courant ≥ 6/7, use dt < $(dt * 6 / 7 / courant)")

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
        coords = Tuple(c3[1:N])
    end
    off = coords .* (nl .- 2H)
    R = ntuple(a -> off[a]+1:off[a]+nl[a], N)
    dev(A) = (B = KernelAbstractions.allocate(backend, eltype(A), size(A)); copyto!(B, A); B)

    # medium on this rank
    loc(A) = dev(T.(A[R...]))
    col(A) = dev(A[R[1:N-1]...])
    gs = ntuple(a -> dev.(T.(v[R[a]]) for v in cerjan(g.n[a], g.d[a], na; top=a < N)), N)
    gnames = N == 3 ? (:gx_c, :gx_b, :gy_c, :gy_b, :gz_c, :gz_b) : (:gx_c, :gx_b, :gz_c, :gz_b)
    M = merge((; rho=loc(med.rho), lam=loc(med.lam), mu=loc(med.mu), taup=loc(med.taup), taus=loc(med.taus),
                 fs_top=col(med.fs_top), fs_bot=col(med.fs_bot), ob_top=col(med.ob_top), ob_bot=col(med.ob_bot)),
              NamedTuple{gnames}(Tuple(Iterators.flatten(gs))))
    c = ntuple(a -> (T(9 / 8 / g.d[a]), T(1 / 24 / g.d[a]), T(1 / g.d[a])), N)
    vc = memory_coefficients(med.ts, dt, T)

    # sources: stress-glut at the cell containing the source (normalized by the total moment, as OpenSWPC)
    M0 = sum(s.M0 for s in sources)
    position(x, y, z) = N == 3 ? (cell_index(g, 1, x), cell_index(g, 2, y), cell_index(g, 3, -z)) : (cell_index(g, 1, x), cell_index(g, 2, -z))
    inside(I) = all(H + 1 .<= I .<= g.n .+ H)
    S = filter(sources) do s
        I = position(s.x, s.y, s.z)
        inside(I) || error("source at ($(s.x), $(s.y), $(s.z)) is outside the model")
        all(2 .<= I .- off .<= nl)       # the stencil reaches i-1, k-1; ghost copies are overwritten by the halo update
    end
    if !isempty(S)
        I = [position(s.x, s.y, s.z) .- off for s in S]
        Mint = [s.M[m] * (m in (4, 5) ? -1 : 1) for s in S, m in 1:6]   # z up -> z down
        stab = [s.M0 / M0 * moment_rate(stf, tbeg + (it - 0.5) * dt, s.t0, s.trise) * dt / prod(g.d) for s in S, it in 1:nt]
        idx = NamedTuple{N == 3 ? (:i, :j, :k) : (:i, :k)}(ntuple(a -> dev(Int32[p[a] for p in I]), N))
        Src = merge(idx, (; M=dev(T.(Mint)), s=dev(T.(stab))))
    end

    # stations owned by this rank
    function station_cell(st)
        I = position(st.x, st.y, st.z)
        st.surface && (I = (I[1:N-1]..., med.kfs[I[1:N-1]...] + 1 + H))
        inside(I) || error("station $(st.name) is outside the model")
        return I
    end
    SI = map(station_cell, stations)
    own = [n for n in eachindex(SI) if all(H + 1 .<= SI[n][1:N-1] .- off[1:N-1] .<= nl[1:N-1] .- H)]
    St = NamedTuple{N == 3 ? (:i, :j, :k) : (:i, :k)}(ntuple(a -> dev(Int32[SI[n][a] - off[a] for n in own]), N))
    components = mode === :psv ? (:x, :z) : mode === :sh ? (:y,) : (:x, :y, :z)
    nrec = (nt - 1) ÷ record_every + 1
    rec = KernelAbstractions.zeros(backend, T, nrec, length(components), max(length(own), 1))

    # wavefields
    z() = KernelAbstractions.zeros(backend, T, nl...)
    zr() = KernelAbstractions.zeros(backend, T, nl..., NM)
    if mode === :psv
        F = (; Vx=z(), Vz=z(), Sxx=z(), Szz=z(), Sxz=z(), Rxx=zr(), Rzz=zr(), Rxz=zr())
        V, Σ, kern = (F.Vx, F.Vz), (F.Sxx, F.Szz, F.Sxz), (vel_psv!, stress_psv!, inject_psv!, record_psv!)
    elseif mode === :sh
        F = (; Vy=z(), Syz=z(), Sxy=z(), Ryz=zr(), Rxy=zr())
        V, Σ, kern = (F.Vy,), (F.Syz, F.Sxy), (vel_sh!, stress_sh!, inject_sh!, record_sh!)
    else
        F = (; Vx=z(), Vy=z(), Vz=z(), Sxx=z(), Syy=z(), Szz=z(), Syz=z(), Sxz=z(), Sxy=z(),
               Rxx=zr(), Ryy=zr(), Rzz=zr(), Ryz=zr(), Rxz=zr(), Rxy=zr())
        V, Σ, kern = (F.Vx, F.Vy, F.Vz), (F.Sxx, F.Syy, F.Szz, F.Syz, F.Sxz, F.Sxy), (vel_3d!, stress_3d!, inject_3d!, record_3d!)
    end
    vel!, stress!, inject!, record! = map(k -> k(backend), kern)
    scale = M0 * (N == 3 ? 1e-15 : 1e-12)     # wavefield units -> m/s (OpenSWPC convention)

    movie = (snapshot_every > 0 && me == 0) ? movie_paraview(name=joinpath(mkpath(snapshot_dir), "wavefield")) : nothing
    ndr = nl .- 2H
    dtT = T(dt)
    for it in 1:nt
        if (it - 1) % record_every == 0 && !isempty(own)
            record!(rec, F, St, (it - 1) ÷ record_every + 1; ndrange=length(own))
        end
        if snapshot_every > 0 && (it - 1) % snapshot_every == 0
            KernelAbstractions.synchronize(backend)
            movie = snapshot(movie, model, F, mode, nl, dims, parallel, me, scale, tbeg + (it - 1) * dt, snapshot_dir, it)
        end
        stress!(F, M, c, vc, dtT; ndrange=ndr)
        isempty(S) || inject!(F, Src, it; ndrange=1)
        parallel && (KernelAbstractions.synchronize(backend); IGG.update_halo!(Σ...))
        vel!(F, M, c, dtT; ndrange=ndr)
        parallel && (KernelAbstractions.synchronize(backend); IGG.update_halo!(V...))
    end
    KernelAbstractions.synchronize(backend)
    movie === nothing || movie_paraview(pvd=movie, Finalize=true)

    v = zeros(nrec, length(components), length(stations))
    v[:, :, own] .= Array(rec)[:, :, 1:length(own)] .* scale
    parallel && (MPI.Allreduce!(v, +, MPI.COMM_WORLD); IGG.finalize_global_grid(finalize_MPI=false))
    t = tbeg .+ (0:nrec-1) .* (record_every * dt)
    return (; t, v, stations=[s.name for s in stations], components)
end

"Write the cell-centred velocity field (m/s, z up) as a ParaView file and add it to the `movie` collection."
function snapshot(movie, model, F, mode, nl, dims, parallel, me, scale, t, dir, it)
    N = length(nl)
    in_ = ntuple(a -> H+1:nl[a]-H, N)
    centred(A, a) = (A[in_...] .+ A[ntuple(b -> b == a ? in_[b] .- 1 : in_[b], N)...]) ./ 2
    loc = mode === :psv ? (centred(F.Vx, 1), -centred(F.Vz, 2)) :
          mode === :sh ? (F.Vy[in_...],) : (centred(F.Vx, 1), centred(F.Vy, 2), -centred(F.Vz, 3))
    glob = map(loc) do A
        A = Array(A)
        parallel || return A
        G = me == 0 ? similar(A, size(A) .* dims) : nothing
        IGG.gather!(A, G)
        G
    end
    me == 0 || return movie
    f(A) = reshape(reverse(Float64.(A) .* scale, dims=N), size(model))
    zero_ = zeros(size(model))
    V = mode === :psv ? (f(glob[1]), zero_, f(glob[2])) : mode === :sh ? (zero_, f(glob[1]), zero_) : f.(glob)
    data = CartData(model.x.val, model.y.val, model.z.val, (; V))
    return write_paraview(data, "wavefield_$(lpad(it - 1, 6, '0'))"; directory=dir, pvd=movie, time=t, verbose=false)
end
