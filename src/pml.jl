# Perfectly matched layer (OpenSWPC m_absorb_p): ADE CFS-PML (Zhang & Shen, 2010) at the sides and the bottom, `na` cells
# thick. The interior and the PML zones are separate cell ranges; the kernels add the PML terms only in the zones.

"""
Damping coefficients `(g1, g2, g3, g4)` at distance `x` from the model edge `0` (model width `L`), for a PML of thickness `h`
(OpenSWPC `damping_profile`; P velocity assumed 6 km/s, `fcut` the highest source frequency).
"""
function damping_profile(x, h, L, na, dt, fcut)
    R0 = 10^(-(log10(na) - 1) / log10(2.0) - 3.0)           # reflection coefficient
    d0 = -(1 / 2h) * 3 * 6.0 * log(R0)
    xx = x <= h ? h - x : x >= L - h ? x - (L - h) : 0.0
    d = d0 * (xx / h)^2
    a = π * fcut * (1 - xx / h)
    b = 1 + 6 * (xx / h)^3
    q = 1 + dt / 2 * (a + d / b)
    return ((1 + dt / 2 * a) / b / q, -1 / b / q, (1 - dt / 2 * (a + d / b)) / q, d / b / q)
end

"PML profiles at the cell centres and edges along an axis of `n` cells (padded by `H`), and the 2nd-order edge cells."
function pml_axis(n, d, na, dt, fcut, ::Type{T}) where {T}
    i = 1-H:n+H
    return pml_axis((i .- 0.5) .* d, i .* d, na * d, n * d, n, na, dt, fcut, T)
end
"""
The same from the positions (distance from the first edge) of the padded centres `xc` and edges `xe`, the PML thickness `h`
and the length `L` of the axis (a stretched axis).
"""
function pml_axis(xc, xe, h, L, n, na, dt, fcut, ::Type{T}) where {T}
    p(x) = T.(damping_profile(x, h, L, na, dt, fcut))
    i = 1-H:n+H
    return p.(xc), p.(xe), [i in (1, 2, n - 1, n) for i in i]
end

"""
Cell ranges (global, physical) of the interior and of the PML zones: sides in x (and y) over the whole depth, bottom below
the interior. As OpenSWPC, no PML at the top.
"""
function zones(n::NTuple{2,Int}, na)
    x, z = n
    inner = (na+1:x-na, 1:z-na)
    return inner, ((1:na, 1:z), (x-na+1:x, 1:z), (na+1:x-na, z-na+1:z))
end
function zones(n::NTuple{3,Int}, na)
    x, y, z = n
    inner = (na+1:x-na, na+1:y-na, 1:z-na)
    return inner, ((1:x, 1:na, 1:z), (1:x, y-na+1:y, 1:z), (1:na, na+1:y-na, 1:z), (x-na+1:x, na+1:y-na, 1:z),
                   (na+1:x-na, na+1:y-na, z-na+1:z))
end

"Offset `lo` and size of the part of the global range `r` (per axis) on this rank (array indices `lo .+ (1:ndr)`)."
function local_range(r, off, nl)
    first_ = ntuple(a -> max(first(r[a]), off[a] + 1), length(r))
    last_ = ntuple(a -> min(last(r[a]), off[a] + nl[a] - 2H), length(r))
    return first_ .- off .+ (H - 1), max.(last_ .- first_ .+ 1, 0)
end

"PML memory variables (one per derivative and axis), made by `z()`."
pml_memory(::Val{:psv}, z) = (; axSxx=z(), azSzz=z(), axSxz=z(), azSxz=z(), axVx=z(), azVz=z(), axVz=z(), azVx=z())
pml_memory(::Val{:sh}, z) = (; azSyz=z(), axSxy=z(), axVy=z(), azVy=z())
pml_memory(::Val{:p3d}, z) = (; axSxx=z(), aySxy=z(), azSxz=z(), axSxy=z(), aySyy=z(), azSyz=z(), axSxz=z(), aySyz=z(), azSzz=z(),
                                axVx=z(), ayVy=z(), azVz=z(), axVy=z(), ayVx=z(), axVz=z(), azVx=z(), ayVz=z(), azVy=z())

"""
Cell ranges `(offset, size, zone)` of this rank for the PML: the interior (zone `nothing`) and the PML zones with their
memory variables and damping profiles.
"""
function pml_regions(::Val{mode}, g::Grid{N}, na, dt, fcut, off, nl, R, dev, backend, ::Type{T}) where {mode,N,T}
    top = g.zf[H] # edge above the first cell
    prof = ntuple(a -> a < N || g.uniform ? pml_axis(g.n[a], g.d[a], na, dt, fcut, T) :
                                 pml_axis(g.zc .- top, g.zf .- top, g.zf[end-H] - g.zf[end-H-na], g.zf[end-H] - top, g.n[a], na, dt, fcut, T), Val(N))
    loc(v, a) = dev(v[R[a]])
    profiles = N == 3 ? (; gxc=loc(prof[1][1], 1), gxe=loc(prof[1][2], 1), gyc=loc(prof[2][1], 2), gye=loc(prof[2][2], 2),
                           gzc=loc(prof[3][1], 3), gze=loc(prof[3][2], 3), ox=loc(prof[1][3], 1), oy=loc(prof[2][3], 2)) :
                (; gxc=loc(prof[1][1], 1), gxe=loc(prof[1][2], 1), gzc=loc(prof[2][1], 2), gze=loc(prof[2][2], 2), ox=loc(prof[1][3], 1))
    inner, boxes = zones(g.n, na)
    pml = map(boxes) do b
        lo, ndr = local_range(b, off, nl)
        memory = pml_memory(Val(mode), () -> KernelAbstractions.zeros(backend, T, ndr...))
        (lo, ndr, merge((; lo=CartesianIndex(lo)), profiles, (; mem=memory)))
    end
    return ((local_range(inner, off, nl)..., nothing), pml...)
end

"All PML memory variables of the `regions`, as a tuple of arrays (for checkpoints)."
pml_state(regions) = flatten_tuple(map(r -> r[3] === nothing ? () : values(r[3].mem), regions))
flatten_tuple(t::Tuple) = isempty(t) ? () : (first(t)..., flatten_tuple(Base.tail(t))...)

"""
The interior region `(lo, ndr, nothing)` split into 5 blocks of rows: rows without 2nd-order differences (`Plain`) and the rows
of the zones next to free surfaces and ocean bottoms of any column (`nothing`); empty blocks have size 0. Without `split`
(GPUs: every block is a kernel launch, and deciding per cell costs little there) all rows are in one block.
"""
function plain_blocks((lo, ndr, _)::Tuple{NTuple{N,Int},NTuple{N,Int},Any}, med, off; split=true) where {N}
    cols = CartesianIndices(ntuple(a -> lo[a]+1+off[a]:lo[a]+ndr[a]+off[a], Val(N - 1)))   # global padded columns
    function zone(top, bot)                                       # the rows of a zone over all columns
        t, b = typemax(Int), typemin(Int)
        for C in cols
            top[C] <= bot[C] && (t = min(t, top[C]); b = max(b, bot[C]))
        end
        return t, b
    end
    A, B = split ? (zone(med.fs_top, med.fs_bot), zone(med.ob_top, med.ob_bot)) : ((lo[N] + 1, lo[N] + ndr[N]), (typemax(Int), typemin(Int)))
    A[1] > A[2] && ((A, B) = (B, A))                              # (an empty zone last)
    B[1] <= B[2] && A[1] <= A[2] && B[1] < A[1] && ((A, B) = (B, A))
    if A[1] <= A[2] && B[1] <= B[2] && B[1] <= A[2] + 1           # overlapping: merged
        A, B = (A[1], max(A[2], B[2])), (typemax(Int), typemin(Int))
    end
    k1, k2 = lo[N] + 1, lo[N] + ndr[N]
    block(a, b, P) = (a = max(a, k1); b = min(b, k2); (ntuple(d -> d < N ? lo[d] : a - 1, Val(N)), ntuple(d -> d < N ? ndr[d] : max(b - a + 1, 0), Val(N)), P))
    ea, eb = A[1] <= A[2], B[1] <= B[2]
    a1, a2 = ea ? A : (k2 + 1, k2)
    b1, b2 = eb ? B : (k2 + 1, k2)
    return (block(k1, a1 - 1, Plain()), block(a1, a2, nothing), block(a2 + 1, b1 - 1, Plain()), block(b1, b2, nothing),
            block(max(a2, b2) + 1, k2, Plain()))
end
