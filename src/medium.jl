# Material setup on the global grid (host), ported from OpenSWPC `m_medium.f90` / `m_absorb_c.f90`.
#
# Internal layout: arrays are indexed (i, [j,] k) with k = 1 at the TOP of the model (depth increases with k),
# padded by H = 2 ghost cells on every side. Physical cell i lives at array index i + H.

const H = 2

struct Grid{N}
    n::NTuple{N,Int}          # physical cells (x, [y,] z)
    d::NTuple{N,Float64}      # spacing [km]; in depth the mean spacing if the grid is stretched
    beg::NTuple{N,Float64}    # cell-edge start; for z this is the DEPTH of the top edge
    zc::Vector{Float64}       # depths of the cell centres, padded (index k)
    zf::Vector{Float64}       # depths of the faces below them (between centres k and k + 1)
    uniform::Bool             # constant spacing in depth
end
Grid{N}(n, d, beg) where {N} = (z = beg[N] .+ ((1:n[N]+2H) .- H .- 0.5) .* d[N]; Grid{N}(n, d, beg, z, z .+ d[N] / 2, true))

"""
Concretely typed copy of the parts of a `CartData` the solver needs (GMG stores them untyped):
coordinate axes and the fields `rho`, `vp`, `vs`, `Qp`, `Qs` as `Array{Float64,3}` (Q = Inf if missing).
"""
function model_arrays(d::CartData)
    axis(A, I...) = Vector{Float64}(A[I...])::Vector{Float64}
    x, y, z = axis(d.x.val, :, 1, 1), axis(d.y.val, 1, :, 1), axis(d.z.val, 1, 1, :)
    function field(name)
        haskey(d.fields, name) && return convert(Array{Float64,3}, d.fields[name])::Array{Float64,3}
        name in (:Qp, :Qs) || error("the CartData model needs fields `rho`, `vp` and `vs` (and optionally `Qp`, `Qs`)")
        return fill(Inf, length(x), length(y), length(z))
    end
    return (; x, y, z, rho=field(:rho), vp=field(:vp), vs=field(:vs), Qp=field(:Qp), Qs=field(:Qs))
end

"""
Grid of the model: every `CartData` node becomes a cell centre. The spacing is constant in x (and y); in depth it may vary
smoothly (a stretched grid, e.g. fine near the surface): the faces between the cells are then interpolated (4th order) from
the centres.
"""
function Grid(m::NamedTuple, ::Val{N}) where {N}
    axes = N == 3 ? (m.x, m.y, -reverse(m.z)) : (m.x, -reverse(m.z))
    for a in axes[1:N-1]
        length(a) > 1 && all(≈(a[2] - a[1]), diff(a)) || error("CartData must be a regular grid in x (and y), with at least 2 points per axis")
    end
    z = axes[N]
    dz = diff(z)
    length(z) > 1 && all(>(0), dz) || error("CartData needs at least 2 increasing points in z")
    uniform = all(≈(dz[1]), dz)
    d = (map(a -> a[2] - a[1], axes[1:N-1])..., uniform ? dz[1] : (z[end] - z[1]) / (length(z) - 1))   # stretched: the mean
    uniform && return Grid{N}(map(length, axes), d, map((a, h) -> a[1] - h / 2, axes, d))
    r = maximum(i -> max(dz[i+1] / dz[i], dz[i] / dz[i+1]), 1:length(dz)-1)
    r <= 1.25 || error("the spacing in z must change smoothly (at most 25 % from cell to cell, here $(round(100(r - 1)))) %)")
    # padded centres (ghost cells continue the spacing at the ends) and faces (4th-order interpolation; 2nd order at the ends)
    n = length(z)
    zc = [z[1] .- (H:-1:1) .* dz[1]; z; z[end] .+ (1:H) .* dz[end]]
    zf = [k in 2:n+2H-2 ? 9 / 16 * (zc[k] + zc[k+1]) - 1 / 16 * (zc[k-1] + zc[k+2]) : (zc[k] + zc[min(k + 1, end)]) / 2 for k in 1:n+2H]
    zf[end] = zc[end] + dz[end] / 2
    top = zc[H+1] - (zf[H+1] - zc[H+1])
    return Grid{N}(map(length, axes), d, (map((a, h) -> a[1] - h / 2, axes[1:N-1], d[1:N-1])..., top), zc, zf, false)
end

"""
Courant number of the time step `dt`: on a stretched grid per layer, with its largest P velocity and its spacing in depth
(stable below 6/7).
"""
function courant_number(med, g::Grid{N}, dt) where {N}
    g.uniform && return med.vmax * dt * sqrt(sum(1 ./ g.d .^ 2))
    vp = sqrt.((med.lam .+ 2 .* med.mu) ./ med.rho)
    hx = sum(1 ./ Base.front(g.d) .^ 2)
    c = 0.0
    for k in H+1:H+g.n[N]
        h = min(g.zc[k+1] - g.zc[k], g.zc[k] - g.zc[k-1])
        c = max(c, maximum(selectdim(vp, N, k)) * dt * sqrt(hx + 1 / h^2))
    end
    return c
end

"Padded array index of the cell that contains coordinate `c` along axis `a` (z given as depth)."
cell_index(g::Grid{N}, a, c) where {N} =
    a < N || g.uniform ? ceil(Int, (c - g.beg[a]) / g.d[a]) + H : clamp(searchsortedfirst(g.zf, c), 1, length(g.zf))

"Thickness of cell `k` (padded index) in depth."
thickness(g::Grid, k) = g.uniform ? g.d[end] : g.zf[k] - g.zf[k-1]

"""
Coefficients of the depth derivatives (4th order: 9/8 and 1/24, and 2nd order) divided by the local spacing, per layer `k`:
`centre` for the derivatives of fields at the cell centres in depth (located at the faces k + 1/2) and `face` for those of
fields at the faces (located at the centres k). Constant on a uniform grid.
"""
function depth_coefficients(g::Grid{N}, ::Type{T}) where {N,T}
    m = g.n[N] + 2H
    if g.uniform                                    # the same constants as along x: identical results
        c = (a=fill(T(9 / 8 / g.d[N]), m), b=fill(T(1 / 24 / g.d[N]), m), c=fill(T(1 / g.d[N]), m))
        return c, c
    end
    zc, zf = g.zc, g.zf
    j4(x, k) = 2 <= k <= m - 2 ? 9 / 8 * (x[k+1] - x[k]) - 1 / 24 * (x[k+2] - x[k-1]) : x[min(k + 1, m)] - x[k]
    hf4 = [j4(zc, k) for k in 1:m]                                  # at the faces k + 1/2
    hf2 = [zc[min(k + 1, m)] - zc[k] for k in 1:m]
    hc2 = [k == 1 ? 2(zf[1] - zc[1]) : zf[k] - zf[k-1] for k in 1:m]  # at the centres k
    hc4 = [k == 1 ? hc2[1] : j4(zf, k - 1) for k in 1:m]
    hf2[m] = hf2[m-1]
    coef(h4, h2) = (a=T.(9 / 8 ./ h4), b=T.(1 / 24 ./ h4), c=T.(1 ./ h2))
    return coef(hf4, hf2), coef(hc4, hc2)
end

"Depth coefficients of layer `k` of the point `I` (a tuple `(9/8, 1/24, 2nd order) / spacing`); constant tuples along x, y."
@inline coefat(c::NTuple{3,Number}, I) = c
Base.@propagate_inbounds coefat(c::NamedTuple, I) = (k = last(Tuple(I)); (c.a[k], c.b[k], c.c[k]))

"Field `A` as an internal array: z flipped so that k = 1 is the top, squeezed to (x, z) in 2D."
function internal(A::Array{Float64,3}, ::Val{N}) where {N}
    f = reverse(A, dims=3)
    return N == 3 ? f : f[:, 1, :]
end

"""
Pad `A` (physical cells `n`) by `H` ghost cells on every side by replication; the absorber zone (`na` cells at the sides
and the bottom) is homogenized, as in OpenSWPC.
"""
function pad_absorber(A::Array{Float64,N}, n::NTuple{N,Int}, na) where {N}
    P = Array{Float64,N}(undef, n .+ 2H)
    for I in CartesianIndices(P)
        J = ntuple(a -> a == N ? clamp(I[a] - H, 1, n[a] - na) : clamp(I[a] - H, na + 1, n[a] - na), Val(N))
        P[I] = A[J...]
    end
    return P
end

"""
Build padded material arrays `(rho, lam, mu, taup, taus)`, the 2nd-order zones near free surfaces
`(fs_top, fs_bot, ob_top, ob_bot)` and the maximum P velocity.
"""
function setup_medium(m::NamedTuple, g::Grid{N}, na, fq_min, fq_max, fq_ref) where {N}
    rho, vp, vs, qp, qs = internal(m.rho, Val(N)), internal(m.vp, Val(N)), internal(m.vs, Val(N)), internal(m.Qp, Val(N)), internal(m.Qs, Val(N))
    rho = max.(rho, 0.001)   # air needs a small non-zero density (as in OpenSWPC)

    n = g.n
    rho, vp, vs, qp, qs = pad_absorber(rho, n, na), pad_absorber(vp, n, na), pad_absorber(vs, n, na), pad_absorber(qp, n, na), pad_absorber(qs, n, na)

    ts = relaxation_times(fq_min, fq_max)
    zeta = constq_zeta(fq_min, fq_max, ts)
    taup, taus = NM * zeta ./ qp, NM * zeta ./ qs
    mu = @. rho * vs^2 / visco_chi((ts,), taus, fq_ref)^2       # unrelaxed moduli
    lam = @. rho * vp^2 / visco_chi((ts,), taup, fq_ref)^2 - 2mu
    vmax = maximum(sqrt.((lam .+ 2mu) ./ rho))

    return (; rho, lam, mu, taup, taus, ts, vmax, surface_zones(lam, mu, n)...)
end

"""
Free-surface (`fs`) and ocean-bottom (`ob`) zones, per column, where 2nd-order instead of
4th-order differences are used (OpenSWPC `surface_detection`).
"""
function surface_zones(lam, mu, n::NTuple{N}) where {N}
    ϵ = eps(Float32)
    nz = n[N]
    hn = Base.front(n)
    # kfs/kob: last k (physical) with vacuum above material; 0 if none. Detected on columns 0:n+2, as in OpenSWPC.
    kfs = zeros(Int, hn .+ (2H + 1))       # physical columns -1 : n+3  ->  index i + 2
    kob = zeros(Int, hn .+ (2H + 1))
    for C in CartesianIndices(kfs)
        ip = Tuple(C) .- 2                # physical column
        all(0 .<= ip .<= hn .+ 2) || continue
        col = CartesianIndex(ip .+ H)
        for k in 1:nz-1
            abs(mu[col, k+H]) < ϵ && abs(mu[col, k+1+H]) > ϵ && (kob[C] = k)
            abs(lam[col, k+H]) < ϵ && abs(lam[col, k+1+H]) > ϵ && (kfs[C] = k)
        end
    end
    # zone = padded k-range [top, bot]; ghost columns get an empty zone (top > bot)
    function zone(kk)
        top, bot = fill(Int32(1), hn .+ 2H), fill(Int32(0), hn .+ 2H)
        for C in CartesianIndices(hn)
            w = kk[CartesianIndices(map(c -> c:c+5, Tuple(C)))]   # physical columns i-2 : i+3
            P = C + CartesianIndex(ntuple(_ -> H, N - 1))
            top[P] = max(minimum(w) - 2, 1) + H
            bot[P] = min(maximum(w) + 2, nz) + H
        end
        return top, bot
    end
    (fs_top, fs_bot), (ob_top, ob_bot) = zone(kfs), zone(kob)
    return (; fs_top, fs_bot, ob_top, ob_bot, kfs, kob)   # kfs, kob are indexed by padded column
end

"""
Cerjan sponge factors `(centre, boundary)` along one axis of `n` cells (padded). No damping at the top (`top=false`).
"""
function cerjan(n, d, na; top=true)
    α, L = 0.09, na * d
    gc, gb = ones(n + 2H), ones(n + 2H)
    for i in 1:n
        if i <= na && top
            gc[i+H] = exp(-α * (1 - (i - 0.5) * d / L)^2)
            gb[i+H] = exp(-α * (1 - i * d / L)^2)
        elseif i >= n - na + 1
            gc[i+H] = exp(-α * (1 - (n - i + 1) * d / L)^2)
            gb[i+H] = exp(-α * (1 - (n - i + 0.5) * d / L)^2)
        end
    end
    return gc, gb
end
