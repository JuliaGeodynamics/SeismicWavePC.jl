# Marmousi2 (Martin, Wiley & Marfurt, 2006): elastic model, 17 km × 3.5 km, vp, vs and density on a 1.25 m grid, with a
# 451 m deep water layer. The SEG-Y files are downloaded from the University of Houston (AGL) into examples/data.
using Downloads, CodecZlib, GeophysicalModelGenerator

const MARMOUSI2_URL = "https://www.agl.uh.edu/downloads"
const DATA = joinpath(@__DIR__, "data")

"IBM 32-bit float (bits in native order) to Float32."
function ibm2float(u::UInt32)
    frac = (u & 0x00ffffff) / 2.0^24
    return Float32(((u >> 31) == 0 ? 1 : -1) * frac * 16.0^(Int((u >> 24) & 0x7f) - 64))
end

"Read a SEG-Y file (IBM or IEEE floats, e.g. Marmousi2) from `io` as a matrix (samples × traces)."
function read_segy(io::IO)
    seek(io, 3220); ns = Int(ntoh(read(io, UInt16)))
    seek(io, 3224); ibm = ntoh(read(io, UInt16)) == 1
    seekend(io); ntr = (position(io) - 3600) ÷ (240 + 4ns)
    A, buf = Matrix{Float32}(undef, ns, ntr), Vector{UInt32}(undef, ns)
    seek(io, 3600)
    for t in 1:ntr
        skip(io, 240); read!(io, buf)
        A[:, t] .= ibm ? ibm2float.(ntoh.(buf)) : reinterpret.(Float32, ntoh.(buf))
    end
    return A
end

"Marmousi2 field `name` (vp, vs, density) on the 1.25 m grid (depth × x), downloaded once."
function marmousi2_field(name)
    file = joinpath(mkpath(DATA), "$(name)_marmousi-ii.segy.gz")
    if !isfile(file)
        println("downloading $name of Marmousi2 …")
        Downloads.download("$MARMOUSI2_URL/$(name)_marmousi-ii.segy.gz", file; timeout=300)
    end
    return read_segy(IOBuffer(transcode(GzipDecompressor, read(file))))
end

"""
    model = marmousi2(dx=10.0; air=2)

Marmousi2 as a GMG `CartData` (2D, coordinates in km, z up, sea surface at z = 0) with `rho` [kg/m³], `vp`, `vs` [km/s] on a grid
of spacing `dx` [m] (a multiple of 1.25 m; the model is decimated), with `air` cells of air above the sea. Cached in examples/data.
"""
function marmousi2(dx=10.0; air=2)
    f = round(Int, dx / 1.25)
    f * 1.25 ≈ dx || error("dx must be a multiple of 1.25 m")
    cache = joinpath(mkpath(DATA), "marmousi2_$(dx)m.bin")
    fields = if isfile(cache)
        open(io -> (nz = read(io, Int); nx = read(io, Int); [read!(io, Matrix{Float32}(undef, nz, nx)) for _ in 1:3]), cache)
    else
        F = [marmousi2_field(n)[1:f:end, 1:f:end] for n in ("vp", "vs", "density")]
        open(io -> (write(io, size(F[1], 1), size(F[1], 2)); foreach(A -> write(io, A), F)), cache, "w")
        F
    end
    top(A, v) = vcat(fill(Float32(v), air, size(A, 2)), A)          # air above the sea
    vp, vs, rho = top(fields[1], 0), top(fields[2], 0), top(fields[3], 0.001)
    nz, nx = size(vp)
    d = dx / 1000
    x = range(0, step=d, length=nx)
    z = range(-(nz - air - 0.5) * d, step=d, length=nz)              # cell centres, z up; the sea surface is at z = 0
    X, Y, Z = xyz_grid(x, 0.0, z)
    tofield(A) = reshape(Float64.(permutedims(reverse(A, dims=1))), nx, 1, nz)
    return CartData(X, Y, Z, (; rho=1000 .* tofield(rho), vp=tofield(vp), vs=tofield(vs)))
end
