# OpenSWPC source list files (`fn_stf` with `stf_format`; the body-force mode `bf_mode`): one point source per line, e.g.
# the subfaults of a kinematic finite fault.

"""
    read_sources(file; format, proj=nothing, model=nothing, mu=nothing, density_unit=:kg_m3)

The sources of an OpenSWPC source list file (`fn_stf`), one per line (`#` comments and blank lines are skipped): a vector
of [`MomentTensorSource`](@ref)s, or of [`ForceSource`](@ref)s for the body-force formats. `format` is OpenSWPC's
`stf_format`:

- `"xym0ij"`, `"xym0dc"`, `"xymwij"`, `"xymwdc"`, `"xydsdc"`: x, y [km] (OpenSWPC's x north, y east), depth [km], start
  time, rise time [s], then the scalar moment M0 [N m] or the magnitude Mw, or slip D [m] and area S [m²], and the moment
  tensor (xx, yy, zz, yz, xz, xy in OpenSWPC's frame) or strike, dip, rake [°];
- `"llm0ij"`, `"llm0dc"`, `"llmwij"`, `"llmwdc"`, `"lldsdc"`: the same at longitude, latitude (needs `proj`, the
  `ProjectionPoint` of the model);
- `"psmeca"`: longitude, latitude, depth, the moment tensor in spherical coordinates (rr, tt, ff, rt, rf, tf) and the
  exponent of the moment in dyn cm (GMT `psmeca -Sm`, Global CMT); start time 0, the rise time from the moment as OpenSWPC;
- `"xy"`, `"ll"` (body forces): x, y or longitude, latitude, depth, start time, rise time, fx, fy, fz [N] (OpenSWPC's
  frame: north, east, down).

Positions and components are converted to the frame of the model (x east, y north, z up). The slip formats (`"..dsdc"`)
need the shear modulus: `mu` [Pa] (a number or a function `mu(x, y, z)`) or the `model` (ρ vs² at the nearest node).
OpenSWPC's horizontal rotation `phi` is not applied, and the 3-parameter `asymcos` time function is not supported.
"""
function read_sources(file; format, proj=nothing, model=nothing, mu=nothing, density_unit=:kg_m3)
    fmt = String(format)
    forces = fmt in ("xy", "ll")
    fmt in ("xym0ij", "xym0dc", "llm0ij", "llm0dc", "xymwij", "xymwdc", "llmwij", "llmwdc", "xydsdc", "lldsdc", "psmeca", "xy", "ll") ||
        error("read_sources: unknown format \"$fmt\"")
    geo = startswith(fmt, "ll") || fmt == "psmeca"
    geo && proj === nothing && error("read_sources: the format \"$fmt\" needs the `proj` (ProjectionPoint) of the model")
    slip = endswith(fmt, "dsdc")
    slip && (mu === nothing) == (model === nothing) && error("read_sources: the format \"$fmt\" needs `mu` or the `model`")
    μat = !slip ? nothing : model === nothing ? (mu isa Number ? (x, y, z) -> Float64(mu) : mu) : shear_modulus_at(model, density_unit)
    # positions: OpenSWPC (x north, y east, depth) -> (x east, y north, z up)
    position(a, b, depth) = geo ? project(a, b, -depth, proj) : (b, a, -depth)
    # moment tensor (xx, yy, zz, yz, xz, xy) of OpenSWPC's frame -> ours
    tensor(m) = (m[2], m[1], m[3], -m[5], -m[4], m[6])
    sources = Any[]
    for (ln, line) in enumerate(eachline(file))
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        v = parse.(Float64, split(s)[1:min(end, 12)])
        need = fmt == "psmeca" ? 10 : forces ? 8 : endswith(fmt, "ij") ? 12 : endswith(fmt, "dsdc") ? 10 : 9
        length(v) >= need || error("read_sources: line $ln of $file has $(length(v)) values, the format \"$fmt\" needs $need")
        if fmt == "psmeca"
            x, y, z = position(v[1], v[2], v[3])
            mrr, mtt, mff, mrt, mrf, mtf = v[4:9]
            mo = (mzz=mrr, mxx=mtt, myy=mff, mxz=mrt, myz=-mrf, mxy=-mtf)          # (as OpenSWPC: x north = -θ, z down = -r)
            M0t = sqrt(mo.mxx^2 + mo.myy^2 + mo.mzz^2 + 2 * (mo.mxz^2 + mo.myz^2 + mo.mxy^2)) / sqrt(2)
            M0dyn = M0t * 10.0^v[10]
            m = (mo.mxx, mo.myy, mo.mzz, mo.myz, mo.mxz, mo.mxy) ./ M0t
            push!(sources, MomentTensorSource(x, y, z, 1e-7 * M0dyn, tensor(m); t0=0.0, trise=2 * 1.05e-8 * M0dyn^(1 / 3)))
            continue
        end
        x, y, z = position(v[1], v[2], v[3])
        t0, trise = v[4], v[5]
        if forces
            push!(sources, ForceSource(x, y, z; fx=v[7], fy=v[6], fz=-v[8], t0, trise))
        elseif endswith(fmt, "dc")
            M0 = slip ? μat(x, y, z) * v[6] * v[7] : occursin("mw", fmt) ? seismic_moment(v[6]) : v[6]
            strike, dip, rake = slip ? v[8:10] : v[7:9]
            push!(sources, DoubleCoupleSource(x, y, z; M0, strike, dip, rake, t0, trise))
        else
            M0 = occursin("mw", fmt) ? seismic_moment(v[6]) : v[6]
            push!(sources, MomentTensorSource(x, y, z, M0, tensor(v[7:12]); t0, trise))
        end
    end
    return forces ? ForceSource[s for s in sources] : MomentTensorSource[s for s in sources]
end

"""
    read_srf(file; proj, mu=nothing, model=nothing, density_unit=:kg_m3)

The point sources of a kinematic rupture in the Standard Rupture Format (SRF, versions 1 and 2; Graves, SCEC Broadband
Platform): one [`MomentTensorSource`](@ref) per point (a double couple with strike, dip, rake), at its longitude, latitude
and depth (projected with `proj`, the `ProjectionPoint` of the model), starting at the rupture time `TINIT`, with its
slip-velocity function as [`SampledSTF`](@ref) and the moment μ × area × slip. Slip in the second in-plane direction
(rake + 90°) is a second source with its own slip-velocity function; fault opening (u3) is not supported. μ: from the
S velocity and density of each point (version 2), else `mu` [Pa] (number or function `mu(x, y, z)`) or the `model`.
"""
function read_srf(file; proj, mu=nothing, model=nothing, density_unit=:kg_m3)
    lines = [strip(l) for l in eachline(file)]
    lines = [l for l in lines if !isempty(l) && !startswith(l, "#")]
    version = parse(Float64, split(lines[1])[1])
    v2 = version >= 2
    i = findfirst(l -> startswith(l, "POINTS"), lines)
    i === nothing && error("read_srf: no POINTS block in $file")
    np = parse(Int, split(lines[i])[2])
    tok = parse.(Float64, reduce(vcat, split.(lines[i+1:end])))       # (values may wrap across lines)
    μfixed = model !== nothing ? shear_modulus_at(model, density_unit) : mu === nothing ? nothing : mu isa Number ? (x, y, z) -> Float64(mu) : mu
    sources = MomentTensorSource[]
    p = 0
    next(n) = (r = tok[p+1:p+n]; p += n; r)
    for _ in 1:np
        h = next(v2 ? 10 : 8)
        lon, lat, dep, stk, dip, area, tinit, dt = h[1:8]
        rake, slip1, nt1, slip2, nt2, slip3, nt3 = next(7)
        sv = [next(Int(nt)) for nt in (nt1, nt2, nt3)]
        (slip3 != 0 && any(!iszero, sv[3])) && error("read_srf: fault opening (slip in u3) is not supported")
        x, y, z = project(lon, lat, -dep, proj)
        μ = v2 && h[9] > 0 && h[10] > 0 ? 1e3h[10] * (1e-2h[9])^2 :
            μfixed === nothing ? error("read_srf: SRF version $version has no S velocity and density: give `mu` or the `model`") : μfixed(x, y, z)
        for (slip, nt, r, v) in ((slip1, nt1, rake, sv[1]), (slip2, nt2, rake + 90, sv[2]))
            (slip > 0 && nt > 0) || continue
            stf = SampledSTF(dt .* (0:Int(nt)-1), v ./ slip)            # (slip velocity / slip: integral 1)
            push!(sources, DoubleCoupleSource(x, y, z; M0=μ * 1e-4area * 1e-2slip, strike=stk, dip, rake=r, t0=tinit, trise=nt * dt, stf))
        end
    end
    return sources
end

"""
    read_cmtsolution(file; proj)

The point sources of a CMTSOLUTION file (SPECFEM, Global CMT; several blocks for a finite source): moment tensors (Mrr, Mtt,
Mpp, Mrt, Mrp, Mtp in dyn cm) at their latitude, longitude and depth (projected with `proj`), with SPECFEM's source time
function: a Gaussian moment rate of the given half duration, centred at the time shift (`t0`).
"""
function read_cmtsolution(file; proj)
    blocks, cur = Vector{Dict{String,Float64}}(), nothing
    for line in eachline(file)
        s = strip(line)
        isempty(s) && continue
        if !occursin(':', s)                                     # the first line (PDE …) starts a block
            cur = Dict{String,Float64}()
            push!(blocks, cur)
            continue
        end
        k, v = strip.(split(s, ':'; limit=2))
        cur === nothing && error("read_cmtsolution: $file does not start with an event line")
        k in ("event name",) || (cur[lowercase(k)] = parse(Float64, split(v)[1]))
    end
    return map(blocks) do b
        x, y, z = project(b["longitude"], b["latitude"], -b["depth"], proj)
        # (r, θ, φ) = (up, south, east) -> (x east, y north, z up)
        M = (b["mpp"], b["mtt"], b["mrr"], -b["mrt"], b["mrp"], -b["mtp"]) .* 1e-7      # [N m]
        M0 = sqrt(sum(abs2, M[1:3]) + 2sum(abs2, M[4:6])) / sqrt(2)
        hdur = b["half duration"]
        α = 1.628 / max(hdur, eps())                             # (SPECFEM: hdur / 1.628 the Gaussian width)
        stf = (t, t0, tr) -> α / sqrt(π) * exp(-(α * (t - t0))^2)
        MomentTensorSource(x, y, z, M0, M ./ M0; t0=b["time shift"], trise=max(2hdur, eps()), stf)
    end
end
