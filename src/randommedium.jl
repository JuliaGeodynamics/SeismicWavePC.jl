# Random media (OpenSWPC gen_rmed2d / gen_rmed3d, m_rmedia, and the *_rmed velocity models).

"""
    xi = random_medium(x, y, z; spectrum=:vonkarman, a, epsilon, kappa=0.5, seed=nothing)

Random fluctuations ξ on the grid `x, y, z` [km] (as for `xyz_grid`; `y = 0.0` for 2D), made as by OpenSWPC's `gen_rmed3d`
and `gen_rmed2d`: white noise is given the power spectrum of an autocorrelation function and transformed back; the result has
mean 0 and RMS `epsilon`.
- `spectrum`: `:gaussian`, `:exponential` or `:vonkarman` (with the Hurst exponent `kappa`; `kappa = 0.5` is exponential)
- `a` [km]: correlation length, a number or one per axis (`(ax, [ay,] az)`; anisotropic media)
- `seed`: an integer for reproducible media
The medium is generated on a grid of the next powers of two (so that it does not repeat within the model) and cut to size.
Returns an `Array{Float64,3}` of the size of the grid; apply it with [`perturb`](@ref).
"""
function random_medium(x, y, z; spectrum=:vonkarman, a, epsilon, kappa=0.5, seed=nothing)
    spectrum in (:gaussian, :exponential, :vonkarman) || error("spectrum must be :gaussian, :exponential or :vonkarman")
    axes_ = filter(v -> length(v) > 1, [collect(Float64, x), collect(Float64, y), collect(Float64, z)])
    D = length(axes_)
    n = length.(axes_)
    d = [v[2] - v[1] for v in axes_]
    as = a isa Number ? fill(Float64(a), D) : collect(Float64, a)
    length(as) == D || error("give one correlation length or one per axis ($D)")
    N = nextpow.(2, n)
    rng = seed === nothing ? Random.default_rng() : Random.Xoshiro(seed)
    B = fft(rand(rng, N...))
    B ./= max.(abs.(B), floatmin())                                    # random phases, unit amplitude
    k = [2π .* fftfreq(N[i], 1 / d[i]) for i in 1:D]                  # angular wavenumbers [1/km]
    for I in CartesianIndices(B)
        m2 = sum((k[i][I[i]] * as[i])^2 for i in 1:D)
        P = spectrum === :gaussian ? exp(-m2 / 4) : (1 + m2)^(-((spectrum === :exponential ? 0.5 : kappa) + D / 2))
        B[I] *= sqrt(P)
    end
    B[1] = 0
    xi = real(ifft(B))[ntuple(i -> 1:n[i], D)...]
    xi .-= sum(xi) / length(xi)
    xi .*= epsilon / sqrt(sum(abs2, xi) / length(xi))
    return reshape(xi, length(x), length(y), length(z))
end

"""
    perturbed = perturb(model::CartData, xi; rho_factor=0.8)

Random velocity model, as OpenSWPC's random-medium models (`vmodel_type = "uni_rmed"`, `"lhm_rmed"`, `"grd_rmed"`): in the
solid part of `model` (`vs > 0`) `vp` and `vs` are multiplied by `1 + ξ` and `rho` by `1 + rho_factor ξ`. `xi` is an array of
the size of the model (from [`random_medium`](@ref)) or, as in OpenSWPC, a smaller one that is repeated periodically
(starting at the top of the model). Other fields are kept.
"""
function perturb(model::CartData, xi::AbstractArray{<:Real,3}; rho_factor=0.8)
    f = model.fields
    all(haskey(f, n) for n in (:rho, :vp, :vs)) || error("the model needs the fields rho, vp and vs")
    nx, ny, nz = size(model)
    mx, my, mz = size(xi)
    ξ(i, j, k) = xi[mod1(i, mx), mod1(j, my), mz - mod1(nz - k + 1, mz) + 1]
    solid = f.vs .> 0
    s = [solid[i, j, k] ? ξ(i, j, k) : 0.0 for i in 1:nx, j in 1:ny, k in 1:nz]
    any(s .<= -1) && error("fluctuations of -100% or more: use a smaller epsilon")
    return CartData(model.x.val, model.y.val, model.z.val,
                    merge(f, (; rho=f.rho .* (1 .+ rho_factor .* s), vp=f.vp .* (1 .+ s), vs=f.vs .* (1 .+ s))))
end

"""
    xi = read_random_medium(file)

Random medium written by OpenSWPC's `gen_rmed3d` or `gen_rmed2d` (netCDF), as an array for [`perturb`](@ref) (z up).
Requires `using NCDatasets`.
"""
function read_random_medium end
read_random_medium(args::Any...) = error(NETCDF_HINT)
