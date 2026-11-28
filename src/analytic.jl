# Reference solutions and misfit criteria for validating simulations: Garvin's problem (a line explosion under the free
# surface of a half-space, exact by the Cagniard–de Hoop method) and the time-frequency envelope and phase misfits of
# Kristeková et al. (2006).

"""
    vx, vz = garvin(x, depth, t; vp, vs, rho, M0=1.0, stf=:ricker, t0=0.0, trise=1.0)

Exact velocities [m/s] at the free surface of a homogeneous half-space (`vp`, `vs` [km/s], `rho` [kg/m³]) for Garvin's
problem (Garvin, 1956): a line explosion (2D, isotropic moment `M0` [N m/m], moment rate `M0 * moment_rate(stf, t, t0,
trise)`, as an [`ExplosionSource`](@ref)) at `depth` [km] below the surface, recorded at the horizontal offsets `x` [km]
(a vector) on the surface, at the times `t` [s]. `vx[it, i]` is along the surface (positive away from the source for
`x > 0`), `vz` along the outward normal (up), as the seismograms of a 2D P-SV simulation with the surface at z = 0.
Computed with the Cagniard–de Hoop method (as in Aki & Richards, 2002, ch. 6), the response to an impulse is
convolved with the source time function; for a tilted surface rotate the simulated velocities into the surface frame.
"""
function garvin(x::AbstractVector, depth, t::AbstractVector; vp, vs, rho, M0=1.0, stf=:ricker, t0=0.0, trise=1.0)
    α, β, h = 1e3vp, 1e3vs, 1e3depth                  # SI units
    dt = t[2] - t[1]
    δ = 1e-3dt
    dm(τ) = (moment_rate(stf, τ + δ, t0, trise) - moment_rate(stf, τ - δ, t0, trise)) / 2δ     # d/dt of the moment rate
    ts = min(t[1], 0.0)                               # the moment rate is zero before ts
    gl = ((-0.8611363115940526, 0.3478548451374538), (-0.3399810435848563, 0.6521451548625461),
          (0.3399810435848563, 0.6521451548625461), (0.8611363115940526, 0.3478548451374538))   # Gauss-Legendre, 4 points
    vx, vz = zeros(length(t), length(x)), zeros(length(t), length(x))
    for (i, xi) in enumerate(x)
        X = 1e3xi
        r = hypot(X, h)
        ta = r / α                                    # arrival of the P wave
        T = t[end] - ts - ta
        T > 0 || continue
        # the impulse response u(t) = -1/(4πρα²) ∫ g(τ) Ṁ0(t - τ) dτ, integrated over bins of τ (dt/8 wide) in the variable
        # w = √(τ - ta), which removes the 1/√ singularity at the arrival
        nb = ceil(Int, 8T / dt)
        Gx, Gz = zeros(nb), zeros(nb)
        for b in 1:nb
            w1, w2 = sqrt((b - 1) * T / nb), sqrt(b * T / nb)
            for (ξ, wq) in gl
                w = (w1 + w2) / 2 + (w2 - w1) / 2 * ξ
                gx, gz = garvin_kernel(X, h, r, α, β, ta + w^2, w)
                Gx[b] += wq * (w2 - w1) / 2 * gx
                Gz[b] += wq * (w2 - w1) / 2 * gz
            end
        end
        C = -M0 / (4π * rho * α^2)
        for (k, tk) in enumerate(t)
            sx = sz = 0.0
            for b in 1:nb
                τ = ta + (b - 0.5) * T / nb
                τ > tk - ts && break
                d = dm(tk - τ)
                sx += Gx[b] * d
                sz += Gz[b] * d
            end
            vx[k, i], vz[k, i] = C * sx, -C * sz       # z up
        end
    end
    return vx, vz
end

"""
The Cagniard–de Hoop integrand times dτ/dw = 2w at τ = ta + w²: on the paths p±(τ) = (iXτ ± h√(τ² - ta²))/r² of the
horizontal slowness, Re[F(p₊) p₊' - F(p₋) p₋'] for the surface displacements (x, z down) per unit Ṁ0 (without -1/(4πρα²)),
with Fx = 4ipηβ/(β²R), Fz = 2γ/(β²R), γ = 1/β² + 2p², R = γ² - 4p²ηαηβ (the Rayleigh function).
"""
function garvin_kernel(X, h, r, α, β, τ, w)
    s = w * sqrt(τ + r / α)                           # √(τ² - ta²)
    gx = gz = 0.0
    for sgn in (1, -1)
        p = (im * X * τ + sgn * h * s) / r^2
        dpdw = 2 * (im * X * w + sgn * h * τ / sqrt(τ + r / α)) / r^2   # p'(τ) · 2w, finite at the arrival
        ηα, ηβ = sqrt(1 / α^2 + p^2), sqrt(1 / β^2 + p^2)
        γ = 1 / β^2 + 2p^2
        R = γ^2 - 4p^2 * ηα * ηβ
        gx += sgn * real(4im * p * ηβ / (β^2 * R) * dpdw)
        gz += sgn * real(2γ / (β^2 * R) * dpdw)
    end
    return gx, gz
end

"""
    EM, PM = tf_misfits(syn, ref, t; band, nf=64, w0=6.0)

Single-valued time-frequency envelope and phase misfits (Kristeková, Kristek, Moczo & Day, 2006) of the seismogram `syn`
against the reference `ref` (vectors, or arrays `[it, …]` for one value per trace) at the times `t` [s]:

    EM = √(Σ ΔE² / Σ |W_ref|²),  PM = √(Σ ΔP² / Σ |W_ref|²),  ΔE = |W_syn| - |W_ref|,  ΔP = |W_ref| Δφ / π,

summed over time and `nf` log-spaced frequencies in `band = (f1, f2)` [Hz], with the continuous wavelet transforms `W`
(Morlet wavelet, `w0 = 6`) and their phase difference Δφ. EM measures amplitude, PM phase (timing) errors, both relative
(0.01 = 1 %); the usual criteria to compare seismograms of different codes. Multiply by 100 for percent.
"""
function tf_misfits(syn::AbstractVector, ref::AbstractVector, t; band, nf=64, w0=6.0)
    length(syn) == length(ref) == length(t) || error("syn, ref and t must have the same length")
    n = length(syn)
    dt = t[2] - t[1]
    L = nextpow(2, 2n)
    ω = 2π .* (0:L-1) ./ (L * dt)
    S, R = fft([Float64.(syn); zeros(L - n)]), fft([Float64.(ref); zeros(L - n)])
    e = p = d = 0.0
    for f in exp.(range(log(band[1]), log(band[2]), length=nf))
        a = w0 / 2π / f
        ψ = [k <= L ÷ 2 ? sqrt(a) * exp(-(a * ω[k] - w0)^2 / 2) : 0.0 for k in 1:L]   # Morlet wavelet, frequency domain
        Ws, Wr = ifft(S .* ψ)[1:n], ifft(R .* ψ)[1:n]
        e += sum(abs2, abs.(Ws) .- abs.(Wr))
        p += sum(abs2, abs.(Wr) .* angle.(Ws .* conj.(Wr)) ./ π)
        d += sum(abs2, Wr)
    end
    d > 0 || error("tf_misfits: the reference has no signal in the band $band")
    return sqrt(e / d), sqrt(p / d)
end

function tf_misfits(syn::AbstractArray, ref::AbstractArray, t; kwargs...)
    size(syn) == size(ref) || error("syn and ref must have the same size")
    m = [tf_misfits(syn[:, I], ref[:, I], t; kwargs...) for I in CartesianIndices(size(syn)[2:end])]
    return first.(m), last.(m)
end
