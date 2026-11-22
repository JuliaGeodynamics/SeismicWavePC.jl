# Generalized Zener body with NM relaxation mechanisms (tau-method of Blanch et al., 1995),
# ported from OpenSWPC `m_fdtool.f90` (visco_set_relaxtime, visco_constq_zeta, visco_chi).

const NM = 3

"Relaxation times, log-spaced between `fmin` and `fmax`."
relaxation_times(fmin, fmax) = ntuple(m -> 1 / (2π * fmin * (fmax / fmin)^((m - 1) / (NM - 1))), NM)

"Constant-Q scaling `zeta`, so that `tau = NM * zeta / Q`."
function constq_zeta(fmin, fmax, ts)
    oa, ob = 2π * fmin, 2π * fmax
    i0 = sum(t -> (log(1 + ob^2 * t^2) - log(1 + oa^2 * t^2)) / (2t), ts)
    i1 = sum(t -> ((atan(ob * t) - ob * t / (1 + ob^2 * t^2)) - (atan(oa * t) - oa * t / (1 + oa^2 * t^2))) / (2t), ts)
    i2 = 0.0
    for m in 1:NM-1, k in m+1:NM
        a, b = ts[m], ts[k]
        i2 += a * b / (b^2 - a^2) * ((atan(ob * a) / a - atan(ob * b) / b) - (atan(oa * a) / a - atan(oa * b) / b))
    end
    return i0 / (i1 + 2i2)
end

"Velocity dispersion factor at frequency `f`: unrelaxed modulus = relaxed modulus / chi^2."
function visco_chi(ts, tau, f)
    ω = 2π * f
    cc = sum(t -> (1 - im * ω * t * (1 + tau)) / (1 - im * ω * t), ts) / NM
    return 1 / real(cc^(-0.5))
end

"Crank-Nicolson coefficients `(c1, c2, d1, d2)` of the memory-variable update."
function memory_coefficients(ts, dt, ::Type{T}) where {T}
    c1 = ntuple(m -> T((2ts[m] - dt) / (2ts[m] + dt)), NM)
    c2 = ntuple(m -> T(2 / (2ts[m] + dt) / NM), NM)
    d1 = ntuple(m -> T(2ts[m] / (2ts[m] - dt)), NM)
    d2 = T(sum(t -> dt / (2t - dt), ts) / NM)
    return (c1, c2, d1, d2)
end
