# Misfit gradients: the time loop is differentiated with Enzyme (extension SeismicWavePCEnzymeExt); here the documentation
# and the chain rule from the internal moduli back to the fields of the model.

"""
    J, gradient = misfit_gradient(model, observed::Seismograms; checkpoint_steps=nothing, kwargs...)

Least-squares misfit `J = ½ Σ (v - v_observed)²` of the velocity seismograms [m/s] (all samples, components and stations
of `observed`), and its gradient with respect to the model, computed exactly (for the discrete simulation) with reverse-mode
automatic differentiation (Enzyme.jl). Requires `using Enzyme`.

`kwargs` are those of [`simulate`](@ref) (`dt`, `nt`, `sources`, `record_every`, `absorber`, `T`, `backend`, …) and must
describe the simulation that `observed` is compared with; its stations are `observed.stations`. Use `T = Float64` for
accurate gradients. `gradient` is a `CartData` on the grid of `model` with the fields `rho`, `vp`, `vs`, `Qp`, `Qs`:
∂J/∂ of the model field in its units (per kg/m³ (or g/cm³ with `density_unit = :g_cm3`), per km/s, per unit Q). Cells in
the absorbing layers are homogenized in the simulation, so their sensitivity is added to the cells next to the layers.

The wavefields are saved every `checkpoint_steps` steps (default `√nt`) and the steps in between are recomputed in the
reverse pass, so that the memory stays small. So far on the CPU (threaded) and one process (no MPI). Plane waves enter as a
fixed initial wavefield (their dependence on the medium is not differentiated).
"""
function misfit_gradient end
misfit_gradient(args...; kwargs...) = error("misfit_gradient needs the optional package Enzyme: add it and load it with `using Enzyme`")

"d chi / d tau of `visco_chi(ts, tau, f)`."
function visco_dchi(ts, tau, f)
    ω = 2π * f
    cc = sum(t -> (1 - im * ω * t * (1 + tau)) / (1 - im * ω * t), ts) / NM
    dcc = sum(t -> (-im * ω * t) / (1 - im * ω * t), ts) / NM
    return -visco_chi(ts, tau, f)^2 * real(-0.5 * cc^(-1.5) * dcc)
end

"""
Gradient with respect to the model fields from the gradient `dM` with respect to the internal (padded) arrays `rho`, `lam`,
`mu`, `taup`, `taus` (chain rule of `setup_medium`, the absorber padding and the units). `s` is the setup of `simulate`.
"""
function model_gradient(s, dM)
    m, g, N = s.m, s.g, length(s.g.n)
    ts = s.med.ts
    pad(A) = pad_absorber(internal(A, Val(N)), g.n, s.na)
    rho_in = pad(m.rho)
    rho, vp, vs, qp, qs = max.(rho_in, 0.001), pad(m.vp), pad(m.vs), pad(m.Qp), pad(m.Qs)
    taup, taus = s.med.taup, s.med.taus
    gr, gl, gm, gtp, gts = (Float64.(Array(getfield(dM, k))) for k in (:rho, :lam, :mu, :taup, :taus))
    G = map(CartesianIndices(rho)) do I
        χp, χs = visco_chi(ts, taup[I], s.fq_ref), visco_chi(ts, taus[I], s.fq_ref)
        gmt = gm[I] - 2gl[I]                                           # lam = rho vp²/χp² - 2 mu
        drho = (gr[I] + gl[I] * vp[I]^2 / χp^2 + gmt * vs[I]^2 / χs^2) * (rho_in[I] >= 0.001)
        dvp = gl[I] * 2rho[I] * vp[I] / χp^2
        dvs = gmt * 2rho[I] * vs[I] / χs^2
        dtp = gtp[I] - gl[I] * 2rho[I] * vp[I]^2 / χp^3 * visco_dchi(ts, taup[I], s.fq_ref)
        dts = gts[I] - gmt * 2rho[I] * vs[I]^2 / χs^3 * visco_dchi(ts, taus[I], s.fq_ref)
        dqp = isfinite(qp[I]) ? -dtp * taup[I] / qp[I] : 0.0                # tau = NM zeta / Q
        dqs = isfinite(qs[I]) ? -dts * taus[I] / qs[I] : 0.0
        (drho / s.rho_unit, dvp, dvs, dqp, dqs)
    end
    # adjoint of the padding (absorber zones and ghost cells copy cells of the model) and of `internal`
    n = g.n
    fields = map(1:5) do q
        A = zeros(n)
        for I in CartesianIndices(G)
            J = ntuple(a -> a == N ? clamp(I[a] - H, 1, n[a] - s.na) : clamp(I[a] - H, s.na + 1, n[a] - s.na), N)
            A[J...] += G[I][q]
        end
        reverse(reshape(A, size(m.rho)), dims=3)
    end
    x, y, z = s.model.x.val, s.model.y.val, s.model.z.val
    return CartData(x, y, z, (; rho=fields[1], vp=fields[2], vs=fields[3], Qp=fields[4], Qs=fields[5]))
end
