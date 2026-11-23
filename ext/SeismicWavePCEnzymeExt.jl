module SeismicWavePCEnzymeExt

# Misfit gradients by reverse-mode automatic differentiation of the time loop, with checkpointing: the forward pass saves
# the state at the start of every segment of `checkpoint_steps` steps; the reverse pass restores each segment, differentiates
# it with Enzyme (the adjoint of the state is carried from segment to segment) and accumulates the gradient of the medium.

using SeismicWavePC, Enzyme
using GeophysicalModelGenerator: CartData
const S = SeismicWavePC

struct GradientDriver
    observed::Seismograms
    checkpoint_steps::Union{Nothing,Int}
end

"Run the steps `a:b` (as `timeloop!`, without output) and return their part of the misfit."
function steps!(F, M, Src, FSrc, St, rec, acc, regions, obs, a, b, p)
    (; c, vc, dtT, nosource, noforce, nown, record_every, pw_edges, edges, backend, scale, nc) = p
    vel!, stress!, inject!, force!, record! = p.kernels
    V, Σ = S.velocities(F), S.stresses(F)
    J = 0.0
    for it in a:b
        itw = (it - 1) % record_every == 0 ? (it - 1) ÷ record_every + 1 : 0
        nown > 0 && S.foreach_item!(backend, record!, (rec, acc, F, St, c, dtT, itw), nown)
        if itw > 0
            for n in 1:nown, k in 1:nc
                r = rec[itw, k, n] * scale - obs[itw, k, n]
                J += r * r / 2
            end
        end
        pw_edges && S.extrapolate_sides!(V, edges)
        map(((lo, ndr, P),) -> S.foreach_cell!(backend, stress!, (F, M, c, vc, dtT, P), lo, ndr), regions)
        nosource || S.run_once!(backend, inject!, (F, Src, it))
        pw_edges && S.extrapolate_sides!(Σ, edges)
        map(((lo, ndr, P),) -> S.foreach_cell!(backend, vel!, (F, M, c, dtT, P), lo, ndr), regions)
        noforce || S.run_once!(backend, force!, (F, M, FSrc, it))
    end
    return J
end

state(s) = (values(s.F)..., s.acc, S.pml_state(s.regions)...)

function (d::GradientDriver)(s)
    s.backend isa S.CPU || error("misfit_gradient runs on the CPU so far")
    obs = d.observed
    length(obs.t) == s.nrec && obs.components == s.components ||
        error("the observed seismograms do not match the simulation (samples $(length(obs.t)) vs $(s.nrec), components)")
    o = obs.v[:, :, s.own]
    p = (; s.c, s.vc, s.dtT, s.nosource, s.noforce, s.nown, s.record_every, s.pw_edges, s.edges, s.backend, s.scale, s.nc,
         kernels=s.kernels[1:5])
    K = something(d.checkpoint_steps, ceil(Int, sqrt(s.nt)))
    starts = collect(1:K:s.nt)
    seg(a) = (a, min(a + K - 1, s.nt))
    args(a) = (s.F, s.M, s.Src, s.FSrc, s.St, s.rec, s.acc, s.regions, o, seg(a)..., p)

    # forward pass: misfit, and the state at the start of every segment
    saved = map(starts) do a
        st = map(copy, state(s))
        J = steps!(args(a)...)
        (st, J)
    end
    J = sum(last, saved)

    # reverse pass, segment by segment
    dF, dacc, dregions, dM = Enzyme.make_zero(s.F), Enzyme.make_zero(s.acc), Enzyme.make_zero(s.regions), Enzyme.make_zero(s.M)
    for (i, a) in Iterators.reverse(enumerate(starts))
        map(copyto!, state(s), saved[i][1])                                    # restore the state at the segment start
        # source and station tables as Duplicated (gradients unused): no mixed activity in the argument tuples of the kernels
        autodiff(set_runtime_activity(Reverse), steps!, Active, Duplicated(s.F, dF), Duplicated(s.M, dM), Duplicated(s.Src, Enzyme.make_zero(s.Src)),
                 Duplicated(s.FSrc, Enzyme.make_zero(s.FSrc)), Duplicated(s.St, Enzyme.make_zero(s.St)),
                 Duplicated(s.rec, Enzyme.make_zero(s.rec)), Duplicated(s.acc, dacc), Duplicated(s.regions, dregions),
                 Const(o), Const(seg(a)[1]), Const(seg(a)[2]), Const(p))
    end
    return J, S.model_gradient(s, dM)
end

function S.misfit_gradient(model::CartData, observed::Seismograms; checkpoint_steps=nothing, kwargs...)
    bad = intersect(keys(kwargs), (:stations, :snapshots, :callback, :checkpoint, :restart, :driver, :nproc))
    isempty(bad) || error("misfit_gradient does not take the keywords $(join(bad, ", "))")
    return simulate(model; kwargs..., stations=observed.stations, verbose=false,
                    driver=GradientDriver(observed, checkpoint_steps === nothing ? nothing : Int(checkpoint_steps)))
end

end
