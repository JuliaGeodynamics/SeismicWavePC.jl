module SeismicWavePCEnzymeExt

# Misfit gradients by reverse-mode automatic differentiation of the time loop, with checkpointing: the forward pass saves
# the state at the start of every segment of `checkpoint_steps` steps; the reverse pass restores each segment, differentiates
# it with Enzyme (the adjoint of the state is carried from segment to segment) and accumulates the gradient of the medium.

using SeismicWavePC, Enzyme
const S = SeismicWavePC

struct GradientDriver
    observed::Seismograms
    checkpoint_steps::Union{Nothing,Int}
end

"Run the steps `a:b` (as `timeloop!`, without output) and return their part of the misfit."
function steps!(F, M, Src, FSrc, St, rec, acc, regions, obs, a, b, p)
    J = 0.0
    for it in a:b
        S.forward_step!(F, M, Src, FSrc, St, rec, acc, regions, it, p)
        itw = (it - 1) % p.record_every == 0 ? (it - 1) ÷ p.record_every + 1 : 0
        if itw > 0
            for n in 1:p.nown, k in 1:p.nc
                r = rec[itw, k, n] * p.scale - obs[itw, k, n]
                J += r * r / 2
            end
        end
    end
    return J
end

state(s) = (values(s.F)..., s.acc, S.pml_state(s.regions)...)

function (d::GradientDriver)(s)
    s.backend isa S.CPU && !s.parallel || error("misfit_gradient(...; method = :ad) runs on the CPU and one process (no MPI)")
    o = S.observed_velocities(d.observed, s)
    p = (; s.c, s.vc, s.dtT, s.nosource, s.noforce, s.nown, s.record_every, s.pw_edges, s.edges, s.backend, s.scale, s.nc, s.parallel, s.nb,
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

S.ad_driver(observed::Seismograms, K) = GradientDriver(observed, K)

end
