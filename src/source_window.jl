# Source amplitudes and source sensitivities of many sources on GPUs.
#
# The amplitude of every source at every time step (`nsrc × nt`), and in the adjoint the sensitivities per step, would not
# fit on a GPU for finite faults of thousands of subfaults and long simulations. They are kept on the device in a window of
# time steps instead: the whole simulation when it fits within `SOURCE_BUDGET` values (then nothing changes), otherwise
# blocks of steps. When the time loop leaves a block, the outputs of the block are passed to the host (`reduce!`) and set
# to zero, and the amplitudes of the next block are computed on the host and copied to the device. Kernels index the window
# with the local step `window_step!(W, it)`: `S.s[n, l]`.

"The largest number of values of a window array (per array; 2²⁴: 64 MB in Float32)."
const SOURCE_BUDGET = Ref(1 << 24)

mutable struct SourceWindow{A,H,F,O,R,B}
    s::A                # the amplitudes on the device (nsrc × K)
    host::H             # the same on the host
    rows::F             # rows(n): the amplitude of source n as a function of the time step
    out::O              # outputs on the device (arrays with the window as last dimension), or ()
    reduce!::R          # reduce!(outputs on the host, steps, amplitudes on the host) (accumulating), or nothing
    K::Int              # time steps per block
    nt::Int
    first::Int          # the first time step of the block on the device (0: none yet)
    backend::B
end

"""
    source_window(rows, nsrc, nt, dev, FT, backend; outputs=(), reduce=nothing)

A window of the amplitudes of `nsrc` sources over `nt` time steps (`rows(n)`: the amplitude of source n as a function of the
time step, see `amp_row`; `dev`: host array -> device array), with optional outputs of sizes `(outputs..., K)` whose blocks
are passed to `reduce(host outputs, steps, host amplitudes)`.
"""
function source_window(rows, nsrc, nt, dev, FT, backend; outputs=(), reduce=nothing)
    per = max(nsrc, maximum((prod(o) for o in outputs); init=0), 1)       # values per time step of the largest array
    K = per * nt <= SOURCE_BUDGET[] ? nt : clamp(SOURCE_BUDGET[] ÷ per, 1, nt)
    host = Matrix{FT}(undef, nsrc, K)
    out = Tuple(KernelAbstractions.zeros(backend, FT, o..., K) for o in outputs)
    W = SourceWindow(dev(copy(host)), host, rows, out, reduce, K, nt, 0, backend)
    switch!(W, 1)
    return W
end

"The local index of time step `it` in the window (moving the window if needed)."
function window_step!(W::SourceWindow, it)
    first = (it - 1) ÷ W.K * W.K + 1
    first == W.first || switch!(W, first)
    return it - first + 1
end
window_step!(::Nothing, it) = it

function switch!(W::SourceWindow, first)
    flush!(W)
    W.first > 0 && KernelAbstractions.synchronize(W.backend)      # (kernels of earlier steps may still read the block)
    last = min(first + W.K - 1, W.nt)
    for n in axes(W.host, 1)
        fill_row!(W.host, n, W.rows(n), first, last)    # (a function barrier: typed in the source time function of source n)
    end
    copyto!(W.s, W.host)
    W.first = first
    return nothing
end

"Pass the outputs of the block on the device to `reduce!` and set them to zero (also at the end of the time loop)."
function flush!(W::SourceWindow)
    (W.first == 0 || isempty(W.out)) && return nothing
    KernelAbstractions.synchronize(W.backend)
    W.reduce!(map(Array, W.out), W.first:min(W.first + W.K - 1, W.nt), W.host)
    foreach(o -> fill!(o, 0), W.out)
    return nothing
end
flush!(::Nothing) = nothing

function fill_row!(H, n, f::F, first, last) where {F}
    for (l, it) in enumerate(first:last)
        H[n, l] = f(it)
    end
end

"""
The amplitude of a source at time step `it`, `a · rate(tbeg + (it - shift) dt) · b / c` with its source time function `stf`
(`source_rate`): a function of `it`, typed in `stf` (the sources of a simulation may have source time functions of any type).
"""
amp_row(stf, a, b, c, tbeg, shift, dt, t0, tr) = it -> a * source_rate(stf, tbeg + (it - shift) * dt, t0, tr, dt) * b / c

"Whether the window holds the whole simulation (no refills)."
whole(W::SourceWindow) = W.K == W.nt
