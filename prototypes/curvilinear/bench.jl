# Cost per time step: a fused leapfrog step of the curvilinear scheme (interior of a block: order-7 dual-pairing
# upwind stencils, metric terms, elastic; plain threaded loops as SeismicWavePC's CPU kernels) against SeismicWavePC
# (staggered, 4th order, elastic, PML) on the same number of points, in Float64 and Float32.
#
#   julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/bench.jl
using SummationByPartsOperators, LinearAlgebra, Printf, SeismicWavePC, GeophysicalModelGenerator

const NX, NZ = 2000, 1000
"interior stencil (offsets, coefficients × h) of an operator"
function stencil(D)
    M = Matrix(D)
    r = size(M, 1) ÷ 2
    h = step(grid(D))
    offs = [j - r for j in axes(M, 2) if M[r, j] != 0]
    Tuple(offs), Tuple(M[r, r.+offs] .* h)
end
U = upwind_operators(MattssonNiemeläWinters2026, derivative_order=1, accuracy_order=7, xmin=0.0, xmax=1.0, N=200)
const OM, CM = stencil(U.minus)          # divergence
const OP, CP = stencil(U.plus)           # gradient
const R = maximum(abs, (OM..., OP...))
println("stencils: D- offsets ", OM, ", D+ offsets ", OP)

@inline dξ(A, i, k, O, C) = (s = zero(eltype(A)); for j in eachindex(O); @inbounds s += C[j] * A[i+O[j], k]; end; s)
@inline dη(A, i, k, O, C) = (s = zero(eltype(A)); for j in eachindex(O); @inbounds s += C[j] * A[i, k+O[j]]; end; s)

function step!(F, M, W, dt, h, CM, CP)
    v1, v2, s11, s22, s12 = F
    J, Jηx, iρJ, lam, mu, Mm = M
    A, B, G, E = W
    # momentum: ρ J ∂t v_i = Dξ(Jξ_j σ_ij) + Dη(Jη_j σ_ij)
    Threads.@threads for k in 1:NZ
        @inbounds @simd for i in 1:NX
            A[i, k] = J[i, k] * s11[i, k]; B[i, k] = Jηx[i, k] * s11[i, k] + s12[i, k]
            G[i, k] = J[i, k] * s12[i, k]; E[i, k] = Jηx[i, k] * s12[i, k] + s22[i, k]
        end
    end
    Threads.@threads for k in R+1:NZ-R
        @inbounds @simd for i in R+1:NX-R
            v1[i, k] += dt / h * (dξ(A, i, k, OM, CM) + dη(B, i, k, OM, CM)) * iρJ[i, k]
            v2[i, k] += dt / h * (dξ(G, i, k, OM, CM) + dη(E, i, k, OM, CM)) * iρJ[i, k]
        end
    end
    # stress: ∂x = Dξ + (Jηx / J) Dη, ∂z = Dη / J
    Threads.@threads for k in R+1:NZ-R
        @inbounds @simd for i in R+1:NX-R
            a, b = dξ(v1, i, k, OP, CP) / h, dη(v1, i, k, OP, CP) / h
            c, e = dξ(v2, i, k, OP, CP) / h, dη(v2, i, k, OP, CP) / h
            q = Jηx[i, k] / J[i, k]
            ex, ez = a + q * b, e / J[i, k]
            s11[i, k] += dt * (Mm[i, k] * ex + lam[i, k] * ez)
            s22[i, k] += dt * (lam[i, k] * ex + Mm[i, k] * ez)
            s12[i, k] += dt * mu[i, k] * (b / J[i, k] + c + q * e)
        end
    end
end

X, Y, Z = xyz_grid(range(0, (NX - 1) * 0.5, length=NX), 0, range(-(NZ - 1) * 0.5, 0, length=NZ))
model = CartData(X, Y, Z, (; rho=fill(2700.0, size(X)), vp=fill(6.0, size(X)), vs=fill(3.464, size(X))))
src = ForceSource(400.0, 0.0, -200.0; fz=1e10, trise=1.0)
@printf("%d × %d points, %d threads\n", NX, NZ, Threads.nthreads())
for FT in (Float64, Float32)
    F = Tuple(FT.(rand(NX, NZ) .* 1e-6) for _ in 1:5)
    Mt = map(x -> FT.(x), (ones(NX, NZ), 0.1 .* rand(NX, NZ), fill(1 / 2700, NX, NZ), fill(3.2e10, NX, NZ), fill(3.2e10, NX, NZ), fill(9.6e10, NX, NZ)))
    W = Tuple(zeros(FT, NX, NZ) for _ in 1:4)
    cm, cp = FT.(CM), FT.(CP)
    step!(F, Mt, W, FT(1e-4), FT(500), cm, cp)
    tc = minimum(@elapsed(for _ in 1:20; step!(F, Mt, W, FT(1e-4), FT(500), cm, cp); end) for _ in 1:3) / 20
    # SeismicWavePC: elastic, 2D P-SV, the same number of cells
    swpc(nt) = @elapsed simulate(model; dt=0.02, nt, sources=src, T=FT, verbose=false, wavefield=false)
    swpc(10)
    ts = minimum((swpc(210) - swpc(10)) / 200 for _ in 1:2)
    @printf("%s: curvilinear (fused leapfrog step, order 7) %6.2f ms/step = %5.2f ns/point;  SeismicWavePC (staggered) %6.2f ms/step = %5.2f ns/point;  ratio %.2f\n",
            FT, 1e3tc, 1e9tc / (NX * NZ), 1e3ts, 1e9ts / (NX * NZ), tc / ts)
end
