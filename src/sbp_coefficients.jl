# The SBP operators of the curvilinear solver: dual-pairing upwind operators (Mattsson, Niemelä & Winters, 2026: boundary-
# optimized closures; order 7) of SummationByPartsOperators.jl (H. Ranocha, https://github.com/ranocha/SummationByPartsOperators.jl).
# At precompile time the operators are built and turned into the tables the kernels use (compile-time constants):
# interior stencils, boundary blocks, the non-equidistant boundary nodes and the norm weights, scaled by the interior
# spacing h. D- (divergence) and D+ (gradient) satisfy H D+ + (H D-)ᵀ = B; the structure assumed by the kernels is checked.
import SummationByPartsOperators as SBPO

function sbp_tables(source, order)
    N0 = 60
    U = SBPO.upwind_operators(source, derivative_order=1, accuracy_order=order, xmin=0.0, xmax=1.0, N=N0)
    Dm, Dp = Matrix(U.minus), Matrix(U.plus)
    x = collect(SBPO.grid(U.minus))
    h = x[N0÷2+1] - x[N0÷2]
    H = LinearAlgebra.diag(Matrix(SBPO.mass_matrix(U.minus)))
    r = N0 ÷ 2
    offs(M) = [j - r for j in 1:N0 if abs(M[r, j]) > 1e-14]
    OM, OP = offs(Dm), offs(Dp)
    CM = Float64.(rationalize.(Dm[r, r.+OM] .* h; tol=1e-11))           # exact fractions (1/140, …)
    CP = Float64.(rationalize.(Dp[r, r.+OP] .* h; tol=1e-11))
    interior(M, O, C, i) = all(abs(M[i, i+o] * h - c) < 1e-10 for (o, c) in zip(O, C)) && count(abs.(M[i, :]) .> 1e-14) == length(O)
    nb(M, O, C) = findfirst(i -> 1 - minimum(O) <= i <= N0 - maximum(O) && interior(M, O, C, i), 1:N0) - 1
    NB = max(nb(Dm, OM, CM), nb(Dp, OP, CP))
    NC = maximum(i -> maximum(j for j in 1:N0 if abs(Dm[i, j]) > 1e-14 || abs(Dp[i, j]) > 1e-14), 1:NB)
    LM, LP = Dm[1:NB, 1:NC] .* h, Dp[1:NB, 1:NC] .* h
    # the kernels assume: the right boundary mirrors the left one (D-[N+1-i, N+1-j] = -D+[i, j]), interior stencils
    # elsewhere, and a grid of NP non-equidistant nodes at either end, uniform in between
    maximum(abs, Dm[N0:-1:N0-NB+1, N0:-1:N0-NC+1] .+ Dp[1:NB, 1:NC]) < 1e-11 || error("SBP operators: no mirror symmetry")
    maximum(abs, Dp[N0:-1:N0-NB+1, N0:-1:N0-NC+1] .+ Dm[1:NB, 1:NC]) < 1e-11 || error("SBP operators: no mirror symmetry")
    all(i -> interior(Dm, OM, CM, i) && interior(Dp, OP, CP, i), NB+1:N0-NB) || error("SBP operators: unexpected interior")
    NP = findfirst(j -> abs((x[j+1] - x[j]) / h - 1) < 1e-12 && abs(H[j+1] / h - 1) < 1e-12, 1:N0)
    P, W = (x[1:NP] .- x[1]) ./ h, H[1:NP] ./ h
    Bd = zeros(N0, N0); Bd[1, 1] = -1; Bd[N0, N0] = 1
    maximum(abs, LinearAlgebra.Diagonal(H) * Dp + (LinearAlgebra.Diagonal(H) * Dm)' - Bd) < 1e-10 || error("SBP operators: not dual-pairing")
    return (; OM=Tuple(OM), CM=Tuple(CM), OP=Tuple(OP), CP=Tuple(CP), LM=Tuple(Tuple(LM[i, :]) for i in 1:NB),
            LP=Tuple(Tuple(LP[i, :]) for i in 1:NB), P=Tuple(P), W=Tuple(W), NB, NC, NP)
end

const SBP_TABLES = sbp_tables(SBPO.MattssonNiemeläWinters2026, 7)
const SBP_OM, SBP_CM, SBP_OP, SBP_CP = SBP_TABLES.OM, SBP_TABLES.CM, SBP_TABLES.OP, SBP_TABLES.CP
const SBP_LM, SBP_LP = SBP_TABLES.LM, SBP_TABLES.LP
const SBP_P, SBP_W = SBP_TABLES.P, SBP_TABLES.W
const SBP_NB, SBP_NC, SBP_NP = SBP_TABLES.NB, SBP_TABLES.NC, SBP_TABLES.NP
const SBP_NMIN = 13                     # the fewest points of a block
