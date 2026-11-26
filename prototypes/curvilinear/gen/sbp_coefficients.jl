# Extracts the dual-pairing upwind SBP operators of Mattsson, Niemelä & Winters (2026, boundary-optimized; order 7) from
# SummationByPartsOperators.jl (H. Ranocha) into src/sbp_coefficients.jl, so the package does not depend on it, and checks
# the structure the kernels rely on.
#
#   julia --project=prototypes/curvilinear prototypes/curvilinear/gen/sbp_coefficients.jl
using SummationByPartsOperators, LinearAlgebra, Printf

const ORDER = 7
op(N; L=1.0) = upwind_operators(MattssonNiemeläWinters2026, derivative_order=1, accuracy_order=ORDER, xmin=0.0, xmax=L, N=N)

N0 = 60
U = op(N0)
Dm, Dp = Matrix(U.minus), Matrix(U.plus)
x = collect(grid(U.minus))
h = x[N0÷2+1] - x[N0÷2]
H = diag(Matrix(mass_matrix(U.minus)))

# boundary rows: those that differ from the interior stencil
r = N0 ÷ 2
offs(M) = [j - r for j in 1:N0 if abs(M[r, j]) > 1e-14]
OM, OP = offs(Dm), offs(Dp)
CM, CP = Dm[r, r.+OM] .* h, Dp[r, r.+OP] .* h
CM, CP = Float64.(rationalize.(CM; tol=1e-11)), Float64.(rationalize.(CP; tol=1e-11))   # exact fractions (1/140, ...)
interior(M, O, C, i) = all(abs(M[i, i+o] * h - c) < 1e-10 for (o, c) in zip(O, C)) && count(abs.(M[i, :]) .> 1e-14) == length(O)
nbm = findfirst(i -> 1 - minimum(OM) <= i <= N0 - maximum(OM) && interior(Dm, OM, CM, i), 1:N0) - 1
nbp = findfirst(i -> 1 - minimum(OP) <= i <= N0 - maximum(OP) && interior(Dp, OP, CP, i), 1:N0) - 1
NB = max(nbm, nbp)
NC = maximum(i -> maximum(j for j in 1:N0 if abs(Dm[i, j]) > 1e-14 || abs(Dp[i, j]) > 1e-14), 1:NB)
LM, LP = Dm[1:NB, 1:NC] .* h, Dp[1:NB, 1:NC] .* h
# the right boundary: D-[N+1-i, N+1-j] = -D+[i, j]
@assert maximum(abs, Dm[N0:-1:N0-NB+1, N0:-1:N0-NC+1] .+ Dp[1:NB, 1:NC]) < 1e-11
@assert maximum(abs, Dp[N0:-1:N0-NB+1, N0:-1:N0-NC+1] .+ Dm[1:NB, 1:NC]) < 1e-11
# rows NB+1 … N-NB are the interior stencils
@assert all(i -> interior(Dm, OM, CM, i) && interior(Dp, OP, CP, i), NB+1:N0-NB)
# boundary nodes and norm: positions p_j = (x_j - x_1)/h and weights H_j/h of the first NP nodes; the rest is uniform
NP = findfirst(j -> abs((x[j+1] - x[j]) / h - 1) < 1e-12 && abs(H[j+1] / h - 1) < 1e-12, 1:N0) 
P = (x[1:NP] .- x[1]) ./ h
W = H[1:NP] ./ h
# h for any N: L = h (2 P[NP] + N - 2NP + 1)
for N in (13, 17, 40, 333)
    Un = op(N; L=7.0)
    xn = collect(grid(Un.minus))
    hn = 7.0 / (2P[end] + N - 2NP + 1)
    @assert abs((xn[N÷2+1] - xn[N÷2]) - hn) < 1e-12 * 7
    @assert maximum(abs, xn[1:NP] .- hn .* P) < 1e-11 && maximum(abs, xn[end:-1:end-NP+1] .- (7.0 .- hn .* P)) < 1e-11
    Hn = diag(Matrix(mass_matrix(Un.minus)))
    @assert maximum(abs, Hn[1:NP] .- hn .* W) < 1e-12 && all(abs.(Hn[NP+1:end-NP] .- hn) .< 1e-12)
    # the closures and stencils reproduce the operator
    @assert maximum(abs, Matrix(Un.minus)[1:NB, 1:NC] .* hn .- LM) < 1e-10
end
println("order $ORDER: interior D- ", OM, ", D+ ", OP, "; boundary rows NB = $NB, columns NC = $NC; nonuniform nodes NP = $NP")

fmt(v) = join((repr(Float64(a)) for a in v), ", ")
tup(v) = "(" * fmt(v) * (length(v) == 1 ? ",)" : ")")
open(joinpath(@__DIR__, "..", "..", "..", "src", "sbp_coefficients.jl"), "w") do io
    println(io, """
# Dual-pairing upwind SBP operators (Mattsson, Niemelä & Winters, 2026: boundary-optimized closures; order $ORDER), from
# SummationByPartsOperators.jl (H. Ranocha), extracted by prototypes/curvilinear/gen/sbp_coefficients.jl. Coefficients are
# multiplied by the interior spacing h. D- (divergence) and D+ (gradient) satisfy H D+ + (H D-)ᵀ = B.
# Rows 1…NB use the boundary blocks LM, LP (columns 1…NC); the last rows the mirrored ones: D-[N+1-i, N+1-j] = -D+[i, j].
# The first NP nodes are not equidistant: x_j = h P[j] (and mirrored), with the norm weights H_j = h W[j]; elsewhere h.
const SBP_OM = $(Tuple(OM))
const SBP_CM = $(tup(CM))
const SBP_OP = $(Tuple(OP))
const SBP_CP = $(tup(CP))
const SBP_LM = ($(join(("(" * fmt(LM[i, :]) * ")" for i in 1:NB), ",\n                ")))
const SBP_LP = ($(join(("(" * fmt(LP[i, :]) * ")" for i in 1:NB), ",\n                ")))
const SBP_P = $(tup(P))
const SBP_W = $(tup(W))
const SBP_NB, SBP_NC, SBP_NP = $NB, $NC, $NP
const SBP_NMIN = 13                     # the fewest points of a block""")
end
println("written src/sbp_coefficients.jl")
