# find_blocks: the interface of cartdata_example.jl is found at the same row whichever side the nodes on it are given
# (GMG's strict below_surface: above; or <=: below); the results are the same.
include(joinpath(@__DIR__, "cartdata_example.jl"))
res = map((("Z .< moho (GMG)", (z, s) -> z < s), ("Z .<= moho", (z, s) -> z <= s))) do (name, b)
    m = model(0.5; below=b)
    rows, bel = find_blocks(m.z.val[:, 1, :], 1e3m.fields.vp[:, 1, :], 1e3m.fields.vs[:, 1, :])
    println(rpad(name, 18), "interface rows ", rows, " of ", size(m.z.val, 3), ", interface nodes with the material below: ",
            count(bel[1]), "/", length(bel[1]))
    run_cartdata(m; case="hill").traces
end
d = maximum(maximum(abs, res[1][k][c] .- res[2][k][c]) / maximum(abs, res[1][k][c]) for k in keys(res[1]), c in 1:2)
println("largest relative difference of the traces: ", d)
@assert d < 1e-12
