# Type stability: JET must not find runtime dispatch in SeismicWavePC, except at the boundaries with
# GeophysicalModelGenerator (whose CartData stores coordinates and fields untyped) and in Printf output.
using Test, JET

@testset "type stability (JET)" begin
    model = test_model(:psv)
    kw = (; dt=DT, nt=10, sources=test_source(), stations=test_stations(:psv), verbose=false)
    elastic = CartData(model.x.val, model.y.val, model.z.val, Base.structdiff(model.fields, NamedTuple{(:Qp, :Qs)}))
    for mode in (:psv, :sh), interp in (:cell, :sinc), (m, att) in ((model, true), (elastic, false))
        reports = JET.get_reports(@report_opt target_modules=(SeismicWavePC,) simulate(m, Val(mode), Val(interp), Val(att); kw...))
        # allowed: reading the untyped CartData fields once (convert, axis), building the output CartData, @printf
        boundary = ("convert(::Array{Float64, 3}", "axis(", "CartData(", "format(")
        hot = filter(r -> !any(b -> occursin(b, sprint(show, r)), boundary), reports)
        foreach(r -> println(sprint(show, r)), hot)
        @test isempty(hot)
    end
end
