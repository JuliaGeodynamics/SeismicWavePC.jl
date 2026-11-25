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
        boundary = ("convert(::Array{Float64, 3}", "axis(", "CartData(", "format(",
                    # Julia 1.10 gives up optimizing the (once per simulation) setup of the interpolation tables
                    (VERSION < v"1.11" ? ("failed to optimize due to recursion",) : ())...)
        hot = filter(r -> !any(b -> occursin(b, sprint(show, r)), boundary), reports)
        foreach(r -> println(sprint(show, r)), hot)
        @test isempty(hot)
    end
    # VTI (anisotropy: the medium has other fields, resolved at compile time)
    vti = CartData(model.x.val, model.y.val, model.z.val, merge(model.fields, (; epsilon=fill(0.1, size(model)), delta=fill(0.05, size(model)))))
    for mode in (:psv, :sh)
        reports = JET.get_reports(@report_opt target_modules=(SeismicWavePC,) simulate(vti, Val(mode), Val(:cell), Val(true), Val(true); kw...))
        boundary = ("convert(::Array{Float64, 3}", "axis(", "CartData(", "format(", (VERSION < v"1.11" ? ("failed to optimize due to recursion",) : ())...)
        hot = filter(r -> !any(b -> occursin(b, sprint(show, r)), boundary), reports)
        foreach(r -> println(sprint(show, r)), hot)
        @test isempty(hot)
    end
    # free_surface = :corners (another average of the shear moduli, resolved at compile time)
    reports = JET.get_reports(@report_opt target_modules=(SeismicWavePC,) simulate(model, Val(:psv), Val(:sinc), Val(true), Val(false), Val(:corners); kw...))
    boundary = ("convert(::Array{Float64, 3}", "axis(", "CartData(", "format(", (VERSION < v"1.11" ? ("failed to optimize due to recursion",) : ())...)
    hot = filter(r -> !any(b -> occursin(b, sprint(show, r)), boundary), reports)
    foreach(r -> println(sprint(show, r)), hot)
    @test isempty(hot)
    # a teleseismic wave (injected along a box)
    tw = TeleseismicWave(:p; angle=20.0, box=(-3.5, 3.5, -3.0), trise=0.3)
    reports = JET.get_reports(@report_opt target_modules=(SeismicWavePC,) simulate(elastic, Val(:psv), Val(:cell), Val(false); kw..., sources=tw))
    boundary = ("convert(::Array{Float64, 3}", "axis(", "CartData(", "format(", (VERSION < v"1.11" ? ("failed to optimize due to recursion",) : ())...)
    hot = filter(r -> !any(b -> occursin(b, sprint(show, r)), boundary), reports)
    foreach(r -> println(sprint(show, r)), hot)
    @test isempty(hot)
end
