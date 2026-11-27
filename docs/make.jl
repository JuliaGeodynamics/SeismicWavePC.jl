# Build the documentation: julia --project=docs docs/make.jl
using Documenter, SeismicWavePC

makedocs(;
    sitename="SeismicWavePC.jl",
    modules=[SeismicWavePC],
    authors="Boris Kaus and contributors",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true", canonical="https://juliageodynamics.github.io/SeismicWavePC.jl",
                           size_threshold_ignore=["api.md"], edit_link="main"),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Getting started" => "getting_started.md",
        "User guide" => [
            "Models" => "models.md",
            "Sources and stations" => "sources_stations.md",
            "Running simulations" => "simulations.md",
            "Output" => "output.md",
        ],
        "Gradients and inversion" => [
            "Misfit gradients" => "gradients.md",
            "Inversion and sampling packages" => "inversion.md",
        ],
        "Examples" => "examples.md",
        "Coming from OpenSWPC" => "openswpc.md",
        "Validation" => "validation.md",
        "Curvilinear solver: method" => "curvilinear_method.md",
        "Performance" => "performance.md",
        "API reference" => "api.md",
    ],
    checkdocs=:exports,
    warnonly=[:missing_docs, :cross_references],
)

deploydocs(; repo="github.com/JuliaGeodynamics/SeismicWavePC.jl.git", devbranch="main", push_preview=true)
