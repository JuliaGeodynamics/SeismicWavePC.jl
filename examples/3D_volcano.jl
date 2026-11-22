# 3D visco-elastic waves in a volcano with a magma chamber, built with GeophysicalModelGenerator.
# Stations are given in longitude/latitude and projected with the same ProjectionPoint as the model.
#
#   julia -t auto --project=examples examples/3D_volcano.jl            # CPU (or add: cuda | amdgpu | metal)
#
using SeismicWavePC, GeophysicalModelGenerator
include("backend.jl")
const MPI_RUN = @isdefined(NPROC)

proj = ProjectionPoint(Lon=14.43, Lat=40.82)                # model origin (the summit)
x = range(-8, 8, length=160)                                 # 0.1 km grid spacing
y = range(-8, 8, length=160)
z = range(-10, 2.5, length=126)
model = CartData(xyz_grid(x, y, z))

Phase, Temp = zeros(Int, size(model)), zeros(size(model))
add_layer!(Phase, Temp, model; zlim=(-20, 0), phase=ConstantPhase(1))                     # crust below sea level
add_volcano!(Phase, Temp, model; volcanic_phase=2, center=(0, 0, 0), height=2.0, radius=6.0, crater=0.5, base=0.0)
add_sphere!(Phase, Temp, model; cen=(0, 0, -4), radius=1.5, phase=ConstantPhase(3))      # magma chamber

#             air     crust    volcanic   magma
rho_ = Dict(0 => 0.001, 1 => 2.7, 2 => 2.3, 3 => 2.5)
vp_  = Dict(0 => 0.0,   1 => 5.5, 2 => 3.0, 3 => 3.5)
vs_  = Dict(0 => 0.0,   1 => 3.2, 2 => 1.6, 3 => 1.0)
Q_   = Dict(0 => 10.0,  1 => 300.0, 2 => 60.0, 3 => 20.0)
f(d) = getindex.(Ref(d), Phase)
model = addfield(model, (; rho=f(rho_), vp=f(vp_), vs=f(vs_), Qp=2 .* f(Q_), Qs=f(Q_), Phase))

source = DoubleCoupleSource(14.45, 40.80, -3.0, proj; Mw=3.5, strike=120, dip=60, rake=-90, trise=0.3)
stations = [Station("SUMMIT", 14.43, 40.82, 0.0, proj; surface=true),
            Station("NORTH",  14.43, 40.86, 0.0, proj; surface=true),
            Station("EAST",   14.48, 40.82, 0.0, proj; surface=true)]

result = simulate(model; dt=0.006, nt=1000, sources=source, stations, backend, nproc=(@isdefined(NPROC) ? NPROC : 1),
                  snapshot_every=250, snapshot_dir="volcano_snapshots")
if !MPI_RUN || MPI.Comm_rank(MPI.COMM_WORLD) == 0
    display(result)
    write_paraview(model, "volcano_model")    # velocity model, to view together with the snapshots
end
