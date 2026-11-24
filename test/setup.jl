# Test model shared by the tests (identical to the setup used to create the OpenSWPC references in test/reference).
using SeismicWavePC, GeophysicalModelGenerator

"""
2D (`mode = :psv` or `:sh`) or 3D (`:p3d`) model: two layers, air above z = -0.2 km, and (if `step`) a 0.15 km high
plateau for x > 1 km. Q = 50.
"""
function test_model(mode; step=mode === :p3d, stretched=false)
    dx = 0.05
    nx, ny, nz = mode === :p3d ? (80, 60, 60) : (200, 1, 100)
    x = range(-(nx - 1) / 2 * dx, (nx - 1) / 2 * dx, length=nx)
    y = mode === :p3d ? range(-(ny - 1) / 2 * dx, (ny - 1) / 2 * dx, length=ny) : 0.0
    # stretched: the spacing in depth grows smoothly from 0.6 dx at the top to 1.4 dx at the bottom
    z = stretched ? 0.5 .- reverse(cumsum(dx .* range(1.4, 0.6, length=nz))) .+ 0.6dx : range(-(nz - 1) * dx + 0.5, 0.5, length=nz)
    X, Y, Z = xyz_grid(x, y, z)
    air = Z .> -0.01 .- 0.2 .* (X .<= (step ? 1.0 : 100.0))
    layer = Z .< -1.5
    vp = ifelse.(air, 0.0, ifelse.(layer, 4.0, 3.0))
    vs = ifelse.(air, 0.0, ifelse.(layer, 2.3, 1.7))
    rho = ifelse.(air, 1.0, ifelse.(layer, 2600.0, 2300.0))   # kg/m³
    Qp = ifelse.(air, 10.0, 50.0)
    Qs = copy(Qp)
    return CartData(X, Y, Z, (; rho, vp, vs, Qp, Qs))
end

# OpenSWPC's x axis points north: this is the same double couple (strike 30, dip 60, rake 70) as used there
function test_source()
    dc = DoubleCoupleSource(-0.4, 0.1, -1.0; Mw=3.0, strike=30.0, dip=60.0, rake=70.0, t0=0.05, trise=0.3)
    m = dc.M
    return MomentTensorSource(dc.x, dc.y, dc.z, dc.M0, (m[2], m[1], m[3], m[5], m[4], m[6]); t0=dc.t0, trise=dc.trise)
end

test_stations(mode) = [Station("A", mode === :p3d ? -1.2 : -2.0, 0.3, -0.3), Station("B", 1.5, -0.4, -0.3), Station("C", 0.6, 0.2, -0.3)]

const DT, NT = 0.004, 600
