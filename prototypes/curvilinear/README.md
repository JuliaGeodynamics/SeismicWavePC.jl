# Prototype: terrain-following grid with SBP-SAT (2D P-SV)

A standalone prototype, not part of the package. Does a grid that follows the topography, with a provably stable free
surface, fix the accuracy of the staircase free surface near slopes?

**Scheme** ([`curvi.jl`](curvi.jl)):
- grid: x = ξ, z = η + h(ξ) w(η), with the surface h(x) at η = 0. Either w = 1 (columns sheared with the surface
  everywhere) or w goes smoothly from 1 at the surface to 0 at a depth D (terrain-following near the surface only,
  Cartesian below);
- collocated velocity-stress, material parameters as point values (averaged over each node's cell);
- dual-pairing upwind SBP operators from [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl):
  H D₊ + (H D₋)ᵀ = B, D₋ for the divergence, D₊ for the gradient;
- free surface by SAT: the traction is removed weakly (also at the other boundaries, which are far away);
- metric terms computed with the divergence operators;
- RK4.

The divergence and the gradient are adjoint in the J-weighted SBP norm, so the semi-discrete energy is conserved for any
mapping and any material. The scheme is stable over 40 s on 45° slopes and hills; the energy changes by 4·10⁻⁵ (RK4).

The tests are the topography cases of Nobuaki Fuji (flexOPT.jl `dev/cutCells`, as
[`examples/topography_benchmark.jl`](../../examples/topography_benchmark.jl)): tilted planes and hills, a vertical force
20 km deep, and receivers 10 km ("deep") and 1 km ("shallow") below the surface. The references are SeismicWavePC runs
on a 62.5 m grid: the exact flat case for the tilts, `:corners` for the hills. Errors are relative L2, the mean over
5 receivers.

## 1. Free surface on topography (500 m grid, 16 points per S wavelength)

[`compare.jl`](compare.jl); curvilinear with the operators of Mattsson, Niemelä & Winters (2026), order 7:

| case | staircase deep / shallow | `:corners` | curvilinear |
|---|---|---|---|
| flat | 1.2 / 2.9 % | 1.2 / 2.9 % | 0.5 / 0.2 % |
| 10° | 4.2 / 31 % | 3.0 / 21 % | 0.8 / 0.6 % |
| 20° | 4.5 / 20 % | 3.0 / 13 % | 2.0 / 1.0 % |
| 30° | 4.0 / 24 % | 2.7 / 16 % | 1.3 / 0.6 % |
| hill | 2.2 / 18 % | 1.9 / 11 % | 0.6 / 0.8 % |

![curvilinear vs staircase](curvilinear_vs_staircase.png)

## 2. Which operators: higher order is not always better

[`operators.jl`](operators.jl), deep / shallow:

| operators | flat | 20° | hill |
|---|---|---|---|
| Mattsson (2017), order 5 | 1.7 / 3.2 % | 3.1 / 2.7 % | 1.9 / 3.4 % |
| Mattsson (2017), order 7 | 1.0 / 16.5 % | 2.2 / 12.9 % | 0.9 / 16.7 % |
| Williams & Duru (2024, dispersion-relation preserving), order 5 | 0.7 / 2.1 % | 2.1 / 1.9 % | 0.8 / 2.2 % |
| Williams & Duru (2024), order 7 | 2.7 / 21.6 % | 3.1 / 24.5 % | 1.7 / 20.9 % |
| **Mattsson, Niemelä & Winters (2026, boundary-optimized), order 7** | **0.5 / 0.2 %** | **2.0 / 1.0 %** | **0.6 / 0.8 %** |
| Mattsson, Niemelä & Winters (2026), order 9 | 0.3 / 0.2 % | 1.9 / 1.0 % | 0.4 / 0.8 % |

- **Standard closures:** near the free surface they are less accurate and wider for higher orders. At 500 m the closure
  of order 7 spans the top 4–5 km, so the shallow line gets worse with the order. Even orders are worse in the interior.
- **Boundary-optimized closures** (a few non-equidistant points at the boundary) remove this: higher order pays again.
  Their order 8 is stable but stiff (spectral radius 18/h; order 7: 4.6/h, Mattsson order 5: 2.0/h), so it needs a
  much smaller time step.
- **Central SBP operators fail.** With them (Mattsson & Nordström 2004), point forces excite spurious grid-scale modes
  of the collocated grid: 5–9 % at 500 m, growing under refinement. The dual-pairing operators behave like a staggered
  grid.

## 3. Terrain-following near the surface only

[`experiments.jl`](experiments.jl), part 1; deep / shallow, 500 m grid:

| mapping | hill (σ = 5 km) | steep hill (σ = 2 km, slopes to 42°) |
|---|---|---|
| staircase / `:corners` | 2.2 / 18 %, 1.9 / 11 % | 2.5 / 16 %, 1.6 / 8.9 % |
| curvilinear, sheared everywhere | 0.6 / 0.8 % | 1.1 / 0.8 % |
| flattened at D = 20 km (J ≤ 1.22) | 0.5 / 0.8 % | 0.5 / 0.8 % |
| D = 10 km (J ≤ 1.45) | 0.5 / 0.8 % | 0.5 / 0.8 % |
| D = 6 km (J ≤ 1.75) | 0.5 / 1.0 % | 0.5 / 0.7 % |

Flattening costs nothing; with steep topography it even improves the interior (no sheared cells there). On a 45° tilted
plane, sheared everywhere, the deep line is worse (3.6 %, shallow 1.3 %).

## 4. A low-velocity layer under the surface

[`experiments.jl`](experiments.jl), part 2. A layer 2 km thick (vp = 3.5, vs = 2.0 km/s, ρ = 2200 kg/m³) parallel to the
tilted surfaces; the exact reference is the flat layered case with the interface on cell faces. 250 m grid, deep /
shallow:

| case | staircase | `:corners` | curvilinear, 1 block | curvilinear, 2 blocks |
|---|---|---|---|---|
| flat | 0.3 / 1.5 % | 0.3 / 1.5 % | 0.9 / 3.2 % | 0.1 / 0.1 % |
| 10° | 1.2 / 13 % | 1.2 / 8.8 % | 0.8 / 1.5 % | 0.8 / 0.7 % |
| 20° | 1.9 / 10 % | 1.9 / 9.3 % | 2.1 / 3.2 % | 1.8 / 1.0 % |
| 30° | 1.0 / 8.5 % | 0.9 / 8.4 % | 1.2 / 2.3 % | 0.7 / 0.5 % |
| hill (D = 10 km) | 0.8 / 9.3 % | 0.8 / 9.1 % | 0.5 / 2.9 % | 0.2 / 1.3 % |

- **Heterogeneity and stability:** heterogeneity is no problem for stability; the energy proof holds for any material.
- **One block:** a sharp interface between grid nodes limits the accuracy (section 5).
- **Two blocks:** the interface is a grid line (blocks stacked in η, each with its own η grid and operators, ≥ 13 points).
  The blocks are coupled by SAT, enforcing continuity of velocity and traction. The coupling is energy-neutral: half of
  the traction mismatch goes into the momentum, half of the velocity mismatch into the stress of each side.
- **Flattening under the hill** starts below the layer (`flatten_from`), so the interface stays a grid line.
- **The hill reference** (staggered, 62.5 m) is itself a staircase at the interface: the 2-block result is the same at
  250 and 125 m (1.3 %), i.e. it is limited by the reference.
- **The time step** follows from the finest block.

## 5. Order of accuracy at a sharp interface

[`interface_order.jl`](interface_order.jl): a flat free surface over a layer 1937.5 m thick. The interface lies on cell
faces of the 62.5 m reference, but between them at 125–500 m. Deep / shallow line (the shallow one is inside the layer):

| | 500 m | 250 m | 125 m |
|---|---|---|---|
| staggered (SeismicWavePC) | 4.7 / 17 % | 2.7 / 8.8 % | 2.6 / 7.8 % |
| curvilinear, 1 block | 2.0 / 7.6 % | 1.0 / 2.7 % | 0.58 / 1.5 % |
| curvilinear, 2 blocks | 0.40 / 0.24 % | 0.08 / 0.08 % | 0.08 / 0.09 % (the reference) |

- **The staggered code samples the material at cell centres**, so an interface moves to the nearest cell face, which can
  be up to half a cell away. Here it is misplaced by 62.5 m on all three grids, so the error does not converge. 62.5 m is
  1/50 of the S wavelength in the layer, and it still costs 8 % in the layer.
- **One curvilinear block** averages the material over each node's cell (4 × 4 samples: arithmetic ρ, harmonic μ and
  λ + 2μ). That gives first order (0.8–0.9).
- **Two blocks** reach the reference accuracy already at 250 m.

The same sub-cell averaging (effective media, Moczo et al. 2002) would also help the staggered code: average when the
model is given finer than the grid.

## Not done

- absorbing boundaries;
- attenuation;
- 3D;
- leapfrog time stepping (instead of RK4: the cost per step here is not representative);
- gradients.

## Run

    julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/compare.jl        # section 1 (~1 min)
    julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/operators.jl      # section 2 (~2 min)
    julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/experiments.jl    # sections 3, 4 (~6 min)
    julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/interface_order.jl  # section 5 (~4 min)
    FAMILY=Mattsson2017 ORDER=5 julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/curvi.jl hill 500 40 10000
