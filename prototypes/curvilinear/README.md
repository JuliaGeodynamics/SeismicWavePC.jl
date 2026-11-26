# Prototype: terrain-following grid with SBP-SAT (2D P-SV)

A standalone stage-1 prototype, not part of the package. Does a grid that follows the topography, with a provably stable
free surface, fix the accuracy of the staircase free surface near slopes?

**Scheme** ([`curvi.jl`](curvi.jl)):
- grid columns sheared with the surface: x = ξ, z = η + h(ξ) (J = 1);
- collocated velocity-stress;
- dual-pairing upwind SBP operators from [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl)
  (Mattsson 2017, order 5): H D₊ + (H D₋)ᵀ = B, D₋ for the divergence, D₊ for the gradient;
- free surface by SAT: the traction is removed weakly (also at the other boundaries, which are far away);
- RK4.

The divergence and the gradient are adjoint in the J-weighted SBP norm, so the semi-discrete energy is conserved.

**Test**: the topography cases of Nobuaki Fuji (flexOPT.jl `dev/cutCells`, as
[`examples/topography_benchmark.jl`](../../examples/topography_benchmark.jl)): tilted planes, a hill, 500 m grid. The
references are SeismicWavePC runs on a 62.5 m grid: the exact flat case for the tilts, `:corners` for the hill.

Relative L2 error (mean over 5 receivers):

| case | line | staircase | `:corners` | curvilinear |
|---|---|---|---|---|
| flat | deep / shallow | 1.2 / 2.9 % | 1.2 / 2.9 % | 1.7 / 3.2 % |
| 10° | deep / shallow | 4.2 / 31 % | 3.0 / 21 % | 1.9 / 3.0 % |
| 20° | deep / shallow | 4.5 / 20 % | 3.0 / 13 % | 3.1 / 2.7 % |
| 30° | deep / shallow | 4.0 / 24 % | 2.7 / 16 % | 3.3 / 1.9 % |
| hill | deep / shallow | 2.2 / 18 % | 1.9 / 11 % | 1.9 / 3.4 % |

![curvilinear vs staircase](curvilinear_vs_staircase.png)

- **Accuracy:** on slopes the shallow line drops from 11–31 % to 2–3.4 %, about the level of the flat case.
- **Convergence:** at 250 m the error is 0.5–0.7 % (30°) and 0.2–0.7 % (hill, limited by the reference).
- **Stability:** stable over 40 s at 30°, 45° and on the hill; after the source stops, the energy is constant to 4·10⁻⁵ (RK4).

**Lessons:**
- **Central SBP fails.** With central operators (Mattsson & Nordström 2004), point forces excite spurious grid-scale modes:
  errors of 5–9 % at 500 m that grow under refinement. Dual-pairing operators behave like a staggered grid.
- **Near the surface, order and pairing matter at 16 points per S wavelength.** Order 5 with D₋ for the divergence is good.
  The other pairing, and orders 6–7 (wider boundary closures), give 7–26 % on the shallow line.
- **At 45°** the shallow line is fine (1.7 %), but the deep line degrades (7.7 %): the strongly sheared grid
  (h′ = 1) costs resolution inside the domain. A mapping that flattens with depth would avoid that.

**Not done:**
- absorbing boundaries;
- attenuation;
- layered media;
- a mapping that flattens with depth;
- 3D;
- leapfrog time stepping (instead of RK4: the cost per step here is not representative);
- gradients.

Run:

    julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/compare.jl        # all cases, figure (~1 min)
    julia -t auto --project=prototypes/curvilinear prototypes/curvilinear/curvi.jl tilt30 250 40
