# rcad-mbd — roadmap

The milestone plan for the standalone multibody-dynamics library. Extracted
from the kernel repository's route card (`rcad-pro/docs/rcad-mbd-route-card.md`)
when the crate became an independent library (2026-11-14).

## 0. One line

**A standalone crate: maximal-coordinate multibody dynamics — rigid bodies
(mass / inertia / frame) + joints + drivers + force elements in, per-time
state trajectories and joint reactions out. Position-level solving reuses
`rcad-gcs`'s rigid-pose family; velocity/acceleration levels and time
integration are homegrown (index-3 HHT-α).** Kernel-neutral: any CAD kernel
or application can drive it; adapters live with the consumer. The benchmark tier is SolidWorks Motion
(Adams kernel, the Inventor Dynamic Simulation lineage); like `rcad-gcs`:
pure math, zero OCCT dependency, consumes numbers not B-rep, deterministic
output.

## 1. Shape and boundaries

| Dimension | Decision |
|---|---|
| Dependencies | glam / serde / thiserror / **rcad-gcs** (position-level IC). Zero OCCT, zero rcad-kernel/B-rep |
| Input | Numbers and frames: bodies (mass, com, inertia about com), markers, joints, driver curves, force parameters. Mass properties arrive as numbers (whatever mass-property tool the consumer uses, converted to SI by the caller) |
| Output | Per-time state (per-body position/pose/velocity/acceleration) + **joint reactions (λ)** + energy/momentum audit |
| Determinism | Entities numbered by stable key order (OndselSolver's lesson: name-dictionary order → reproducible equation numbering); no parallelism; same input → same output |
| Errors | `{code, detail, hint}` data, never a panic at the boundary |
| Consumers | **Kernel-neutral library.** Any CAD kernel or application may use it; every adapter (assembly → `System` conversion, mate mapping, units, result presentation) is the consumer's and lives in the consumer's code. This crate ships no kernel-specific code and no kernel dependency |
| File discipline | single file < 2000 lines; English comments; `cargo check -p rcad-mbd` after each edit |

**No kernel in the dependency graph**: the crate depends on math crates and
`rcad-gcs` only; it never depends on a CAD kernel, B-rep or document model,
and no kernel is assumed by its API.

## 2. Benchmark definition: what counts as "commercial grade"

Benchmark: SolidWorks Motion (SIMULIA/Adams kernel), Inventor Dynamic
Simulation (same Adams lineage). Capability → milestone mapping:

| Commercial capability | Milestone | Note |
|---|---|---|
| Joint set (revolute/prismatic/cylindrical/spherical/planar/universal + fixed) | M0/M1 | The lower pairs every CAD mate set maps onto |
| Closed-loop assembly simulation (no user break) | M1 | The natural advantage of maximal coordinates |
| Drivers (displacement/velocity/acceleration curves, constant/piecewise-linear/function) | M1 | `MotorProfile` semantics folded in |
| Kinematic output: position/velocity/acceleration | M1 | = OndselSolver's `runKINEMATIC` tier |
| Gravity / spring / damper / applied force-torque | M3 | |
| **Dynamic simulation + time integration** (index-3 HHT-α, variable step, error control) | M2 | Adams HHT tier |
| **Joint reactions** (bearing loads — the commercial headline feature) | M2 | λ recovery |
| Result trajectories / energy / momentum audit | M2 | |
| Joint friction | M5 | Commercial staple but mathematically dirty (Stribeck); after reactions |
| Contact/impact | M5 | |
| Flexible bodies / control co-simulation | excluded | Different order of magnitude |

**Red line**: no M5 field is pre-seeded in M0–M4 models.

## 3. Architecture: maximal coordinates + constrained DAE

Three schools: maximal coordinates (6 DOF per body, joints as Lagrange
multiplier constraints, DAE index 3 — Adams/OndselSolver), minimal
coordinates / joint space (Featherstone spatial algebra — RecurDyn/game
engines), hybrid. **Maximal coordinates chosen**, because:

1. Closed loops are free — real CAD assemblies almost always have loops;
   minimal coordinates need constraint cutting.
2. Isomorphic to `rcad-gcs::rigid3d` (6N unknowns + constraint residuals /
   Jacobians) — the position-level IC reuses the existing solver; M1's
   foundation is ready.
3. OndselSolver uses the same route and its position level is proven
   commercially usable.

Costs (recorded honestly): constraint drift needs stabilization; the
Jacobian was dense (slow past ~100 bodies) — the post-M2 performance
channel (**done**, after M3) made the per-step work sparse: owner-local
Jacobian, coloured differences for the iteration matrix, a sparse LDLᵀ of
the saddle-point systems with a constrained minimum-degree ordering (dense
rank-revealing QR as the diagnosing fallback). A 100-body chain simulates
one second in ≈ 2.3 s, 200 bodies in ≈ 6 s (`examples/scaling.rs`); one-off
analyses (`diagnose`, static equilibrium, reactions) stay dense.

State representation: position = translation + unit quaternion (same manifold
as gcs); velocity/acceleration = translational + angular, **fixed at M1 to the
inertial (world) frame**: per free body, the linear rate of the body-frame
origin and the angular rate, both world-frame (`Twist`, 6 tangent
coordinates; every body-fixed vector `a` then has `ȧ = ω×a`, which keeps the
Jacobian a set of cross products). M2's Euler equations use the world-frame
inertia `R·I·Rᵀ`. The unit constraint (quaternion norm) joins the joint
equations only at the position level; the tangent-space levels have none.

## 4. Milestones

### M0 Data model + skeleton (small card) — **done**
`Body` / `Marker` / `Joint` / `Driver` / `ForceElement` / `System`
(BTreeMap stable order) + serde + golden. Gates: golden bytes stable +
`cargo test -p rcad-mbd`.

### M1 Kinematic three levels (= the full OndselSolver tier, the real body of closed-loop Tier 1) — **done**
- Position-level IC: the joint residuals (own formulation, every row a dot
  product) on the `rcad-gcs` damped Newton core; seedable
  (`position_ic_from`) so a sweep stays on one assembly branch; rotation
  drivers wrap their residual so a motor may run any number of turns.
  **Done.**
- Velocity level: `Φ_q·qd = −Φ_t + driver velocity`, one linear solve
  (OndselSolver's VelIC approach: no iteration). **Done.**
- Acceleration level: `Φ_q·qdd = γ(q, qd, t)` (velocity-quadratic right
  side) + driver acceleration. **Done.**
- The Jacobian and `γ` are analytic: the residuals are evaluated on
  second-order jets (forward mode), so the three levels share one
  formulation. `rcad-gcs` exposes only its finite-difference Newton
  publicly; its analytic Jacobian, pivoted QR and diagnosis are
  `pub(crate)`, so the crate carries its own dense rank-revealing QR
  (`linalg.rs`) — a candidate to move into `rcad-gcs` once it exports one.
- Redundancy (Risk 3): rank-revealing QR of `Φ_qᵀ` → DOF, mobility,
  redundant and conflicting equations (`diagnose`); redundant rows are
  skipped by the linear solves and listed per frame. **Done.**
- Drivers: constant rate, keyframes, harmonic, `STEP5`, clamped cubic
  spline, each with exact first/second derivatives. **Done.** (Drivers on
  general relative coordinates / point motion: see M1.5.)
- Sweep: `kinematic_sweep` with a second-order Taylor predictor seed;
  free DOF and toggle positions are data (`underdetermined_motion`,
  `kinematic_singular`). **Done.**
- Gates: §6 Tier A crank-slider (3 revolutions, position/velocity/
  acceleration vs closed form) and four-bar (Freudenstein residual, branch
  continuity, rocker rate/acceleration) **green** (`tests/kinematics_benchmarks.rs`);
  OndselSolver `.asmt` reference-pose comparison (Tier B, reusing the gap
  doc Phase 5 methodology — the `.asmt` text format is dependency-free;
  the samples are exported from the test systems by `write_asmt`, an
  independent OndselSolver exe produces the reference JSON): **green**
  (`src/asmt.rs`, `examples/tier_b.rs`, `tests/tier_b/` — offset
  crank-slider, spatial RSUR, nested cylindrical/fixed/prismatic chain;
  position/orientation/velocity/acceleration of every part at 101 instants
  agree to ≤ 4e-10 with OndselSolver `09d6175`).

**M1 is complete.**

### M1.5 CAD-grade kinematics — **done**
What the commercial CAD motion tools have at the kinematic level that the
seven lower pairs do not cover (benchmarked against SolidWorks Motion /
Inventor Dynamic Simulation / Creo Mechanism, and the kinematic-only tier
Onshape / Fusion / CATIA DMU):
- Joint primitives (Adams JPRIM: `InLine`, `InPlane`, `Orientation`,
  `ParallelAxes`, `Perpendicular`) so CAD mates map one to one; the lower
  pairs are exactly their combinations (same rows). **Done** — Tier A
  (decomposed crank-slider ≡ lower pairs) + Tier B (primitive crank-slider
  and RSUR vs OndselSolver's PointInLine/PointInPlane/NoRotation/
  ParallelAxes/Perpendicular joints, ≤ 4e-10).
- Couplers (Adams COUPLER `Σ cᵢ·Δqᵢ = 0` over joint coordinates relative
  to the authored configuration): gear, rack-and-pinion, screw, general.
  Rotation coordinates are tracked unwrapped across turns in the state
  (`joint_angle`). **Done** — Tier A (gear pair driving the crank-slider
  over six motor turns, rack and pinion, screw over three turns). Not
  exported to `.asmt` (OndselSolver's gear/rack/screw couple markers, not
  joint coordinates).
- Rigid-group merging (`merge_fixed_groups`): welded bodies collapse to one
  body with combined mass properties (parallel-axis theorem), placed by the
  welds themselves; `MergedSystem::expand` maps frames back. **Done** —
  the merged nested chain (6 coordinates fewer) still matches
  OndselSolver's unmerged reference to 1e-14.
- Measures (`KinematicFrame::point / marker / relative / joint_coordinate`):
  trace paths, point and marker kinematics, Adams-style relative measures
  expressed in a reference marker, joint coordinates with rate and
  acceleration. **Done** — Tier A against the crank-slider closed form.
- Left for later: a distance-between-points primitive (Ondsel
  `SphSphJoint`), mate-level conversion from `rcad-gcs::rigid3d`
  constraints (the consumer's adapter).

### M2 review — decided (2026-11-14)
**Index-3 HHT-α** (the Adams HHT integrator), **no Baumgarte** (that belongs
to the index-1 ODE route). Default α = −0.3 (Adams' default: light
numerical damping of the high frequencies, negligible at low ones) with
variable-step local error control; the conservation audits run α = 0 at a
fixed step. Rotations advance by a Lie-group increment `R ← exp(Δθ)·R`
(Newmark formulas in the world-frame tangent space); the Newton matrix is
scaled by `1/(β·h²)` (Negrut et al.). Index 3 holds the position
constraints only — velocities drift O(h²) off the velocity constraints —
so **output frames are projected**: velocity, acceleration and λ are
re-solved on the velocity/acceleration constraint levels at each output
instant, the integrator state itself is not touched (stabilized index-2
is a possible later option). M2 also adds per-body initial velocities and
static equilibrium (Creo static / Inventor unknown force).

Correction to the earlier plan: OndselSolver *does* take part — its
kinematic output carries per-joint/driver reaction series
(`FXonI … TZonI`, inverse dynamics), a Tier B reference for λ recovery.

### M2 Forward dynamics + integration (the largest card) — **done**
Cards (one commit each):
1. **Done** — equations of motion, dynamic IC, static equilibrium
   (`dynamics` module): mass matrix about the body origin with the
   center-of-mass offset (`R·I·Rᵀ`), velocity-quadratic term, gravity and
   the M0 force elements, forward dynamics as a rank-revealing
   saddle-point solve (redundancy listed, minimum-norm λ; a fully driven
   mechanism = inverse dynamics), mass-weighted velocity IC, Newton static
   equilibrium, energy/momentum audits; Tier A instantaneous benchmarks.
2. **Done** — HHT-α index-3 integrator (`integrate::simulate`, fixed
   nominal step landing on output times, step halving on Newton failure,
   projected output frames) + Tier A time benchmarks (free fall, pendulum
   period + energy, spring-mass ω = √(k/m), damping three regimes, closed
   four-bar energy error second order in h).
3. **Done** — **λ recovery = joint reactions** (`reaction` module: force
   on I / torque about marker I, world frame; driver efforts; statically
   indeterminate directions under redundancy) + Tier B against
   OndselSolver's joint/motion reaction series on all seven cases (two
   with gravity): worst 1.4e-10 N.
4. **Done** — variable-step error control (Zienkiewicz–Xie estimate,
   hysteresis, default tol 1e-6 m/rad) + long-run angular-momentum drift
   audit (unloaded tumbling body and spherical chain: drift second order
   in h). Found and fixed on the way: classical HHT drops to first order
   with the configuration-dependent world inertia — the integrator now
   uses the Arnold–Brüls generalized-α form (second order for any α).
   Recorded: α = 0 is only weakly stable on index-3 constraint modes.

- Equations of motion (maximal coordinates): `M·v̇ + b = Q(q, v, t) − Jᵀ·λ`;
- Mass matrix `M` (per-body 6×6 composition; the parallel-axis theorem's
  numbers come from the caller's mass-property tool);
- Gates: §6 Tier A dynamics benchmarks + long-run momentum-drift audit +
  Tier B reactions against OndselSolver.
- Estimate: the largest card on the route (the gap doc's T4 estimate of 4+
  weeks is reasonable).

### M3 Force library — **done**
Gravity / linear spring-damper / torsional spring / applied force-torque
(time- and state-functions). Gates: Tier A spring-mass (ω=√(k/m)), forced
oscillation, energy audit.

Cards:
1. **Done** — function forces (`Function` = `DriverProfile` curve of time,
   joint coordinate / rate, marker distance / rate), `Force` / `Torque`
   (world or reference-frame direction: follower loads), `LineForce`,
   spring preload, `power` (work-energy audit). Tier A: forced oscillation
   amplitude/phase in three regimes, follower vs world-fixed equilibria,
   torsional pendulum, work-energy theorem (second order in h).
2. **Done** — joint spring-dampers on joint coordinates (rotation
   multi-turn, translation; Inventor joint springs), 6-DOF bushing (Adams
   BUSHING), their potentials; Tier A damped torsional pendulum, static and
   dynamic multi-turn winding, bushing deflection, frequencies, energy.

### M4 Integration surface (kernel-neutral) — proposed
*Re-scoped 2026-11-15: the crate is a standalone, kernel-neutral library;
wiring it into any particular CAD kernel or application (rcad included)
is that consumer's work, in that consumer's repository.* What this crate
owes every consumer instead:
- a **study API**: an analysis described as data (kinematic sweep, dynamic
  simulation, static equilibrium; output grid, integrator settings,
  requested measures) run through one entry point to one result type;
- **result tables**: per-frame poses, velocities, accelerations, joint
  reactions and driver efforts, requested measures, energy/momentum
  audit — serde-stable (golden tests) plus a flat CSV export;
- an **integrator's guide** with a neutral sample adapter (an example that
  maps a generic "parts + mates + motors" description onto `System`),
  unit-conversion and marker-placement conventions, error-code catalogue;
- API-stability notes (what is public and versioned).
Gates: serde goldens of studies and results, the sample adapter running
in CI, doc examples compiling.

### M5 (separate card, not in this roadmap's schedule)
Joint friction (Stribeck), contact/impact (Chrono/MuJoCo class problems),
flexible/control co-simulation.

## 5. Module layout

```text
src/
├── lib.rs          // exports + crate docs (unit system fixed at M0)
├── body.rs         // Body / Marker (mass properties, frames)
├── joint.rs        // joint vocabulary (PairType mapping) + constraint residuals/Jacobians
├── driver.rs       // drivers (curves + slots)
├── force.rs        // force elements (gravity/spring/damper/applied)
├── system.rs       // System assembly + stable-order numbering + IC entry
├── jet.rs          // second-order jets: residual + Jacobian column + γ from one formulation
├── linalg.rs       // dense rank-revealing QR (redundant/conflicting rows, min-norm solves)
├── kinematics.rs   // three levels: position (delegates to gcs) / velocity / acceleration + sweep
├── dynamics.rs     // mass matrix, right-hand side, λ recovery
├── integrate.rs    // BDF(1-3) + step control + stabilization
├── results.rs      // trajectories / reactions / energy audit (serializable)
└── error.rs        // {code, detail, hint}
```

(`kinematics.rs` / `jet.rs` / `linalg.rs` / `driver.rs` landed with M1;
`dynamics.rs` / `integrate.rs` / `results.rs` arrive with M2.)

## 6. Verification system (three layers, spanning all milestones)

| Layer | Oracle | Note |
|---|---|---|
| **A Analytic closed form** | mathematical truth | free fall a=g; small-angle pendulum period 2π√(l/g) + energy conservation; spring-mass ω=√(k/m); damped oscillation three-regime closed form; crank-slider x=r·cosθ+√(l²−r²sin²θ); four-bar Freudenstein; gear ratio and torque balance ω₁T₁=ω₂T₂; **long-run angular-momentum drift with no external force** (integrator quality metric) |
| **B OndselSolver reference poses + reactions** | `.asmt` samples → independent exe → reference JSON | position/velocity/acceleration levels (all three complete) + joint/motion reaction series (inverse dynamics, with gravity; M2). The gap doc Phase 5 methodology; OndselSolver is local |
| **C Open-source MBD cross-check** | Project Chrono / MuJoCo (Python bindings via uv) | tolerance-level sanity (different integrator/contact paths); cross-check only, no alignment. Introduced from M2 |

Plus: crate-level golden (system serialization bytes stable) and
determinism tests (same input → same trajectory).

## 7. Relationship to existing assets

- `rcad-gcs`: depended on (position-level IC); the four upgrade mechanisms
  the gap doc names (weighted-least-squares IC / redundant-constraint
  reporting / sparse ladder / analytic derivatives) enter `rcad-gcs`'s own
  ledger on demand, not as a prerequisite of this card.
- Consumers (rcad or any other kernel / application): they build `System`s
  from their own assembly models and present the results; their adapters
  and any legacy special-case solvers they retire are their own business.
- OndselSolver gap doc: this card is the crate-ization of its Phase 4 plus
  the landing point of the Phase 5 reference test system; Phase 3 (assembly
  solver alignment) remains an independent prerequisite, not blocked by this
  card.

## 8. Risks and non-goals

- **Non-goals**: flexible bodies, control co-simulation, real-time solving,
  GPU — all excluded; contact/friction deferred to the separate M5 card.
- Risk 1: BDF + stabilization tuning (Baumgarte β/γ) — pin every parameter
  against the Tier A benchmarks; "it runs" is not acceptance.
- Risk 2: dense-Jacobian performance wall — correctness first, speed later;
  resolved by the performance channel (sparse per-step work; the
  determinism discipline still holds: same input, same trajectory).
- Risk 3: redundant constraints (real CAD assemblies are over-constrained
  often) — M1 must detect and report them (OndselSolver's pivoted
  factorization mechanism is portable), or a commercial assembly crashes on
  first feed.
- Unit system: M0 fixes one (SI kernel + document-layer conversion, the gcs
  unit-audit pattern).

## 9. Construction order

M0 → M1 (including the Tier B reference pipeline) → M2 (largest card) → M3
→ performance → M4 (integration surface) → Tier C → (M5 separate card). Each card is an independent commit + full gate;
if stopped midway before M2, M1's output (closed-loop kinematic simulation)
is independently usable and better than the current Tier-1 chained
special case.
