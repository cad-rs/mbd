# rcad-mbd — roadmap

The milestone plan for the standalone multibody-dynamics library. Extracted
from the kernel repository's route card (`rcad-pro/docs/rcad-mbd-route-card.md`)
when the crate became an independent library (2026-11-14).

## 0. One line

**A standalone crate: maximal-coordinate multibody dynamics — rigid bodies
(mass / inertia / frame) + joints + drivers + force elements in, per-time
state trajectories and joint reactions out. Position-level solving reuses
`rcad-gcs`'s rigid-pose family; velocity/acceleration levels and time
integration are homegrown (BDF).** The benchmark tier is SolidWorks Motion
(Adams kernel, the Inventor Dynamic Simulation lineage); like `rcad-gcs`:
pure math, zero OCCT dependency, consumes numbers not B-rep, deterministic
output.

## 1. Shape and boundaries

| Dimension | Decision |
|---|---|
| Dependencies | glam / serde / thiserror / **rcad-gcs** (position-level IC). Zero OCCT, zero rcad-kernel/B-rep |
| Input | Numbers and frames: bodies (mass, com, inertia about com), markers, joints, driver curves, force parameters. Mass properties arrive as numbers (the kernel's GProp output, converted to SI by the caller) |
| Output | Per-time state (per-body position/pose/velocity/acceleration) + **joint reactions (λ)** + energy/momentum audit |
| Determinism | Entities numbered by stable key order (OndselSolver's lesson: name-dictionary order → reproducible equation numbering); no parallelism; same input → same output |
| Errors | `{code, detail, hint}` data, never a panic at the boundary |
| Wiring | `rcad-assembly` depends on it and keeps the Assembly→system conversion in the assembly domain |
| File discipline | single file < 2000 lines; English comments; `cargo check -p rcad-mbd` after each edit |

**Zero contact with the parity chain**: `rcad-kernel` / `rcad-brep` /
`rcad-algo` / `rcad-modeling` never depend on this crate (direction is
assembly → mbd → gcs, all product layer).

## 2. Benchmark definition: what counts as "commercial grade"

Benchmark: SolidWorks Motion (SIMULIA/Adams kernel), Inventor Dynamic
Simulation (same Adams lineage). Capability → milestone mapping:

| Commercial capability | Milestone | Note |
|---|---|---|
| Joint set (revolute/prismatic/cylindrical/spherical/planar/universal + fixed) | M0/M1 | Same vocabulary as the assembly domain's `PairType` |
| Closed-loop assembly simulation (no user break) | M1 | The natural advantage of maximal coordinates |
| Drivers (displacement/velocity/acceleration curves, constant/piecewise-linear/function) | M1 | `MotorProfile` semantics folded in |
| Kinematic output: position/velocity/acceleration | M1 | = OndselSolver's `runKINEMATIC` tier |
| Gravity / spring / damper / applied force-torque | M3 | |
| **Dynamic simulation + time integration** (BDF variable order/step, error control) | M2 | Adams GSTIFF tier |
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
Jacobian is dense (slow past ~100 bodies) — performance work (sparsification
/ body partitioning) is a post-M2 channel, not an early milestone.

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
  constraints (an M4 concern, in the assembly domain).

### M2 review note (proposed)
The index-3 implicit route (Adams GSTIFF / HHT-α with scaling) does not
need Baumgarte; Baumgarte belongs to the index-1 ODE route. Pick one route
at the M2 review. M2 also needs per-body initial velocities in the data
model and naturally yields static equilibrium (Creo static / Inventor
unknown force) at little cost.

### M2 Forward dynamics + integration (the largest card)
- Equations of motion (maximal coordinates): `M·qdd = Q(q, qd, t) − Φᵀ·λ`;
- Mass matrix `M` (per-body 6×6 composition; the parallel-axis theorem's
  numbers come from the kernel's GProp);
- Index-3 DAE: implicit BDF multistep (orders 1–3 to start) + Baumgarte
  stabilization (or projection stabilization — pick one and record the
  trade-off), variable-step error control;
- **λ recovery = joint reactions** (auditable every step);
- Gates: §6 Tier A dynamics benchmarks + long-run momentum-drift audit
  (angular-momentum drift on an unloaded system is the industry metric of
  integrator quality) + OndselSolver does not participate (it has none).
- Estimate: the largest card on the route (the gap doc's T4 estimate of 4+
  weeks is reasonable).

### M3 Force library
Gravity / linear spring-damper / torsional spring / applied force-torque
(time- and state-functions). Gates: Tier A spring-mass (ω=√(k/m)), forced
oscillation, energy audit.

### M4 Assembly-domain wiring (document/session/tool surface)
Assembly → mbd system conversion (instances + mass properties → Body;
`joint_derive` → Joint; `Motor`/`MotionStudy` → Driver/Study);
`assembly.study.run` switches to the real backend; result tables (frame
poses + reactions) reach tool output. `rcad-assembly::motion`'s chained
`simulate` is downgraded to a chain special case or retired (decided at the
Rule-4 review). Gates: tool-surface six points + `engine_tool_coverage` +
existing gates all green.

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
| **B OndselSolver reference poses** | `.asmt` samples → independent exe → reference JSON | position/velocity/acceleration levels (all three complete). The gap doc Phase 5 methodology; OndselSolver is local |
| **C Open-source MBD cross-check** | Project Chrono / MuJoCo (Python bindings via uv) | tolerance-level sanity (different integrator/contact paths); cross-check only, no alignment. Introduced from M2 |

Plus: crate-level golden (system serialization bytes stable) and
determinism tests (same input → same trajectory).

## 7. Relationship to existing assets

- `rcad-gcs`: depended on (position-level IC); the four upgrade mechanisms
  the gap doc names (weighted-least-squares IC / redundant-constraint
  reporting / sparse ladder / analytic derivatives) enter `rcad-gcs`'s own
  ledger on demand, not as a prerequisite of this card.
- `rcad-assembly::kinematics` / `transmission` / `joint_derive` / `motion`:
  the assembly domain's vocabulary and the Tier-1 chained special case; M4
  reviews the fate of `simulate` when wiring.
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
  sparsification is registered as a formal departure (the parity discipline
  does not apply to this crate, but the determinism discipline does).
- Risk 3: redundant constraints (real CAD assemblies are over-constrained
  often) — M1 must detect and report them (OndselSolver's pivoted
  factorization mechanism is portable), or a commercial assembly crashes on
  first feed.
- Unit system: M0 fixes one (SI kernel + document-layer conversion, the gcs
  unit-audit pattern).

## 9. Construction order

M0 → M1 (including the Tier B reference pipeline) → M2 (largest card) → M3
→ M4 → (M5 separate card). Each card is an independent commit + full gate;
if stopped midway before M2, M1's output (closed-loop kinematic simulation)
is independently usable and better than the current Tier-1 chained
special case.
