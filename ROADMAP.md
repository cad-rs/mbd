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
as gcs); velocity/acceleration = translational + angular (body or inertial
frame — M1 fixes one and the whole crate is consistent). The unit constraint
(quaternion norm) joins the joint equations in the constraint set.

## 4. Milestones

### M0 Data model + skeleton (small card) — **done**
`Body` / `Marker` / `Joint` / `Driver` / `ForceElement` / `System`
(BTreeMap stable order) + serde + golden. Gates: golden bytes stable +
`cargo test -p rcad-mbd`.

### M1 Kinematic three levels (= the full OndselSolver tier, the real body of closed-loop Tier 1)
- Position-level IC: joints → gcs constraints, `rigid3d` solve (assembly
  initial pose).
- Velocity level: `Φ_q·qd = −Φ_t + driver velocity`, one linear solve
  (OndselSolver's VelIC approach: no iteration).
- Acceleration level: `Φ_q·qdd = γ(q, qd, t)` (velocity-quadratic right
  side) + driver acceleration.
- Drivers: `Driver` (body/joint + displacement/velocity/acceleration curve,
  `MotorProfile` semantics).
- Gates: §6 Tier A analytic benchmarks all green + OndselSolver `.asmt`
  reference-pose comparison (Tier B, reusing the gap doc Phase 5 methodology
  — the `.asmt` text format is dependency-free; hand-written samples feed an
  independent OndselSolver exe producing reference JSON).

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
├── kinematics.rs   // three levels: position (delegates to gcs) / velocity / acceleration
├── dynamics.rs     // mass matrix, right-hand side, λ recovery
├── integrate.rs    // BDF(1-3) + step control + stabilization
├── results.rs      // trajectories / reactions / energy audit (serializable)
└── error.rs        // {code, detail, hint}
```

(`kinematics.rs` / `dynamics.rs` / `integrate.rs` / `results.rs` arrive with
M1/M2; `driver.rs` splits out of `joint.rs` when the driver curves land.)

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
