# Changelog

## 0.1.0 (2026-11-14)

- Extracted from the rcad kernel repository into a standalone library
  (same shape as `rcad-gcs`): own repository, own license, `[workspace]`
  root, consumed by rcad as a git dependency.
- M0 data model: `Body` / `Marker` / `Joint` / `Driver` / `ForceElement` /
  `System` with deterministic `BTreeMap` tables and `{code,detail,hint}`
  validation gates; serde golden round-trip tests.
- Dependency on `rcad-gcs` declared for the M1 position-level solver.

## Unreleased (M1 + M1.5 + M2 + M3 + M4 complete, performance channel, Tier C, M5-1 joint friction, M5-2 contact)

### M5-2 — contact and impact

- `ForceElement::Contact` (new `contact` module): penalty contact between
  primitives placed by markers (`ContactGeometry`: point, sphere, capsule,
  box, plane). Modelled pairs are point / sphere / capsule against each
  other and point / sphere / capsule / box against a plane; anything else
  is the new `contact_unsupported`. Mesh and B-rep contact stay with the
  consumer, which maps its geometry onto these primitives.
- Normal force (`Impact`, the Adams `IMPACT` form):
  `max(0, k·d^e + c·ḋ·STEP(d, 0, 0, d_damp, 1))` per contact point — the
  damping ramps in so first touch is continuous, and the clamp means a
  contact never pulls. `Impact::default()` is Adams' default set in SI.
  Tangential: optional `ContactFriction`, regularized Coulomb on the slip
  in the contact plane, sharing M5-1's Stribeck curve.
- Contact points per pair are fixed and stable across frames: the closest
  pair for two swept shapes, one for a sphere or point on a plane, two for
  a capsule on a plane (its end spheres), eight for a box (its corners).
  `DynamicFrame::contacts(&sys)` reports penetration, closing speed,
  normal force, slip and friction force per point (`ContactLoad`),
  including points that carry nothing.
- `energy()` counts the penalty's stored deformation
  `k·d^(e+1)/(e+1)` as elastic energy, so the audit balances through an
  impact; the damping and friction are dissipative and show up in
  `power()`.
- Contact is not exported to MJCF (MuJoCo's soft contact is a different
  model): `mjcf_unsupported`.
- Tier A: `tests/contact_benchmarks.rs` (14 tests) — Hertzian peak
  penetration, restitution, energy and impulse; the linear contact's
  `π·√(m/k)` duration; damping lowering the restitution; resting
  penetration under gravity for a sphere, a box and a capsule; two
  spheres' momentum and restitution; a sliding box's `μ·g`; crossed
  capsules; a point as a zero-radius sphere; a bouncing ball under error
  control; separation leaving the trajectory and the integrator statistics
  bit-identical; the data gates and the document form.
- Note for callers: a stiff contact sizes the step. On a bouncing ball at
  `k = 1e8` the worst energy error over three bounces is 1.3e-2 at the
  default `error_tol` (1e-6), 4.2e-4 at 1e-8 and 2.5e-5 at 1e-10 — see
  `docs/INTEGRATING.md` §5b.
- Docs: two stray table rows that slipped into the top of `ROADMAP.md`
  with the M5-1 commit are removed.

### M5-1 — joint friction

- `ForceElement::JointFriction` on revolute / prismatic / cylindrical /
  spherical joints (`friction` module): Adams-style **load-dependent**
  friction `mu(slip) * N`, where the friction load `N` is the joint's own
  reaction through the bearing geometry — radial force at the pin radius,
  axial force at the thrust-face radius, bending moment over the bearing
  spacing, moment about a sliding axis at its radius, plus a preload. A
  zero arm switches its term off, `reaction_force: false` switches the
  reaction force off (Adams' "input forces to friction"), and with all of
  them off the friction is the constant `mu(slip) * preload`. A
  cylindrical joint shares one friction cone between sliding and turning;
  a spherical joint's couple follows the relative angular velocity.
- Coefficient: the regularized Stribeck curve (`mu_static`, `mu_dynamic`,
  `stiction_velocity`, `friction_velocity`) — a smooth ramp to the static
  coefficient, the `STEP` fall to the dynamic one, constant beyond.
- The equations of motion are now `Q(q, v, lambda)`: the integrator's
  iteration matrix carries the `dr/dlambda` block (differences over the
  rubbing joints' rows, applied as an exact low-rank correction of the
  symmetric saddle factorization) and the instantaneous solves iterate to
  the fixed point of the multipliers (`friction_not_converged` when they
  do not settle). On a rubbing pendulum the block costs 4.4 Newton
  iterations per step against 2.99 frictionless, and 12 matrix rebuilds
  against 49 without it.
- `static_equilibrium` holds mechanisms with stiction: at rest the slip
  measure is the displacement from the pose the solve started at,
  saturating at `max_stiction_deformation` (`0` leaves friction out of
  statics). `StaticEquilibrium::friction` and `DynamicFrame::friction`
  report each element's friction load, slip, coefficient and friction
  force / torque (`FrictionLoad`).
- New codes: `friction_unsupported` (a joint kind without a friction
  model, or two elements on one joint), `friction_not_converged`.
- MJCF export carries load-independent friction as a joint's
  `frictionloss`; load-dependent friction is `mjcf_unsupported`. New Tier
  C case `rubbing_pendulum` (limited by the differing regularizations, not
  by integration error — `tests/tier_c/README.md`); the Tier C energy
  limit is now per case.
- Tier A: `tests/friction_benchmarks.rs` (12 tests: closed-form
  benchmarks, the data gates, the document form).
  Systems without friction elements are bit-identical to before.

### Code review — deduplication and simplification

- API: `write_mjcf(sys)` takes no options (`MjcfOptions` removed; the
  timestep 0.1 ms and RK4 were its only settings). `DEFAULT_ERROR_TOL` is
  re-exported next to `DEFAULT_ALPHA`.
- One copy of each solver helper (pose retraction, driver levels,
  acceleration right side, tangent-vector conversions, max norm, dense
  saddle assembly); one frame builder and one saddle solve in dynamics;
  the integrator's Newton loop reuses the residual a fresh matrix was
  built at; a dead loop in the sparse ordering removed; reactions resolve
  each owner's markers once.
- Fix: a fixed-step simulation no longer keeps the halved step after a
  Newton failure (the no-growth-after-reject rule now applies to error
  control only).
- Tests: shared builders in `tests/common`, duplicate checks of the same
  system merged, calibration prints and stale tests removed (93 tests).
- Outputs bit-identical to before (Tier B/C trajectories, integrator
  statistics, reactions, statics, diagnostics, MJCF / asmt / CSV exports).

### Tier C — MuJoCo cross-check

- `mjcf` module: `write_mjcf(sys)` (RK4 at 0.1 ms) — tree mechanisms
  (hinge / slide / ball / hinge+slide / nested welds, free bodies as
  freejoints with their initial velocity in a keyframe), revolute /
  spherical loop closures as `connect` equalities, gravity, spring-dampers
  as spatial tendons, joint spring-dampers as joint stiffness / damping /
  springref; contacts off, energy on. New code `mjcf_unsupported`.
- `tests/tier_c/`: five cases (double pendulum, spatial ball+hinge chain,
  free tumbler about its intermediate axis, sprung cart-pendulum with a
  tendon spring, four-bar closed by a connect equality), the MuJoCo runner
  (`run_mujoco.py`, Python via uv), committed references from MuJoCo 4.0.1
  (RK4, 0.1 ms); `tests/tier_c.rs`: worst ≤ 2.5e-6 m, ≤ 8.9e-5 rad (the
  slender spatial link about its own axis), energy ≤ 1.4e-6 J, the
  remaining difference being this crate's integration error at a 1e-10
  tolerance (MuJoCo's RK4 holds energy to 1e-14).
- Fix found on the way: the Newton targets now follow the error tolerance
  (velocity corrections below `1e-4·tol/h`, constraint residuals below
  `1e-2·tol`, never below the index-3 rounding floor, accepted when
  stalled there). The fixed 1e-9 target let corrector error accumulate
  over the many steps of a tight tolerance: at 1e-10 the double
  pendulum's energy drift falls 3.2e-6 → 2.5e-7 J, the tumbler's 3.1e-7 →
  4.8e-8 J. Default-tolerance runs are unchanged (same steps).

### M4 card 2 — integrator's guide and sample adapter (M4 complete)

- `docs/INTEGRATING.md`: the contract an adapter writes against — scope,
  units and frames, marker placement (`local = R_bodyᵀ·(world − t_body)`),
  a CAD-mate → `JointKind` table, drivers / couplers / forces, studies,
  result tables and the reaction convention, redundancy, the error-code
  catalogue, determinism and serialization, an adapter checklist, API
  stability.
- `examples/adapter.rs`: a neutral CAD-style model (mm, kg·mm², world-frame
  mates, rpm motors) → `System` with an id lookup, run as a kinematic study
  with reactions, printed as CSV; `tests/adapter.rs` runs it and checks the
  stroke against `r·cos θ + √(l² − r²·sin²θ)`, the three redundant
  equations and the reaction rows.

### M4 card 1 — study API and result tables

- `study` module: `Study { name, analysis, output, measures, reactions }`
  as serde data — `Analysis::Kinematic` (sweep; with `reactions`, the
  inverse-dynamics loads and motor efforts per frame), `Dynamic {
  integrator }` (HHT-α; `Integrator` is the serializable face of
  `HhtSettings`, every field defaulted), `Static` (quasi-static
  equilibrium at every output time, seeded from the previous);
  `TimeGrid::{Uniform, Times}`; `MeasureRequest::{Point, Marker, Relative,
  JointCoordinate}` with labels. `run_study(sys, study)` → `StudyResult`
  (per frame: body poses / velocities / accelerations, `ReactionResult`s
  with effort and uniqueness, measure values, energy, momentum; redundant
  equations; integrator stats). `StudyResult::to_csv`: one wide table.
  New code `bad_study`.
- serde on `Twist`, `EquationOwner`/`EquationRef`, the measure motions,
  `Energy`, `Momentum`, `IntegrationStats`.
- Tests (`tests/study.rs`): the study JSON form pinned as a contract and
  round-tripped (sparse documents fill defaults); kinematic study efforts =
  inverse dynamics; dynamic study ≡ `simulate`, JSON round trip of the
  results, CSV shape and columns; quasi-static holding torque
  `m·g·l·sin θ(t)`; refused studies. (`serde_json`'s `float_roundtrip`
  feature for exact float round trips in the tests.)

### Performance — sparse dynamics and kinematics

`examples/scaling.rs` (N-link pendulum chain under gravity, one simulated
second, default settings): 40 bodies 88 s → 0.7 s; 100 bodies ≈ 2.3 s,
200 bodies ≈ 6 s; dynamic IC at 200 bodies 5.1 s → 9 ms. Step counts and
energy drift are unchanged (the arithmetic of every step is the same up to
the linear solver).

- Owner-local constraint Jacobian (`Problem::jacobian_sparse`): each joint /
  coupler / driver is differentiated along its own bodies only — O(rows)
  jets instead of O(rows × coordinates), entry-for-entry equal to the old
  assembly (unit test).
- Coloured forward differences for the HHT iteration matrix: a distance-2
  colouring of the body coupling graph (joints, couplers, drivers, force
  elements) perturbs non-interfering bodies together — three colours for
  a chain of any length; bit-for-bit equal to column-by-column differences
  (unit test).
- `sparse` module: sparse LDLᵀ (Davis' LDL: elimination tree, up-looking
  numeric factorization) of the saddle-point systems `[A Jᵀ; J 0]` with a
  constrained minimum-degree ordering (a constraint row only after the
  bodies it touches), linear fill for chains and trees (regression test);
  redundant rows are removed on the spot (multiplier 0, the Adams /
  OndselSolver convention; the dense fallback keeps the minimum-norm
  split); iterative refinement against the unshifted matrix; whatever it
  cannot settle (conflicts, unresisted loads, an indefinite iteration
  matrix) falls back to the dense rank-revealing QR, which still names the
  offending equations. Used by forward dynamics, the velocity IC, output
  projection, the integrator, the kinematic velocity / acceleration levels
  (minimum-norm solves with `A = I`) and a new sparse minimum-norm position
  Newton (backtracking; the `rcad-gcs` dense Newton remains the fallback).
- Still dense: `diagnose`, `static_equilibrium`, `reactions` (null-space
  analysis) — one-off analyses, not per-step work.

### M3 card 2 — joint spring-dampers, bushings (M3 complete)

- `ForceElement::JointSpringDamper` on a joint coordinate (rotation or
  translation; Inventor joint spring / SolidWorks hinge torsion spring):
  `−k·(q − q₀) − c·q̇ + preload` applied through the coordinate's exact
  tangent gradient (jets), rotation unwrapped across turns — winds any
  number of revolutions. Inactive while its joint is suppressed.
- `ForceElement::Bushing` (Adams BUSHING): diagonal translational and
  torsional stiffness / damping in marker I's axes on the displacement and
  rotation vector of J relative to I (rate as seen from I); reaction on I
  at the same point. Small-deflection element, exact about one I axis.
- `energy` carries both potentials; `static_equilibrium` re-anchors the
  angle unwrapping after every Newton step (a spring may wind past half a
  turn on the way) and allows 200 iterations.
- Tier A: damped torsional pendulum on a hinge spring (≤ 1e-5 of θ₀);
  static winding to 6π + 0.5 rad and a dynamic 2.5-turn winding (peak
  1e-5, energy 1e-5); prismatic joint spring ≡ line spring-damper (1e-10);
  bushing static deflection `d_y = −mg/k_y`, `θ_z = T/k_tz` (1e-12);
  bushing translational / torsional vibration at `√(k/m)` / `√(k_t/I)`
  (1e-7) with energy conserved to 1e-14; validation and serde.

### M3 card 1 — function forces, follower loads, preload, work audit

- `force::Function { argument, curve }`: any `DriverProfile` as a curve
  of `Argument::Time`, `JointCoordinate` / `JointRate` (rotation unwrapped
  across turns), `Distance` / `DistanceRate` between markers
  (constructors `constant`, `linear`, `of_time`) — the SolidWorks Motion
  force-function / Adams function-expression subset CAD users reach for.
- New elements: `Force` (action-only, unit direction in a reference
  marker's frame or the world — a reference on the loaded body is a
  follower force), `Torque` (same for moments), `LineForce` (action-
  reaction along the marker line, Adams translational SFORCE).
  `SpringDamper::preload` (Adams convention, pushes apart; serde-defaulted)
  with its potential in `energy`. `ForceElement::name` / `check` /
  `measured_joint`; validation of curves, directions and measured joints
  (new code `bad_force_element`).
- `power(sys, time, state, velocity)`: `Σ Q·v` of all applied loads — the
  work-energy audit.
- Tier A (`tests/force_benchmarks.rs`): forced damped oscillation at
  Ω/ω = 0.5 / 1 / 2 against the steady-state amplitude and phase
  (≤ 4e-5 of X at resonance); follower vs world-fixed push
  (sin θ vs tan θ equilibria, 1e-10); torque linear in the hinge angle =
  torsional pendulum ω = √(k/I_o); line forces of distance and rate
  reproduce the preloaded spring-damper to 1e-12; preloaded static
  extension `l₀ + (mg + P)/k` and energy conservation; work-energy theorem
  with shaker + rate-dependent drag, error second order in h (1.4e-6 →
  3.6e-7 J of 2.2 J); validation codes and serde round trip.

### M2 card 4 — variable step, momentum audit (M2 complete)

- **Fix: HHT now in the Arnold–Brüls generalized-α form** — the equations
  of motion hold at `t + h` with the true accelerations, an auxiliary
  acceleration carries the α weighting into Newmark. The classical
  recursion (card 2) drops to first order with the configuration-dependent
  world inertia `R·I·Rᵀ`: a free tumbling body at α = −0.3 lost 6.8 % of
  its angular momentum in 20 s; now 7.3e-4, second order like α = 0.
  Results at α = 0 are unchanged.
- Variable step (`HhtSettings::error_tol`, default `Some(1e-6)` m/rad ≈
  Adams `ERROR = 1e-3` mm; `HhtSettings::fixed(α, h)` for fixed steps):
  Zienkiewicz–Xie estimate on the true accelerations, `0.9·r^(−1/3)` with
  hysteresis, no growth after a rejection, steps spread evenly to each
  output time (a sliver step re-excites the index-3 λ oscillation: 16 % →
  0 % rejections on the chain audit). `min_step` / `max_step` bounds;
  `DEFAULT_ERROR_TOL`. Iteration matrix reused across step changes (its
  scaled constraint blocks are h-independent); Newton tolerance 1e-9 and
  five stale iterations before a rebuild (four-bar: 31k → 12k residual
  evaluations, results unchanged).
- Tier A: free fall grows the step (exact, < 30 steps); global error
  ∝ tol^(2/3) on the pendulum (observed order 0.64 / 0.65); a stiff
  critically damped spring (k = 4e4) resolved in 292 steps; **unloaded
  momentum drift audit** — free body tumbling about its intermediate axis
  (20 s) and a spinning two-body spherical chain (10 s): angular momentum
  drift second order in h (5.2e-4 → 1.3e-4), chain 2.1e-5 at α = −0.3.
  Documented: α = 0 is only weakly stable on index-3 constraint modes.

### M2 card 3 — joint reactions (λ recovery) + Tier B reactions

- `reaction` module: `reactions(sys, time, state, multipliers)` and
  `DynamicFrame::reactions` / `StaticEquilibrium::reactions` — per joint
  and driver the force on marker I's body and the torque about marker I's
  origin (world frame, the OndselSolver / Adams convention), the J side,
  the driver `effort` along its coordinate, and the statically
  `indeterminate` wrench directions under redundancy (image of the null
  space of `Jᵀ`; `is_unique`, `determinate_part`).
- `.asmt` export now writes constant gravity (force elements still
  refused).
- Tier B: the pipeline reads OndselSolver's joint/motion reaction series
  (`FXonI … TZonI`); two gravity cases (`gravity_rsur`,
  `gravity_nested_chain`); `tier_b_reactions_match_ondsel` compares inverse
  dynamics on every frame of all seven cases — worst 1.4e-10 N / 1.7e-11
  N·m (peak 40 N), redundant crank-sliders on their determinate part.
  The earlier references' part series reproduced byte for byte.
- Tier A: pendulum hinge load `m·(a_com − g)`, motor effort = inverse
  dynamics torque, static hinge load = weight + push, coaxial double hinge
  (indeterminate split, exact total), planar four-bar (in-plane loads
  determinate, out-of-plane split flagged).

### M2 card 2 — index-3 HHT-α integrator

- `integrate` module: `simulate(sys, times, &HhtSettings)` — dynamic IC at
  the first output time, then index-3 HHT-α steps (`β = (1−α)²/4`,
  `γ = ½ − α`, default α = −0.3 = `DEFAULT_ALPHA`) landing on every output
  time. Lie-group rotation increment `R ← exp(Δθ)·R`; constraint rows
  scaled by `1/(β·h²)`; iteration matrix `[∂r/∂a (1+α)Jᵀ; J 0]` by forward
  differences, reused across steps (modified Newton) and rebuilt when
  convergence slows; a failed step is halved (down to `step/1024`).
  Output frames are projected (velocity onto `J·v = ṡ`, then acceleration
  and λ on the acceleration level); the integrator state is untouched.
  `IntegrationStats` counts steps, rejections, Newton iterations and
  matrix builds. New codes `bad_integrator_settings`, `integration_failed`.
- Tier A time benchmarks: free fall with spin exact at any α (1e-10);
  θ₀ = 0.5 rad pendulum period against the elliptic-integral period
  (1.3e-6 at α = 0, 1.9e-6 at α = −0.3) with energy error 2e-8 / 1e-7 of
  m·g·l over 6 s; small-angle period 2π·√(I_o/mgl); spring-mass
  ω = √(k/m) (6e-6) and trajectory; damped oscillator ζ = 0.1 / 1 / 2.5
  against the closed forms (≤ 1e-6 of A); a four-bar of 3D revolutes
  (three redundant equations) stays closed to 1e-9 on positions and
  projected velocities, with energy error bounded and second order in h
  (3.2e-4 → 8.0e-5 J of 1.95 J); driven pendulum holds its driver and its
  λ is the inverse-dynamics torque; bitwise determinism.

### M2 card 1 — equations of motion, dynamic IC, static equilibrium

- Integrator route decided at the M2 review: index-3 HHT-α (Adams HHT;
  default α = −0.3, variable step), no Baumgarte; rotations advance by a
  Lie-group increment; output frames project velocity/acceleration/λ onto
  the constraint levels (integrator state untouched).
- `Body::velocity` / `Body::angular_velocity`: authored initial velocities
  (world frame, body origin — Adams PART `VX…WZ`), serde-defaulted and
  omitted when zero; a merged group takes its root's.
- `dynamics` module: `mass_matrix` (6×6 about the body origin with the
  center-of-mass offset, world inertia `R·I·Rᵀ`) and the velocity-quadratic
  term; generalized forces from gravity, `SpringDamper` (zero-length
  springs isotropic), `AppliedForce`, `AppliedTorque`;
  `forward_dynamics` (saddle-point solve `[M Jᵀ; J 0]` through the
  rank-revealing factorization — redundant equations listed, minimum-norm
  multipliers; a fully driven mechanism gives inverse-dynamics efforts);
  `velocity_ic` (kinetic-energy-norm projection of authored velocities);
  `dynamic_ic`; `static_equilibrium` (Newton on `Q = Jᵀλ, Φ = 0`, tangent
  stiffness by central differences, line search; driver multipliers are
  holding efforts); `energy` / `momentum` audits.
- New codes `dynamics_singular`, `static_not_converged`,
  `degenerate_force`.
- Tests (`tests/dynamics_benchmarks.rs`, Tier A): free tumbling body (com
  at g, Euler equations), pendulum θ̈ = −(mgl/I_o)·sin θ at five states,
  velocity IC keeps hinge-axis momentum, redundant coaxial hinge, driven
  pendulum = inverse dynamics (motor torque), spring-mass-damper, energy
  and momentum audit, static spring extension mg/k, side-loaded pendulum
  tan θ = F·L/(m·g·l), driver holding torque, failure codes; unit tests
  for the mass matrix (energy, SPD), the bias (momentum rate) and the
  hinge multipliers (= hinge load).

### M1.5 CAD-grade kinematics

- Joint primitives `InLine`, `InPlane`, `Orientation`, `ParallelAxes`,
  `Perpendicular` (Adams JPRIM; CAD mates map one to one), exported to the
  matching OndselSolver joints.
- Couplers (`coupler` module, `System::couplers`, serde-defaulted for old
  documents): `Σ cᵢ·Δqᵢ = 0` over joint coordinates, with `gear`,
  `rack_pinion` and `screw` constructors; new code `bad_coupler`.
  `KinematicState` tracks rotation coordinates unwrapped across turns
  (`joint_angle`). Couplers are refused by the `.asmt` export.
- Measures (`measure` module): `KinematicFrame::point`, `marker`,
  `relative` (Adams-style, expressed in the reference marker) and
  `joint_coordinate` (value, rate, acceleration).
- Rigid-group merging (`merge` module): `merge_fixed_groups` collapses
  welded bodies (combined mass properties, weld-implied placement), checks
  weld loops and multiply-grounded groups (new code `inconsistent_weld`);
  `MergedSystem::expand` maps frames back.
- Tests: decomposed crank-slider ≡ lower pairs; gear pair over six motor
  turns, rack and pinion, screw; measures against closed forms; merged
  nested chain against OndselSolver; two primitive Tier B cases
  (`primitive_crank_slider`, `primitive_rsur`) against OndselSolver.

### Tier B references — green

- `tests/tier_b/*.json`: reference trajectories from OndselSolver
  `09d6175` (release build) for the three cases; `tier_b_matches_ondsel`
  is live. Worst deviation 3.8e-10 (spatial RSUR angular acceleration),
  everything else ≤ 5e-11; limits keep ≥ 25× headroom.
- `write_asmt`: the rotation motion of a joint driven in both rotation and
  translation gets its own alias I marker — OndselSolver lets two motions
  sharing an I marker overwrite each other.
- Reference runner: reads the `.asmt` itself (`assemblyFromFile` pops the
  header inside an `assert()`, which a release build drops).

### Tier B pipeline

- `asmt` module: `write_asmt` exports a system to OndselSolver's `.asmt`
  (the FreeCAD Assembly solver format): parts, grounding fixed joints, the
  seven lower pairs, constant-rate and harmonic motions. New code
  `asmt_unsupported`.
- `tests/tier_b/`: three cases Tier A does not cover (offset crank-slider,
  spatial RSUR with spherical + universal joints, a nested chain with two
  drivers on a cylindrical joint and a prismatic joint in a rotating frame),
  the reference runner source, and `examples/tier_b.rs` (export / convert
  OndselSolver's time series to JSON). The comparison test is ignored until
  the reference JSON is generated.

### Second slice: velocity / acceleration levels, redundancy, sweep

- Velocity and acceleration levels: `solve_frame` (one instant, all three
  levels) and `kinematic_sweep` (a time grid, each instant seeded from the
  previous one by a second-order predictor). Velocities are world-frame
  `Twist`s (linear rate of the body-frame origin + angular rate).
- Analytic Jacobian and `γ`: the joint residuals are written once over
  second-order jets (`jet.rs`); position residual, Jacobian column and
  velocity-quadratic term come from the same code.
- Redundancy diagnosis: a dense rank-revealing QR (`linalg.rs`);
  `diagnose` reports DOF, mobility, redundant and conflicting equations;
  the linear levels skip redundant rows; a failed position IC names the
  conflicting equations.
- Driver profiles (`driver.rs`, split out of `joint.rs` and re-exported
  there): `Harmonic`, `Step5`, `Spline` (clamped cubic) next to `Constant`
  and `Keyframes`, all with exact first/second derivatives
  (`Driver::sample`).
- `position_ic_from`: seeded position IC.
- New codes: `duplicate_driver`, `bad_driver_profile`, `state_mismatch`,
  `underdetermined_motion`, `kinematic_singular`, `bad_time_grid`.
- Tier A benchmarks (`tests/kinematics_benchmarks.rs`): crank-slider over
  three revolutions and four-bar (Freudenstein) over two, position /
  velocity / acceleration against closed forms; both carry three redundant
  equations; undriven and over-driven four-bars report as data.

### Fixes

- A rotation driver past ±π never converged (the residual compared an
  `atan2` angle with the unbounded prescribed one); the residual is now
  wrapped, so motors may run any number of revolutions.
- Validation accepted planar translation and universal rotation drivers
  that the solver silently ignored (a zero equation); they are rejected
  now (`slot_unsupported`), and planar rotation is actually solved.
- **Breaking (format):** `DriverProfile::Keyframes(Vec<_>)` becomes
  `Keyframes { points }` — serde cannot serialize an internally tagged
  newtype variant holding a sequence, so the old form never round-tripped.
- A joint slot carrying two drivers is rejected (`duplicate_driver`), as
  the field docs always said.

### First slice

- `kinematics` module: the flat per-body pose state (translation +
  quaternion), the joint constraint residuals for all seven joint kinds
  (revolute / prismatic / cylindrical / spherical / planar / universal /
  fixed), the driver equations, and the position initial condition as a
  damped Newton-Raphson solve on `rcad-gcs::numerics::newton_solve`.

### Next

- Open: M5 (friction, contact) as its own card; Project Chrono as a second
  Tier C engine if wanted.
