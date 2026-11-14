# Changelog

## 0.1.0 (2026-11-14)

- Extracted from the rcad kernel repository into a standalone library
  (same shape as `rcad-gcs`): own repository, own license, `[workspace]`
  root, consumed by rcad as a git dependency.
- M0 data model: `Body` / `Marker` / `Joint` / `Driver` / `ForceElement` /
  `System` with deterministic `BTreeMap` tables and `{code,detail,hint}`
  validation gates; serde golden round-trip tests.
- Dependency on `rcad-gcs` declared for the M1 position-level solver.

## Unreleased (M1 + M1.5 + M2 complete, M3 in progress)

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

- M3 card 2: joint spring-dampers (multi-turn), bushings; then performance (sparse Jacobian, analytic
  iteration matrix) once systems pass ~100 bodies.
