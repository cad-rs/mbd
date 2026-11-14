# Changelog

## 0.1.0 (2026-11-14)

- Extracted from the rcad kernel repository into a standalone library
  (same shape as `rcad-gcs`): own repository, own license, `[workspace]`
  root, consumed by rcad as a git dependency.
- M0 data model: `Body` / `Marker` / `Joint` / `Driver` / `ForceElement` /
  `System` with deterministic `BTreeMap` tables and `{code,detail,hint}`
  validation gates; serde golden round-trip tests.
- Dependency on `rcad-gcs` declared for the M1 position-level solver.

## Unreleased (M1 + M1.5 complete, M2 in progress)

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

- M2 card 2: the HHT-α index-3 integrator + Tier A time-integration
  benchmarks (free fall, pendulum period + energy, spring-mass ω, damping
  regimes).
