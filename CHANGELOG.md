# Changelog

## 0.1.0 (2026-11-14)

- Extracted from the rcad kernel repository into a standalone library
  (same shape as `rcad-gcs`): own repository, own license, `[workspace]`
  root, consumed by rcad as a git dependency.
- M0 data model: `Body` / `Marker` / `Joint` / `Driver` / `ForceElement` /
  `System` with deterministic `BTreeMap` tables and `{code,detail,hint}`
  validation gates; serde golden round-trip tests.
- Dependency on `rcad-gcs` declared for the M1 position-level solver.

## Unreleased (M1 complete)

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

- OndselSolver `.asmt` reference poses (Tier B); then the M1.5 proposals
  (joint primitives, couplers, rigid-group merging, measures) or M2.
