# Changelog

## 0.1.0 (2026-11-14)

- Extracted from the rcad kernel repository into a standalone library
  (same shape as `rcad-gcs`): own repository, own license, `[workspace]`
  root, consumed by rcad as a git dependency.
- M0 data model: `Body` / `Marker` / `Joint` / `Driver` / `ForceElement` /
  `System` with deterministic `BTreeMap` tables and `{code,detail,hint}`
  validation gates; serde golden round-trip tests.
- Dependency on `rcad-gcs` declared for the M1 position-level solver.

## Unreleased (M1 first slice)

- `kinematics` module: the flat per-body pose state (translation +
  quaternion), the joint constraint residuals for all seven joint kinds
  (revolute / prismatic / cylindrical / spherical / planar / universal /
  fixed), the driver equations, and the position initial condition as a
  damped Newton-Raphson solve on `rcad-gcs::numerics::newton_solve`.
- Tests: residual layout consistency, the driver angle held by the solve
  for a hinge, the suppressed-joint layout, and the validation gates.
- Next: the velocity and acceleration levels (linear solves against the
  same constraint Jacobian), then the four-bar / crank-slider analytic
  benchmarks and the OndselSolver `.asmt` reference poses.
