# Integrating rcad-mbd

`rcad-mbd` is a kernel-neutral multibody-dynamics library. Any CAD kernel
or application can use it; **the adapter from your assembly model to the
crate's data model is yours** and lives in your code. This guide is the
contract that adapter writes against. A complete, runnable sample adapter —
a millimetre-unit "parts + world-frame mates + motors" model turned into a
`System`, run as a study and exported as CSV — is
[`examples/adapter.rs`](../examples/adapter.rs) (`cargo run --example
adapter`; `tests/adapter.rs` checks it against the closed form).

## 1. What you hand over, what you get back

| You provide | You get |
|---|---|
| Rigid bodies with mass properties and an assembled pose | Per output time: every body's pose, velocity, acceleration |
| Markers (frames on bodies) and joints between them | Joint and driver reactions (force, torque, actuator effort) |
| Drivers (prescribed joint motion), couplers, force elements, gravity | Measures you ask for (points, markers, relative motion, joint coordinates) |
| A `Study`: analysis, output times, measures | Energy / momentum audit, redundancy report, integrator statistics |

Nothing in the crate reads geometry, B-rep or a document model: mass
properties, frames and joint choices arrive as numbers.

## 2. Units, frames, conventions

- **SI everywhere**: kg, m, s, rad, N, N·m. Convert at the boundary
  (mm → ×1e-3; kg·mm² → ×1e-6; rpm → ×2π/60).
- **World frame** results. A body's frame is the frame its `translation` /
  `rotation` place in the world (use your part frame). `com` is in body
  coordinates; `inertia_com` is the tensor **about the center of mass, in
  body axes** (any symmetric tensor; principal axes are not required).
- **Velocities** are `Twist { linear, angular }`: the world-frame velocity
  of the body-frame *origin* and the world-frame angular velocity.
- **Quaternions** are glam `DQuat` (serialized `[x, y, z, w]`).
- **Grounded** bodies are fixed in the world (mass may be 0); every free
  body needs a positive mass.
- Author the system **at an assembled pose**; the position solve closes
  small gaps (minimum-norm Newton), but it is not a mate-placement solver.

## 3. Markers and joints

A joint connects two markers: `marker1` is **I**, `marker2` is **J**.
Joint equations use the common Z axis (revolute: rotation about it;
prismatic: translation along it; planar: motion in the XY plane). At the
assembled pose the two markers of a joint normally coincide; place both at
the mate origin with Z along the mate axis, each expressed in its own body
frame: `local = R_bodyᵀ·(world − t_body)` (see `to_system` in the sample).

| Typical CAD mate | `JointKind` |
|---|---|
| concentric + coincident (hinge) | `Revolute` |
| slider / translational | `Prismatic` |
| concentric only (pin in slot) | `Cylindrical` |
| ball / point-to-point coincident | `Spherical` |
| face coincident with free in-plane motion | `Planar` |
| universal / Cardan | `Universal` |
| lock / weld / rigid group | `Fixed` (or merge with `merge_fixed_groups`) |
| point on line / point on plane | `InLine` / `InPlane` |
| parallel / perpendicular axes | `ParallelAxes` / `Perpendicular` |
| same orientation, translation free | `Orientation` |

Mates that map onto several primitives (concentric + coincident written
as two mates) work as given: the resulting redundancy is normal and
reported (§6). Gears, rack-and-pinion and screws are `Coupler`s over joint
coordinates.

**Drivers** prescribe a joint coordinate: rotation is the angle of J's X
axis about I's Z from I's X (unbounded across turns); translation is J's
origin along I's Z. Profiles: constant rate, keyframes, spline, harmonic,
STEP5 (`DriverProfile`).

**Forces**: gravity on `System`; `ForceElement` covers spring-dampers
(with preload), joint spring-dampers, bushings, constant and function
forces / torques (`Function` = a curve of time or of a joint / distance
measure), follower loads (a direction in a body marker's frame) and line
forces.

## 4. Running

```rust
let result = run_study(&sys, &Study {
    name: "one revolution".into(),
    analysis: Analysis::Kinematic,        // or Dynamic { integrator }, Static
    output: TimeGrid::Uniform { start: 0.0, end: 1.0, step: 0.01 },
    measures: vec![/* MeasureRequest::… */],
    reactions: true,
})?;
```

- **Kinematic**: every degree of freedom driven; with `reactions` the
  joint loads and motor efforts are inverse dynamics.
- **Dynamic**: index-3 HHT-α (α = −0.3, error-controlled steps by
  default; `Integrator` exposes the settings). Initial velocities come
  from `Body::velocity` / `angular_velocity`, projected onto the joints.
- **Static**: equilibrium at every output time with drivers held — a
  quasi-static sweep.

The lower-level entry points (`position_ic`, `solve_frame`,
`kinematic_sweep`, `dynamic_ic`, `forward_dynamics`, `simulate`,
`static_equilibrium`, `reactions`, `energy`, `power`, `diagnose`) are
public too.

## 5. Results

`StudyResult` serializes with serde (stable field names; the study
document format is pinned by a golden test) and flattens to one wide CSV
table with `to_csv()` (`b{id}.x…`, `j{id}.fx…`, `d{index}.effort`,
measures by name, `energy.*`, `momentum.*`).

Reaction convention (Adams / OndselSolver): the **force on marker I's
body** and the **torque on it about marker I's origin**, world frame;
`Reaction::force_on_j` / `torque_on_j` give the J side. A driver's
`effort` is the torque (rotation) or force (translation) the actuator
applies to J's body along the driven coordinate.

## 5a. Joint friction

`ForceElement::JointFriction` gives a revolute, prismatic, cylindrical or
spherical joint Adams-style friction: `μ(slip)·N`, where the *friction
load* `N` comes from the joint's own reaction (the radial force at the pin
radius, the axial force at the thrust-face radius, the bending moment over
the bearing spacing, a preload), in the joint's generalized units — N·m
for a rotation, N for a translation. What an adapter needs to know:

- **Set the arms you have**, leave the rest at zero: a zero arm switches
  its term off. `reaction_force: false` with every arm zero leaves
  `μ(slip)·preload`, the load-independent Coulomb friction.
- **The coefficient is regularized** (`stiction_velocity`,
  `friction_velocity`, `mu_static`, `mu_dynamic`): a smooth ramp to
  `mu_static`, the Stribeck fall to `mu_dynamic`, constant beyond. Make
  the transition velocities small against the joint's working speed, but
  not so small that the error controller has to crawl through reversals.
- **Nothing sticks for ever in a dynamic run** — that is the commercial
  compromise too. `static_equilibrium` instead measures the slip as the
  displacement from the pose the solve started at, saturating at
  `max_stiction_deformation`, so a mechanism can rest held by stiction
  (`StaticEquilibrium::friction` reports what holds it);
  `max_stiction_deformation: 0.0` keeps friction out of statics entirely.
- **Report it** with `DynamicFrame::friction(&sys)` (and the field on
  `StaticEquilibrium`): per element the friction load, the slip, the
  coefficient and the friction force / torque. The friction is *not* part
  of `Reaction` — a reaction is the constraint load, the friction is an
  applied load the joint's own reaction creates.
- **One element per joint**, and only on the four kinds above
  (`friction_unsupported` otherwise). Friction makes the loads depend on
  the multipliers, so the instantaneous solves iterate to the fixed point
  of `λ` (`friction_not_converged` if an extreme coefficient and arm never
  settle) and the integrator carries the `∂r/∂λ` block.
- Exports: the MJCF cross-check carries load-independent friction as
  MuJoCo's `frictionloss`; anything load-dependent is
  `mjcf_unsupported`. The `.asmt` export does not carry friction.

## 6. Redundancy

Real assemblies are over-constrained (a planar linkage built from 3D
joints carries redundant equations). The crate detects them, keeps
solving, lists them (`StudyResult::redundant`, `KinematicFrame::redundant`,
`diagnose`) and marks every affected reaction: `unique == false`, with
`Reaction::indeterminate` the wrench directions whose split between joints
the model cannot determine. Present those loads accordingly.

## 7. Failures

Every failure is an `MbdError { code, detail, hint }`, never a panic. The
codes are stable strings (`codes::*`):

| Code | Meaning |
|---|---|
| `dangling_body` / `dangling_marker` / `dangling_joint` | a reference to something not in the system |
| `duplicate_id` / `duplicate_driver` | an id used twice / two drivers on one joint slot |
| `bad_mass_properties` | a free body without positive finite mass |
| `slot_unsupported` | a driver / measure on a coordinate the joint kind lacks |
| `bad_driver_profile` / `bad_coupler` / `bad_force_element` | unusable element data |
| `inconsistent_weld` | fixed joints that cannot all hold |
| `kinematics_not_converged` | the position solve failed (names conflicting equations) |
| `underdetermined_motion` | a kinematic study with free degrees of freedom |
| `kinematic_singular` | a lock-up / toggle position, or over-prescribing drivers |
| `dynamics_singular` | inconsistent dynamics equations, or a load nothing resists |
| `static_not_converged` | no static equilibrium found from the seed |
| `degenerate_force` | a force element without a line of action |
| `bad_integrator_settings` / `integration_failed` | integrator data / a step that cannot complete |
| `bad_time_grid` / `bad_study` | unusable output times or study data |
| `state_mismatch` | a state or multipliers from another system |
| `friction_unsupported` | friction on a joint kind without a friction model, or two friction elements on one joint |
| `friction_not_converged` | the friction load and the reaction it reads never settled |
| `asmt_unsupported` / `mjcf_unsupported` | the OndselSolver `.asmt` / MuJoCo MJCF export cannot represent the input |

## 8. Determinism and serialization

Tables are `BTreeMap`s keyed by stable ids: equation numbering, results
and serialized bytes depend only on the input. The same input gives the
same trajectory, bit for bit. `System`, `Study` and `StudyResult` all
serialize; enable `serde_json`'s `float_roundtrip` feature if you need
exact float round trips through JSON.

## 9. Adapter checklist

1. One `Body` per rigid part (or merge welded groups); mass properties in
   SI about the center of mass, in the part axes.
2. Two markers per mate at the mate frame, each in its own body's frame;
   pick the `JointKind` per §3.
3. Motors → `Driver`s; springs, loads, gravity → force elements; bearing
   friction → `JointFriction` (§5a).
4. `sys.validate()` and surface `MbdError` codes to your users.
5. Describe the analysis as a `Study`, `run_study`, map `StudyResult` back
   onto your parts and mates (keep your own id lookup, as the sample's
   `Lookup` does).

## 10. Stability

The public API is what `lib.rs` re-exports. Until 1.0, breaking changes
are listed in `CHANGELOG.md`; documents (`System`, `Study`) add fields
with serde defaults so older documents keep loading.
