# rcad-mbd — multibody dynamics

A standalone, kernel-neutral multibody-dynamics library: rigid bodies,
markers, joints, drivers and force elements in, state trajectories and joint
reactions out. Pure math and data — no OCCT, no B-rep, no CAD-kernel
dependency. Any CAD kernel or application can use it; the adapter from its
own assembly model to a `System` (and the presentation of the results) is
the consumer's code.

| Milestone | What it is | State |
|-----------|------------|-------|
| M0 | Data model: `Body` / `Marker` / `Joint` / `Driver` / `ForceElement` / `System`, deterministic tables, `{code,detail,hint}` gates | **done** |
| M1 | Kinematic three levels: position IC on the `rcad-gcs` Newton core, velocity and acceleration linear solves on the analytic Jacobian, redundancy diagnosis, smooth driver profiles, time sweep | **done** (Tier A closed forms + Tier B OndselSolver references, ≤ 4e-10) |
| M1.5 | CAD-grade kinematics: joint primitives (mate vocabulary), gear / rack-pinion / screw couplers, rigid-group merging, measures | **done** |
| M2 | Forward dynamics, index-3 HHT-α with error control, static equilibrium, joint reactions | **done** (Tier A closed forms, momentum audit; Tier B reactions vs OndselSolver ≤ 1.4e-10 N) |
| M3 | Force library: function forces (time / state curves), follower loads, line forces, preloaded springs, joint spring-dampers, bushings, work audit | **done** |
| perf | Sparse per-step work: owner-local Jacobian, coloured differences, sparse LDLᵀ | **done** (100-body chain ≈ 2.3 s per simulated second) |
| Tier C | MuJoCo cross-check via the MJCF export (5 cases, ≤ 2.5e-6 m) | **done** |
| M4 | Kernel-neutral integration surface: study API, result tables (serde + CSV), integrator's guide ([`docs/INTEGRATING.md`](docs/INTEGRATING.md)) + sample adapter | **done** |

## How it models

- **Maximal coordinates**: every body carries 6 DOF; joints are constraint
  equations with Lagrange multipliers (the joint reactions). Closed loops
  cost nothing.
- **SI units** inside the crate: kg, m, s, rad, N. Callers convert.
- **Velocities in the world frame**: per body, the linear rate of the body
  frame origin and the angular rate (`Twist`).
- **Deterministic**: tables are `BTreeMap`s keyed by stable ids, so equation
  numbering and serialized bytes are reproducible.
- **Failures are data**: `MbdError` `{code, detail, hint}`; the boundary never
  panics.

## Quick start

```rust
use rcad_mbd::*;
use glam::{DMat3, DQuat, DVec3};

let mut sys = System::new("pendulum").with_standard_gravity();
sys.add_body(Body {
    id: BodyId(1), name: "base".into(),
    mass_kg: 0.0, com: DVec3::ZERO, inertia_com: DMat3::ZERO,
    translation: DVec3::ZERO, rotation: DQuat::IDENTITY, grounded: true,
    velocity: DVec3::ZERO, angular_velocity: DVec3::ZERO,
})?;
// ... add the link body, hinge markers, revolute joint, driver ...
sys.validate()?;

// Kinematic analysis: pose, velocity and acceleration at every output time.
let times: Vec<f64> = (0..=100).map(|k| k as f64 * 0.01).collect();
let frames = kinematic_sweep(&sys, &times)?;
let link = frames[50].velocity[&BodyId(2)].angular; // world-frame ω

// Dynamics (without the driver): index-3 HHT-α with error control;
// every frame is consistent and carries the joint reactions.
let sim = simulate(&sys, &times, &HhtSettings::default())?;
let loads = sim.frames[50].reactions(&sys)?; // force / torque per joint
```

Over-constrained layouts (a planar loop built from 3D joints) are normal:
`diagnose` reports the degrees of freedom, the mobility and the redundant
equations, and the linear levels skip the redundant ones.

Writing an adapter for your CAD kernel or application: see
[`docs/INTEGRATING.md`](docs/INTEGRATING.md) and `examples/adapter.rs`.
See `ROADMAP.md` for the milestone plan and the benchmark gates.

## License

Dual-licensed under MIT or Apache-2.0, at your option. See `LICENSE`.
