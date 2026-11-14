# rcad-mbd — multibody dynamics

A standalone multibody-dynamics backend usable outside of rcad: rigid bodies,
markers, joints, drivers and force elements in, state trajectories and joint
reactions out. Pure math and data — no OCCT, no B-rep.

| Milestone | What it is | State |
|-----------|------------|-------|
| M0 | Data model: `Body` / `Marker` / `Joint` / `Driver` / `ForceElement` / `System`, deterministic tables, `{code,detail,hint}` gates | **done** |
| M1 | Kinematic three levels: position IC via `rcad-gcs` rigid-pose, velocity and acceleration linear solves, drivers | next |
| M2 | Forward dynamics + BDF integration + joint-reaction recovery | planned |
| M3 | Force library: gravity, spring-damper, torque, applied loads | planned |
| M4 | Assembly wiring (Assembly → system conversion, tool surface) | planned |

## How it models

- **Maximal coordinates**: every body carries 6 DOF; joints are constraint
  equations (Lagrange multipliers from M2). Closed loops cost nothing.
- **SI units** inside the crate: kg, m, s, rad, N. Callers convert.
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
})?;
// ... add the link body, hinge markers, revolute joint, driver ...
sys.validate()?;
```

See `ROADMAP.md` for the milestone plan and the benchmark gates.

## License

Dual-licensed under MIT or Apache-2.0, at your option. See `LICENSE`.
