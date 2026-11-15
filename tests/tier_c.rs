//! Tier C (roadmap §6): `simulate` against MuJoCo on the same systems
//! (MJCF export, `tests/tier_c/README.md`). A different formulation
//! (minimal coordinates) and integrator (RK4): agreement at tolerance level
//! is the cross-check; the references are committed, `cargo test` never
//! runs MuJoCo.

#[path = "tier_c/cases.rs"]
mod cases;

use std::path::PathBuf;

use glam::{DQuat, DVec3};
use rcad_mbd::*;

use cases::Reference;

fn load(name: &str) -> Reference {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/tier_c/{name}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e} — see tests/tier_c/README.md", path.display()));
    serde_json::from_str(&text).unwrap()
}

/// Every case exports to MJCF.
#[test]
fn tier_c_cases_export() {
    for case in cases::all() {
        let xml = write_mjcf(&case.system)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert!(
            xml.contains("<mujoco") && xml.contains("</mujoco>"),
            "{}",
            case.name
        );
    }
}

/// Body poses and energy against MuJoCo over every case's horizon, with
/// this crate at a 1e-10 local tolerance (observed, MuJoCo 4.0.1 RK4 at
/// 0.1 ms: ≤ 2.5e-6 m, ≤ 8.9e-5 rad — the slender spatial-chain link
/// spinning about its own axis —, ≤ 1.4e-6 J; the limits in `cases.rs` keep
/// ≥ 4x headroom). The rubbing pendulum is the exception: its two engines
/// regularize Coulomb friction differently, so its limits are set by that
/// (`cases.rs`), and the friction itself is checked against closed forms in
/// `tests/friction_benchmarks.rs`.
#[test]
fn tier_c_matches_mujoco() {
    let settings = HhtSettings {
        error_tol: Some(1e-10),
        ..HhtSettings::default()
    };
    let mut failures = Vec::new();
    for case in cases::all() {
        let reference = load(case.name);
        let sim = simulate(&case.system, &reference.times, &settings).unwrap();
        let (mut worst_p, mut worst_r, mut worst_e): (f64, f64, f64) = (0.0, 0.0, 0.0);
        let e0_ours = energy(&case.system, &sim.frames[0].state, &sim.frames[0].velocity)
            .unwrap()
            .total();
        let e0_theirs = reference.energy[0][0] + reference.energy[0][1];
        let (mut drift_ours, mut drift_theirs): (f64, f64) = (0.0, 0.0);
        for (k, frame) in sim.frames.iter().enumerate() {
            for (name, samples) in &reference.bodies {
                let id: u64 = name[1..].parse().unwrap();
                let s = &samples[k];
                let (p, r) = frame.state.body_pose(BodyId(id)).unwrap();
                worst_p = worst_p.max((p - DVec3::new(s[0], s[1], s[2])).length());
                let theirs = DQuat::from_xyzw(s[4], s[5], s[6], s[3]);
                let turn = r.inverse() * theirs;
                worst_r = worst_r.max(2.0 * turn.xyz().length().atan2(turn.w.abs()));
            }
            let e = energy(&case.system, &frame.state, &frame.velocity).unwrap();
            let theirs = reference.energy[k][0] + reference.energy[k][1];
            worst_e = worst_e.max((e.total() - theirs).abs());
            drift_ours = drift_ours.max((e.total() - e0_ours).abs());
            drift_theirs = drift_theirs.max((theirs - e0_theirs).abs());
        }
        println!(
            "{}: worst |Δ| position {worst_p:.2e} m, rotation {worst_r:.2e} rad, energy {worst_e:.2e} J; energy drift ours {drift_ours:.1e} / MuJoCo {drift_theirs:.1e} J ({})",
            case.name, reference.source
        );
        if worst_p >= case.limits.0 || worst_r >= case.limits.1 || worst_e >= case.limits.2 {
            failures.push(format!(
                "{}: position {worst_p:e}, rotation {worst_r:e}, energy {worst_e:e}",
                case.name
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// What the MJCF export cannot represent is refused as data.
#[test]
fn mjcf_refuses_what_it_cannot_express() {
    let mut driven = cases::all().remove(0).system;
    driven.add_driver(Driver {
        joint: JointId(1),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Constant {
            start: 0.0,
            rate: 1.0,
        },
    });
    assert_eq!(
        write_mjcf(&driven)
            .unwrap_err()
            .code,
        codes::MJCF_UNSUPPORTED
    );
    let mut loaded = cases::all().remove(0).system;
    loaded
        .add_force(ForceElement::AppliedTorque {
            id: ForceId(9),
            name: "torque".into(),
            marker: MarkerId(2),
            torque: DVec3::Z,
        })
        .unwrap();
    assert_eq!(
        write_mjcf(&loaded)
            .unwrap_err()
            .code,
        codes::MJCF_UNSUPPORTED
    );
}
