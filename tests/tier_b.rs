//! Tier B (roadmap §6): `kinematic_sweep` against OndselSolver reference
//! trajectories. The references (`tests/tier_b/<case>.json`) are produced
//! offline by an independent OndselSolver build from the `.asmt` export of
//! the same systems (`examples/tier_b.rs`, `tests/tier_b/README.md`).

#[path = "tier_b/cases.rs"]
mod cases;

use std::path::PathBuf;

use glam::{DQuat, DVec3, EulerRot};
use rcad_mbd::*;

use cases::Reference;

/// Every case is a fully prescribed mechanism this crate solves on its own.
#[test]
fn tier_b_cases_solve() {
    for case in cases::all() {
        let frames = kinematic_sweep(&case.system, &cases::times(&case.sim))
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let diag = diagnose(&case.system, &frames[0].state, 0.0).unwrap();
        assert_eq!(diag.dof, 0, "{}: {diag:?}", case.name);
        assert!(diag.conflicting.is_empty(), "{}", case.name);
        write_asmt(&case.system, &case.sim).unwrap_or_else(|e| panic!("{}: {e}", case.name));
    }
}

fn load(name: &str) -> Reference {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/tier_b/{name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} — regenerate with tests/tier_b/README.md",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap()
}

/// Pose, velocity and acceleration of every part at every reference time.
#[test]
fn tier_b_matches_ondsel() {
    for case in cases::all() {
        let reference = load(case.name);
        let frames = kinematic_sweep(&case.system, &reference.times).unwrap();
        let mut worst = [0.0_f64; 6];
        for (k, frame) in frames.iter().enumerate() {
            for id in case.system.bodies.keys() {
                let part = format!("b{}", id.0);
                let s = &reference.parts[&part][k];
                let v3 = |i: usize| DVec3::new(s[i], s[i + 1], s[i + 2]);
                let (pos, rot) = frame.state.body_pose(*id).unwrap();
                let ref_rot = DQuat::from_euler(EulerRot::XYZ, s[3], s[4], s[5]);
                let vel = frame.velocity[id];
                let acc = frame.acceleration[id];
                let turn = rot.inverse() * ref_rot;
                let errors = [
                    (pos - v3(0)).length(),
                    2.0 * turn.xyz().length().atan2(turn.w.abs()),
                    (vel.linear - v3(6)).length(),
                    (vel.angular - v3(9)).length(),
                    (acc.linear - v3(12)).length(),
                    (acc.angular - v3(15)).length(),
                ];
                for (w, e) in worst.iter_mut().zip(errors) {
                    *w = w.max(e);
                }
            }
        }
        let shown: Vec<String> = worst.iter().map(|w| format!("{w:.2e}")).collect();
        println!(
            "{}: worst |Δ| pos/rot/vel/omega/acc/alpha = {}",
            case.name,
            shown.join(" ")
        );
        // Observed (OndselSolver 09d6175, release): worst 3.8e-10 (spatial_rsur
        // alpha), everything else <= 5e-11; the limits keep >= 25x headroom.
        let limits = [1e-10, 1e-9, 1e-9, 1e-9, 1e-8, 1e-8];
        for ((w, limit), what) in worst.iter().zip(limits).zip([
            "position",
            "rotation",
            "velocity",
            "omega",
            "acceleration",
            "alpha",
        ]) {
            assert!(
                *w < limit,
                "{}: {what} differs by {w:.3e} (limit {limit:e})",
                case.name
            );
        }
    }
}
