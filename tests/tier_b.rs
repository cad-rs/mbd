//! Tier B (roadmap §6): `kinematic_sweep` and the joint reactions against
//! OndselSolver reference trajectories and reaction series. The references
//! (`tests/tier_b/<case>.json`) are produced offline by an independent
//! OndselSolver build from the `.asmt` export of
//! the same systems (`examples/tier_b.rs`, `tests/tier_b/README.md`).

#[path = "tier_b/cases.rs"]
mod cases;

use std::path::PathBuf;

use glam::{DQuat, DVec3, EulerRot};
use rcad_mbd::*;

use cases::Reference;

/// Every case is a fully prescribed mechanism (its sweep is checked
/// against the references below) and exports.
#[test]
fn tier_b_cases_solve() {
    for case in cases::all() {
        let state = position_ic(&case.system, 0.0).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let diag = diagnose(&case.system, &state, 0.0).unwrap();
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

/// Worst deviation of `frames` from `reference` over every body of `sys`:
/// position, rotation, velocity, angular velocity, acceleration, angular
/// acceleration.
fn worst_deviation(sys: &System, frames: &[KinematicFrame], reference: &Reference) -> [f64; 6] {
    let mut worst = [0.0_f64; 6];
    for (k, frame) in frames.iter().enumerate() {
        for id in sys.bodies.keys() {
            let s = &reference.parts[&format!("b{}", id.0)][k];
            let v3 = |i: usize| DVec3::new(s[i], s[i + 1], s[i + 2]);
            let (pos, rot) = frame.state.body_pose(*id).unwrap();
            let turn = rot.inverse() * DQuat::from_euler(EulerRot::XYZ, s[3], s[4], s[5]);
            let (vel, acc) = (frame.velocity[id], frame.acceleration[id]);
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
    worst
}

/// Observed (OndselSolver 09d6175, release): worst 3.8e-10 (spatial_rsur
/// alpha), everything else <= 5e-11; the limits keep >= 25x headroom.
const LIMITS: [f64; 6] = [1e-10, 1e-9, 1e-9, 1e-9, 1e-8, 1e-8];

fn check(name: &str, worst: [f64; 6]) {
    let shown: Vec<String> = worst.iter().map(|w| format!("{w:.2e}")).collect();
    println!(
        "{name}: worst |Δ| pos/rot/vel/omega/acc/alpha = {}",
        shown.join(" ")
    );
    let what = [
        "position",
        "rotation",
        "velocity",
        "omega",
        "acceleration",
        "alpha",
    ];
    for ((w, limit), what) in worst.iter().zip(LIMITS).zip(what) {
        assert!(
            *w < limit,
            "{name}: {what} differs by {w:.3e} (limit {limit:e})"
        );
    }
}

/// Pose, velocity and acceleration of every part at every reference time.
#[test]
fn tier_b_matches_ondsel() {
    for case in cases::all() {
        let reference = load(case.name);
        let frames = kinematic_sweep(&case.system, &reference.times).unwrap();
        check(
            case.name,
            worst_deviation(&case.system, &frames, &reference),
        );
    }
}

/// Joint and driver reactions (λ recovery, inverse dynamics on the
/// kinematic frames) against OndselSolver's `FXonI … TZonI` series: the
/// force on marker I's part and the torque about marker I, world frame.
/// Under redundancy only the determinate part is compared — OndselSolver
/// removes redundant equations, this crate reports the minimum-norm split;
/// valid splits differ only along the reported indeterminate directions.
#[test]
fn tier_b_reactions_match_ondsel() {
    for case in cases::all() {
        let reference = load(case.name);
        assert!(!reference.reactions.is_empty(), "{}: no reaction series", case.name);
        let frames = kinematic_sweep(&case.system, &reference.times).unwrap();
        let (mut worst_f, mut worst_t, mut peak) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut indeterminate = 0;
        for (k, frame) in frames.iter().enumerate() {
            let dynamic = forward_dynamics(&case.system, frame.time, &frame.state, &frame.velocity).unwrap();
            for r in dynamic.reactions(&case.system).unwrap() {
                let key = match r.owner {
                    EquationOwner::Joint(id) => format!("j{}", id.0),
                    EquationOwner::Driver(i) => format!("d{i}"),
                    EquationOwner::Coupler(_) => unreachable!("no couplers in Tier B"),
                };
                let s = &reference.reactions[&key][k];
                let ours = r.wrench_on_i();
                let diff: Vec<f64> = ours.iter().zip(s).map(|(a, b)| a - b).collect();
                let d = r.determinate_part(diff.try_into().unwrap());
                worst_f = worst_f.max(DVec3::new(d[0], d[1], d[2]).length());
                worst_t = worst_t.max(DVec3::new(d[3], d[4], d[5]).length());
                peak = peak.max(r.force_on_i.length());
                if !r.is_unique() && k == 0 {
                    indeterminate += 1;
                }
            }
        }
        println!(
            "{}: worst |Δ| force {worst_f:.2e} N, torque {worst_t:.2e} N·m (peak force {peak:.3} N, {indeterminate} indeterminate)",
            case.name
        );
        assert!(worst_f < REACTION_LIMITS[0], "{}: force differs by {worst_f:e}", case.name);
        assert!(worst_t < REACTION_LIMITS[1], "{}: torque differs by {worst_t:e}", case.name);
    }
}

/// Observed (OndselSolver 09d6175, release): worst force 1.4e-10 N
/// (spatial / gravity RSUR, peak 28 N), torque 1.7e-11 N·m; the limits keep
/// >= 25x headroom.
const REACTION_LIMITS: [f64; 2] = [5e-9, 5e-10];

/// Rigid-group merging: the nested chain's arm is welded to its carriage.
/// Merged (one body, six coordinates and six equations fewer), solved, and
/// expanded back onto the original bodies, it still matches OndselSolver's
/// unmerged reference.
#[test]
fn merged_nested_chain_matches_ondsel() {
    let case = cases::all()
        .into_iter()
        .find(|c| c.name == "nested_chain")
        .unwrap();
    let merged = merge_fixed_groups(&case.system).unwrap();
    assert_eq!(merged.system.bodies.len() + 1, case.system.bodies.len());
    assert_eq!(merged.merged_joints, vec![JointId(2)]);
    let reference = load(case.name);
    let reduced = kinematic_sweep(&merged.system, &reference.times).unwrap();
    let full = kinematic_sweep(&case.system, &reference.times[..1]).unwrap();
    let d_full = diagnose(&case.system, &full[0].state, 0.0).unwrap();
    let d_merged = diagnose(&merged.system, &reduced[0].state, 0.0).unwrap();
    assert_eq!(d_full.coordinates - d_merged.coordinates, 6);
    assert_eq!(d_merged.dof, 0);
    let expanded: Vec<KinematicFrame> = reduced
        .iter()
        .map(|f| merged.expand(&case.system, f))
        .collect();
    check(
        "nested_chain (merged)",
        worst_deviation(&case.system, &expanded, &reference),
    );
}
