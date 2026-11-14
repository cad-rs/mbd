//! M4: the sample adapter (`examples/adapter.rs`) runs and its results are
//! right — the integrator's guide stays honest.

#[path = "../examples/adapter.rs"]
mod adapter;

use rcad_mbd::*;

/// The adapted crank-slider (mm / rpm on the CAD side) solves one
/// revolution: the stroke follows `r·cos θ + √(l² − r²·sin²θ)` in metres,
/// the planar loop of 3D joints reports its three redundant equations,
/// and every frame carries the four joint reactions and the motor effort.
#[test]
fn sample_adapter_runs_and_is_right() {
    let (sys, lookup, result) = adapter::run().expect("adapter study");
    assert_eq!(lookup.body_of_part.len(), 4);
    assert_eq!(sys.joints.len(), 4);
    assert_eq!(result.redundant.len(), 3);
    let (r, l) = (0.05_f64, 0.2_f64);
    let omega = std::f64::consts::TAU; // 60 rpm
    for frame in &result.frames {
        let theta = omega * frame.time;
        let x = r * theta.cos() + (l * l - (r * theta.sin()).powi(2)).sqrt();
        match &frame.measures[0] {
            MeasureValue::JointCoordinate { motion, .. } => {
                assert!(
                    (motion.value - (x - (r + l))).abs() < 1e-10,
                    "t = {}",
                    frame.time
                )
            }
            other => panic!("unexpected measure {other:?}"),
        }
        assert_eq!(frame.reactions.len(), 5);
        let motor = frame
            .reactions
            .iter()
            .find(|r| r.owner == EquationOwner::Driver(0))
            .unwrap();
        assert!(motor.effort.is_some_and(f64::is_finite));
    }
    let csv = result.to_csv();
    assert!(csv.starts_with("time,b1.x"));
    assert_eq!(csv.lines().count(), result.frames.len() + 1);
}
