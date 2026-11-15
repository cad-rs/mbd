//! M4: the study API and result tables — the kernel-neutral integration
//! surface (data contracts, one entry point, CSV export).

mod common;

use common::*;
use glam::{DQuat, DVec3};
use rcad_mbd::*;

/// The benchmark pendulum at 0.6 rad with a marker at the tip.
fn swing() -> System {
    let mut sys = pendulum(0.6);
    marker(&mut sys, 3, 2, DVec3::new(0.0, -0.9, 0.0), DQuat::IDENTITY);
    sys
}

fn angle_measure() -> MeasureRequest {
    MeasureRequest::JointCoordinate {
        name: "hinge".into(),
        joint: JointId(1),
        slot: DriverSlot::Rotation,
    }
}

/// The study document is a stable contract: the JSON form is pinned, it
/// round-trips, and a sparse document fills in the integrator defaults.
#[test]
fn study_documents_are_stable() {
    let study = Study {
        name: "swing".into(),
        analysis: Analysis::Dynamic {
            integrator: Integrator {
                error_tol: Some(1e-7),
                ..Integrator::default()
            },
        },
        output: TimeGrid::Uniform {
            start: 0.0,
            end: 1.0,
            step: 0.01,
        },
        measures: vec![
            angle_measure(),
            MeasureRequest::Point {
                name: "tip".into(),
                body: BodyId(2),
                local: DVec3::new(0.0, -0.9, 0.0),
            },
        ],
        reactions: true,
    };
    let json = serde_json::to_string(&study).unwrap();
    assert_eq!(
        json,
        r#"{"name":"swing","analysis":{"kind":"dynamic","integrator":{"alpha":-0.3,"step":0.001,"error_tol":1e-7,"min_step":1e-9,"max_step":null,"max_newton":12,"newton_tol":1e-9}},"output":{"grid":"uniform","start":0.0,"end":1.0,"step":0.01},"measures":[{"measure":"joint_coordinate","name":"hinge","joint":1,"slot":"rotation"},{"measure":"point","name":"tip","body":2,"local":[0.0,-0.9,0.0]}],"reactions":true}"#
    );
    assert_eq!(serde_json::from_str::<Study>(&json).unwrap(), study);
    let sparse: Study = serde_json::from_str(
        r#"{"name":"s","analysis":{"kind":"dynamic"},"output":{"grid":"times","times":[0.0,0.5]}}"#,
    )
    .unwrap();
    assert_eq!(
        sparse.analysis,
        Analysis::Dynamic {
            integrator: Integrator::default()
        }
    );
    assert!(sparse.reactions && sparse.measures.is_empty());
}

/// A kinematic study of the driven pendulum reports the motor effort
/// (inverse dynamics, `I_o·θ̈ + m·g·l·sin θ`) and the hinge angle measure
/// on every frame.
#[test]
fn kinematic_study_reports_efforts() {
    let mut sys = swing();
    sys.add_driver(Driver {
        joint: JointId(1),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Harmonic {
            offset: 0.2,
            amplitude: 0.7,
            frequency_hz: 0.8,
            phase: 0.0,
        },
    });
    let study = Study {
        name: "driven".into(),
        analysis: Analysis::Kinematic,
        output: TimeGrid::Uniform {
            start: 0.0,
            end: 1.0,
            step: 0.1,
        },
        measures: vec![angle_measure()],
        reactions: true,
    };
    let result = run_study(&sys, &study).unwrap();
    assert_eq!(result.frames.len(), 11);
    for frame in &result.frames {
        let s = sys.drivers[0].sample(frame.time);
        let motor = frame
            .reactions
            .iter()
            .find(|r| r.owner == EquationOwner::Driver(0))
            .unwrap();
        close(
            motor.effort.unwrap(),
            i_hinge() * s.acceleration + M * G * L * s.value.sin(),
            1e-9,
            "effort",
        );
        match &frame.measures[0] {
            MeasureValue::JointCoordinate { motion, .. } => {
                close(motion.value, s.value, 1e-10, "angle")
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}

/// A dynamic study is `simulate` plus the tables: the same trajectory, the
/// reactions and audits on every frame, a JSON round trip and a CSV table
/// with one row per output time.
#[test]
fn dynamic_study_tables() {
    let sys = swing();
    let study = Study {
        name: "swing".into(),
        analysis: Analysis::Dynamic {
            integrator: Integrator::default(),
        },
        output: TimeGrid::Uniform {
            start: 0.0,
            end: 0.5,
            step: 0.05,
        },
        measures: vec![
            angle_measure(),
            MeasureRequest::Marker {
                name: "tip".into(),
                marker: MarkerId(3),
            },
        ],
        reactions: true,
    };
    let result = run_study(&sys, &study).unwrap();
    let times: Vec<f64> = (0..=10).map(|k| k as f64 * 0.05).collect();
    let sim = simulate(&sys, &times, &HhtSettings::default()).unwrap();
    assert_eq!(result.frames.len(), sim.frames.len());
    for (r, f) in result.frames.iter().zip(&sim.frames) {
        assert_eq!(
            r.bodies[&BodyId(2)].position,
            f.state.body_pose(BodyId(2)).unwrap().0
        );
        assert_eq!(r.reactions.len(), 1);
        assert!(r.reactions[0].unique);
    }
    assert_eq!(result.stats, Some(sim.stats));
    let e0 = result.frames[0].energy;
    let e1 = result.frames.last().unwrap().energy;
    close(
        e1.kinetic + e1.gravity,
        e0.kinetic + e0.gravity,
        1e-5,
        "energy audit",
    );
    let json = serde_json::to_string(&result).unwrap();
    assert_eq!(serde_json::from_str::<StudyResult>(&json).unwrap(), result);
    let csv = result.to_csv();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 12, "header + 11 rows");
    let header: Vec<&str> = lines[0].split(',').collect();
    for column in [
        "time",
        "b2.x",
        "b2.qw",
        "b2.alz",
        "j1.fx",
        "j1.tz",
        "hinge.q",
        "tip.wz",
        "energy.kinetic",
        "momentum.hz",
    ] {
        assert!(header.contains(&column), "missing column {column}");
    }
    assert!(lines.iter().all(|l| l.split(',').count() == header.len()));
}

/// A static study is a quasi-static sweep: the motor slowly lifting the
/// pendulum holds it with `m·g·l·sin θ(t)` at every output time.
#[test]
fn static_study_is_quasi_static() {
    let mut sys = swing();
    sys.add_driver(Driver {
        joint: JointId(1),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Constant {
            start: 0.0,
            rate: 0.5,
        },
    });
    let study = Study {
        name: "lift".into(),
        analysis: Analysis::Static,
        output: TimeGrid::Times {
            times: vec![0.0, 0.5, 1.0, 2.0, 3.0],
        },
        measures: Vec::new(),
        reactions: true,
    };
    let result = run_study(&sys, &study).unwrap();
    for frame in &result.frames {
        let theta = 0.5 * frame.time;
        let motor = frame
            .reactions
            .iter()
            .find(|r| r.owner == EquationOwner::Driver(0))
            .unwrap();
        close(
            motor.effort.unwrap(),
            M * G * L * theta.sin(),
            1e-9,
            "holding torque",
        );
        assert_eq!(frame.bodies[&BodyId(2)].velocity, Twist::default());
    }
}

/// Unusable studies are data.
#[test]
fn bad_studies_are_refused() {
    let sys = swing();
    let base = Study {
        name: "x".into(),
        analysis: Analysis::Dynamic {
            integrator: Integrator::default(),
        },
        output: TimeGrid::Uniform {
            start: 0.0,
            end: 1.0,
            step: 0.0,
        },
        measures: Vec::new(),
        reactions: false,
    };
    assert_eq!(run_study(&sys, &base).unwrap_err().code, codes::BAD_STUDY);
    let missing = Study {
        output: TimeGrid::Times { times: vec![0.0] },
        measures: vec![MeasureRequest::Marker {
            name: "ghost".into(),
            marker: MarkerId(42),
        }],
        ..base.clone()
    };
    assert_eq!(
        run_study(&sys, &missing).unwrap_err().code,
        codes::BAD_STUDY
    );
    let wrong_slot = Study {
        output: TimeGrid::Times { times: vec![0.0] },
        measures: vec![MeasureRequest::JointCoordinate {
            name: "slide".into(),
            joint: JointId(1),
            slot: DriverSlot::Translation,
        }],
        ..base
    };
    assert_eq!(
        run_study(&sys, &wrong_slot).unwrap_err().code,
        codes::BAD_STUDY
    );
}
