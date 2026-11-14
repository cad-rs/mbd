//! Tier B cases (roadmap §6): mechanisms exported to `.asmt`, solved by an
//! independent OndselSolver build, and compared against `kinematic_sweep`.
//! Shared by `examples/tier_b.rs` (export / reference conversion) and
//! `tests/tier_b.rs` (comparison), so both sides see the same system.
//!
//! Chosen to cover what Tier A does not: an offset slider, a spatial loop
//! (non-parallel pivot axes, spherical + universal), and a nested open chain
//! with two drivers on one cylindrical joint, a fixed joint and a prismatic
//! joint sliding in a rotating frame (Coriolis terms).

#![allow(dead_code)]

use glam::{DMat3, DQuat, DVec3};
use rcad_mbd::*;

pub struct Case {
    pub name: &'static str,
    pub system: System,
    pub sim: AsmtSimulation,
}

pub fn all() -> Vec<Case> {
    vec![offset_crank_slider(), spatial_rsur(), nested_chain()]
}

fn body(id: u64, at: DVec3, rotation: DQuat, grounded: bool) -> Body {
    Body {
        id: BodyId(id),
        name: format!("b{id}"),
        mass_kg: if grounded { 0.0 } else { 1.0 + 0.1 * id as f64 },
        com: if grounded {
            DVec3::ZERO
        } else {
            DVec3::new(0.01, 0.0, 0.0)
        },
        inertia_com: if grounded {
            DMat3::ZERO
        } else {
            DMat3::from_diagonal(DVec3::new(0.01, 0.02, 0.03))
        },
        translation: at,
        rotation,
        grounded,
    }
}

fn marker(sys: &mut System, id: u64, body: u64, at: DVec3, rotation: DQuat) {
    sys.add_marker(Marker {
        id: MarkerId(id),
        name: format!("m{id}"),
        body: BodyId(body),
        translation: at,
        rotation,
    })
    .unwrap();
}

fn joint(sys: &mut System, id: u64, kind: JointKind, m1: u64, m2: u64) {
    sys.add_joint(Joint {
        id: JointId(id),
        name: format!("j{id}"),
        kind,
        marker1: MarkerId(m1),
        marker2: MarkerId(m2),
        suppressed: false,
    })
    .unwrap();
}

fn drive(sys: &mut System, joint: u64, slot: DriverSlot, profile: DriverProfile) {
    sys.add_driver(Driver {
        joint: JointId(joint),
        slot,
        profile,
    });
}

/// Z → X (a joint axis along world/body X).
fn z_to_x() -> DQuat {
    DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2)
}

const SIM: AsmtSimulation = AsmtSimulation {
    tstart: 0.0,
    tend: 2.0,
    hout: 0.02,
};

/// Crank `r` at the origin, rod `l`, slider on the line `y = e` (prismatic
/// along world X); the motor turns one revolution per second.
fn offset_crank_slider() -> Case {
    let (r, l, e): (f64, f64, f64) = (0.05, 0.2, 0.02);
    let tip = DVec3::new(r, 0.0, 0.0);
    let slider = DVec3::new(r + (l * l - e * e).sqrt(), e, 0.0);
    let mut sys = System::new("offset-crank-slider");
    let id = DQuat::IDENTITY;
    sys.add_body(body(1, DVec3::ZERO, id, true)).unwrap();
    sys.add_body(body(2, DVec3::ZERO, id, false)).unwrap();
    sys.add_body(body(3, tip, id, false)).unwrap();
    sys.add_body(body(4, slider, id, false)).unwrap();
    marker(&mut sys, 1, 1, DVec3::ZERO, id);
    marker(&mut sys, 2, 1, DVec3::new(0.0, e, 0.0), z_to_x());
    marker(&mut sys, 3, 2, DVec3::ZERO, id);
    marker(&mut sys, 4, 2, tip, id);
    marker(&mut sys, 5, 3, DVec3::ZERO, id);
    marker(&mut sys, 6, 3, slider - tip, id);
    marker(&mut sys, 7, 4, DVec3::ZERO, id);
    marker(&mut sys, 8, 4, DVec3::ZERO, z_to_x());
    joint(&mut sys, 1, JointKind::Revolute, 1, 3);
    joint(&mut sys, 2, JointKind::Revolute, 4, 5);
    joint(&mut sys, 3, JointKind::Revolute, 6, 7);
    joint(&mut sys, 4, JointKind::Prismatic, 2, 8);
    drive(
        &mut sys,
        1,
        DriverSlot::Rotation,
        DriverProfile::Constant {
            start: 0.0,
            rate: std::f64::consts::TAU,
        },
    );
    Case {
        name: "offset_crank_slider",
        system: sys,
        sim: SIM,
    }
}

/// Spatial RSUR: crank about world Z at the origin, a coupler on a
/// spherical joint at the crank tip and a universal joint at the rocker,
/// the rocker on a revolute about world X through `P`. One DOF, no
/// redundancy, non-parallel pivot axes.
fn spatial_rsur() -> Case {
    let (a, c): (f64, f64) = (0.04, 0.1);
    let p = DVec3::new(0.1, 0.15, 0.0);
    let b2 = a * a + p.length_squared() + c * c; // b² = mean |B−P|² + c²
    let b0 = DVec3::new(a, 0.0, 0.0);
    // Rocker angle ψ0 at crank angle 0: (B−P)·u(ψ) = K, u = (0, cosψ, sinψ).
    let k = ((b0 - p).length_squared() + c * c - b2) / (2.0 * c);
    let psi0 = (k / (b0.y - p.y)).acos();
    let c0 = p + c * DVec3::new(0.0, psi0.cos(), psi0.sin());
    let along = (c0 - b0).normalize();
    let cross_pin = DVec3::X.cross(along).normalize();
    let mut sys = System::new("spatial-rsur");
    let id = DQuat::IDENTITY;
    sys.add_body(body(1, DVec3::ZERO, id, true)).unwrap();
    sys.add_body(body(2, DVec3::ZERO, id, false)).unwrap();
    sys.add_body(body(3, b0, id, false)).unwrap();
    sys.add_body(body(4, p, id, false)).unwrap();
    marker(&mut sys, 1, 1, DVec3::ZERO, id);
    marker(&mut sys, 2, 2, DVec3::ZERO, id);
    marker(&mut sys, 3, 2, b0, id);
    marker(&mut sys, 4, 3, DVec3::ZERO, id);
    marker(
        &mut sys,
        5,
        3,
        c0 - b0,
        DQuat::from_rotation_arc(DVec3::Z, cross_pin),
    );
    marker(&mut sys, 6, 4, c0 - p, z_to_x());
    marker(&mut sys, 7, 4, DVec3::ZERO, z_to_x());
    marker(&mut sys, 8, 1, p, z_to_x());
    joint(&mut sys, 1, JointKind::Revolute, 1, 2);
    joint(&mut sys, 2, JointKind::Spherical, 3, 4);
    joint(&mut sys, 3, JointKind::Universal, 5, 6);
    joint(&mut sys, 4, JointKind::Revolute, 8, 7);
    drive(
        &mut sys,
        1,
        DriverSlot::Rotation,
        DriverProfile::Constant {
            start: 0.0,
            rate: std::f64::consts::TAU,
        },
    );
    Case {
        name: "spatial_rsur",
        system: sys,
        sim: SIM,
    }
}

/// Nested open chain: a carriage on a cylindrical joint along world Z
/// (rotation at a constant rate, harmonic stroke), an arm fixed to it at an
/// offset and a tilt, and a slider on a prismatic joint along the arm's X
/// axis with a harmonic stroke — sliding in a rotating frame.
fn nested_chain() -> Case {
    let arm_at = DVec3::new(0.2, 0.0, 0.1);
    let arm_rot = DQuat::from_rotation_x(0.3);
    let mut sys = System::new("nested-chain");
    let id = DQuat::IDENTITY;
    sys.add_body(body(1, DVec3::ZERO, id, true)).unwrap();
    sys.add_body(body(2, DVec3::ZERO, id, false)).unwrap();
    sys.add_body(body(3, arm_at, arm_rot, false)).unwrap();
    sys.add_body(body(4, arm_at, arm_rot, false)).unwrap();
    marker(&mut sys, 1, 1, DVec3::ZERO, id);
    marker(&mut sys, 2, 2, DVec3::ZERO, id);
    marker(&mut sys, 3, 2, arm_at, arm_rot);
    marker(&mut sys, 4, 3, DVec3::ZERO, id);
    marker(&mut sys, 5, 3, DVec3::ZERO, z_to_x());
    marker(&mut sys, 6, 4, DVec3::ZERO, z_to_x());
    joint(&mut sys, 1, JointKind::Cylindrical, 1, 2);
    joint(&mut sys, 2, JointKind::Fixed, 3, 4);
    joint(&mut sys, 3, JointKind::Prismatic, 5, 6);
    drive(
        &mut sys,
        1,
        DriverSlot::Rotation,
        DriverProfile::Constant {
            start: 0.0,
            rate: 1.5,
        },
    );
    drive(
        &mut sys,
        1,
        DriverSlot::Translation,
        DriverProfile::Harmonic {
            offset: 0.0,
            amplitude: 0.05,
            frequency_hz: 0.5,
            phase: 0.0,
        },
    );
    drive(
        &mut sys,
        3,
        DriverSlot::Translation,
        DriverProfile::Harmonic {
            offset: 0.05,
            amplitude: 0.1,
            frequency_hz: 1.0,
            phase: 0.4,
        },
    );
    Case {
        name: "nested_chain",
        system: sys,
        sim: SIM,
    }
}

/// The output times of a case (`tstart + k·hout` up to `tend`).
pub fn times(sim: &AsmtSimulation) -> Vec<f64> {
    let n = ((sim.tend - sim.tstart) / sim.hout).round() as usize;
    (0..=n).map(|k| sim.tstart + k as f64 * sim.hout).collect()
}

/// One reference frame of one part: the 18 per-part series OndselSolver
/// writes, in its order — position (3), Bryant angles x-y-z (3), velocity
/// (3), angular velocity (3), acceleration (3), angular acceleration (3);
/// all world frame, of the part frame origin.
pub type PartSample = [f64; 18];

/// The series labels OndselSolver writes per part, in [`PartSample`] order.
pub const PART_SERIES: [&str; 18] = [
    "X", "Y", "Z", "Bryantx", "Bryanty", "Bryantz", "VX", "VY", "VZ", "OmegaX", "OmegaY", "OmegaZ",
    "AX", "AY", "AZ", "AlphaX", "AlphaY", "AlphaZ",
];

/// A reference trajectory: times plus, per part name (`b{id}`), one sample
/// per time.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Reference {
    pub source: String,
    pub times: Vec<f64>,
    pub parts: std::collections::BTreeMap<String, Vec<Vec<f64>>>,
}
