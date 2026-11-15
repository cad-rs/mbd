//! Shared builders and checks of the Tier A benchmarks.

#![allow(dead_code)]

use glam::{DMat3, DQuat, DVec3};
use rcad_mbd::*;

pub const G: f64 = STANDARD_GRAVITY;

/// The benchmark pendulum: mass, hinge-to-center-of-mass length, inertia
/// about the hinge axis at the center of mass.
pub const M: f64 = 2.0;
pub const L: f64 = 0.6;
pub const IZZ: f64 = 0.03;

/// The pendulum's inertia about its hinge.
pub fn i_hinge() -> f64 {
    IZZ + M * L * L
}

/// Body 1, grounded at the origin.
pub fn ground() -> Body {
    Body {
        id: BodyId(1),
        name: "ground".into(),
        mass_kg: 0.0,
        com: DVec3::ZERO,
        inertia_com: DMat3::ZERO,
        translation: DVec3::ZERO,
        rotation: DQuat::IDENTITY,
        grounded: true,
        velocity: DVec3::ZERO,
        angular_velocity: DVec3::ZERO,
    }
}

/// A free body at rest (principal inertia `inertia` about the center of
/// mass).
pub fn part(id: u64, mass: f64, com: DVec3, inertia: DVec3, at: DVec3, rotation: DQuat) -> Body {
    Body {
        id: BodyId(id),
        name: format!("part{id}"),
        mass_kg: mass,
        com,
        inertia_com: DMat3::from_diagonal(inertia),
        translation: at,
        rotation,
        grounded: false,
        velocity: DVec3::ZERO,
        angular_velocity: DVec3::ZERO,
    }
}

/// Body 2 with the pendulum's mass properties.
pub fn link(com: DVec3, at: DVec3, rotation: DQuat) -> Body {
    part(2, M, com, DVec3::new(0.04, 0.01, IZZ), at, rotation)
}

pub fn marker(sys: &mut System, id: u64, body: u64, at: DVec3, rotation: DQuat) {
    sys.add_marker(Marker {
        id: MarkerId(id),
        name: format!("m{id}"),
        body: BodyId(body),
        translation: at,
        rotation,
    })
    .unwrap();
}

pub fn joint(sys: &mut System, id: u64, kind: JointKind, m1: u64, m2: u64) {
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

/// A compound pendulum: hinge (revolute 1 about world Z, markers 1 / 2) at
/// the origin, center of mass `L` below the hinge in the body frame,
/// gravity −Y, the link authored at hinge angle `theta`.
pub fn pendulum(theta: f64) -> System {
    let mut sys = System::new("pendulum");
    sys.gravity = Some(DVec3::new(0.0, -G, 0.0));
    sys.add_body(ground()).unwrap();
    sys.add_body(link(
        DVec3::new(0.0, -L, 0.0),
        DVec3::ZERO,
        DQuat::from_rotation_z(theta),
    ))
    .unwrap();
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    joint(&mut sys, 1, JointKind::Revolute, 1, 2);
    sys
}

/// `0, step, …, end`.
pub fn grid(end: f64, step: f64) -> Vec<f64> {
    let n = (end / step).round() as usize;
    (0..=n).map(|k| k as f64 * step).collect()
}

pub fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

pub fn close_v(a: DVec3, b: DVec3, tol: f64, what: &str) {
    assert!(
        (a - b).length() <= tol * (1.0 + b.length()),
        "{what}: {a:?} vs {b:?}"
    );
}
