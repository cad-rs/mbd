//! Tier C cases (roadmap §6): systems exported to MJCF, simulated by
//! MuJoCo (minimal coordinates, RK4), compared against `simulate` at
//! tolerance level. Shared by `examples/tier_c.rs` (export) and
//! `tests/tier_c.rs` (comparison).

#![allow(dead_code)]

use glam::{DMat3, DQuat, DVec3};
use rcad_mbd::*;

pub struct Case {
    pub name: &'static str,
    pub system: System,
    pub end: f64,
    pub hout: f64,
    /// Worst deviation allowed: position (m), rotation (rad) and energy
    /// (J). The energy limit is the same for every case whose only
    /// difference is integration error; a case whose two engines model the
    /// same physics differently says so where it is set.
    pub limits: (f64, f64, f64),
}

pub fn all() -> Vec<Case> {
    vec![
        double_pendulum(),
        spatial_chain(),
        tumbler(),
        cart_pendulum(),
        four_bar(),
        rubbing_pendulum(),
    ]
}

fn ground(sys: &mut System) {
    sys.add_body(Body {
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
    })
    .unwrap();
}

fn link(
    sys: &mut System,
    id: u64,
    mass: f64,
    com: DVec3,
    inertia: DVec3,
    at: DVec3,
    rotation: DQuat,
) {
    sys.add_body(Body {
        id: BodyId(id),
        name: format!("b{id}"),
        mass_kg: mass,
        com,
        inertia_com: DMat3::from_diagonal(inertia),
        translation: at,
        rotation,
        grounded: false,
        velocity: DVec3::ZERO,
        angular_velocity: DVec3::ZERO,
    })
    .unwrap();
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

fn gravity(sys: &mut System, g: DVec3) {
    sys.gravity = Some(g * STANDARD_GRAVITY);
}

/// Two links on parallel hinges (world Z), released from 0.9 / −0.4 rad.
fn double_pendulum() -> Case {
    let mut sys = System::new("double pendulum");
    gravity(&mut sys, -DVec3::Y);
    ground(&mut sys);
    let (r1, r2) = (DQuat::from_rotation_z(0.9), DQuat::from_rotation_z(0.5));
    let tip = r1 * DVec3::new(0.0, -0.4, 0.0);
    link(
        &mut sys,
        2,
        1.2,
        DVec3::new(0.0, -0.2, 0.0),
        DVec3::new(0.02, 0.002, 0.02),
        DVec3::ZERO,
        r1,
    );
    link(
        &mut sys,
        3,
        0.8,
        DVec3::new(0.0, -0.15, 0.0),
        DVec3::new(0.01, 0.001, 0.01),
        tip,
        r2,
    );
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 3, 2, DVec3::new(0.0, -0.4, 0.0), DQuat::IDENTITY);
    marker(&mut sys, 4, 3, DVec3::ZERO, DQuat::IDENTITY);
    joint(&mut sys, 1, JointKind::Revolute, 1, 2);
    joint(&mut sys, 2, JointKind::Revolute, 3, 4);
    Case {
        name: "double_pendulum",
        system: sys,
        end: 2.0,
        hout: 0.01,
        limits: (1e-6, 5e-6, 1e-5),
    }
}

/// A spatial chain: a ball joint at the ground, then hinges about the
/// links' X and a tilted axis; asymmetric inertia, offset centres of mass,
/// gravity −Z.
fn spatial_chain() -> Case {
    let mut sys = System::new("spatial chain");
    gravity(&mut sys, -DVec3::Z);
    ground(&mut sys);
    let r1 = DQuat::from_euler(glam::EulerRot::XYZ, 0.5, 0.2, 0.0);
    let p2 = r1 * DVec3::new(0.3, 0.0, 0.0);
    // Joint 2 turns link 2 about link 1's X, joint 3 link 3 about a tilted
    // axis of link 2.
    let tilt_axis = DVec3::new(0.0, 0.6, 0.8);
    let r2 = r1 * DQuat::from_rotation_x(-0.4);
    let p3 = p2 + r2 * DVec3::new(0.25, 0.0, 0.0);
    let r3 = r2 * DQuat::from_axis_angle(tilt_axis, 0.7);
    link(
        &mut sys,
        2,
        1.0,
        DVec3::new(0.15, 0.01, 0.0),
        DVec3::new(0.001, 0.008, 0.009),
        DVec3::ZERO,
        r1,
    );
    link(
        &mut sys,
        3,
        0.7,
        DVec3::new(0.12, 0.0, 0.02),
        DVec3::new(0.0008, 0.005, 0.0055),
        p2,
        r2,
    );
    link(
        &mut sys,
        4,
        0.5,
        DVec3::new(0.1, -0.02, 0.0),
        DVec3::new(0.0005, 0.003, 0.0032),
        p3,
        r3,
    );
    let tilted = DQuat::from_rotation_arc(DVec3::Z, tilt_axis);
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    let z_to_x = DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2);
    marker(&mut sys, 3, 2, DVec3::new(0.3, 0.0, 0.0), z_to_x);
    marker(
        &mut sys,
        4,
        3,
        DVec3::ZERO,
        DQuat::from_rotation_x(0.4) * z_to_x,
    );
    marker(&mut sys, 5, 3, DVec3::new(0.25, 0.0, 0.0), tilted);
    marker(
        &mut sys,
        6,
        4,
        DVec3::ZERO,
        DQuat::from_axis_angle(tilt_axis, -0.7) * tilted,
    );
    joint(&mut sys, 1, JointKind::Spherical, 1, 2);
    joint(&mut sys, 2, JointKind::Revolute, 3, 4);
    joint(&mut sys, 3, JointKind::Revolute, 5, 6);
    Case {
        name: "spatial_chain",
        system: sys,
        end: 1.5,
        hout: 0.01,
        limits: (1e-5, 5e-4, 1e-5),
    }
}

/// A free asymmetric body spun near its intermediate axis (the tumbling
/// Dzhanibekov case) with an offset centre of mass, no gravity.
fn tumbler() -> Case {
    let mut sys = System::new("tumbler");
    sys.add_body(Body {
        id: BodyId(2),
        name: "b2".into(),
        mass_kg: 2.0,
        com: DVec3::new(0.05, -0.02, 0.01),
        inertia_com: DMat3::from_diagonal(DVec3::new(0.01, 0.02, 0.03)),
        translation: DVec3::new(0.1, 0.2, 0.3),
        rotation: DQuat::from_rotation_x(0.3),
        grounded: false,
        velocity: DVec3::new(0.2, 0.0, -0.1),
        angular_velocity: DVec3::new(0.05, 6.0, 0.02),
    })
    .unwrap();
    Case {
        name: "tumbler",
        system: sys,
        end: 3.0,
        hout: 0.01,
        limits: (1e-5, 1e-5, 1e-5),
    }
}

/// A cart on a slide (world X) held by a joint spring-damper, a pendulum
/// hinged on it, a tendon spring from the ground to the pendulum tip.
fn cart_pendulum() -> Case {
    let mut sys = System::new("cart pendulum");
    gravity(&mut sys, -DVec3::Y);
    ground(&mut sys);
    let along_x = DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2);
    let r = DQuat::from_rotation_z(0.6);
    link(
        &mut sys,
        2,
        3.0,
        DVec3::ZERO,
        DVec3::new(0.02, 0.03, 0.02),
        DVec3::new(0.05, 0.0, 0.0),
        DQuat::IDENTITY,
    );
    link(
        &mut sys,
        3,
        0.8,
        DVec3::new(0.0, -0.25, 0.0),
        DVec3::new(0.02, 0.001, 0.02),
        DVec3::new(0.05, 0.0, 0.0),
        r,
    );
    marker(&mut sys, 1, 1, DVec3::ZERO, along_x);
    marker(&mut sys, 2, 2, DVec3::new(-0.05, 0.0, 0.0), along_x);
    marker(&mut sys, 3, 2, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 4, 3, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 5, 1, DVec3::new(-0.4, -0.3, 0.0), DQuat::IDENTITY);
    marker(&mut sys, 6, 3, DVec3::new(0.0, -0.5, 0.0), DQuat::IDENTITY);
    joint(&mut sys, 1, JointKind::Prismatic, 1, 2);
    joint(&mut sys, 2, JointKind::Revolute, 3, 4);
    sys.add_force(ForceElement::JointSpringDamper {
        id: ForceId(1),
        name: "cart spring".into(),
        joint: JointId(1),
        slot: DriverSlot::Translation,
        stiffness: 120.0,
        damping: 1.5,
        free_value: 0.0,
        preload: 0.0,
    })
    .unwrap();
    sys.add_force(ForceElement::SpringDamper {
        id: ForceId(2),
        name: "tip spring".into(),
        marker1: MarkerId(5),
        marker2: MarkerId(6),
        stiffness: 40.0,
        damping: 0.3,
        free_length: 0.3,
        preload: 0.0,
    })
    .unwrap();
    Case {
        name: "cart_pendulum",
        system: sys,
        end: 2.0,
        hout: 0.01,
        limits: (1e-5, 1e-5, 1e-5),
    }
}

/// The four-bar's crank, coupler, rocker and ground lengths.
pub const FOUR_BAR: (f64, f64, f64, f64) = (0.1, 0.3, 0.25, 0.3);

/// A planar four-bar of revolutes about world Z swinging under gravity;
/// MuJoCo closes the loop with a (soft) connect equality.
fn four_bar() -> Case {
    let (a, b, c, d) = FOUR_BAR;
    let mut sys = System::new("four-bar");
    gravity(&mut sys, -DVec3::Y);
    ground(&mut sys);
    let bar = |sys: &mut System, id, len: f64, at: DVec3, angle: f64| {
        link(
            sys,
            id,
            0.5 + len,
            DVec3::new(len / 2.0, 0.0, 0.0),
            DVec3::new(1e-4, 0.004, 0.004),
            at,
            DQuat::from_rotation_z(angle),
        )
    };
    bar(&mut sys, 2, a, DVec3::ZERO, std::f64::consts::FRAC_PI_2);
    bar(&mut sys, 3, b, DVec3::new(0.0, a, 0.0), 0.5455);
    bar(&mut sys, 4, c, DVec3::new(d, 0.0, 0.0), 1.7239);
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 1, DVec3::new(d, 0.0, 0.0), DQuat::IDENTITY);
    marker(&mut sys, 3, 2, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 4, 2, DVec3::new(a, 0.0, 0.0), DQuat::IDENTITY);
    marker(&mut sys, 5, 3, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 6, 3, DVec3::new(b, 0.0, 0.0), DQuat::IDENTITY);
    marker(&mut sys, 7, 4, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 8, 4, DVec3::new(c, 0.0, 0.0), DQuat::IDENTITY);
    joint(&mut sys, 1, JointKind::Revolute, 1, 3);
    joint(&mut sys, 2, JointKind::Revolute, 4, 5);
    joint(&mut sys, 4, JointKind::Revolute, 2, 7);
    joint(&mut sys, 3, JointKind::Revolute, 6, 8);
    Case {
        name: "four_bar",
        system: sys,
        end: 1.0,
        hout: 0.01,
        limits: (2e-6, 2e-5, 1e-5),
    }
}

/// A pendulum whose hinge rubs with a constant friction torque (M5-1,
/// every arm zero so the load is the preload alone) — the one friction
/// model MuJoCo also has, as a joint's `frictionloss`. The two
/// regularizations differ (a solver-side Coulomb constraint against this
/// crate's Stribeck curve of the slip), so the agreement is the loosest of
/// the cases: it checks the torque's size and sign through a reversal, not
/// the regularization.
fn rubbing_pendulum() -> Case {
    let mut sys = System::new("rubbing pendulum");
    gravity(&mut sys, -DVec3::Y);
    ground(&mut sys);
    link(
        &mut sys,
        2,
        1.0,
        DVec3::new(0.0, -0.3, 0.0),
        DVec3::new(0.02, 0.002, 0.02),
        DVec3::ZERO,
        DQuat::from_rotation_z(1.0),
    );
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    joint(&mut sys, 1, JointKind::Revolute, 1, 2);
    sys.add_force(ForceElement::JointFriction {
        id: ForceId(1),
        name: "hinge friction".into(),
        joint: JointId(1),
        mu_static: 0.5,
        mu_dynamic: 0.5,
        stiction_velocity: 1.0e-4,
        friction_velocity: 1.0e-4,
        reaction_force: false,
        pin_radius: 0.0,
        thrust_arm: 0.0,
        bending_arm: 0.0,
        preload: 0.2,
        max_stiction_deformation: 1.0e-4,
    })
    .unwrap();
    Case {
        name: "rubbing_pendulum",
        system: sys,
        end: 1.0,
        hout: 0.01,
        // The friction regularizations differ, so this case is limited by
        // the models, not by integration error: observed 3.6e-3 rad and
        // 5.4e-4 J of the 0.3 J the friction takes out over the second.
        limits: (1e-6, 1e-2, 2e-3),
    }
}

/// A MuJoCo reference: per body `b{id}`, per output time
/// `[x, y, z, qw, qx, qy, qz]` of the body frame; per time
/// `[potential, kinetic]` energy.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Reference {
    pub source: String,
    pub times: Vec<f64>,
    pub bodies: std::collections::BTreeMap<String, Vec<Vec<f64>>>,
    pub energy: Vec<Vec<f64>>,
}
