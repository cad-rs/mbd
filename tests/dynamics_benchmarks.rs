//! Tier A dynamics benchmarks (roadmap §6) — closed-form checks of the
//! equations of motion, the dynamic initial condition and static
//! equilibrium. (Time integration benchmarks join with the integrator.)

use glam::{DMat3, DQuat, DVec3};
use rcad_mbd::*;

const G: f64 = STANDARD_GRAVITY;

fn ground() -> Body {
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

fn part(id: u64, mass: f64, com: DVec3, inertia: DVec3, at: DVec3, rotation: DQuat) -> Body {
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

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

fn close_v(a: DVec3, b: DVec3, tol: f64, what: &str) {
    assert!(
        (a - b).length() <= tol * (1.0 + b.length()),
        "{what}: {a:?} vs {b:?}"
    );
}

fn multiplier(frame: &[(EquationRef, f64)], owner: EquationOwner) -> Vec<f64> {
    frame
        .iter()
        .filter(|(r, _)| r.owner == owner)
        .map(|(_, l)| *l)
        .collect()
}

// ── free body ───────────────────────────────────────────────────────────

/// A free, asymmetric body with an offset center of mass, tumbling under
/// gravity: the center of mass falls at exactly `g` and the rotation obeys
/// the torque-free Euler equations `J_c·α = −ω×(J_c·ω)`. The authored
/// velocity of an unconstrained body is kept as is.
#[test]
fn free_body_falls_at_g_and_obeys_euler() {
    let mut sys = System::new("free").with_standard_gravity();
    let r = DQuat::from_euler(glam::EulerRot::XYZ, 0.3, -0.6, 1.2);
    let mut b = part(
        2,
        3.0,
        DVec3::new(0.2, -0.1, 0.4),
        DVec3::new(0.05, 0.11, 0.16),
        DVec3::new(1.0, 2.0, 3.0),
        r,
    );
    b.velocity = DVec3::new(0.5, -0.2, 1.0);
    b.angular_velocity = DVec3::new(2.0, -1.5, 3.0);
    sys.add_body(b.clone()).unwrap();
    let frame = dynamic_ic(&sys, 0.0).unwrap();
    let v = frame.velocity[&BodyId(2)];
    let a = frame.acceleration[&BodyId(2)];
    close_v(v.linear, b.velocity, 1e-12, "authored linear velocity kept");
    close_v(
        v.angular,
        b.angular_velocity,
        1e-12,
        "authored angular velocity kept",
    );
    let c = r * b.com;
    let w = v.angular;
    let a_com = a.linear + a.angular.cross(c) + w.cross(w.cross(c));
    close_v(
        a_com,
        DVec3::new(0.0, 0.0, -G),
        1e-10,
        "center of mass falls at g",
    );
    let rm = DMat3::from_quat(r);
    let jc = rm * b.inertia_com * rm.transpose();
    let alpha = -(jc.inverse() * w.cross(jc * w));
    close_v(a.angular, alpha, 1e-10, "Euler's equations");
    assert!(frame.multipliers.is_empty() && frame.redundant.is_empty());
}

// ── pendulum ────────────────────────────────────────────────────────────

const M: f64 = 2.0;
const L: f64 = 0.6;
const IZZ: f64 = 0.03;

/// A compound pendulum: hinge (revolute about world Z) at the origin,
/// center of mass `L` below the hinge in the body frame, gravity −Y, the
/// body authored at hinge angle `theta`.
fn pendulum(theta: f64) -> System {
    let mut sys = System::new("pendulum");
    sys.gravity = Some(DVec3::new(0.0, -G, 0.0));
    sys.add_body(ground()).unwrap();
    sys.add_body(part(
        2,
        M,
        DVec3::new(0.0, -L, 0.0),
        DVec3::new(0.04, 0.01, IZZ),
        DVec3::ZERO,
        DQuat::from_rotation_z(theta),
    ))
    .unwrap();
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    joint(&mut sys, 1, JointKind::Revolute, 1, 2);
    sys
}

fn i_hinge() -> f64 {
    IZZ + M * L * L
}

/// The pendulum's angular acceleration is `θ̈ = −(m·g·l / I_o)·sin θ` at
/// every angle and rate (the rate does not enter: the hinge carries the
/// centripetal load).
#[test]
fn pendulum_angular_acceleration() {
    for (theta, rate) in [(0.0, 0.0), (0.4, 0.0), (1.3, 2.0), (-2.5, -3.0), (3.0, 1.0)] {
        let mut sys = pendulum(theta);
        sys.bodies.get_mut(&BodyId(2)).unwrap().angular_velocity = DVec3::Z * rate;
        let frame = dynamic_ic(&sys, 0.0).unwrap();
        let a = frame.acceleration[&BodyId(2)];
        let expected = -(M * G * L / i_hinge()) * theta.sin();
        close(a.angular.z, expected, 1e-10, "θ̈");
        assert!(
            a.angular.truncate().length() < 1e-10,
            "no out-of-plane rotation"
        );
        assert!(a.linear.length() < 1e-10, "the hinge point stays put");
        close_v(
            frame.velocity[&BodyId(2)].angular,
            DVec3::Z * rate,
            1e-12,
            "consistent rate kept",
        );
    }
}

/// An inconsistent authored velocity (the hinge point moving, an
/// out-of-plane spin) is projected in the kinetic-energy norm: the hinge
/// point comes to rest and the angular momentum about the hinge axis is
/// kept, `ω = ω₀ + m·(c × v₀)_z / I_o`.
#[test]
fn velocity_ic_keeps_hinge_axis_momentum() {
    let theta = 0.7;
    let mut sys = pendulum(theta);
    let (w0, vx) = (1.5, 0.8);
    {
        let b = sys.bodies.get_mut(&BodyId(2)).unwrap();
        b.angular_velocity = DVec3::new(0.9, -0.4, w0);
        b.velocity = DVec3::new(vx, 0.0, 0.0);
    }
    let frame = dynamic_ic(&sys, 0.0).unwrap();
    let v = frame.velocity[&BodyId(2)];
    assert!(
        v.linear.length() < 1e-12,
        "hinge point at rest: {:?}",
        v.linear
    );
    assert!(
        v.angular.truncate().length() < 1e-12,
        "spin about the hinge axis only"
    );
    // c = R(θ)·(0, −L, 0) = (L sin θ, −L cos θ, 0); (c × v₀)_z = L cos θ · vx.
    let expected = w0 + M * L * theta.cos() * vx / i_hinge();
    close(v.angular.z, expected, 1e-12, "projected rate");
}

/// Two coaxial revolutes on one hinge: five redundant equations, the same
/// motion as a single revolute, the redundancy reported.
#[test]
fn redundant_hinge_has_the_same_dynamics() {
    let theta = 0.9;
    let mut sys = pendulum(theta);
    marker(&mut sys, 3, 1, DVec3::new(0.0, 0.0, 0.1), DQuat::IDENTITY);
    marker(&mut sys, 4, 2, DVec3::new(0.0, 0.0, 0.1), DQuat::IDENTITY);
    joint(&mut sys, 2, JointKind::Revolute, 3, 4);
    let frame = dynamic_ic(&sys, 0.0).unwrap();
    let expected = -(M * G * L / i_hinge()) * theta.sin();
    close(
        frame.acceleration[&BodyId(2)].angular.z,
        expected,
        1e-10,
        "θ̈",
    );
    assert_eq!(frame.redundant.len(), 5, "{:?}", frame.redundant);
}

/// A fully driven pendulum is inverse dynamics: forward dynamics returns
/// the kinematic acceleration, and the driver multiplier is the motor
/// torque `I_o·θ̈ + m·g·l·sin θ` (with the multiplier sign convention
/// `−Jᵀλ` = load on the bodies: λ = −torque on the link).
#[test]
fn driven_pendulum_is_inverse_dynamics() {
    let mut sys = pendulum(0.0);
    sys.add_driver(Driver {
        joint: JointId(1),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Harmonic {
            offset: 0.3,
            amplitude: 0.8,
            frequency_hz: 0.7,
            phase: 0.2,
        },
    });
    for t in [0.0, 0.31, 0.77, 1.4] {
        let kin = solve_frame(&sys, t, None).unwrap();
        let dynamic = forward_dynamics(&sys, t, &kin.state, &kin.velocity).unwrap();
        let s = sys.drivers[0].sample(t);
        close(
            dynamic.acceleration[&BodyId(2)].angular.z,
            s.acceleration,
            1e-9,
            "kinematic θ̈",
        );
        let torque = i_hinge() * s.acceleration + M * G * L * s.value.sin();
        let lambda = multiplier(&dynamic.multipliers, EquationOwner::Driver(0));
        close(lambda[0], -torque, 1e-9, "motor torque");
    }
}

// ── spring-mass-damper ──────────────────────────────────────────────────

const K: f64 = 400.0;
const FREE: f64 = 0.25;

/// A slider on a prismatic joint along world X, a spring-damper from the
/// ground origin to the slider origin, gravity −Y (carried by the joint).
fn spring_mass(x: f64, damping: f64) -> System {
    let mut sys = System::new("spring-mass");
    sys.gravity = Some(DVec3::new(0.0, -G, 0.0));
    sys.add_body(ground()).unwrap();
    sys.add_body(part(
        2,
        M,
        DVec3::new(0.0, 0.05, 0.0),
        DVec3::splat(0.01),
        DVec3::new(x, 0.0, 0.0),
        DQuat::IDENTITY,
    ))
    .unwrap();
    let along_x = DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2);
    marker(&mut sys, 1, 1, DVec3::ZERO, along_x);
    marker(&mut sys, 2, 2, DVec3::ZERO, along_x);
    joint(&mut sys, 1, JointKind::Prismatic, 1, 2);
    sys.add_force(ForceElement::SpringDamper {
        id: ForceId(1),
        name: "spring".into(),
        marker1: MarkerId(1),
        marker2: MarkerId(2),
        stiffness: K,
        damping,
        free_length: FREE,
    })
    .unwrap();
    sys
}

/// `m·ẍ = −k·(x − l₀) − c·ẋ` along the slide; nothing moves across it.
#[test]
fn spring_mass_damper_acceleration() {
    for (x, v, c) in [
        (0.3, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (0.4, 1.2, 15.0),
        (0.2, -0.7, 3.0),
    ] {
        let mut sys = spring_mass(x, c);
        sys.bodies.get_mut(&BodyId(2)).unwrap().velocity = DVec3::new(v, 0.0, 0.0);
        let frame = dynamic_ic(&sys, 0.0).unwrap();
        let a = frame.acceleration[&BodyId(2)];
        close(a.linear.x, -(K * (x - FREE) + c * v) / M, 1e-10, "ẍ");
        assert!(a.linear.y.abs() < 1e-10 && a.angular.length() < 1e-10);
    }
}

/// The energy audit: kinetic, gravity and spring potential at a known state.
#[test]
fn energy_and_momentum_audit() {
    let (x, v) = (0.35, 0.9);
    let mut sys = spring_mass(x, 0.0);
    sys.bodies.get_mut(&BodyId(2)).unwrap().velocity = DVec3::new(v, 0.0, 0.0);
    let frame = dynamic_ic(&sys, 0.0).unwrap();
    let e = energy(&sys, &frame.state, &frame.velocity).unwrap();
    close(e.kinetic, 0.5 * M * v * v, 1e-12, "kinetic");
    close(e.elastic, 0.5 * K * (x - FREE).powi(2), 1e-12, "elastic");
    close(
        e.gravity,
        M * G * 0.05,
        1e-12,
        "gravity (com 5 cm above the origin)",
    );
    let p = momentum(&sys, &frame.state, &frame.velocity).unwrap();
    close_v(
        p.linear,
        DVec3::new(M * v, 0.0, 0.0),
        1e-12,
        "linear momentum",
    );
    // About the world origin: r_com × m·v, r_com = (x, 0.05, 0).
    close_v(
        p.angular,
        DVec3::new(x, 0.05, 0.0).cross(DVec3::new(M * v, 0.0, 0.0)),
        1e-12,
        "angular momentum",
    );
}

// ── static equilibrium ──────────────────────────────────────────────────

/// A hanging spring-mass (slide along world Y, gravity −Y) settles at the
/// static extension `m·g/k` below its free length.
#[test]
fn static_spring_extension() {
    let mut sys = System::new("hanging");
    sys.gravity = Some(DVec3::new(0.0, -G, 0.0));
    sys.add_body(ground()).unwrap();
    sys.add_body(part(
        2,
        M,
        DVec3::ZERO,
        DVec3::splat(0.01),
        DVec3::new(0.0, -0.1, 0.0),
        DQuat::IDENTITY,
    ))
    .unwrap();
    let along_y = DQuat::from_rotation_x(-std::f64::consts::FRAC_PI_2); // Z → Y
    marker(&mut sys, 1, 1, DVec3::ZERO, along_y);
    marker(&mut sys, 2, 2, DVec3::ZERO, along_y);
    joint(&mut sys, 1, JointKind::Prismatic, 1, 2);
    sys.add_force(ForceElement::SpringDamper {
        id: ForceId(1),
        name: "spring".into(),
        marker1: MarkerId(1),
        marker2: MarkerId(2),
        stiffness: K,
        damping: 5.0,
        free_length: FREE,
    })
    .unwrap();
    let eq = static_equilibrium(&sys, 0.0, None).unwrap();
    let (t, _) = eq.state.body_pose(BodyId(2)).unwrap();
    close(t.y, -(FREE + M * G / K), 1e-10, "static extension");
    assert!(t.x.abs() < 1e-12 && t.z.abs() < 1e-12);
    assert!(eq.iterations >= 1);
}

/// A pendulum pushed sideways by a constant horizontal force at its tip
/// settles at `tan θ = F·L_tip / (m·g·l)`.
#[test]
fn static_pendulum_under_side_load() {
    let mut sys = pendulum(0.0);
    let (f, tip) = (7.0, 0.9);
    marker(&mut sys, 3, 2, DVec3::new(0.0, -tip, 0.0), DQuat::IDENTITY);
    sys.add_force(ForceElement::AppliedForce {
        id: ForceId(1),
        name: "push".into(),
        marker: MarkerId(3),
        force: DVec3::new(f, 0.0, 0.0),
    })
    .unwrap();
    let eq = static_equilibrium(&sys, 0.0, None).unwrap();
    let theta = eq.state.joint_angle(JointId(1)).unwrap();
    close(
        theta,
        (f * tip / (M * G * L)).atan(),
        1e-10,
        "equilibrium angle",
    );
}

/// A motor holding the pendulum at a fixed angle: the driver multiplier is
/// the holding torque `m·g·l·sin θ₀` (Inventor's "unknown force").
#[test]
fn static_holding_torque_of_a_driver() {
    let theta0 = 0.8;
    let mut sys = pendulum(0.0);
    sys.add_driver(Driver {
        joint: JointId(1),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Constant {
            start: theta0,
            rate: 0.0,
        },
    });
    let eq = static_equilibrium(&sys, 0.0, None).unwrap();
    close(
        eq.state.joint_angle(JointId(1)).unwrap(),
        theta0,
        1e-10,
        "held angle",
    );
    let lambda = multiplier(&eq.multipliers, EquationOwner::Driver(0));
    close(lambda[0], -M * G * L * theta0.sin(), 1e-9, "holding torque");
}

/// Failures are data: a load along a free, unresisted slide has no
/// equilibrium; a spring with coincident markers and a free length has no
/// line of action.
#[test]
fn dynamics_failures_are_data() {
    let mut slide = spring_mass(0.3, 0.0);
    slide.forces.clear();
    slide.gravity = Some(DVec3::new(-G, 0.0, 0.0));
    assert_eq!(
        static_equilibrium(&slide, 0.0, None).unwrap_err().code,
        codes::DYNAMICS_SINGULAR
    );
    let coincident = spring_mass(0.0, 0.0);
    assert_eq!(
        dynamic_ic(&coincident, 0.0).unwrap_err().code,
        codes::DEGENERATE_FORCE
    );
}
