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

// ── time integration (index-3 HHT-α) ───────────────────────────────────

fn grid(end: f64, step: f64) -> Vec<f64> {
    let n = (end / step).round() as usize;
    (0..=n).map(|k| k as f64 * step).collect()
}

fn hht(alpha: f64, step: f64) -> HhtSettings {
    HhtSettings {
        alpha,
        step,
        ..HhtSettings::default()
    }
}

/// Complete elliptic integral of the first kind by the AGM.
fn elliptic_k(k: f64) -> f64 {
    let (mut a, mut b) = (1.0_f64, (1.0 - k * k).sqrt());
    for _ in 0..30 {
        let next = ((a + b) / 2.0, (a * b).sqrt());
        a = next.0;
        b = next.1;
    }
    std::f64::consts::PI / (2.0 * a)
}

/// Mean period of a signal from its upward zero crossings (linear
/// interpolation between samples).
fn period(samples: &[(f64, f64)]) -> f64 {
    let crossings: Vec<f64> = samples
        .windows(2)
        .filter(|w| w[0].1 < 0.0 && w[1].1 >= 0.0)
        .map(|w| w[0].0 + (w[1].0 - w[0].0) * (-w[0].1) / (w[1].1 - w[0].1))
        .collect();
    assert!(
        crossings.len() >= 2,
        "need two crossings, got {}",
        crossings.len()
    );
    (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64
}

/// Free fall with a spinning, isotropic body: the Newmark family is exact
/// for constant acceleration, and the Lie-group rotation update is exact
/// for constant angular velocity — at any α.
#[test]
fn free_fall_is_exact() {
    for alpha in [0.0, DEFAULT_ALPHA] {
        let mut sys = System::new("fall").with_standard_gravity();
        let mut b = part(
            2,
            1.5,
            DVec3::ZERO,
            DVec3::splat(0.02),
            DVec3::new(0.0, 0.0, 10.0),
            DQuat::IDENTITY,
        );
        b.velocity = DVec3::new(1.0, -0.5, 3.0);
        b.angular_velocity = DVec3::new(0.3, 2.0, -1.0);
        sys.add_body(b.clone()).unwrap();
        let sim = simulate(&sys, &grid(1.0, 0.1), &hht(alpha, 0.01)).unwrap();
        for f in &sim.frames {
            let t = f.time;
            let (p, r) = f.state.body_pose(BodyId(2)).unwrap();
            let expected = b.translation + b.velocity * t + 0.5 * DVec3::new(0.0, 0.0, -G) * t * t;
            close_v(p, expected, 1e-10, "parabola");
            let spin = DQuat::from_scaled_axis(b.angular_velocity * t);
            assert!(r.dot(spin).abs() > 1.0 - 1e-12, "rotation at t = {t}");
            close_v(
                f.velocity[&BodyId(2)].angular,
                b.angular_velocity,
                1e-10,
                "constant spin",
            );
        }
        assert_eq!(sim.stats.rejected, 0);
    }
}

/// Large-amplitude pendulum (θ₀ = 0.5 rad) released from rest: the period
/// matches the exact elliptic-integral period `4·√(I_o/mgl)·K(sin θ₀/2)`
/// and the energy holds: α = 0 (trapezoidal) is conservative, and α = −0.3
/// dissipates only at O((ω·h)³) per step — invisible at the pendulum frequency.
#[test]
fn pendulum_period_and_energy() {
    let theta0 = 0.5;
    let sys = pendulum(theta0);
    let tau = (i_hinge() / (M * G * L)).sqrt();
    let exact = 4.0 * tau * elliptic_k((theta0 / 2.0).sin());
    let scale = M * G * L;
    for (alpha, period_tol, energy_tol) in [(0.0, 2e-6, 2e-7), (DEFAULT_ALPHA, 2e-6, 1e-6)] {
        let sim = simulate(&sys, &grid(6.0, 0.002), &hht(alpha, 1e-3)).unwrap();
        let samples: Vec<(f64, f64)> = sim
            .frames
            .iter()
            .map(|f| (f.time, f.state.joint_angle(JointId(1)).unwrap()))
            .collect();
        let measured = period(&samples);
        let e0 = energy(&sys, &sim.frames[0].state, &sim.frames[0].velocity)
            .unwrap()
            .total();
        let drift = sim
            .frames
            .iter()
            .map(|f| (energy(&sys, &f.state, &f.velocity).unwrap().total() - e0).abs() / scale)
            .fold(0.0, f64::max);
        eprintln!(
            "alpha {alpha}: period {measured} vs {exact} (rel {:.2e}), energy drift {drift:.2e}, stats {:?}",
            (measured - exact).abs() / exact,
            sim.stats
        );
        close(measured, exact, period_tol, "period");
        assert!(drift < energy_tol, "energy drift {drift:e}");
    }
}

/// Small-angle pendulum: the period is `2π·√(I_o/(m·g·l))` (for a point
/// mass `2π·√(l/g)`).
#[test]
fn small_angle_pendulum_period() {
    let sys = pendulum(0.01);
    let sim = simulate(&sys, &grid(5.0, 0.002), &hht(DEFAULT_ALPHA, 1e-3)).unwrap();
    let samples: Vec<(f64, f64)> = sim
        .frames
        .iter()
        .map(|f| (f.time, f.state.joint_angle(JointId(1)).unwrap()))
        .collect();
    let expected = std::f64::consts::TAU * (i_hinge() / (M * G * L)).sqrt();
    close(period(&samples), expected, 3e-5, "small-angle period");
}

/// Spring-mass released from a stretch: `x(t) = l₀ + A·cos(ω·t)`,
/// `ω = √(k/m)`.
#[test]
fn spring_mass_oscillates_at_sqrt_k_over_m() {
    let amplitude = 0.05;
    let sys = spring_mass(FREE + amplitude, 0.0);
    let omega = (K / M).sqrt();
    let sim = simulate(&sys, &grid(2.0, 0.001), &hht(DEFAULT_ALPHA, 5e-4)).unwrap();
    let samples: Vec<(f64, f64)> = sim
        .frames
        .iter()
        .map(|f| (f.time, f.state.body_pose(BodyId(2)).unwrap().0.x - FREE))
        .collect();
    let worst = samples
        .iter()
        .map(|(t, x)| (x - amplitude * (omega * t).cos()).abs())
        .fold(0.0, f64::max);
    let measured = std::f64::consts::TAU / period(&samples);
    eprintln!(
        "spring: omega {measured} vs {omega}, worst {worst:.2e}, stats {:?}",
        sim.stats
    );
    close(measured, omega, 2e-5, "ω = √(k/m)");
    assert!(worst < 1e-3 * amplitude, "trajectory error {worst:e}");
}

/// Damped oscillator in its three regimes (ζ = 0.1 under-, 1 critically,
/// 2.5 over-damped) against the closed forms.
#[test]
fn damped_oscillator_three_regimes() {
    let amplitude = 0.04;
    let omega = (K / M).sqrt();
    for zeta in [0.1, 1.0, 2.5] {
        let c = 2.0 * zeta * (K * M).sqrt();
        let sys = spring_mass(FREE + amplitude, c);
        let exact = |t: f64| -> f64 {
            if zeta < 1.0 {
                let wd = omega * (1.0 - zeta * zeta).sqrt();
                amplitude
                    * (-zeta * omega * t).exp()
                    * ((wd * t).cos() + zeta * omega / wd * (wd * t).sin())
            } else if zeta == 1.0 {
                amplitude * (-omega * t).exp() * (1.0 + omega * t)
            } else {
                let root = (zeta * zeta - 1.0).sqrt();
                let (s1, s2) = (-omega * (zeta - root), -omega * (zeta + root));
                amplitude * (s2 * (s1 * t).exp() - s1 * (s2 * t).exp()) / (s2 - s1)
            }
        };
        let sim = simulate(&sys, &grid(1.5, 0.005), &hht(DEFAULT_ALPHA, 5e-4)).unwrap();
        let worst = sim
            .frames
            .iter()
            .map(|f| (f.state.body_pose(BodyId(2)).unwrap().0.x - FREE - exact(f.time)).abs())
            .fold(0.0, f64::max);
        eprintln!("zeta {zeta}: worst {worst:.2e}");
        assert!(worst < 2e-4 * amplitude, "ζ = {zeta}: error {worst:e}");
    }
}

const FOUR_BAR: (f64, f64, f64, f64) = (0.1, 0.3, 0.25, 0.3);

/// A planar four-bar (crank `a`, coupler `b`, rocker `c`, ground `d`) from
/// four 3D revolutes about world Z, gravity −Y, authored near crank angle
/// 90° (the position IC closes the loop).
fn four_bar() -> System {
    let (a, b, c, d) = FOUR_BAR;
    let mut sys = System::new("four-bar");
    sys.gravity = Some(DVec3::new(0.0, -G, 0.0));
    sys.add_body(ground()).unwrap();
    let link = |id, len: f64, at: DVec3, angle: f64| {
        part(
            id,
            0.5 + len,
            DVec3::new(len / 2.0, 0.0, 0.0),
            DVec3::new(1e-4, 0.004, 0.004),
            at,
            DQuat::from_rotation_z(angle),
        )
    };
    sys.add_body(link(2, a, DVec3::ZERO, std::f64::consts::FRAC_PI_2))
        .unwrap();
    sys.add_body(link(3, b, DVec3::new(0.0, a, 0.0), 0.5455))
        .unwrap();
    sys.add_body(link(4, c, DVec3::new(d, 0.0, 0.0), 1.7239))
        .unwrap();
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
    joint(&mut sys, 3, JointKind::Revolute, 6, 8);
    joint(&mut sys, 4, JointKind::Revolute, 2, 7);
    sys
}

/// The four-bar (three redundant equations) swinging under gravity from
/// rest: index 3 keeps the loop closed, the projected output velocities
/// satisfy the velocity constraints, and the trapezoidal rule holds the
/// energy.
#[test]
fn four_bar_swings_closed_and_conservative() {
    let (_, b, c, _) = FOUR_BAR;
    let sys = four_bar();
    // Energy error at two step sizes: bounded (no secular drift) and
    // second order — halving the step quarters it.
    let run = |step: f64| {
        let sim = simulate(&sys, &grid(2.0, 0.01), &hht(0.0, step)).unwrap();
        let e0 = energy(&sys, &sim.frames[0].state, &sim.frames[0].velocity).unwrap();
        let mut drift: f64 = 0.0;
        let mut swing: f64 = 0.0;
        for f in &sim.frames {
            assert_eq!(f.redundant.len(), 3, "planar loop of 3D revolutes");
            let p = f
                .state
                .world_point(BodyId(3), DVec3::new(b, 0.0, 0.0))
                .unwrap();
            let q = f
                .state
                .world_point(BodyId(4), DVec3::new(c, 0.0, 0.0))
                .unwrap();
            assert!((p - q).length() < 1e-9, "loop closed at t = {}", f.time);
            let arm3 = p - f.state.body_pose(BodyId(3)).unwrap().0;
            let arm4 = q - f.state.body_pose(BodyId(4)).unwrap().0;
            let vp = f.velocity[&BodyId(3)].point_velocity(arm3);
            let vq = f.velocity[&BodyId(4)].point_velocity(arm4);
            assert!(
                (vp - vq).length() < 1e-9,
                "velocity constraint at t = {}",
                f.time
            );
            let e = energy(&sys, &f.state, &f.velocity).unwrap();
            drift = drift.max((e.total() - e0.total()).abs());
            swing = swing.max(e.kinetic);
        }
        eprintln!(
            "four-bar h = {step}: energy error {drift:.2e} J of peak kinetic {swing:.3} J, {:?}",
            sim.stats
        );
        (drift, swing)
    };
    let (coarse, peak) = run(1e-3);
    let (fine, _) = run(5e-4);
    assert!(peak > 0.1, "the linkage actually swings: {peak} J");
    let order = (coarse / fine).log2();
    assert!(
        (order - 2.0).abs() < 0.2,
        "second-order energy error: observed order {order}"
    );
    assert!(fine < 1e-4 * peak, "energy error {fine:e} J of {peak} J");
}

/// A driven pendulum in the integrator: the rheonomic driver equation is
/// held exactly and the driver multiplier is the inverse-dynamics torque.
#[test]
fn driven_pendulum_simulates_its_driver() {
    let mut sys = pendulum(0.0);
    sys.add_driver(Driver {
        joint: JointId(1),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Harmonic {
            offset: 0.0,
            amplitude: 0.6,
            frequency_hz: 0.5,
            phase: 0.0,
        },
    });
    let sim = simulate(&sys, &grid(2.0, 0.05), &HhtSettings::default()).unwrap();
    for f in &sim.frames {
        let s = sys.drivers[0].sample(f.time);
        close(
            f.state.joint_angle(JointId(1)).unwrap(),
            s.value,
            1e-9,
            "driver angle",
        );
        close(
            f.velocity[&BodyId(2)].angular.z,
            s.velocity,
            1e-9,
            "driver rate",
        );
        let torque = i_hinge() * s.acceleration + M * G * L * s.value.sin();
        let lambda = multiplier(&f.multipliers, EquationOwner::Driver(0));
        close(lambda[0], -torque, 1e-8, "motor torque");
    }
}

/// Same input, same trajectory (bit for bit); bad settings and grids are
/// data.
#[test]
fn integration_is_deterministic_and_validated() {
    let sys = pendulum(1.0);
    let times = grid(0.5, 0.05);
    let a = simulate(&sys, &times, &HhtSettings::default()).unwrap();
    let b = simulate(&sys, &times, &HhtSettings::default()).unwrap();
    assert_eq!(a, b);
    for bad in [
        hht(0.1, 1e-3),
        hht(-0.5, 1e-3),
        hht(0.0, 0.0),
        hht(0.0, f64::NAN),
    ] {
        assert_eq!(
            simulate(&sys, &times, &bad).unwrap_err().code,
            codes::BAD_INTEGRATOR_SETTINGS
        );
    }
    assert_eq!(
        simulate(&sys, &[0.0, 1.0, 0.5], &HhtSettings::default())
            .unwrap_err()
            .code,
        codes::BAD_TIME_GRID
    );
    assert_eq!(
        simulate(&sys, &[], &HhtSettings::default())
            .unwrap_err()
            .code,
        codes::BAD_TIME_GRID
    );
}

// ── joint reactions ─────────────────────────────────────────────────────

/// The swinging pendulum's hinge: the force on the ground (marker I) is
/// `m·g − m·a_com` (the link pulls on the hinge), the force on the link the
/// opposite, no hinge torque; unique.
#[test]
fn pendulum_hinge_reaction() {
    let mut sys = pendulum(1.1);
    sys.bodies.get_mut(&BodyId(2)).unwrap().angular_velocity = DVec3::Z * 2.3;
    let frame = dynamic_ic(&sys, 0.0).unwrap();
    let r = &frame.reactions(&sys).unwrap()[0];
    assert_eq!(r.owner, EquationOwner::Joint(JointId(1)));
    assert!(r.is_unique() && r.effort.is_none());
    let (v, a) = (frame.velocity[&BodyId(2)], frame.acceleration[&BodyId(2)]);
    let c = DQuat::from_rotation_z(1.1) * DVec3::new(0.0, -L, 0.0);
    let a_com = a.linear + a.angular.cross(c) + v.angular.cross(v.angular.cross(c));
    let g = DVec3::new(0.0, -G, 0.0);
    close_v(r.force_on_j, M * (a_com - g), 1e-10, "hinge force on the link");
    close_v(r.force_on_i, -M * (a_com - g), 1e-10, "hinge force on the ground");
    assert!(r.torque_on_i.length() < 1e-10 && r.torque_on_j.length() < 1e-10);
}

/// A motor's effort is the torque it applies to the driven body; the
/// driver's wrench is a pure torque about the joint axis.
#[test]
fn driver_effort_is_the_motor_torque() {
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
    let t = 0.77;
    let kin = solve_frame(&sys, t, None).unwrap();
    let frame = forward_dynamics(&sys, t, &kin.state, &kin.velocity).unwrap();
    let reactions = frame.reactions(&sys).unwrap();
    let motor = reactions.iter().find(|r| r.owner == EquationOwner::Driver(0)).unwrap();
    let s = sys.drivers[0].sample(t);
    let torque = i_hinge() * s.acceleration + M * G * L * s.value.sin();
    close(motor.effort.unwrap(), torque, 1e-9, "effort");
    close_v(motor.torque_on_j, DVec3::Z * torque, 1e-9, "torque on the link");
    assert!(motor.force_on_i.length() < 1e-12);
    let hinge = reactions.iter().find(|r| r.owner == EquationOwner::Joint(JointId(1))).unwrap();
    assert!(hinge.torque_on_i.z.abs() < 1e-10, "the hinge carries no axial torque");
}

/// Static equilibrium reactions: the side-loaded pendulum's hinge carries
/// the weight plus the push.
#[test]
fn static_hinge_reaction() {
    let mut sys = pendulum(0.0);
    marker(&mut sys, 3, 2, DVec3::new(0.0, -0.9, 0.0), DQuat::IDENTITY);
    let push = DVec3::new(7.0, 0.0, 0.0);
    sys.add_force(ForceElement::AppliedForce {
        id: ForceId(1),
        name: "push".into(),
        marker: MarkerId(3),
        force: push,
    })
    .unwrap();
    let eq = static_equilibrium(&sys, 0.0, None).unwrap();
    let r = &eq.reactions(&sys).unwrap()[0];
    close_v(r.force_on_i, push + DVec3::new(0.0, -M * G, 0.0), 1e-9, "load on the ground");
    assert!(r.torque_on_i.length() < 1e-9);
}

/// Redundancy: a coaxial double hinge shares the load in an arbitrary way
/// — each reaction is flagged indeterminate, yet the two together carry
/// exactly the hinge load. The planar four-bar of 3D revolutes determines
/// every in-plane load (Fx, Fy, Tz) of each joint and leaves only the
/// out-of-plane split indeterminate.
#[test]
fn indeterminate_reactions_are_flagged() {
    let mut sys = pendulum(0.9);
    marker(&mut sys, 3, 1, DVec3::new(0.0, 0.0, 0.1), DQuat::IDENTITY);
    marker(&mut sys, 4, 2, DVec3::new(0.0, 0.0, 0.1), DQuat::IDENTITY);
    joint(&mut sys, 2, JointKind::Revolute, 3, 4);
    let frame = dynamic_ic(&sys, 0.0).unwrap();
    let both = frame.reactions(&sys).unwrap();
    assert!(both.iter().all(|r| !r.is_unique()), "{both:?}");
    let total: DVec3 = both.iter().map(|r| r.force_on_j).sum();
    let a = frame.acceleration[&BodyId(2)];
    let c = DQuat::from_rotation_z(0.9) * DVec3::new(0.0, -L, 0.0);
    let a_com = a.linear + a.angular.cross(c);
    close_v(total, M * (a_com - DVec3::new(0.0, -G, 0.0)), 1e-9, "total hinge load");

    let sys = four_bar();
    let frame = dynamic_ic(&sys, 0.0).unwrap();
    let reactions = frame.reactions(&sys).unwrap();
    assert_eq!(reactions.len(), 4);
    for r in &reactions {
        assert!(!r.is_unique(), "{:?}", r.owner);
        for e in &r.indeterminate {
            assert!(
                e[0].abs() < 1e-9 && e[1].abs() < 1e-9 && e[5].abs() < 1e-9,
                "{:?}: in-plane loads must be determinate: {e:?}",
                r.owner
            );
        }
        assert!(r.torque_on_i.z.abs() < 1e-9, "revolutes carry no axial torque");
    }
}
