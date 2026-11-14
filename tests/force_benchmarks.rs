//! Tier A force-library benchmarks (roadmap §6, M3): function forces
//! (time and state curves), follower loads, line forces, preloads and the
//! work-energy audit, against closed forms.

use glam::{DMat3, DQuat, DVec3};
use rcad_mbd::*;

const G: f64 = STANDARD_GRAVITY;
const M: f64 = 2.0;
const L: f64 = 0.6;
const IZZ: f64 = 0.03;
const K: f64 = 400.0;
const FREE: f64 = 0.25;

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

fn part(com: DVec3, at: DVec3, rotation: DQuat) -> Body {
    Body {
        id: BodyId(2),
        name: "part".into(),
        mass_kg: M,
        com,
        inertia_com: DMat3::from_diagonal(DVec3::new(0.04, 0.01, IZZ)),
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

fn joint(sys: &mut System, kind: JointKind) {
    sys.add_joint(Joint {
        id: JointId(1),
        name: "j1".into(),
        kind,
        marker1: MarkerId(1),
        marker2: MarkerId(2),
        suppressed: false,
    })
    .unwrap();
}

fn add(sys: &mut System, element: ForceElement) {
    sys.add_force(element).unwrap();
}

fn grid(end: f64, step: f64) -> Vec<f64> {
    let n = (end / step).round() as usize;
    (0..=n).map(|k| k as f64 * step).collect()
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

/// Compound pendulum: hinge about world Z at the origin, center of mass
/// `L` below it, gravity −Y, authored at hinge angle `theta`.
fn pendulum(theta: f64, gravity: bool) -> System {
    let mut sys = System::new("pendulum");
    if gravity {
        sys.gravity = Some(DVec3::new(0.0, -G, 0.0));
    }
    sys.add_body(ground()).unwrap();
    sys.add_body(part(
        DVec3::new(0.0, -L, 0.0),
        DVec3::ZERO,
        DQuat::from_rotation_z(theta),
    ))
    .unwrap();
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    joint(&mut sys, JointKind::Revolute);
    sys
}

fn i_hinge() -> f64 {
    IZZ + M * L * L
}

/// A slider on a prismatic joint along world X, no load across the slide
/// (gravity −Y is carried by the joint).
fn slider(x: f64) -> System {
    let mut sys = System::new("slider");
    sys.gravity = Some(DVec3::new(0.0, -G, 0.0));
    sys.add_body(ground()).unwrap();
    sys.add_body(part(DVec3::ZERO, DVec3::new(x, 0.0, 0.0), DQuat::IDENTITY))
        .unwrap();
    let along_x = DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2);
    marker(&mut sys, 1, 1, DVec3::ZERO, along_x);
    marker(&mut sys, 2, 2, DVec3::ZERO, along_x);
    joint(&mut sys, JointKind::Prismatic);
    sys
}

fn spring(sys: &mut System, id: u64, damping: f64, preload: f64) {
    add(
        sys,
        ForceElement::SpringDamper {
            id: ForceId(id),
            name: "spring".into(),
            marker1: MarkerId(1),
            marker2: MarkerId(2),
            stiffness: K,
            damping,
            free_length: FREE,
            preload,
        },
    );
}

fn x_of(f: &DynamicFrame) -> f64 {
    f.state.body_pose(BodyId(2)).unwrap().0.x
}

/// Harmonic forcing `F₀·sin(Ω·t)` of a damped spring-mass (ζ = 0.1) at
/// Ω/ω = 0.5, 1 (resonance) and 2: after the transient the response is
/// `X·sin(Ω·t − φ)` with `X = (F₀/k)/√((1 − r²)² + (2ζr)²)` and
/// `φ = atan2(2ζr, 1 − r²)`.
#[test]
fn forced_oscillation_steady_state() {
    let (f0, zeta) = (10.0, 0.1);
    let omega = (K / M).sqrt();
    let c = 2.0 * zeta * (K * M).sqrt();
    for r in [0.5, 1.0, 2.0] {
        let big_omega = r * omega;
        let mut sys = slider(FREE);
        spring(&mut sys, 1, c, 0.0);
        add(
            &mut sys,
            ForceElement::Force {
                id: ForceId(2),
                name: "shaker".into(),
                marker: MarkerId(2),
                reference: None,
                direction: DVec3::X,
                magnitude: Function::of_time(DriverProfile::Harmonic {
                    offset: 0.0,
                    amplitude: f0,
                    frequency_hz: big_omega / std::f64::consts::TAU,
                    phase: 0.0,
                }),
            },
        );
        let settings = HhtSettings {
            error_tol: Some(1e-7),
            ..HhtSettings::default()
        };
        let sim = simulate(&sys, &grid(12.0, 0.01), &settings).unwrap();
        let amplitude = (f0 / K) / ((1.0 - r * r).powi(2) + (2.0 * zeta * r).powi(2)).sqrt();
        let phase = (2.0 * zeta * r).atan2(1.0 - r * r);
        let worst = sim
            .frames
            .iter()
            .filter(|f| f.time >= 10.0)
            .map(|f| (x_of(f) - FREE - amplitude * (big_omega * f.time - phase).sin()).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "r = {r}: X = {amplitude:.5e}, worst {worst:.2e}, {:?}",
            sim.stats
        );
        assert!(
            worst < 2e-3 * amplitude,
            "r = {r}: steady-state error {worst:e} of {amplitude:e}"
        );
    }
}

/// A follower force (fixed in the link frame, perpendicular to it) holds
/// the pendulum at `sin θ = F·d/(m·g·l)`; the same force fixed in the
/// world at `tan θ = F·d/(m·g·l)`.
#[test]
fn follower_and_world_forces_in_equilibrium() {
    let (tip, ratio) = (0.9, 0.6);
    let f = ratio * M * G * L / tip;
    for (follower, expected) in [(true, ratio.asin()), (false, ratio.atan())] {
        let mut sys = pendulum(0.0, true);
        marker(&mut sys, 3, 2, DVec3::new(0.0, -tip, 0.0), DQuat::IDENTITY);
        add(
            &mut sys,
            ForceElement::Force {
                id: ForceId(1),
                name: "push".into(),
                marker: MarkerId(3),
                reference: follower.then_some(MarkerId(3)),
                direction: DVec3::X,
                magnitude: Function::constant(f),
            },
        );
        let eq = static_equilibrium(&sys, 0.0, None).unwrap();
        let theta = eq.state.joint_angle(JointId(1)).unwrap();
        close(
            theta,
            expected,
            1e-10,
            if follower { "follower" } else { "world-fixed" },
        );
    }
}

/// A torque that is a linear function of the hinge angle, `−k_t·θ`, makes
/// a torsional pendulum: `θ(t) = θ₀·cos(√(k_t/I_o)·t)`.
#[test]
fn torque_of_joint_coordinate_is_a_torsion_spring() {
    let (kt, theta0) = (3.0, 0.3);
    let mut sys = pendulum(theta0, false);
    add(
        &mut sys,
        ForceElement::Torque {
            id: ForceId(1),
            name: "torsion".into(),
            marker: MarkerId(2),
            reference: Some(MarkerId(1)),
            direction: DVec3::Z,
            magnitude: Function::linear(
                Argument::JointCoordinate {
                    joint: JointId(1),
                    slot: DriverSlot::Rotation,
                },
                0.0,
                -kt,
            ),
        },
    );
    let omega = (kt / i_hinge()).sqrt();
    let sim = simulate(&sys, &grid(3.0, 0.01), &HhtSettings::fixed(0.0, 1e-3)).unwrap();
    let worst = sim
        .frames
        .iter()
        .map(|f| (f.state.joint_angle(JointId(1)).unwrap() - theta0 * (omega * f.time).cos()).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-5 * theta0, "torsional pendulum error {worst:e}");
}

/// Line forces built from functions of the distance and its rate
/// reproduce the spring-damper with preload exactly.
#[test]
fn line_forces_reproduce_the_spring_damper() {
    let (c, preload) = (8.0, 15.0);
    let mut reference = slider(FREE + 0.05);
    spring(&mut reference, 1, c, preload);
    let mut built = slider(FREE + 0.05);
    let distance = Argument::Distance {
        marker1: MarkerId(1),
        marker2: MarkerId(2),
    };
    let rate = Argument::DistanceRate {
        marker1: MarkerId(1),
        marker2: MarkerId(2),
    };
    for (id, tension) in [
        (1, Function::linear(distance, -K * FREE - preload, K)),
        (2, Function::linear(rate, 0.0, c)),
    ] {
        add(
            &mut built,
            ForceElement::LineForce {
                id: ForceId(id),
                name: format!("line{id}"),
                marker1: MarkerId(1),
                marker2: MarkerId(2),
                tension,
            },
        );
    }
    let times = grid(1.0, 0.01);
    let a = simulate(&reference, &times, &HhtSettings::default()).unwrap();
    let b = simulate(&built, &times, &HhtSettings::default()).unwrap();
    for (fa, fb) in a.frames.iter().zip(&b.frames) {
        assert!((x_of(fa) - x_of(fb)).abs() < 1e-12, "t = {}", fa.time);
    }
}

/// A preloaded hanging spring settles at `l = l₀ + (m·g + preload)/k`, and
/// the energy audit carries the preload's potential.
#[test]
fn preloaded_spring_statics_and_energy() {
    let preload = 12.0;
    let mut sys = System::new("hanging");
    sys.gravity = Some(DVec3::new(0.0, -G, 0.0));
    sys.add_body(ground()).unwrap();
    sys.add_body(part(
        DVec3::ZERO,
        DVec3::new(0.0, -0.3, 0.0),
        DQuat::IDENTITY,
    ))
    .unwrap();
    let along_y = DQuat::from_rotation_x(-std::f64::consts::FRAC_PI_2);
    marker(&mut sys, 1, 1, DVec3::ZERO, along_y);
    marker(&mut sys, 2, 2, DVec3::ZERO, along_y);
    joint(&mut sys, JointKind::Prismatic);
    spring(&mut sys, 1, 0.0, preload);
    let eq = static_equilibrium(&sys, 0.0, None).unwrap();
    let y = eq.state.body_pose(BodyId(2)).unwrap().0.y;
    close(
        -y,
        FREE + (M * G + preload) / K,
        1e-10,
        "preloaded extension",
    );
    // Released from 5 cm above the equilibrium, undamped: energy holds.
    let mut moving = sys.clone();
    moving.bodies.get_mut(&BodyId(2)).unwrap().translation.y = y + 0.05;
    let sim = simulate(&moving, &grid(2.0, 0.01), &HhtSettings::fixed(0.0, 1e-3)).unwrap();
    let e0 = energy(&moving, &sim.frames[0].state, &sim.frames[0].velocity)
        .unwrap()
        .total();
    for f in &sim.frames {
        let e = energy(&moving, &f.state, &f.velocity).unwrap().total();
        assert!(
            (e - e0).abs() < 1e-6 * e0.abs().max(1.0),
            "energy at t = {}: {e} vs {e0}",
            f.time
        );
    }
}

/// Work-energy audit: with a time-varying shaker, a damper and a follower
/// torque the kinetic energy changes by exactly the work of the applied
/// forces, `T(t) − T(0) = ∫ Σ Q·v dt` (the joints do no work), up to the
/// integrator's second-order truncation error (Simpson quadrature).
#[test]
fn work_energy_audit() {
    let mut sys = pendulum(0.4, true);
    marker(&mut sys, 3, 2, DVec3::new(0.0, -0.9, 0.0), DQuat::IDENTITY);
    add(
        &mut sys,
        ForceElement::Force {
            id: ForceId(1),
            name: "shaker".into(),
            marker: MarkerId(3),
            reference: None,
            direction: DVec3::new(1.0, 0.3, 0.0),
            magnitude: Function::of_time(DriverProfile::Harmonic {
                offset: 1.0,
                amplitude: 6.0,
                frequency_hz: 1.3,
                phase: 0.2,
            }),
        },
    );
    add(
        &mut sys,
        ForceElement::Torque {
            id: ForceId(2),
            name: "drag".into(),
            marker: MarkerId(2),
            reference: Some(MarkerId(2)),
            direction: DVec3::Z,
            magnitude: Function::linear(
                Argument::JointRate {
                    joint: JointId(1),
                    slot: DriverSlot::Rotation,
                },
                0.0,
                -0.4,
            ),
        },
    );
    let dt = 1e-3;
    let run = |h: f64| -> (f64, f64) {
        let sim = simulate(&sys, &grid(3.0, dt), &HhtSettings::fixed(DEFAULT_ALPHA, h)).unwrap();
        let kinetic: Vec<f64> = sim
            .frames
            .iter()
            .map(|f| energy(&sys, &f.state, &f.velocity).unwrap().kinetic)
            .collect();
        let power: Vec<f64> = sim
            .frames
            .iter()
            .map(|f| power(&sys, f.time, &f.state, &f.velocity).unwrap())
            .collect();
        let peak = kinetic.iter().fold(0.0_f64, |a, b| a.max(*b));
        let mut work = 0.0;
        let mut worst: f64 = 0.0;
        // Simpson's rule over pairs of output intervals.
        for k in (2..sim.frames.len()).step_by(2) {
            work += dt / 3.0 * (power[k - 2] + 4.0 * power[k - 1] + power[k]);
            worst = worst.max((kinetic[k] - kinetic[0] - work).abs());
        }
        eprintln!("work-energy h = {h}: worst {worst:.2e} J of peak kinetic {peak:.3} J");
        (worst, peak)
    };
    // The mismatch is the integrator's truncation error: second order in h.
    let (coarse, peak) = run(2.5e-4);
    let (fine, _) = run(1.25e-4);
    assert!(peak > 0.5, "the pendulum swings: {peak}");
    let order = (coarse / fine).log2();
    assert!((order - 2.0).abs() < 0.2, "work-energy error order {order}");
    assert!(fine < 1e-6 * peak, "work-energy mismatch {fine:e}");
}

/// The new elements validate and serialize; failures are data.
#[test]
fn force_elements_validate_and_round_trip() {
    let mut sys = pendulum(0.0, true);
    add(
        &mut sys,
        ForceElement::Torque {
            id: ForceId(1),
            name: "t".into(),
            marker: MarkerId(2),
            reference: Some(MarkerId(1)),
            direction: DVec3::Z,
            magnitude: Function {
                argument: Argument::JointCoordinate {
                    joint: JointId(1),
                    slot: DriverSlot::Rotation,
                },
                curve: DriverProfile::Spline {
                    points: vec![(-1.0, 2.0), (0.0, 0.0), (1.0, -2.0)],
                },
            },
        },
    );
    spring(&mut sys, 2, 1.0, 3.0);
    assert!(sys.validate().is_ok());
    let text = serde_json::to_string(&sys).unwrap();
    assert_eq!(serde_json::from_str::<System>(&text).unwrap(), sys);

    let with = |element: ForceElement| {
        let mut s = pendulum(0.0, true);
        s.add_force(element).unwrap();
        s
    };
    let force = |direction: DVec3, magnitude: Function| ForceElement::Force {
        id: ForceId(1),
        name: "f".into(),
        marker: MarkerId(2),
        reference: None,
        direction,
        magnitude,
    };
    let zero_dir = with(force(DVec3::ZERO, Function::constant(1.0)));
    assert_eq!(
        zero_dir.validate().unwrap_err().code,
        codes::BAD_FORCE_ELEMENT
    );
    let empty_curve = with(force(
        DVec3::X,
        Function::of_time(DriverProfile::Keyframes { points: vec![] }),
    ));
    assert_eq!(
        empty_curve.validate().unwrap_err().code,
        codes::BAD_FORCE_ELEMENT
    );
    let missing = with(force(
        DVec3::X,
        Function::linear(
            Argument::JointRate {
                joint: JointId(9),
                slot: DriverSlot::Rotation,
            },
            0.0,
            1.0,
        ),
    ));
    assert_eq!(missing.validate().unwrap_err().code, codes::DANGLING_JOINT);
    let wrong_slot = with(force(
        DVec3::X,
        Function::linear(
            Argument::JointCoordinate {
                joint: JointId(1),
                slot: DriverSlot::Translation,
            },
            0.0,
            1.0,
        ),
    ));
    assert_eq!(
        wrong_slot.validate().unwrap_err().code,
        codes::SLOT_UNSUPPORTED
    );
    let coincident = with(ForceElement::LineForce {
        id: ForceId(1),
        name: "l".into(),
        marker1: MarkerId(1),
        marker2: MarkerId(2),
        tension: Function::constant(5.0),
    });
    assert_eq!(
        dynamic_ic(&coincident, 0.0).unwrap_err().code,
        codes::DEGENERATE_FORCE
    );
}
