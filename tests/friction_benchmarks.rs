//! Tier A benchmarks of joint friction (M5-1): closed forms for the
//! sliding block, the Coulomb pendulum, the loaded pin, the friction cone
//! of a cylindrical joint, the rubbing ball socket and the stiction that
//! holds a static equilibrium.

mod common;

use common::*;
use glam::{DQuat, DVec3};
use rcad_mbd::*;

/// The slope of the sliding-block benchmarks (rad).
const SLOPE: f64 = 20.0_f64.to_radians();

/// A block of mass [`M`] on a prismatic guide down a `slope`, gravity −Y:
/// the joint's axis is the slope direction, so the guide reacts
/// `M·g·cos(slope)` and gravity drives `M·g·sin(slope)` along it.
fn incline(slope: f64) -> (System, DVec3) {
    let down = DVec3::new(slope.cos(), -slope.sin(), 0.0);
    let frame = DQuat::from_rotation_arc(DVec3::Z, down);
    let mut sys = System::new("incline");
    sys.gravity = Some(DVec3::new(0.0, -G, 0.0));
    sys.add_body(ground()).unwrap();
    sys.add_body(part(2, M, DVec3::ZERO, DVec3::splat(0.01), DVec3::ZERO, frame))
        .unwrap();
    marker(&mut sys, 1, 1, DVec3::ZERO, frame);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    joint(&mut sys, 1, JointKind::Prismatic, 1, 2);
    (sys, down)
}

/// The friction parameters, so a benchmark can vary one of them: no
/// Stribeck dip (`μs = μd`), a narrow regularization, no arms and no
/// preload.
#[derive(Debug, Clone, Copy)]
struct Fric {
    mu_static: f64,
    mu_dynamic: f64,
    stiction_velocity: f64,
    friction_velocity: f64,
    reaction_force: bool,
    pin_radius: f64,
    thrust_arm: f64,
    bending_arm: f64,
    preload: f64,
    max_stiction_deformation: f64,
}

impl Default for Fric {
    fn default() -> Self {
        Self {
            mu_static: 0.1,
            mu_dynamic: 0.1,
            stiction_velocity: 1.0e-4,
            friction_velocity: 1.0e-4,
            reaction_force: true,
            pin_radius: 0.0,
            thrust_arm: 0.0,
            bending_arm: 0.0,
            preload: 0.0,
            max_stiction_deformation: 1.0e-4,
        }
    }
}

impl Fric {
    fn element(self, id: u64, joint: u64) -> ForceElement {
        ForceElement::JointFriction {
            id: ForceId(id),
            name: format!("friction{id}"),
            joint: JointId(joint),
            mu_static: self.mu_static,
            mu_dynamic: self.mu_dynamic,
            stiction_velocity: self.stiction_velocity,
            friction_velocity: self.friction_velocity,
            reaction_force: self.reaction_force,
            pin_radius: self.pin_radius,
            thrust_arm: self.thrust_arm,
            bending_arm: self.bending_arm,
            preload: self.preload,
            max_stiction_deformation: self.max_stiction_deformation,
        }
    }
}

/// Constant (load-independent) friction: every arm zero, so the friction
/// load is the preload alone.
fn constant_friction(id: u64, joint: u64, mu: f64, load: f64) -> ForceElement {
    Fric {
        mu_static: mu,
        mu_dynamic: mu,
        reaction_force: false,
        preload: load,
        ..Fric::default()
    }
    .element(id, joint)
}

/// Load-dependent friction at a pin of radius `pin_radius`.
fn pin_friction(id: u64, joint: u64, mu: f64, pin_radius: f64) -> ForceElement {
    Fric {
        mu_static: mu,
        mu_dynamic: mu,
        pin_radius,
        ..Fric::default()
    }
    .element(id, joint)
}

/// The inverse of the stiction ramp `tanh`.
fn stiction_inverse(r: f64) -> f64 {
    0.5 * ((1.0 + r) / (1.0 - r)).ln()
}

/// A sliding block with `μ_d < tan(slope)` accelerates at
/// `g·(sin θ − μ_d·cos θ)` — the first closed form of every friction
/// chapter. The friction load is the guide reaction, not a given constant.
#[test]
fn sliding_block_takes_the_coulomb_acceleration() {
    let mu = 0.15;
    let (mut sys, down) = incline(SLOPE);
    // A prismatic guide rubs the transverse reaction force, which needs no
    // arm — the preload is zero, so all of the friction is bearing load.
    sys.add_force(pin_friction(10, 1, mu, 0.0)).unwrap();
    let expected = G * (SLOPE.sin() - mu * SLOPE.cos());
    let step = 0.05;
    let run = simulate(&sys, &grid(0.5, step), &HhtSettings::default()).unwrap();
    let along = |f: &DynamicFrame| {
        (
            f.state.body_pose(BodyId(2)).unwrap().0.dot(down),
            f.velocity[&BodyId(2)].linear.dot(down),
            f.acceleration[&BodyId(2)].linear.dot(down),
        )
    };
    for frame in run.frames.iter().skip(1) {
        close(along(frame).2, expected, 1.0e-7, "sliding acceleration");
    }
    // The regularized friction is zero at exactly zero slip (as in the
    // commercial solvers), so the start-up carries a little less friction
    // than Coulomb; from the second frame on the motion is the closed form
    // interval by interval.
    for w in run.frames[1..].windows(2) {
        let (x0, v0, _) = along(&w[0]);
        let (x1, v1, _) = along(&w[1]);
        close(v1 - v0, expected * step, 1.0e-6, "velocity gained");
        close(
            x1 - x0,
            v0 * step + 0.5 * expected * step * step,
            1.0e-6,
            "distance covered",
        );
    }
    // The reported friction load is the guide reaction times μ.
    let last = run.frames.last().unwrap();
    let report = last.friction(&sys).unwrap();
    assert_eq!(report.len(), 1);
    close(
        report[0].normal_load,
        M * G * SLOPE.cos(),
        1.0e-7,
        "friction load",
    );
    close(report[0].coefficient, mu, 1.0e-12, "coefficient");
    close(
        report[0].force,
        mu * M * G * SLOPE.cos(),
        1.0e-7,
        "friction force",
    );
}

/// A pendulum with a constant friction torque loses exactly `T_f` per
/// radian swept: between two turning points
/// `M·g·L·(cos θ₂ − cos θ₁) = T_f·|θ₁ − θ₂|` (exact — the ends have no
/// kinetic energy), and in the small-angle limit the amplitude falls by
/// `4·T_f/(M·g·L)` per cycle, the classic linear Coulomb decay.
#[test]
fn coulomb_pendulum_decays_by_the_friction_torque() {
    let (mu, preload) = (0.3, 0.98);
    let torque = mu * preload;
    let mut sys = pendulum(0.4);
    sys.add_force(constant_friction(10, 1, mu, preload)).unwrap();
    let times = grid(3.2, 0.001);
    let run = simulate(&sys, &times, &HhtSettings::default()).unwrap();

    // Turning points: the frames where the hinge rate changes sign.
    let angle = |f: &DynamicFrame| f.state.joint_angle(JointId(1)).unwrap();
    let rate = |f: &DynamicFrame| f.velocity[&BodyId(2)].angular.z;
    let mut turns = vec![angle(&run.frames[0])];
    for w in run.frames.windows(2) {
        if rate(&w[0]) * rate(&w[1]) <= 0.0 && rate(&w[0]) != 0.0 {
            let (a, b) = (rate(&w[0]).abs(), rate(&w[1]).abs());
            turns.push(if a <= b { angle(&w[0]) } else { angle(&w[1]) });
        }
    }
    assert!(
        turns.len() >= 4,
        "expected at least four turning points, got {turns:?}"
    );
    let mgl = M * G * L;
    for pair in turns.windows(2) {
        let (t1, t2) = (pair[0], pair[1]);
        assert!(t1 * t2 < 0.0, "turning points alternate: {t1} and {t2}");
        close(
            mgl * (t2.cos() - t1.cos()),
            torque * (t1 - t2).abs(),
            2.0e-3,
            "energy lost to friction between turning points",
        );
    }
    // The linear decay per cycle (small-angle, so only to a few percent at
    // this amplitude).
    for pair in turns.windows(3) {
        close(
            pair[0].abs() - pair[2].abs(),
            4.0 * torque / mgl,
            0.05,
            "amplitude lost per cycle",
        );
    }
    // Friction only ever removes energy (to the integrator's own energy
    // wobble, which is far below the friction's loss per step).
    let mut previous = f64::INFINITY;
    for f in &run.frames {
        let e = energy(&sys, &f.state, &f.velocity).unwrap().total();
        assert!(e <= previous + 1.0e-5, "energy grew at t = {}", f.time);
        previous = e;
    }
    assert!(
        energy(&sys, &run.frames.last().unwrap().state, &run.frames.last().unwrap().velocity)
            .unwrap()
            .total()
            < energy(&sys, &run.frames[0].state, &run.frames[0].velocity)
                .unwrap()
                .total()
                - 0.5,
        "the run should have lost real energy"
    );
}

/// A driven hinge with a loaded pin: the mechanism is fully prescribed, so
/// the bearing load is the same with and without friction, and the motor
/// effort rises by exactly `μ·R·F_radial` — the load-dependent friction of
/// the commercial tools, and the `∂Q/∂λ` coupling at work.
#[test]
fn pin_friction_adds_its_torque_to_the_motor_effort() {
    let (mu, radius) = (0.2, 0.02);
    let mut plain = pendulum(0.3);
    plain.add_driver(Driver {
        joint: JointId(1),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Constant {
            start: 0.3,
            rate: 2.0,
        },
    });
    let mut rubbing = plain.clone();
    rubbing.add_force(pin_friction(10, 1, mu, radius)).unwrap();

    for time in [0.0, 0.37, 1.1] {
        let free = dynamic_ic(&plain, time).unwrap();
        let worn = dynamic_ic(&rubbing, time).unwrap();
        let hinge = |frame: &DynamicFrame, sys: &System| {
            let r = frame.reactions(sys).unwrap();
            let joint = r
                .iter()
                .find(|r| r.owner == EquationOwner::Joint(JointId(1)))
                .unwrap()
                .clone();
            let effort = r
                .iter()
                .find(|r| matches!(r.owner, EquationOwner::Driver(_)))
                .unwrap()
                .effort
                .unwrap();
            (joint, effort)
        };
        let (free_joint, free_effort) = hinge(&free, &plain);
        let (worn_joint, worn_effort) = hinge(&worn, &rubbing);
        // The prescribed motion fixes the bearing load.
        close_v(
            worn_joint.force_on_i,
            free_joint.force_on_i,
            1.0e-10,
            "bearing load unchanged by friction",
        );
        let axis = DVec3::Z;
        let radial = (free_joint.force_on_i - free_joint.force_on_i.dot(axis) * axis).length();
        let expected = mu * radius * radial;
        let report = worn.friction(&rubbing).unwrap();
        assert_eq!(report.len(), 1);
        close(
            report[0].normal_load,
            radius * radial,
            1.0e-10,
            "friction load from the radial reaction",
        );
        close(report[0].torque, expected, 1.0e-10, "friction torque");
        // The motor carries it: the hinge turns at +2 rad/s, so the
        // friction torque is added to the effort.
        close(
            worn_effort - free_effort,
            expected,
            1.0e-9,
            "friction torque in the motor effort",
        );
    }
}

/// A cylindrical joint sliding and turning at once rubs one surface: the
/// friction force and torque lie on one cone, `√(F² + (T/R)²) = μ·N`, and
/// oppose the surface slip — not two independent frictions.
#[test]
fn cylindrical_friction_shares_one_cone() {
    let (mu, radius) = (0.25, 0.03);
    let (slide, spin) = (0.4, 7.0);
    let axis = DVec3::X;
    let frame = DQuat::from_rotation_arc(DVec3::Z, axis);
    let mut sys = System::new("shaft");
    sys.gravity = Some(DVec3::new(0.0, -G, 0.0));
    sys.add_body(ground()).unwrap();
    let mut shaft = part(2, M, DVec3::ZERO, DVec3::splat(0.02), DVec3::ZERO, frame);
    shaft.velocity = slide * axis;
    shaft.angular_velocity = spin * axis;
    sys.add_body(shaft).unwrap();
    marker(&mut sys, 1, 1, DVec3::ZERO, frame);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    joint(&mut sys, 1, JointKind::Cylindrical, 1, 2);
    sys.add_force(pin_friction(10, 1, mu, radius)).unwrap();

    let frame = dynamic_ic(&sys, 0.0).unwrap();
    let report = frame.friction(&sys).unwrap();
    assert_eq!(report.len(), 1);
    // The shaft's center of mass is on the axis, so the guide carries its
    // weight and nothing else.
    close(report[0].normal_load, M * G, 1.0e-9, "friction load");
    let surface = (slide * slide + (spin * radius).powi(2)).sqrt();
    close(report[0].slip, surface, 1.0e-12, "surface slip speed");
    close(report[0].coefficient, mu, 1.0e-12, "coefficient");
    let cone = mu * M * G;
    close(
        (report[0].force.powi(2) + (report[0].torque / radius).powi(2)).sqrt(),
        cone,
        1.0e-9,
        "the friction stays on its cone",
    );
    // Shared by the slip direction.
    close(report[0].force, cone * slide / surface, 1.0e-9, "slide share");
    close(
        report[0].torque,
        cone * radius * (spin * radius) / surface,
        1.0e-9,
        "turn share",
    );
    // Both components resist: the friction only ever takes power out.
    assert!(
        power(&sys, 0.0, &frame.state, &frame.velocity).unwrap() < 0.0,
        "friction must dissipate"
    );
}

/// A ball socket rubbing under a pure spin: with the center of mass at the
/// ball the socket carries exactly `M·g`, so the spin decays at
/// `μ·R·M·g / I` — a closed form for the spherical friction couple.
#[test]
fn spherical_friction_brakes_a_spin() {
    let (mu, radius) = (0.3, 0.05);
    let (spin, izz) = (5.0, 0.02);
    let mut sys = System::new("socket");
    sys.gravity = Some(DVec3::new(0.0, -G, 0.0));
    sys.add_body(ground()).unwrap();
    let mut ball = part(
        2,
        M,
        DVec3::ZERO,
        DVec3::new(0.02, 0.02, izz),
        DVec3::ZERO,
        DQuat::IDENTITY,
    );
    ball.angular_velocity = DVec3::new(0.0, 0.0, spin);
    sys.add_body(ball).unwrap();
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    joint(&mut sys, 1, JointKind::Spherical, 1, 2);
    sys.add_force(pin_friction(10, 1, mu, radius)).unwrap();

    let frame = dynamic_ic(&sys, 0.0).unwrap();
    let report = frame.friction(&sys).unwrap();
    close(
        report[0].normal_load,
        radius * M * G,
        1.0e-9,
        "the socket carries the weight",
    );
    close(report[0].slip, spin, 1.0e-12, "relative angular speed");
    close(
        report[0].torque,
        mu * radius * M * G,
        1.0e-9,
        "friction couple",
    );
    let decay = -mu * radius * M * G / izz;
    close(
        frame.acceleration[&BodyId(2)].angular.z,
        decay,
        1.0e-9,
        "spin-down rate",
    );
    // A constant braking couple means a linear run-down to rest at
    // `spin / |decay|`, and rest is where it stays.
    let stop = spin / decay.abs();
    let run = simulate(&sys, &grid(1.0, 0.02), &HhtSettings::default()).unwrap();
    let rate = |f: &DynamicFrame| f.velocity[&BodyId(2)].angular.z;
    for f in &run.frames {
        if f.time < 0.9 * stop {
            close(rate(f), spin + decay * f.time, 1.0e-5, "linear run-down");
        } else if f.time > 1.1 * stop {
            assert!(rate(f).abs() < 1.0e-4, "at rest by t = {}", f.time);
        }
    }
    let stopped = run.frames.last().unwrap();
    close(rate(stopped), 0.0, 1.0e-4, "spun down by t = 1 s");
    // Rest: the regularized coefficient has no force left to apply.
    assert!(
        stopped.friction(&sys).unwrap()[0].torque < 1.0e-4 * mu * radius * M * G,
        "no friction left at rest"
    );
}

/// Static equilibrium with stiction: a block whose `μ_s` beats the slope
/// stays put, held within `max_stiction_deformation`. The held friction is
/// exactly the driving load `M·g·sin θ`, and the creep is where the
/// stiction ramp reaches it: `μ_s·N·tanh(Δ/Δ_max) = M·g·sin θ`.
#[test]
fn stiction_holds_a_block_on_a_slope() {
    let (mu_s, deformation) = (0.8, 1.0e-4);
    let (mut sys, down) = incline(SLOPE);
    sys.add_force(
        Fric {
            mu_static: mu_s,
            mu_dynamic: 0.5,
            stiction_velocity: 1.0e-3,
            friction_velocity: 1.0e-2,
            max_stiction_deformation: deformation,
            ..Fric::default()
        }
        .element(10, 1),
    )
    .unwrap();
    assert!(SLOPE.tan() < mu_s, "the slope must be holdable");

    let rest = static_equilibrium(&sys, 0.0, None).unwrap();
    let load = M * G * SLOPE.cos();
    let driving = M * G * SLOPE.sin();
    assert_eq!(rest.friction.len(), 1);
    close(rest.friction[0].normal_load, load, 1.0e-7, "friction load");
    close(rest.friction[0].force, driving, 1.0e-7, "friction holds the block");
    close(
        rest.friction[0].coefficient,
        SLOPE.tan(),
        1.0e-7,
        "only part of μs is needed",
    );
    let creep = rest.state.body_pose(BodyId(2)).unwrap().0.dot(down);
    close(
        creep,
        deformation * stiction_inverse(SLOPE.tan() / mu_s),
        1.0e-5,
        "the stiction creep",
    );
    assert!(
        creep > 0.0 && creep < 2.0 * deformation,
        "creep {creep} out of range"
    );

    // Beyond what stiction can hold, the block cannot rest on the slope:
    // the same solve with a coefficient below the slope does not converge.
    let mut slipping = sys.clone();
    if let Some(ForceElement::JointFriction {
        mu_static,
        mu_dynamic,
        ..
    }) = slipping.forces.get_mut(&ForceId(10))
    {
        *mu_static = 0.2;
        *mu_dynamic = 0.2;
    }
    let err = static_equilibrium(&slipping, 0.0, None).unwrap_err();
    assert!(
        err.code == codes::STATIC_NOT_CONVERGED || err.code == codes::DYNAMICS_SINGULAR,
        "a slope steeper than the stiction has no rest pose: {err}"
    );
}

/// Friction with no stiction deformation is Adams' "sliding only": static
/// equilibrium ignores it (and then a block on a slope has no rest pose),
/// while the dynamic run still rubs.
#[test]
fn sliding_only_friction_leaves_static_equilibrium_alone() {
    let (mut sys, _) = incline(SLOPE);
    sys.add_force(
        Fric {
            mu_static: 0.9,
            mu_dynamic: 0.9,
            preload: 100.0,
            max_stiction_deformation: 0.0,
            ..Fric::default()
        }
        .element(10, 1),
    )
    .unwrap();
    // Nothing resists along the slope at all, which is what the M2
    // static solve says about an unresisted load direction.
    let err = static_equilibrium(&sys, 0.0, None).unwrap_err();
    assert_eq!(err.code, codes::DYNAMICS_SINGULAR, "{err}");
    // The same friction does act in the dynamic run (here it is large
    // enough to nearly stop the block).
    let run = simulate(&sys, &grid(0.1, 0.05), &HhtSettings::default()).unwrap();
    let a = run.frames.last().unwrap().acceleration[&BodyId(2)]
        .linear
        .length();
    assert!(a < 0.1 * G, "the friction should hold it back: {a}");
}

/// The data gates: friction only on the kinds that have a model, once per
/// joint, with usable parameters.
#[test]
fn friction_validation_names_its_codes() {
    let (mut sys, _) = incline(SLOPE);
    // A second element on the same joint.
    let mut twice = sys.clone();
    twice.add_force(constant_friction(10, 1, 0.1, 1.0)).unwrap();
    twice.add_force(constant_friction(11, 1, 0.1, 1.0)).unwrap();
    assert_eq!(
        dynamic_ic(&twice, 0.0).unwrap_err().code,
        codes::FRICTION_UNSUPPORTED
    );
    // A kind with no friction model.
    let mut planar = System::new("planar");
    planar.add_body(ground()).unwrap();
    planar
        .add_body(part(
            2,
            M,
            DVec3::ZERO,
            DVec3::splat(0.01),
            DVec3::ZERO,
            DQuat::IDENTITY,
        ))
        .unwrap();
    marker(&mut planar, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut planar, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    joint(&mut planar, 1, JointKind::Planar, 1, 2);
    planar.add_force(constant_friction(10, 1, 0.1, 1.0)).unwrap();
    assert_eq!(
        dynamic_ic(&planar, 0.0).unwrap_err().code,
        codes::FRICTION_UNSUPPORTED
    );
    // A missing joint.
    let mut dangling = sys.clone();
    dangling.add_force(constant_friction(10, 7, 0.1, 1.0)).unwrap();
    assert_eq!(
        dynamic_ic(&dangling, 0.0).unwrap_err().code,
        codes::DANGLING_JOINT
    );
    // Unusable numbers.
    let usable = Fric {
        preload: 1.0,
        ..Fric::default()
    };
    for bad in [
        Fric {
            mu_static: 0.1,
            mu_dynamic: 0.2,
            ..usable
        },
        Fric {
            stiction_velocity: 0.0,
            ..usable
        },
        Fric {
            stiction_velocity: 1.0,
            friction_velocity: 0.1,
            ..usable
        },
        Fric {
            pin_radius: -0.01,
            ..usable
        },
        Fric {
            preload: f64::NAN,
            ..usable
        },
    ]
    .map(|f| f.element(10, 1))
    {
        let mut broken = sys.clone();
        broken.add_force(bad).unwrap();
        assert_eq!(
            dynamic_ic(&broken, 0.0).unwrap_err().code,
            codes::BAD_FORCE_ELEMENT
        );
    }
    sys.add_force(constant_friction(10, 1, 0.1, 1.0)).unwrap();
    dynamic_ic(&sys, 0.0).unwrap();
}

/// A suppressed joint rubs nothing.
#[test]
fn suppressing_a_joint_suppresses_its_friction() {
    let mut sys = pendulum(0.4);
    sys.add_force(constant_friction(10, 1, 0.3, 1.0)).unwrap();
    sys.joints.get_mut(&JointId(1)).unwrap().suppressed = true;
    let frame = dynamic_ic(&sys, 0.0).unwrap();
    assert!(frame.friction(&sys).unwrap().is_empty());
    // Free fall, nothing holding and nothing rubbing.
    close_v(
        frame.acceleration[&BodyId(2)].linear,
        DVec3::new(0.0, -G, 0.0),
        1.0e-12,
        "free fall",
    );
}

/// Constant friction maps onto MuJoCo's `frictionloss` (the Tier C
/// cross-check); load-dependent friction has no counterpart and says so.
#[test]
fn mjcf_exports_constant_friction_only() {
    let mut sys = pendulum(0.4);
    sys.add_force(constant_friction(10, 1, 0.3, 0.98)).unwrap();
    let xml = write_mjcf(&sys).unwrap();
    assert!(
        xml.contains("frictionloss=\"0.294\""),
        "expected the Coulomb loss in\n{xml}"
    );
    let mut loaded = pendulum(0.4);
    loaded.add_force(pin_friction(10, 1, 0.3, 0.02)).unwrap();
    assert_eq!(
        write_mjcf(&loaded).unwrap_err().code,
        codes::MJCF_UNSUPPORTED
    );
}

/// Load-dependent friction keeps the integrator healthy: the multipliers
/// and the friction load they feed converge together, within a Newton
/// iteration or two of a frictionless run. The `∂r/∂λ` block of the
/// iteration matrix is what buys that — without it this run takes 16% more
/// Newton iterations and four times as many matrix rebuilds (49 against
/// 12), which is what the rebuild gate below guards.
#[test]
fn friction_does_not_cost_the_integrator_its_convergence() {
    let times = grid(2.0, 0.01);
    let plain = pendulum(0.8);
    let mut rubbing = plain.clone();
    rubbing
        .add_force(
            Fric {
                mu_static: 0.5,
                mu_dynamic: 0.4,
                stiction_velocity: 1.0e-3,
                friction_velocity: 1.0e-2,
                pin_radius: 0.08,
                bending_arm: 0.0,
                ..Fric::default()
            }
            .element(10, 1),
        )
        .unwrap();
    let a = simulate(&plain, &times, &HhtSettings::default()).unwrap().stats;
    let b = simulate(&rubbing, &times, &HhtSettings::default())
        .unwrap()
        .stats;
    let per_step = |s: &IntegrationStats| s.newton_iterations as f64 / s.steps as f64;
    assert!(
        per_step(&b) <= per_step(&a) + 2.0,
        "friction cost {:.2} Newton iterations per step against {:.2} without it ({b:?})",
        per_step(&b),
        per_step(&a)
    );
    assert!(
        b.jacobians <= 25,
        "the friction's λ block should keep the matrix fresh for long stretches: {b:?}"
    );
    // A few more rejections than the smooth run are fair — the
    // coefficient swings fastest where the joint reverses — but they must
    // stay a tail of the run.
    assert!(
        20 * b.rejected <= b.steps,
        "too many rejected steps: {b:?} against {a:?}"
    );
}

/// The document form: a friction element round-trips, and the fields a
/// terse document leaves out take their documented defaults (the M5 red
/// line — new fields are `serde(default)`, so no document breaks).
#[test]
fn friction_serializes_with_defaults() {
    let element = Fric {
        mu_static: 0.4,
        mu_dynamic: 0.3,
        stiction_velocity: 1.0e-3,
        friction_velocity: 1.0e-2,
        pin_radius: 0.02,
        thrust_arm: 0.01,
        bending_arm: 0.05,
        preload: 0.5,
        ..Fric::default()
    }
    .element(10, 1);
    let text = serde_json::to_string(&element).unwrap();
    assert_eq!(
        serde_json::from_str::<ForceElement>(&text).unwrap(),
        element,
        "{text}"
    );
    let terse: ForceElement = serde_json::from_str(
        r#"{"kind":"joint_friction","id":10,"name":"friction10","joint":1,
            "mu_static":0.4,"mu_dynamic":0.3,
            "stiction_velocity":0.001,"friction_velocity":0.01}"#,
    )
    .unwrap();
    let ForceElement::JointFriction {
        reaction_force,
        pin_radius,
        thrust_arm,
        bending_arm,
        preload,
        max_stiction_deformation,
        ..
    } = terse
    else {
        panic!("not a friction element");
    };
    assert!(reaction_force, "the reaction force rubs unless told not to");
    assert_eq!(
        (pin_radius, thrust_arm, bending_arm, preload),
        (0.0, 0.0, 0.0, 0.0),
        "no arms and no preload by default"
    );
    close(max_stiction_deformation, 1.0e-4, 1.0e-12, "stiction default");
}
