//! Tier A kinematic benchmarks (roadmap §6): closed-form mechanisms swept
//! over several motor revolutions, checked on all three levels.
//!
//! Both mechanisms are planar loops built from 3D joints — the CAD case:
//! Grübler counts −2 DOF, the true mobility is 1, so the solver must find
//! and skip exactly three redundant equations.

use glam::{DMat3, DQuat, DVec3};
use rcad_mbd::*;

fn body(id: u64, name: &str, at: DVec3, grounded: bool) -> Body {
    Body {
        id: BodyId(id),
        name: name.into(),
        mass_kg: if grounded { 0.0 } else { 1.0 },
        com: DVec3::ZERO,
        inertia_com: if grounded {
            DMat3::ZERO
        } else {
            DMat3::from_diagonal(DVec3::splat(0.01))
        },
        translation: at,
        rotation: DQuat::IDENTITY,
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

fn grid(end: f64, step: f64) -> Vec<f64> {
    let n = (end / step).round() as usize;
    (0..=n).map(|k| k as f64 * step).collect()
}

// ── crank-slider ────────────────────────────────────────────────────────

const R: f64 = 0.05;
const L: f64 = 0.2;
const THETA0: f64 = 0.3;
const OMEGA: f64 = std::f64::consts::TAU; // one revolution per second

/// Ground, crank (revolute at the origin), rod, slider (prismatic along
/// world X), authored at crank angle 0; the motor turns the crank from
/// `THETA0` at `OMEGA`.
fn crank_slider() -> System {
    let mut sys = System::new("crank-slider");
    sys.add_body(body(1, "ground", DVec3::ZERO, true)).unwrap();
    sys.add_body(body(2, "crank", DVec3::ZERO, false)).unwrap();
    sys.add_body(body(3, "rod", DVec3::new(R, 0.0, 0.0), false))
        .unwrap();
    sys.add_body(body(4, "slider", DVec3::new(R + L, 0.0, 0.0), false))
        .unwrap();
    let along_x = DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2); // Z → X
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 1, DVec3::ZERO, along_x);
    marker(&mut sys, 3, 2, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 4, 2, DVec3::new(R, 0.0, 0.0), DQuat::IDENTITY);
    marker(&mut sys, 5, 3, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 6, 3, DVec3::new(L, 0.0, 0.0), DQuat::IDENTITY);
    marker(&mut sys, 7, 4, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 8, 4, DVec3::ZERO, along_x);
    joint(&mut sys, 1, JointKind::Revolute, 1, 3);
    joint(&mut sys, 2, JointKind::Revolute, 4, 5);
    joint(&mut sys, 3, JointKind::Revolute, 6, 7);
    joint(&mut sys, 4, JointKind::Prismatic, 2, 8);
    sys.add_driver(Driver {
        joint: JointId(1),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Constant {
            start: THETA0,
            rate: OMEGA,
        },
    });
    sys
}

/// `x(θ) = r cosθ + √(l² − r² sin²θ)` and its first two θ-derivatives.
fn slider_closed_form(theta: f64) -> (f64, f64, f64) {
    let (s, c) = theta.sin_cos();
    let root = (L * L - R * R * s * s).sqrt();
    let x = R * c + root;
    let dx = -R * s - R * R * s * c / root;
    let ddx = -R * c - R * R * (c * c - s * s) / root - R.powi(4) * s * s * c * c / root.powi(3);
    (x, dx, ddx)
}

#[test]
fn crank_slider_matches_closed_form_over_three_revolutions() {
    let sys = crank_slider();
    let frames = kinematic_sweep(&sys, &grid(3.0, 0.01)).expect("sweep runs");
    let state = &frames[0].state;
    let diag = diagnose(&sys, state, 0.0).unwrap();
    assert_eq!(diag.coordinates, 18);
    assert_eq!((diag.rank, diag.dof, diag.mobility), (18, 0, 1), "{diag:?}");
    assert_eq!(
        diag.redundant.len(),
        3,
        "planar loop of 3D joints: 3 redundant"
    );
    assert!(diag.conflicting.is_empty());
    for frame in &frames {
        let theta = THETA0 + OMEGA * frame.time;
        let (x, dx, ddx) = slider_closed_form(theta);
        let (pos, _) = frame.state.body_pose(BodyId(4)).unwrap();
        let v = frame.velocity[&BodyId(4)].linear;
        let a = frame.acceleration[&BodyId(4)].linear;
        let t = frame.time;
        assert!((pos.x - x).abs() < 1e-9, "t={t}: x {} vs {x}", pos.x);
        assert!(pos.y.abs() < 1e-9 && pos.z.abs() < 1e-9);
        assert!(
            (v.x - OMEGA * dx).abs() < 1e-7,
            "t={t}: v {} vs {}",
            v.x,
            OMEGA * dx
        );
        assert!(
            (a.x - OMEGA * OMEGA * ddx).abs() < 1e-5,
            "t={t}: a {} vs {}",
            a.x,
            OMEGA * OMEGA * ddx
        );
        assert_eq!(frame.redundant.len(), 3);
        // The crank turns at the motor rate about +Z.
        let w = frame.velocity[&BodyId(2)].angular;
        assert!((w - DVec3::Z * OMEGA).length() < 1e-7, "t={t}: crank {w:?}");
    }
}

/// Deterministic: the same sweep twice gives bit-identical frames.
#[test]
fn sweep_is_deterministic() {
    let sys = crank_slider();
    let times = grid(0.5, 0.05);
    let a = kinematic_sweep(&sys, &times).unwrap();
    let b = kinematic_sweep(&sys, &times).unwrap();
    assert_eq!(a, b);
}

// ── four-bar (Freudenstein) ─────────────────────────────────────────────

const A: f64 = 0.1; // crank
const B: f64 = 0.3; // coupler
const C: f64 = 0.25; // rocker
const D: f64 = 0.3; // ground (pivot distance)
const PHI0: f64 = 0.5;
const W: f64 = 3.0;

/// The coupler/rocker joint `C` for crank angle `phi` on the open branch
/// (left of the crank tip → rocker pivot direction).
fn coupler_point(phi: f64) -> DVec3 {
    let b = DVec3::new(A * phi.cos(), A * phi.sin(), 0.0);
    let d = DVec3::new(D, 0.0, 0.0);
    let e = d - b;
    let len = e.length();
    let u = e / len;
    let along = (B * B - C * C + len * len) / (2.0 * len);
    let h = (B * B - along * along).sqrt();
    b + u * along + DVec3::new(-u.y, u.x, 0.0) * h
}

fn rocker_angle(phi: f64) -> f64 {
    let c = coupler_point(phi) - DVec3::new(D, 0.0, 0.0);
    c.y.atan2(c.x)
}

fn four_bar() -> System {
    let b0 = DVec3::new(A * PHI0.cos(), A * PHI0.sin(), 0.0);
    let c0 = coupler_point(PHI0);
    let d = DVec3::new(D, 0.0, 0.0);
    let mut sys = System::new("four-bar");
    sys.add_body(body(1, "ground", DVec3::ZERO, true)).unwrap();
    sys.add_body(body(2, "crank", DVec3::ZERO, false)).unwrap();
    sys.add_body(body(3, "coupler", b0, false)).unwrap();
    sys.add_body(body(4, "rocker", d, false)).unwrap();
    let id = DQuat::IDENTITY;
    marker(&mut sys, 1, 1, DVec3::ZERO, id);
    marker(&mut sys, 2, 2, DVec3::ZERO, id);
    marker(&mut sys, 3, 2, b0, id);
    marker(&mut sys, 4, 3, DVec3::ZERO, id);
    marker(&mut sys, 5, 3, c0 - b0, id);
    marker(&mut sys, 6, 4, c0 - d, id);
    marker(&mut sys, 7, 4, DVec3::ZERO, id);
    marker(&mut sys, 8, 1, d, id);
    joint(&mut sys, 1, JointKind::Revolute, 1, 2);
    joint(&mut sys, 2, JointKind::Revolute, 3, 4);
    joint(&mut sys, 3, JointKind::Revolute, 5, 6);
    joint(&mut sys, 4, JointKind::Revolute, 8, 7);
    sys.add_driver(Driver {
        joint: JointId(1),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Constant {
            start: 0.0,
            rate: W,
        },
    });
    sys
}

#[test]
fn four_bar_satisfies_freudenstein_over_two_revolutions() {
    let sys = four_bar();
    let c0 = coupler_point(PHI0) - DVec3::new(D, 0.0, 0.0);
    let k1 = D / A;
    let k2 = D / C;
    let k3 = (A * A - B * B + C * C + D * D) / (2.0 * A * C);
    let frames = kinematic_sweep(&sys, &grid(2.0 * std::f64::consts::TAU / W, 0.01)).unwrap();
    let h = 1e-5;
    for frame in &frames {
        let t = frame.time;
        let phi = PHI0 + W * t;
        let tip = frame.state.world_point(BodyId(4), c0).unwrap() - DVec3::new(D, 0.0, 0.0);
        let psi = tip.y.atan2(tip.x);
        let freudenstein = k1 * psi.cos() - k2 * phi.cos() + k3 - (phi - psi).cos();
        assert!(
            freudenstein.abs() < 1e-9,
            "t={t}: Freudenstein residual {freudenstein}"
        );
        let want = rocker_angle(phi);
        let off = (psi - want + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        assert!(
            off.abs() < 1e-9,
            "t={t}: stayed on the open branch ({psi} vs {want})"
        );
        // Rocker rate and acceleration against central differences of the
        // closed form (dψ/dt = W·dψ/dφ).
        let rate = W * (rocker_angle(phi + h) - rocker_angle(phi - h)) / (2.0 * h);
        let h2 = 1e-4;
        let accel =
            W * W * (rocker_angle(phi + h2) - 2.0 * want + rocker_angle(phi - h2)) / (h2 * h2);
        let w = frame.velocity[&BodyId(4)].angular;
        let alpha = frame.acceleration[&BodyId(4)].angular;
        assert!(
            (w.z - rate).abs() < 1e-6,
            "t={t}: rocker rate {} vs {rate}",
            w.z
        );
        assert!(
            (alpha.z - accel).abs() < 1e-3 * accel.abs().max(1.0),
            "t={t}: rocker accel {} vs {accel}",
            alpha.z
        );
        assert!(w.x.abs() < 1e-9 && w.y.abs() < 1e-9, "stays planar");
        assert_eq!(frame.redundant.len(), 3);
    }
}

/// Without its motor the four-bar assembles (position IC) but has one free
/// degree of freedom, which the kinematic levels report as data.
#[test]
fn undriven_four_bar_is_underdetermined() {
    let mut sys = four_bar();
    sys.drivers.clear();
    let state = position_ic(&sys, 0.0).expect("assembles");
    let diag = diagnose(&sys, &state, 0.0).unwrap();
    assert_eq!((diag.dof, diag.mobility, diag.redundant.len()), (1, 1, 3));
    let err = solve_frame(&sys, 0.0, None).unwrap_err();
    assert_eq!(err.code, codes::UNDERDETERMINED_MOTION);
}

/// A second motor on the rocker pivot over-prescribes the single DOF: with
/// rates the loop cannot honour, the position solve fails as data and the
/// diagnosis names a conflicting driver equation.
#[test]
fn over_driven_four_bar_reports_the_conflict() {
    let mut sys = four_bar();
    sys.add_driver(Driver {
        joint: JointId(4),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Constant {
            start: 0.0,
            rate: 5.0,
        },
    });
    let err = solve_frame(&sys, 0.4, None).unwrap_err();
    assert_eq!(err.code, codes::KINEMATICS_NOT_CONVERGED, "{err}");
    assert!(err.detail.contains("conflicting"), "{}", err.detail);
}

// ── joint primitives ────────────────────────────────────────────────────

/// The crank-slider with its undriven pairs decomposed into primitives —
/// revolute = spherical + parallel axes, prismatic = in-line + orientation —
/// is the same equation set: identical motion, identical redundancy.
#[test]
fn primitive_decomposition_matches_the_lower_pairs() {
    let pairs = crank_slider();
    let mut prims = crank_slider();
    for (id, kind, m1, m2) in [(2, JointKind::Revolute, 4, 5), (3, JointKind::Revolute, 6, 7)] {
        assert_eq!(prims.joints[&JointId(id)].kind, kind);
        prims.joints.get_mut(&JointId(id)).unwrap().kind = JointKind::Spherical;
        joint(&mut prims, 10 + id, JointKind::ParallelAxes, m1, m2);
    }
    prims.joints.get_mut(&JointId(4)).unwrap().kind = JointKind::InLine;
    joint(&mut prims, 14, JointKind::Orientation, 2, 8);
    let times = grid(1.2, 0.05);
    let a = kinematic_sweep(&pairs, &times).unwrap();
    let b = kinematic_sweep(&prims, &times).unwrap();
    for (fa, fb) in a.iter().zip(&b) {
        assert_eq!(fb.redundant.len(), 3);
        for id in pairs.bodies.keys() {
            let (pa, ra) = fa.state.body_pose(*id).unwrap();
            let (pb, rb) = fb.state.body_pose(*id).unwrap();
            let turn = ra.inverse() * rb;
            let angle = 2.0 * turn.xyz().length().atan2(turn.w.abs());
            assert!((pa - pb).length() < 1e-12 && angle < 1e-10, "{:e} {angle:e}", (pa - pb).length());
            assert!((fa.velocity[id].linear - fb.velocity[id].linear).length() < 1e-10);
            assert!((fa.acceleration[id].linear - fb.acceleration[id].linear).length() < 1e-8);
        }
    }
}

// ── couplers ────────────────────────────────────────────────────────────

/// A motor pinion on its own revolute drives the crank through a gear pair
/// (crank undriven): the crank turns at `ratio · ω_motor`, and the slider
/// follows the crank-slider closed form over six motor revolutions — the
/// rotation coordinates unwrap across turns.
#[test]
fn gear_train_drives_the_crank_slider() {
    let ratio = -0.5;
    let w_motor = std::f64::consts::TAU;
    let mut sys = crank_slider();
    sys.drivers.clear();
    sys.add_body(body(5, "motor", DVec3::new(-0.15, 0.0, 0.0), false)).unwrap();
    marker(&mut sys, 9, 1, DVec3::new(-0.15, 0.0, 0.0), DQuat::IDENTITY);
    marker(&mut sys, 10, 5, DVec3::ZERO, DQuat::IDENTITY);
    joint(&mut sys, 5, JointKind::Revolute, 9, 10);
    sys.add_driver(Driver {
        joint: JointId(5),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Constant { start: 0.0, rate: w_motor },
    });
    sys.add_coupler(Coupler::gear(CouplerId(1), "mesh", JointId(5), JointId(1), ratio))
        .unwrap();
    let frames = kinematic_sweep(&sys, &grid(6.0, 0.02)).unwrap();
    for frame in &frames {
        let t = frame.time;
        let theta = ratio * w_motor * t;
        let (x, dx, ddx) = slider_closed_form(theta);
        let w = ratio * w_motor;
        let (pos, _) = frame.state.body_pose(BodyId(4)).unwrap();
        assert!((pos.x - x).abs() < 1e-9, "t={t}: x {} vs {x}", pos.x);
        assert!((frame.velocity[&BodyId(4)].linear.x - w * dx).abs() < 1e-7, "t={t}");
        assert!((frame.acceleration[&BodyId(4)].linear.x - w * w * ddx).abs() < 1e-5, "t={t}");
        let crank = frame.state.joint_angle(JointId(1)).unwrap();
        assert!((crank - theta).abs() < 1e-9, "t={t}: unwrapped crank {crank} vs {theta}");
        let motor = frame.state.joint_angle(JointId(5)).unwrap();
        assert!((motor - w_motor * t).abs() < 1e-9, "t={t}: unwrapped motor {motor}");
    }
    // Without the motor the coupled train keeps exactly one mobility.
    let mut free = sys.clone();
    free.drivers.clear();
    let state = position_ic(&free, 0.0).unwrap();
    let diag = diagnose(&free, &state, 0.0).unwrap();
    assert_eq!((diag.mobility, diag.dof), (1, 1), "{diag:?}");
}

/// Rack and pinion: a driven pinion moves a prismatic rack by `r · θ`.
#[test]
fn rack_and_pinion() {
    let r = 0.03;
    let mut sys = System::new("rack-pinion");
    sys.add_body(body(1, "ground", DVec3::ZERO, true)).unwrap();
    sys.add_body(body(2, "pinion", DVec3::ZERO, false)).unwrap();
    sys.add_body(body(3, "rack", DVec3::new(0.0, -r, 0.0), false)).unwrap();
    let along_x = DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2);
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 3, 1, DVec3::new(0.0, -r, 0.0), along_x);
    marker(&mut sys, 4, 3, DVec3::ZERO, along_x);
    joint(&mut sys, 1, JointKind::Revolute, 1, 2);
    joint(&mut sys, 2, JointKind::Prismatic, 3, 4);
    sys.add_driver(Driver {
        joint: JointId(1),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Harmonic { offset: 0.0, amplitude: 9.0, frequency_hz: 0.25, phase: 0.0 },
    });
    sys.add_coupler(Coupler::rack_pinion(CouplerId(1), "rack", JointId(1), JointId(2), r))
        .unwrap();
    for frame in kinematic_sweep(&sys, &grid(4.0, 0.02)).unwrap() {
        let s = sys.drivers[0].sample(frame.time);
        let (pos, _) = frame.state.body_pose(BodyId(3)).unwrap();
        assert!((pos.x - r * s.value).abs() < 1e-10, "t={}: {} vs {}", frame.time, pos.x, r * s.value);
        assert!((frame.velocity[&BodyId(3)].linear.x - r * s.velocity).abs() < 1e-9);
        assert!((frame.acceleration[&BodyId(3)].linear.x - r * s.acceleration).abs() < 1e-8);
    }
}

/// A screw on one cylindrical joint: only the rotation is driven, the
/// coupler advances the nut by `lead / 2π` per radian over three turns.
#[test]
fn screw_advances_by_its_lead() {
    let (lead, w) = (0.004, 5.0);
    let mut sys = System::new("screw");
    sys.add_body(body(1, "ground", DVec3::ZERO, true)).unwrap();
    sys.add_body(body(2, "nut", DVec3::ZERO, false)).unwrap();
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    joint(&mut sys, 1, JointKind::Cylindrical, 1, 2);
    sys.add_driver(Driver {
        joint: JointId(1),
        slot: DriverSlot::Rotation,
        profile: DriverProfile::Constant { start: 0.0, rate: w },
    });
    sys.add_coupler(Coupler::screw(CouplerId(1), "thread", JointId(1), lead)).unwrap();
    let t_end = 3.0 * std::f64::consts::TAU / w;
    for frame in kinematic_sweep(&sys, &grid(t_end, 0.01)).unwrap() {
        let theta = w * frame.time;
        let (pos, _) = frame.state.body_pose(BodyId(2)).unwrap();
        let z = lead / std::f64::consts::TAU * theta;
        assert!((pos.z - z).abs() < 1e-12, "t={}: z {} vs {z}", frame.time, pos.z);
        assert!((frame.velocity[&BodyId(2)].linear.z - lead / std::f64::consts::TAU * w).abs() < 1e-12);
        assert!((frame.state.joint_angle(JointId(1)).unwrap() - theta).abs() < 1e-9);
    }
}

/// The coupler gate: missing joints, unmeasurable coordinates and zero
/// coefficients are data.
#[test]
fn coupler_validation() {
    let base = crank_slider();
    let mut dangling = base.clone();
    dangling
        .add_coupler(Coupler::gear(CouplerId(1), "g", JointId(1), JointId(99), 2.0))
        .unwrap();
    assert_eq!(dangling.validate().unwrap_err().code, codes::DANGLING_JOINT);
    let mut wrong_slot = base.clone();
    wrong_slot
        .add_coupler(Coupler::rack_pinion(CouplerId(1), "r", JointId(1), JointId(2), 0.1))
        .unwrap();
    assert_eq!(wrong_slot.validate().unwrap_err().code, codes::BAD_COUPLER, "a revolute has no translation");
    let mut zero = base.clone();
    zero.add_coupler(Coupler::gear(CouplerId(1), "g", JointId(1), JointId(2), 0.0)).unwrap();
    assert_eq!(zero.validate().unwrap_err().code, codes::BAD_COUPLER);
    let mut dup = base;
    dup.add_coupler(Coupler::gear(CouplerId(1), "g", JointId(1), JointId(2), 1.0)).unwrap();
    assert_eq!(
        dup.add_coupler(Coupler::gear(CouplerId(1), "g", JointId(1), JointId(2), 1.0)).unwrap_err().code,
        codes::DUPLICATE_ID
    );
}
