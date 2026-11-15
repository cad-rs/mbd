//! Tier A benchmarks of contact and impact (M5-2): closed forms for the
//! penalty contact — the Hertzian impact's peak penetration, an elastic
//! impact's restitution, energy and impulse, the linear spring contact's
//! duration, resting penetration under gravity, box and capsule contact
//! sets, sliding Coulomb friction, and the data gates.

mod common;

use common::*;
use glam::{DQuat, DVec3};
use rcad_mbd::*;

/// A frictionless Hertzian contact (`e = 3/2`, no damping) of stiffness
/// `k`: conservative, so the closed forms are exact.
fn hertz(k: f64) -> Impact {
    Impact {
        stiffness: k,
        exponent: 1.5,
        damping: 0.0,
        damping_depth: 1.0e-4,
    }
}

fn contact(
    id: u64,
    first: ContactGeometry,
    second: ContactGeometry,
    impact: Impact,
    friction: Option<ContactFriction>,
) -> ForceElement {
    ForceElement::Contact {
        id: ForceId(id),
        name: format!("contact{id}"),
        first,
        second,
        impact,
        friction,
    }
}

fn rubbing(mu: f64) -> Option<ContactFriction> {
    Some(ContactFriction {
        mu_static: mu,
        mu_dynamic: mu,
        stiction_velocity: 1.0e-4,
        friction_velocity: 1.0e-4,
    })
}

/// A ground plane (marker 1 on the ground, the world XY plane, `+Z` out)
/// and one free body of mass [`M`] whose marker 2 sits at its origin.
fn on_the_ground(gravity: bool, at: DVec3, rotation: DQuat, inertia: DVec3) -> System {
    let mut sys = System::new("contact");
    if gravity {
        sys.gravity = Some(DVec3::new(0.0, 0.0, -G));
    }
    sys.add_body(ground()).unwrap();
    sys.add_body(part(2, M, DVec3::ZERO, inertia, at, rotation))
        .unwrap();
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    sys
}

fn plane() -> ContactGeometry {
    ContactGeometry::Plane { marker: MarkerId(1) }
}

fn sphere(marker: u64, radius: f64) -> ContactGeometry {
    ContactGeometry::Sphere {
        marker: MarkerId(marker),
        radius,
    }
}

/// A free sphere of radius `r` whose centre starts exactly `r` above the
/// plane, moving down at `speed`, with no gravity: a pure impact.
fn impact_rig(r: f64, speed: f64, impact: Impact) -> System {
    let mut sys = on_the_ground(false, DVec3::new(0.0, 0.0, r), DQuat::IDENTITY, DVec3::splat(0.01));
    sys.bodies.get_mut(&BodyId(2)).unwrap().velocity = DVec3::new(0.0, 0.0, -speed);
    sys.add_force(contact(10, sphere(2, r), plane(), impact, None))
        .unwrap();
    sys
}

/// A Hertzian impact is conservative: the peak penetration is where the
/// kinetic energy has all gone into the penalty's potential,
/// `½mv² = k·δ^(e+1)/(e+1)`, the sphere leaves at the speed it arrived
/// with, and the impulse is `2mv`.
#[test]
fn hertzian_impact_has_its_closed_form_peak_and_restitution() {
    let (k, e, v0, r) = (1.0e7, 1.5, 1.0, 0.05);
    let settings = HhtSettings {
        error_tol: Some(1.0e-10),
        ..HhtSettings::default()
    };
    let sys = impact_rig(r, v0, hertz(k));
    let times = grid(0.012, 1.0e-5);
    let run = simulate(&sys, &times, &settings).unwrap();

    let expected_peak = ((e + 1.0) * M * v0 * v0 / (2.0 * k)).powf(1.0 / (e + 1.0));
    let mut peak = 0.0_f64;
    let mut impulse = 0.0;
    let mut first = None;
    let mut last = None;
    for (step, frame) in run.frames.iter().enumerate() {
        let loads = frame.contacts(&sys).unwrap();
        assert_eq!(loads.len(), 1, "a sphere on a plane touches at one point");
        let load = loads[0];
        peak = peak.max(load.depth);
        impulse += load.normal_force * 1.0e-5;
        if load.normal_force > 0.0 {
            first.get_or_insert(step);
            last = Some(step);
        }
        // The contact never pulls, and the normal always points up.
        assert!(load.normal_force >= 0.0);
        close_v(load.normal, DVec3::Z, 1.0e-12, "contact normal");
        if load.depth <= 0.0 {
            assert_eq!(load.normal_force, 0.0, "no force without penetration");
        }
    }
    close(peak, expected_peak, 2.0e-4, "peak penetration");
    close(impulse, 2.0 * M * v0, 2.0e-3, "normal impulse");

    // Restitution 1: it leaves as fast as it came, and the energy (with
    // the penalty's stored deformation) never moves.
    let rebound = run.frames.last().unwrap().velocity[&BodyId(2)].linear.z;
    close(rebound, v0, 1.0e-5, "rebound speed");
    let e0 = energy(&sys, &run.frames[0].state, &run.frames[0].velocity)
        .unwrap()
        .total();
    for frame in &run.frames {
        let total = energy(&sys, &frame.state, &frame.velocity).unwrap().total();
        close(total, e0, 1.0e-5, "energy through the impact");
    }
    assert!(first.unwrap() > 0 && last.unwrap() < run.frames.len() - 1);
}

/// A linear penalty (`e = 1`) is a spring: the contact lasts exactly half
/// its period, `π·√(m/k)`, whatever the impact speed.
#[test]
fn a_linear_contact_lasts_half_a_spring_period() {
    let (k, r, v0) = (1.0e6, 0.05, 2.0);
    let impact = Impact {
        stiffness: k,
        exponent: 1.0,
        damping: 0.0,
        damping_depth: 1.0e-4,
    };
    let h = 1.0e-5;
    let settings = HhtSettings {
        error_tol: Some(1.0e-11),
        ..HhtSettings::default()
    };
    let sys = impact_rig(r, v0, impact);
    let run = simulate(&sys, &grid(0.008, h), &settings).unwrap();
    let touching: Vec<f64> = run
        .frames
        .iter()
        .filter(|f| f.contacts(&sys).unwrap()[0].normal_force > 0.0)
        .map(|f| f.time)
        .collect();
    let duration = touching.last().unwrap() - touching.first().unwrap();
    let expected = std::f64::consts::PI * (M / k).sqrt();
    close(duration, expected, 1.0e-2, "contact duration");
    // The peak penetration of a linear spring is v·√(m/k).
    let peak = run
        .frames
        .iter()
        .map(|f| f.contacts(&sys).unwrap()[0].depth)
        .fold(0.0_f64, f64::max);
    close(peak, v0 * (M / k).sqrt(), 1.0e-4, "peak penetration");
}

/// Damping takes energy out of an impact: the rebound speed falls as the
/// damping rises, and the force stays continuous at first touch (no
/// impulsive jump at `d = 0`).
#[test]
fn damping_sets_the_restitution() {
    let (k, r, v0) = (1.0e7, 0.05, 1.0);
    let settings = HhtSettings {
        error_tol: Some(1.0e-10),
        ..HhtSettings::default()
    };
    let mut previous = f64::INFINITY;
    for damping in [0.0, 100.0, 500.0, 2000.0] {
        let impact = Impact {
            damping,
            ..hertz(k)
        };
        let sys = impact_rig(r, v0, impact);
        let run = simulate(&sys, &grid(0.012, 1.0e-4), &settings).unwrap();
        let rebound = run.frames.last().unwrap().velocity[&BodyId(2)].linear.z;
        assert!(
            rebound > 0.0 && rebound <= v0 + 1.0e-6,
            "damping {damping}: rebound {rebound}"
        );
        assert!(
            rebound < previous,
            "damping {damping} should rebound slower than {previous}: {rebound}"
        );
        previous = rebound;
    }
}

/// A sphere resting on a plane settles where the penalty carries its
/// weight: `k·δᵉ = m·g`, and static equilibrium finds it.
#[test]
fn a_resting_sphere_penetrates_by_its_weight() {
    let (k, e, r) = (1.0e6, 1.5, 0.05);
    let mut sys = on_the_ground(
        true,
        DVec3::new(0.0, 0.0, r),
        DQuat::IDENTITY,
        DVec3::splat(0.01),
    );
    sys.add_force(contact(10, sphere(2, r), plane(), hertz(k), None))
        .unwrap();
    let rest = static_equilibrium(&sys, 0.0, None).unwrap();
    let expected = (M * G / k).powf(1.0 / e);
    let height = rest.state.body_pose(BodyId(2)).unwrap().0.z;
    close(r - height, expected, 1.0e-6, "resting penetration");
    // And the report at that pose carries exactly the weight.
    let held = loads_at(&sys, &rest.state);
    close(held[0].normal_force, M * G, 1.0e-6, "the contact holds the weight");
    close(held[0].depth, expected, 1.0e-6, "reported penetration");
}

/// The contact loads at a given pose, at rest.
fn loads_at(sys: &System, state: &KinematicState) -> Vec<ContactLoad> {
    let at_rest = std::collections::BTreeMap::new();
    let frame = forward_dynamics(sys, 0.0, state, &at_rest).unwrap();
    frame.contacts(sys).unwrap()
}

/// Two spheres meeting head-on with a conservative penalty: momentum is
/// exactly conserved, the relative velocity reverses (restitution 1), and
/// the energy does not move.
#[test]
fn two_spheres_bounce_elastically() {
    let (k, r) = (1.0e7, 0.05);
    let (m1, m2) = (2.0, 3.0);
    let mut sys = System::new("spheres");
    let mut a = part(
        1,
        m1,
        DVec3::ZERO,
        DVec3::splat(0.01),
        DVec3::new(-r, 0.0, 0.0),
        DQuat::IDENTITY,
    );
    a.velocity = DVec3::new(1.5, 0.0, 0.0);
    let mut b = part(
        2,
        m2,
        DVec3::ZERO,
        DVec3::splat(0.01),
        DVec3::new(r, 0.0, 0.0),
        DQuat::IDENTITY,
    );
    b.velocity = DVec3::new(-0.5, 0.0, 0.0);
    sys.add_body(a).unwrap();
    sys.add_body(b).unwrap();
    marker(&mut sys, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    sys.add_force(contact(10, sphere(1, r), sphere(2, r), hertz(k), None))
        .unwrap();
    let settings = HhtSettings {
        error_tol: Some(1.0e-10),
        ..HhtSettings::default()
    };
    let run = simulate(&sys, &grid(0.02, 1.0e-4), &settings).unwrap();
    let first = &run.frames[0];
    let last = run.frames.last().unwrap();
    let p = |f: &DynamicFrame| m1 * f.velocity[&BodyId(1)].linear + m2 * f.velocity[&BodyId(2)].linear;
    close_v(p(last), p(first), 1.0e-8, "linear momentum");
    let relative = |f: &DynamicFrame| {
        (f.velocity[&BodyId(2)].linear - f.velocity[&BodyId(1)].linear).x
    };
    close(relative(last), -relative(first), 1.0e-5, "restitution 1");
    let total = |f: &DynamicFrame| energy(&sys, &f.state, &f.velocity).unwrap().total();
    close(total(last), total(first), 1.0e-5, "energy");
    // The normal points from the second sphere to the first: `−X` here.
    let hit = run
        .frames
        .iter()
        .find(|f| f.contacts(&sys).unwrap()[0].normal_force > 0.0)
        .unwrap();
    close_v(
        hit.contacts(&sys).unwrap()[0].normal,
        -DVec3::X,
        1.0e-12,
        "normal from the second toward the first",
    );
}

/// A box on a plane rests on its four lower corners — each carrying a
/// quarter of the weight at the penetration that quarter produces — and
/// reports its upper four as separated.
#[test]
fn a_box_rests_on_its_lower_corners() {
    let (k, e) = (1.0e6, 1.5);
    let half = DVec3::new(0.1, 0.08, 0.05);
    let mut sys = on_the_ground(
        true,
        DVec3::new(0.0, 0.0, half.z),
        DQuat::IDENTITY,
        DVec3::new(0.01, 0.01, 0.01),
    );
    sys.add_force(contact(
        10,
        ContactGeometry::Box {
            marker: MarkerId(2),
            half_extents: half,
        },
        plane(),
        hertz(k),
        None,
    ))
    .unwrap();
    let rest = static_equilibrium(&sys, 0.0, None).unwrap();
    let loads = loads_at(&sys, &rest.state);
    assert_eq!(loads.len(), 8, "a box offers its eight corners");
    let (touching, apart): (Vec<&ContactLoad>, Vec<&ContactLoad>) =
        loads.iter().partition(|l| l.depth > 0.0);
    assert_eq!(touching.len(), 4, "it rests on four of them");
    let expected = (M * G / (4.0 * k)).powf(1.0 / e);
    for load in &touching {
        close(load.depth, expected, 1.0e-6, "corner penetration");
        close(load.normal_force, M * G / 4.0, 1.0e-6, "corner load");
        close_v(load.normal, DVec3::Z, 1.0e-12, "corner normal");
    }
    for load in &apart {
        assert_eq!(load.normal_force, 0.0, "a separated corner carries nothing");
    }
    close(
        touching.iter().map(|l| l.normal_force).sum::<f64>(),
        M * G,
        1.0e-6,
        "the corners carry the weight",
    );
}

/// A box sliding on a plane with contact friction slows at `μ·g` — the
/// Coulomb deceleration, with the normal load its own contacts give. The
/// friction acts at the corners, below the centre of mass, so the box
/// rocks until the contact damping settles it; the closed form is the
/// steady state.
#[test]
fn a_sliding_box_slows_at_the_coulomb_rate() {
    let (k, mu) = (1.0e6, 0.25);
    let impact = Impact {
        damping: 3000.0,
        damping_depth: 1.0e-5,
        ..hertz(k)
    };
    let half = DVec3::new(0.1, 0.08, 0.05);
    let depth = (M * G / (4.0 * k)).powf(1.0 / 1.5);
    let mut sys = on_the_ground(
        true,
        DVec3::new(0.0, 0.0, half.z - depth),
        DQuat::IDENTITY,
        DVec3::new(0.01, 0.01, 0.01),
    );
    sys.bodies.get_mut(&BodyId(2)).unwrap().velocity = DVec3::new(1.0, 0.0, 0.0);
    sys.add_force(contact(
        10,
        ContactGeometry::Box {
            marker: MarkerId(2),
            half_extents: half,
        },
        plane(),
        impact,
        rubbing(mu),
    ))
    .unwrap();
    let settings = HhtSettings {
        error_tol: Some(1.0e-9),
        ..HhtSettings::default()
    };
    let run = simulate(&sys, &grid(0.1, 0.005), &settings).unwrap();
    for frame in run.frames.iter().filter(|f| f.time >= 0.04) {
        let a = frame.acceleration[&BodyId(2)].linear.x;
        close(a, -mu * G, 2.0e-3, "sliding deceleration");
    }
    let v = run.frames.last().unwrap().velocity[&BodyId(2)].linear.x;
    close(v, 1.0 - mu * G * 0.1, 5.0e-3, "speed after 0.1 s");
    // The friction opposes the slip and the normal load is still the weight.
    let loads = run.frames.last().unwrap().contacts(&sys).unwrap();
    let friction: DVec3 = loads.iter().map(|l| l.friction_force).sum();
    let normal: f64 = loads.iter().map(|l| l.normal_force).sum();
    close(normal, M * G, 5.0e-3, "normal load");
    close(friction.x, -mu * M * G, 5.0e-3, "friction force");
}

/// A capsule lying on a plane rests on both end spheres (half the weight
/// each); tilted, only the lower end carries anything.
#[test]
fn a_capsule_rests_on_its_two_ends() {
    let (k, e, r, half_length) = (1.0e6, 1.5, 0.03, 0.2);
    // Lying along X: the capsule's Z axis turned onto X.
    let lying = DQuat::from_rotation_arc(DVec3::Z, DVec3::X);
    let mut sys = on_the_ground(true, DVec3::new(0.0, 0.0, r), lying, DVec3::splat(0.02));
    let capsule = ContactGeometry::Capsule {
        marker: MarkerId(2),
        radius: r,
        half_length,
    };
    sys.add_force(contact(10, capsule, plane(), hertz(k), None))
        .unwrap();
    let rest = static_equilibrium(&sys, 0.0, None).unwrap();
    let loads = loads_at(&sys, &rest.state);
    assert_eq!(loads.len(), 2, "a capsule offers its two ends");
    let expected = (M * G / (2.0 * k)).powf(1.0 / e);
    for load in &loads {
        close(load.depth, expected, 1.0e-6, "end penetration");
        close(load.normal_force, M * G / 2.0, 1.0e-6, "end load");
    }

    // Tilted about Y: one end down, one end up.
    let tilt = 0.01;
    let tilted = DQuat::from_rotation_y(tilt) * lying;
    let mut leaning = on_the_ground(true, DVec3::new(0.0, 0.0, r), tilted, DVec3::splat(0.02));
    leaning
        .add_force(contact(10, capsule, plane(), hertz(k), None))
        .unwrap();
    let frame = dynamic_ic(&leaning, 0.0).unwrap();
    let loads = frame.contacts(&leaning).unwrap();
    assert_eq!(loads.len(), 2);
    assert!(
        (loads[0].depth > 0.0) != (loads[1].depth > 0.0),
        "exactly one end should touch: {loads:?}"
    );
    let lower = loads.iter().find(|l| l.depth > 0.0).unwrap();
    close(
        lower.depth,
        half_length * tilt.sin(),
        1.0e-9,
        "the lower end's penetration is its geometry",
    );
}

/// Two crossed capsules: the normal runs between their axes and the
/// penalty force is the closed form of the overlap.
#[test]
fn crossed_capsules_push_along_their_closest_line() {
    let (k, e, r) = (1.0e6, 1.5, 0.02);
    let overlap = 1.0e-3;
    let mut sys = System::new("crossed");
    sys.gravity = Some(DVec3::new(0.0, 0.0, -G));
    sys.add_body(ground()).unwrap();
    // The ground capsule lies along X at the origin; the free one lies
    // along Y, `2r − overlap` above it.
    sys.add_body(part(
        2,
        M,
        DVec3::ZERO,
        DVec3::splat(0.02),
        DVec3::new(0.0, 0.0, 2.0 * r - overlap),
        DQuat::from_rotation_arc(DVec3::Z, DVec3::Y),
    ))
    .unwrap();
    marker(
        &mut sys,
        1,
        1,
        DVec3::ZERO,
        DQuat::from_rotation_arc(DVec3::Z, DVec3::X),
    );
    marker(&mut sys, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    let capsule = |m: u64| ContactGeometry::Capsule {
        marker: MarkerId(m),
        radius: r,
        half_length: 0.15,
    };
    sys.add_force(contact(10, capsule(2), capsule(1), hertz(k), None))
        .unwrap();
    let frame = dynamic_ic(&sys, 0.0).unwrap();
    let loads = frame.contacts(&sys).unwrap();
    assert_eq!(loads.len(), 1, "two swept shapes touch at their closest pair");
    close(loads[0].depth, overlap, 1.0e-12, "overlap");
    close_v(loads[0].normal, DVec3::Z, 1.0e-12, "normal between the axes");
    let force = k * overlap.powf(e);
    close(loads[0].normal_force, force, 1.0e-12, "penalty force");
    // The free capsule feels that force less its weight.
    close(
        frame.acceleration[&BodyId(2)].linear.z,
        force / M - G,
        1.0e-9,
        "net acceleration",
    );
}

/// A point geometry is a sphere of no radius: against a plane and against
/// a sphere its penetration is pure geometry.
#[test]
fn a_point_touches_like_a_zero_radius_sphere() {
    let (k, e) = (1.0e6, 1.5);
    let depth = 2.0e-4;
    let mut sys = on_the_ground(
        false,
        DVec3::new(0.0, 0.0, -depth),
        DQuat::IDENTITY,
        DVec3::splat(0.01),
    );
    sys.add_force(contact(
        10,
        ContactGeometry::Point { marker: MarkerId(2) },
        plane(),
        hertz(k),
        None,
    ))
    .unwrap();
    let loads = dynamic_ic(&sys, 0.0).unwrap().contacts(&sys).unwrap();
    close(loads[0].depth, depth, 1.0e-12, "point penetration");
    close(loads[0].normal_force, k * depth.powf(e), 1.0e-12, "force");
    close_v(loads[0].point, DVec3::new(0.0, 0.0, -0.5 * depth), 1.0e-12, "contact point");

    // Point against a sphere: the gap is the sphere's radius.
    let r = 0.05;
    let mut pair = System::new("point-sphere");
    pair.add_body(ground()).unwrap();
    pair.add_body(part(
        2,
        M,
        DVec3::ZERO,
        DVec3::splat(0.01),
        DVec3::new(0.0, 0.0, r - depth),
        DQuat::IDENTITY,
    ))
    .unwrap();
    marker(&mut pair, 1, 1, DVec3::ZERO, DQuat::IDENTITY);
    marker(&mut pair, 2, 2, DVec3::ZERO, DQuat::IDENTITY);
    pair.add_force(contact(
        10,
        sphere(2, r),
        ContactGeometry::Point { marker: MarkerId(1) },
        hertz(k),
        None,
    ))
    .unwrap();
    let loads = dynamic_ic(&pair, 0.0).unwrap().contacts(&pair).unwrap();
    close(loads[0].depth, depth, 1.0e-12, "point-sphere penetration");
    close_v(loads[0].normal, DVec3::Z, 1.0e-12, "normal onto the sphere");
}

/// Separated geometries cost nothing: a sphere falling far above the plane
/// is in free fall, and the contact element changes no result.
#[test]
fn separation_costs_nothing() {
    let r = 0.05;
    let plain = on_the_ground(true, DVec3::new(0.0, 0.0, 1.0), DQuat::IDENTITY, DVec3::splat(0.01));
    let mut watched = plain.clone();
    watched
        .add_force(contact(10, sphere(2, r), plane(), hertz(1.0e6), rubbing(0.3)))
        .unwrap();
    let times = grid(0.2, 0.02);
    let a = simulate(&plain, &times, &HhtSettings::default()).unwrap();
    let b = simulate(&watched, &times, &HhtSettings::default()).unwrap();
    for (x, y) in a.frames.iter().zip(&b.frames) {
        assert_eq!(x.state.params(), y.state.params(), "poses must be identical");
        assert_eq!(
            x.velocity[&BodyId(2)],
            y.velocity[&BodyId(2)],
            "velocities must be identical"
        );
    }
    assert_eq!(a.stats, b.stats, "the integrator must take the same steps");
    let loads = b.frames.last().unwrap().contacts(&watched).unwrap();
    assert!(loads[0].depth < 0.0 && loads[0].normal_force == 0.0);
    assert_eq!(loads[0].friction_force, DVec3::ZERO);
    close(
        b.frames.last().unwrap().acceleration[&BodyId(2)].linear.z,
        -G,
        1.0e-12,
        "free fall",
    );
}

/// The data gates: only the modelled pairs, with usable numbers.
#[test]
fn contact_validation_names_its_codes() {
    let base = on_the_ground(true, DVec3::new(0.0, 0.0, 0.05), DQuat::IDENTITY, DVec3::splat(0.01));
    let boxed = |m: u64| ContactGeometry::Box {
        marker: MarkerId(m),
        half_extents: DVec3::splat(0.05),
    };
    for (first, second) in [
        (plane(), ContactGeometry::Plane { marker: MarkerId(2) }),
        (boxed(2), sphere(1, 0.05)),
        (boxed(2), boxed(1)),
    ] {
        let mut sys = base.clone();
        sys.add_force(contact(10, first, second, hertz(1.0e6), None))
            .unwrap();
        assert_eq!(
            dynamic_ic(&sys, 0.0).unwrap_err().code,
            codes::CONTACT_UNSUPPORTED
        );
    }
    for bad in [
        contact(10, sphere(2, -0.1), plane(), hertz(1.0e6), None),
        contact(
            10,
            sphere(2, 0.05),
            plane(),
            Impact {
                stiffness: 0.0,
                ..hertz(1.0e6)
            },
            None,
        ),
        contact(
            10,
            sphere(2, 0.05),
            plane(),
            Impact {
                exponent: 0.5,
                ..hertz(1.0e6)
            },
            None,
        ),
        contact(
            10,
            sphere(2, 0.05),
            plane(),
            Impact {
                damping_depth: 0.0,
                damping: 1.0,
                ..hertz(1.0e6)
            },
            None,
        ),
        contact(
            10,
            sphere(2, 0.05),
            plane(),
            hertz(1.0e6),
            Some(ContactFriction {
                mu_static: 0.1,
                mu_dynamic: 0.4,
                stiction_velocity: 1.0e-3,
                friction_velocity: 1.0e-2,
            }),
        ),
    ] {
        let mut sys = base.clone();
        sys.add_force(bad).unwrap();
        assert_eq!(
            dynamic_ic(&sys, 0.0).unwrap_err().code,
            codes::BAD_FORCE_ELEMENT
        );
    }
    // A dangling marker is caught as everywhere else.
    let mut dangling = base.clone();
    dangling
        .add_force(contact(10, sphere(9, 0.05), plane(), hertz(1.0e6), None))
        .unwrap();
    assert_eq!(
        dynamic_ic(&dangling, 0.0).unwrap_err().code,
        codes::DANGLING_MARKER
    );
}

/// The document form: a contact round-trips, and a contact without a
/// `friction` field reads as frictionless.
#[test]
fn contact_serializes_with_defaults() {
    let element = contact(
        10,
        ContactGeometry::Capsule {
            marker: MarkerId(2),
            radius: 0.03,
            half_length: 0.2,
        },
        plane(),
        Impact::default(),
        rubbing(0.3),
    );
    let text = serde_json::to_string(&element).unwrap();
    assert_eq!(
        serde_json::from_str::<ForceElement>(&text).unwrap(),
        element,
        "{text}"
    );
    let terse: ForceElement = serde_json::from_str(
        r#"{"kind":"contact","id":10,"name":"c","first":{"shape":"sphere","marker":2,"radius":0.05},
            "second":{"shape":"plane","marker":1},
            "impact":{"stiffness":100000.0,"exponent":2.2,"damping":10.0,"damping_depth":0.0001}}"#,
    )
    .unwrap();
    let ForceElement::Contact { friction, impact, .. } = terse else {
        panic!("not a contact");
    };
    assert_eq!(friction, None, "frictionless unless the document says so");
    assert_eq!(impact, Impact::default(), "the Adams IMPACT defaults");
}

/// The integrator through real impact events: a ball bouncing on a plane
/// under gravity with a conservative penalty keeps coming back to the
/// height it was dropped from, and the error controller pays for the
/// impacts with its step size, not with rejected steps everywhere.
///
/// A stiff contact needs a tighter tolerance than the default: with
/// `k = 1e8 N/m^1.5` this run's worst energy error is 1.3e-2 at the
/// default 1e-6, 4.2e-4 at 1e-8 and 2.5e-5 at 1e-10 (and α barely matters
/// — it is the step size, not the numerical damping). The tolerance here
/// is 1e-8, the figure `docs/INTEGRATING.md` recommends for contact.
#[test]
fn a_bouncing_ball_keeps_its_apex_under_error_control() {
    let (k, r, drop) = (1.0e8, 0.05, 0.1);
    let mut sys = on_the_ground(
        true,
        DVec3::new(0.0, 0.0, r + drop),
        DQuat::IDENTITY,
        DVec3::splat(0.01),
    );
    sys.add_force(contact(10, sphere(2, r), plane(), hertz(k), None))
        .unwrap();
    // Three flights: 2·√(2·drop/g) each.
    let flight = 2.0 * (2.0 * drop / G).sqrt();
    let settings = HhtSettings {
        error_tol: Some(1.0e-8),
        ..HhtSettings::default()
    };
    let run = simulate(&sys, &grid(3.1 * flight, 0.002), &settings).unwrap();

    let height = |f: &DynamicFrame| f.state.body_pose(BodyId(2)).unwrap().0.z - r;
    let total = |f: &DynamicFrame| energy(&sys, &f.state, &f.velocity).unwrap().total();
    let e0 = total(&run.frames[0]);
    for frame in &run.frames {
        close(total(frame), e0, 1.0e-3, "energy over the bounces");
    }
    // Every apex comes back to the drop height.
    let mut apexes = Vec::new();
    for w in run.frames.windows(3) {
        if height(&w[1]) > height(&w[0]) && height(&w[1]) >= height(&w[2]) {
            apexes.push(height(&w[1]));
        }
    }
    assert!(apexes.len() >= 3, "expected three apexes, got {apexes:?}");
    for apex in &apexes {
        close(*apex, drop, 2.0e-3, "apex height");
    }
    // The impacts are short, so the controller must shrink the step for
    // them and grow it again: a handful of rejections, not a crawl.
    assert!(
        20 * run.stats.rejected <= run.stats.steps,
        "too many rejected steps through the impacts: {:?}",
        run.stats
    );
    assert!(
        run.stats.steps < 20_000,
        "the run should not crawl: {:?}",
        run.stats
    );
}
