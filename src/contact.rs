//! Contact and impact (M5-2): penalty contact between primitive
//! geometries — the Adams `CONTACT` / `IMPACT` tier, kernel-neutral.
//!
//! This crate consumes numbers, not B-rep, so contact is offered on
//! primitives attached to markers: a point, a sphere, a capsule, a box and
//! a plane. Mesh and B-rep contact belong to the consumer, which can map
//! its geometry onto these (a bearing ball to a sphere, a shaft to a
//! capsule, a table to a plane) or feed contact loads in as force
//! elements.
//!
//! **Normal force** (Adams `IMPACT`), per contact point of penetration
//! depth `d` closing at rate `ḋ`:
//!
//! ```text
//! N = max(0, k·dᵉ + c·ḋ·STEP(d, 0, 0, d_damp, 1))      d > 0
//! N = 0                                                d ≤ 0
//! ```
//!
//! The exponent `e` is the shape of the local deformation (3/2 is the
//! Hertzian sphere, 1.1–2.2 the usual engineering range); the damping is
//! ramped in over `d_damp` so the force is continuous at first touch, and
//! the whole force is clamped at zero so a contact can never pull.
//!
//! **Tangential force**: regularized Coulomb on the slip velocity in the
//! contact plane, `μ(|v_t|)·N`, with the same Stribeck curve joint
//! friction uses ([`crate::friction`]).
//!
//! **Step size.** A penalty contact is a stiff spring that switches on,
//! and the integrator pays for it in step size: on a bouncing ball with
//! `k = 1e8 N/m^1.5` the worst energy error over three bounces is 1.3e-2
//! at the default error tolerance (1e-6), 4.2e-4 at 1e-8 and 2.5e-5 at
//! 1e-10 — the step size, not the numerical damping, is what matters
//! (`α = 0` behaves the same). Tighten [`crate::HhtSettings::error_tol`]
//! for impact work, and prefer the softest stiffness the application can
//! live with.
//!
//! **Contact points.** Each element resolves to a fixed list of candidate
//! points, so a report's indices mean the same thing from frame to frame:
//! a sphere or point against anything gives one, a capsule against a
//! plane gives two (its end spheres — a capsule lying flat rests on both,
//! a tilted one loads the lower end only), a box against a plane gives its
//! eight corners, and two swept shapes give the closest pair. A candidate
//! with no penetration carries no force.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::body::MarkerId;
use crate::error::{codes, MbdError};
use crate::force::{ForceElement, ForceId};
use crate::friction::stribeck;
use crate::kinematics::{read_pose, Problem, Twist};
use crate::system::System;

/// Below this distance two swept shapes have no line of action.
const CENTER_EPS: f64 = 1.0e-12;

/// Below this slip the tangential direction is taken as zero (the
/// coefficient is zero there anyway).
const SLIP_EPS: f64 = 1.0e-300;

/// A contact primitive, placed by a marker.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum ContactGeometry {
    /// The marker's origin, with no extent.
    Point { marker: MarkerId },
    /// A sphere centred at the marker's origin.
    Sphere { marker: MarkerId, radius: f64 },
    /// A capsule along the marker's Z axis: the segment from `−half_length`
    /// to `+half_length` swept by `radius`.
    Capsule {
        marker: MarkerId,
        radius: f64,
        half_length: f64,
    },
    /// A box centred at the marker's origin, with half-extents along the
    /// marker's axes. Only its corners touch (M5-2 scope).
    Box {
        marker: MarkerId,
        half_extents: DVec3,
    },
    /// The marker's XY plane, material on the `−Z` side: `+Z` is the
    /// outward normal.
    Plane { marker: MarkerId },
}

impl ContactGeometry {
    pub fn marker(&self) -> MarkerId {
        match self {
            ContactGeometry::Point { marker }
            | ContactGeometry::Sphere { marker, .. }
            | ContactGeometry::Capsule { marker, .. }
            | ContactGeometry::Box { marker, .. }
            | ContactGeometry::Plane { marker } => *marker,
        }
    }

    /// Whether the numbers are usable.
    pub(crate) fn check(&self) -> Result<(), &'static str> {
        let ok = match self {
            ContactGeometry::Point { .. } | ContactGeometry::Plane { .. } => true,
            ContactGeometry::Sphere { radius, .. } => radius.is_finite() && *radius > 0.0,
            ContactGeometry::Capsule {
                radius,
                half_length,
                ..
            } => {
                radius.is_finite() && *radius > 0.0 && half_length.is_finite() && *half_length > 0.0
            }
            ContactGeometry::Box { half_extents, .. } => {
                half_extents.is_finite() && half_extents.min_element() > 0.0
            }
        };
        ok.then_some(())
            .ok_or("contact geometry needs finite positive dimensions")
    }

    /// How many contact points this geometry contributes against a plane.
    fn points_against_plane(&self) -> usize {
        match self {
            ContactGeometry::Point { .. } | ContactGeometry::Sphere { .. } => 1,
            ContactGeometry::Capsule { .. } => 2,
            ContactGeometry::Box { .. } => 8,
            ContactGeometry::Plane { .. } => 0,
        }
    }
}

/// The normal force law of a contact (Adams `IMPACT`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Impact {
    /// Stiffness (N/mᵉ).
    pub stiffness: f64,
    /// Force exponent: `3/2` for Hertzian spheres, `1.1`–`2.2` in
    /// engineering practice (Adams' default is `2.2`).
    pub exponent: f64,
    /// Full damping coefficient (N·s/m).
    pub damping: f64,
    /// Penetration at which the damping reaches its full value (m): the
    /// `STEP` ramp that keeps the force continuous at first touch.
    pub damping_depth: f64,
}

impl Default for Impact {
    /// The Adams `IMPACT` defaults, in SI.
    fn default() -> Self {
        Self {
            stiffness: 1.0e5,
            exponent: 2.2,
            damping: 10.0,
            damping_depth: 1.0e-4,
        }
    }
}

impl Impact {
    fn check(&self) -> Result<(), &'static str> {
        if ![
            self.stiffness,
            self.exponent,
            self.damping,
            self.damping_depth,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            return Err("non-finite contact parameters");
        }
        if self.stiffness <= 0.0 || self.exponent < 1.0 {
            return Err("a contact needs a positive stiffness and an exponent of at least 1");
        }
        if self.damping < 0.0 || self.damping_depth <= 0.0 {
            return Err("a contact needs damping ≥ 0 over a positive damping depth");
        }
        Ok(())
    }

    /// `k·dᵉ + c·ḋ·STEP(d)`, clamped at zero: a contact pushes, never
    /// pulls. `depth ≤ 0` is no contact.
    fn force(&self, depth: f64, closing: f64) -> f64 {
        if depth <= 0.0 {
            return 0.0;
        }
        let ramp = smooth_step(depth / self.damping_depth);
        (self.stiffness * depth.powf(self.exponent) + self.damping * ramp * closing).max(0.0)
    }

    /// The elastic energy stored at penetration `depth`:
    /// `k·d^(e+1)/(e+1)` (the damping is not a potential).
    fn energy(&self, depth: f64) -> f64 {
        if depth <= 0.0 {
            return 0.0;
        }
        self.stiffness * depth.powf(self.exponent + 1.0) / (self.exponent + 1.0)
    }
}

/// Regularized Coulomb friction in the contact plane.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactFriction {
    pub mu_static: f64,
    pub mu_dynamic: f64,
    /// Slip at which the coefficient reaches `mu_static` (m/s).
    pub stiction_velocity: f64,
    /// Slip at and beyond which it is `mu_dynamic` (m/s).
    pub friction_velocity: f64,
}

impl ContactFriction {
    fn check(&self) -> Result<(), &'static str> {
        if ![
            self.mu_static,
            self.mu_dynamic,
            self.stiction_velocity,
            self.friction_velocity,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            return Err("non-finite contact friction parameters");
        }
        if self.mu_dynamic < 0.0 || self.mu_static < self.mu_dynamic {
            return Err("contact friction needs mu_static ≥ mu_dynamic ≥ 0");
        }
        if self.stiction_velocity <= 0.0 || self.friction_velocity < self.stiction_velocity {
            return Err("contact friction needs 0 < stiction_velocity ≤ friction_velocity");
        }
        Ok(())
    }

    fn coefficient(&self, slip: f64) -> f64 {
        stribeck(
            self.mu_static,
            self.mu_dynamic,
            self.stiction_velocity,
            self.friction_velocity,
            slip,
        )
    }
}

/// One contact point's load at an instant — what the commercial tools plot
/// per contact.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactLoad {
    /// The contact element.
    pub force_element: ForceId,
    /// Which candidate point of that element (stable from frame to frame:
    /// a capsule against a plane is `0` and `1`, a box's corners `0..8`).
    pub index: usize,
    /// Where the force acts (world).
    pub point: DVec3,
    /// Unit normal, the direction the contact pushes the *first* geometry.
    pub normal: DVec3,
    /// Penetration (m); `≤ 0` is separation, and then nothing acts.
    pub depth: f64,
    /// Closing speed along the normal (m/s; positive = penetrating).
    pub closing: f64,
    /// Normal force (N, never negative).
    pub normal_force: f64,
    /// Tangential slip speed in the contact plane (m/s).
    pub slip: f64,
    /// Friction force on the first geometry (N, world).
    pub friction_force: DVec3,
}

/// Whether the system has any contact element (the fast path skips it).
pub(crate) fn has_contact(sys: &System) -> bool {
    sys.forces
        .values()
        .any(|f| matches!(f, ForceElement::Contact { .. }))
}

/// The smooth `STEP` ramp on `[0, 1]`.
fn smooth_step(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    x * x * (3.0 - 2.0 * x)
}

/// A geometry placed in the world: the solver's two shapes.
enum Placed {
    /// `a`..`b` swept by `radius` (`a == b` is a sphere, `radius == 0` a
    /// point).
    Swept { a: DVec3, b: DVec3, radius: f64 },
    /// A plane through `p` with outward unit normal `n`.
    Plane { p: DVec3, n: DVec3 },
    /// A box's eight corners.
    Corners([DVec3; 8]),
}

/// One candidate contact point.
struct Candidate {
    /// Where the force acts (the middle of the overlap).
    point: DVec3,
    /// Unit normal, pointing the way the contact pushes the first shape.
    normal: DVec3,
    /// Penetration depth (positive when overlapping).
    depth: f64,
}

/// Place a geometry in the world at `params`.
fn place(problem: &Problem, params: &[f64], vel: &[Twist], g: &ContactGeometry) -> Placed {
    let frame = problem.frame(params, vel, g.marker());
    let (p, x, y, z) = (frame.p.v, frame.x.v, frame.y.v, frame.z.v);
    match g {
        ContactGeometry::Point { .. } => Placed::Swept {
            a: p,
            b: p,
            radius: 0.0,
        },
        ContactGeometry::Sphere { radius, .. } => Placed::Swept {
            a: p,
            b: p,
            radius: *radius,
        },
        ContactGeometry::Capsule {
            radius,
            half_length,
            ..
        } => Placed::Swept {
            a: p - *half_length * z,
            b: p + *half_length * z,
            radius: *radius,
        },
        ContactGeometry::Box { half_extents, .. } => {
            let (ex, ey, ez) = (
                half_extents.x * x,
                half_extents.y * y,
                half_extents.z * z,
            );
            let mut corners = [DVec3::ZERO; 8];
            for (k, corner) in corners.iter_mut().enumerate() {
                let sign = |bit: usize| if k & (1 << bit) == 0 { -1.0 } else { 1.0 };
                *corner = p + sign(0) * ex + sign(1) * ey + sign(2) * ez;
            }
            Placed::Corners(corners)
        }
        ContactGeometry::Plane { .. } => Placed::Plane { p, n: z },
    }
}

/// A sphere of `radius` at `center` against a plane: penetration along the
/// plane's normal, the force point in the middle of the overlap.
fn sphere_plane(center: DVec3, radius: f64, p: DVec3, n: DVec3) -> Candidate {
    let height = (center - p).dot(n);
    let depth = radius - height;
    Candidate {
        point: center - n * (radius - 0.5 * depth.max(0.0)),
        normal: n,
        depth,
    }
}

/// The closest points of two segments, as parameters in `[0, 1]`: Ericson's
/// algorithm, with parallel segments clamped to the middle of their
/// overlap so the answer is deterministic.
fn closest_parameters(p1: DVec3, q1: DVec3, p2: DVec3, q2: DVec3) -> (f64, f64) {
    let (d1, d2) = (q1 - p1, q2 - p2);
    let r = p1 - p2;
    let (a, e, f) = (d1.dot(d1), d2.dot(d2), d2.dot(r));
    if a <= CENTER_EPS && e <= CENTER_EPS {
        return (0.0, 0.0);
    }
    if a <= CENTER_EPS {
        return (0.0, (f / e).clamp(0.0, 1.0));
    }
    let c = d1.dot(r);
    if e <= CENTER_EPS {
        return ((-c / a).clamp(0.0, 1.0), 0.0);
    }
    let b = d1.dot(d2);
    let denom = a * e - b * b;
    // Parallel (or nearly): the overlap is an interval, so take its middle
    // — projecting the second segment's ends onto the first.
    if denom <= CENTER_EPS * a * e {
        let t0 = (p2 - p1).dot(d1) / a;
        let t1 = (q2 - p1).dot(d1) / a;
        let (lo, hi) = (t0.min(t1).max(0.0), t0.max(t1).min(1.0));
        let s = if lo <= hi {
            0.5 * (lo + hi)
        } else {
            // No overlap: the nearer end of the first segment.
            f64::from(t0.min(t1) > 1.0)
        };
        let t = ((b * s + f) / e).clamp(0.0, 1.0);
        return (s, t);
    }
    let s = ((b * f - c * e) / denom).clamp(0.0, 1.0);
    let t = (b * s + f) / e;
    if t < 0.0 {
        ((-c / a).clamp(0.0, 1.0), 0.0)
    } else if t > 1.0 {
        (((b - c) / a).clamp(0.0, 1.0), 1.0)
    } else {
        (s, t)
    }
}

/// Two swept segments: the closest pair of points, their penetration and
/// the normal from the second toward the first.
fn swept_swept(
    (a1, b1, r1): (DVec3, DVec3, f64),
    (a2, b2, r2): (DVec3, DVec3, f64),
    what: &str,
) -> Result<Candidate, MbdError> {
    let (s, t) = closest_parameters(a1, b1, a2, b2);
    let c1 = a1 + s * (b1 - a1);
    let c2 = a2 + t * (b2 - a2);
    let d = c1 - c2;
    let distance = d.length();
    if distance <= CENTER_EPS {
        return Err(MbdError::new(
            codes::DEGENERATE_FORCE,
            format!("{what}: the two geometries' axes meet, so the contact normal is undefined"),
            "separate the geometries, or lower the stiffness so they cannot pass through each other",
        ));
    }
    let normal = d / distance;
    let depth = r1 + r2 - distance;
    // The middle of the overlap: between the two surface points.
    let point = 0.5 * ((c1 - normal * r1) + (c2 + normal * r2));
    Ok(Candidate {
        point,
        normal,
        depth,
    })
}

/// Every candidate point of one contact element, in a stable order.
fn candidates(
    first: &Placed,
    second: &Placed,
    what: &str,
) -> Result<Vec<Candidate>, MbdError> {
    let flip = |mut c: Vec<Candidate>| {
        for candidate in &mut c {
            candidate.normal = -candidate.normal;
        }
        c
    };
    match (first, second) {
        (Placed::Swept { a, b, radius }, Placed::Plane { p, n }) => {
            let mut out = vec![sphere_plane(*a, *radius, *p, *n)];
            if *a != *b {
                out.push(sphere_plane(*b, *radius, *p, *n));
            }
            Ok(out)
        }
        (Placed::Plane { .. }, Placed::Swept { .. }) => {
            Ok(flip(candidates(second, first, what)?))
        }
        (Placed::Corners(corners), Placed::Plane { p, n }) => Ok(corners
            .iter()
            .map(|c| sphere_plane(*c, 0.0, *p, *n))
            .collect()),
        (Placed::Plane { .. }, Placed::Corners(_)) => Ok(flip(candidates(second, first, what)?)),
        (
            Placed::Swept {
                a: a1,
                b: b1,
                radius: r1,
            },
            Placed::Swept {
                a: a2,
                b: b2,
                radius: r2,
            },
        ) => Ok(vec![swept_swept(
            (*a1, *b1, *r1),
            (*a2, *b2, *r2),
            what,
        )?]),
        _ => unreachable!("validation admits only the supported pairs"),
    }
}

/// Whether this pair of geometries has a contact model; `Err` names what
/// is missing (the validation gate reports it).
pub(crate) fn supported(
    first: &ContactGeometry,
    second: &ContactGeometry,
) -> Result<(), &'static str> {
    use ContactGeometry as G;
    let swept = |g: &ContactGeometry| {
        matches!(g, G::Point { .. } | G::Sphere { .. } | G::Capsule { .. })
    };
    let plane = |g: &ContactGeometry| matches!(g, G::Plane { .. });
    let corners = |g: &ContactGeometry| matches!(g, G::Box { .. });
    if swept(first) && swept(second) {
        return Ok(());
    }
    if (swept(first) || corners(first)) && plane(second) {
        return Ok(());
    }
    if plane(first) && (swept(second) || corners(second)) {
        return Ok(());
    }
    Err("the modelled pairs are point/sphere/capsule against each other, and point/sphere/capsule/box against a plane")
}

/// The velocity of the world point `at` on the body of `marker`.
fn point_velocity(problem: &Problem, params: &[f64], vel: &[Twist], marker: MarkerId, at: DVec3) -> DVec3 {
    let slot = problem.slot_of[&problem.sys.markers[&marker].body];
    let origin = read_pose(params, slot).0;
    vel[slot].linear + vel[slot].angular.cross(at - origin)
}

/// One contact element's loads at an instant (the candidate points in
/// order, including the ones carrying nothing).
fn evaluate(
    problem: &Problem,
    params: &[f64],
    vel: &[Twist],
    element: &ForceElement,
) -> Result<Vec<ContactLoad>, MbdError> {
    let ForceElement::Contact {
        id,
        name,
        first,
        second,
        impact,
        friction,
    } = element
    else {
        return Ok(Vec::new());
    };
    let what = format!("contact {} ({name})", id.0);
    let placed = (
        place(problem, params, vel, first),
        place(problem, params, vel, second),
    );
    let mut out = Vec::new();
    for (index, candidate) in candidates(&placed.0, &placed.1, &what)?
        .into_iter()
        .enumerate()
    {
        let relative = point_velocity(problem, params, vel, first.marker(), candidate.point)
            - point_velocity(problem, params, vel, second.marker(), candidate.point);
        // Closing: the first geometry moving against the normal.
        let closing = -relative.dot(candidate.normal);
        let normal_force = impact.force(candidate.depth, closing);
        let tangential = relative + closing * candidate.normal;
        let slip = tangential.length();
        let friction_force = match friction {
            Some(f) if normal_force > 0.0 && slip > SLIP_EPS => {
                -f.coefficient(slip) * normal_force * tangential / slip
            }
            _ => DVec3::ZERO,
        };
        out.push(ContactLoad {
            force_element: *id,
            index,
            point: candidate.point,
            normal: candidate.normal,
            depth: candidate.depth,
            closing,
            normal_force,
            slip,
            friction_force,
        });
    }
    Ok(out)
}

/// Add every contact's loads to the per-slot applied loads `q` (force,
/// torque about the body origin): the normal force and the friction act on
/// the first geometry's body at the contact point, and the equal and
/// opposite load on the second's at the same point.
pub(crate) fn contact_forces(
    problem: &Problem,
    params: &[f64],
    vel: &[Twist],
    q: &mut [Twist],
) -> Result<(), MbdError> {
    for element in problem.sys.forces.values() {
        let ForceElement::Contact { first, second, .. } = element else {
            continue;
        };
        for load in evaluate(problem, params, vel, element)? {
            let force = load.normal_force * load.normal + load.friction_force;
            if force == DVec3::ZERO {
                continue;
            }
            for (marker, f) in [(first.marker(), force), (second.marker(), -force)] {
                let slot = problem.slot_of[&problem.sys.markers[&marker].body];
                let origin = read_pose(params, slot).0;
                q[slot].linear += f;
                q[slot].angular += (load.point - origin).cross(f);
            }
        }
    }
    Ok(())
}

/// Every contact load at an instant, in force-element id order.
pub(crate) fn contact_report(
    problem: &Problem,
    params: &[f64],
    vel: &[Twist],
) -> Result<Vec<ContactLoad>, MbdError> {
    let mut out = Vec::new();
    for element in problem.sys.forces.values() {
        out.extend(evaluate(problem, params, vel, element)?);
    }
    Ok(out)
}

/// The elastic energy stored in every contact at a pose (the penalty's
/// `k·d^(e+1)/(e+1)`; the damping and friction are not potentials — their
/// work is audited through [`crate::power`]).
pub(crate) fn contact_energy(
    problem: &Problem,
    params: &[f64],
    vel: &[Twist],
) -> Result<f64, MbdError> {
    let mut total = 0.0;
    for element in problem.sys.forces.values() {
        let ForceElement::Contact { impact, .. } = element else {
            continue;
        };
        for load in evaluate(problem, params, vel, element)? {
            total += impact.energy(load.depth);
        }
    }
    Ok(total)
}

/// Whether a contact element's numbers are usable.
pub(crate) fn check(
    first: &ContactGeometry,
    second: &ContactGeometry,
    impact: &Impact,
    friction: &Option<ContactFriction>,
) -> Result<(), &'static str> {
    first.check()?;
    second.check()?;
    impact.check()?;
    if let Some(f) = friction {
        f.check()?;
    }
    Ok(())
}

/// How many contact points an element reports (for documentation and
/// tests: the report's length is fixed by the geometry pair).
pub fn contact_points(first: &ContactGeometry, second: &ContactGeometry) -> usize {
    match (first, second) {
        (ContactGeometry::Plane { .. }, other) | (other, ContactGeometry::Plane { .. }) => {
            other.points_against_plane()
        }
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_impact_force_is_continuous_and_never_pulls() {
        let impact = Impact {
            stiffness: 1.0e5,
            exponent: 1.5,
            damping: 50.0,
            damping_depth: 1.0e-4,
        };
        assert_eq!(impact.force(0.0, 1.0), 0.0);
        assert_eq!(impact.force(-1.0e-6, 1.0), 0.0);
        // The damping ramps in, so first touch is a pure spring.
        assert!(impact.force(1.0e-12, 10.0) < 1.0e-6);
        // Separating fast cannot pull the bodies together.
        assert_eq!(impact.force(1.0e-5, -1.0e6), 0.0);
        // The spring part is the closed form.
        let d = 1.0e-3;
        close(
            impact.force(d, 0.0),
            1.0e5 * d.powf(1.5),
            1.0e-12,
            "spring force",
        );
        // The energy is the integral of the spring part.
        let steps = 200_000;
        let mut work = 0.0;
        for k in 0..steps {
            let x = d * (k as f64 + 0.5) / steps as f64;
            work += impact.force(x, 0.0) * d / steps as f64;
        }
        close(impact.energy(d), work, 1.0e-6, "stored energy");
    }

    #[test]
    fn parallel_capsules_touch_in_the_middle_of_their_overlap() {
        // Two parallel segments along X, overlapping over [0, 1].
        let (s, t) = closest_parameters(
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.1, 0.0),
            DVec3::new(2.0, 0.1, 0.0),
        );
        let c1 = DVec3::new(-1.0, 0.0, 0.0) + s * DVec3::new(2.0, 0.0, 0.0);
        let c2 = DVec3::new(0.0, 0.1, 0.0) + t * DVec3::new(2.0, 0.0, 0.0);
        close(c1.x, 0.5, 1.0e-12, "overlap middle on the first");
        close(c2.x, 0.5, 1.0e-12, "overlap middle on the second");
        close((c1 - c2).length(), 0.1, 1.0e-12, "closest distance");
    }

    #[test]
    fn crossed_segments_meet_at_their_crossing() {
        let (s, t) = closest_parameters(
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, -1.0, 0.2),
            DVec3::new(0.0, 1.0, 0.2),
        );
        close(s, 0.5, 1.0e-12, "crossing on the first");
        close(t, 0.5, 1.0e-12, "crossing on the second");
    }

    fn close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
    }
}
