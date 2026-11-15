//! Joint friction (M5-1): the Adams / SolidWorks Motion friction tier —
//! a friction load that follows the joint's own bearing load.
//!
//! The friction of a lower pair is `μ(slip)·N`, with `N` the *friction
//! load*: the reaction the joint carries, turned into a generalized load
//! by the geometry of the bearing surfaces (a radial force rubs at the pin
//! radius, an axial force at the thrust-face radius, a bending moment is
//! reacted as a force couple over the bearing spacing). The reaction comes
//! from the joint's own multipliers, so the equations of motion become
//!
//! ```text
//! M·v̇ + b + Jᵀλ = Q(q, v, λ)
//! ```
//!
//! — the friction is part of the unknown. The integrator's Newton solves
//! `(v̇, λ)` together and carries the `∂r/∂λ` block of the friction
//! ([`crate::integrate`]); the instantaneous solves iterate the saddle
//! point to the fixed point of `λ`.
//!
//! **Regularization.** Coulomb friction is set-valued at zero slip; like
//! the commercial solvers this crate replaces it with a smooth curve: a
//! ramp `0 → μ_s` over the stiction transition velocity, the Stribeck fall
//! to `μ_d` at the friction transition velocity, `μ_d` beyond. A dynamic
//! run therefore creeps instead of sticking (the creep speed is
//! `v_stiction·load/μ_s·N`, which the transition velocity sizes), and
//! [`crate::static_equilibrium`] measures the slip as a *displacement*
//! from the pose it started at instead — Adams' maximum stiction
//! deformation — so a mechanism can rest held by stiction.

use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::body::MarkerId;
use crate::driver::DriverSlot;
use crate::dynamics::coordinate_gradient;
use crate::force::{ForceElement, ForceId};
use crate::joint::{JointId, JointKind};
use crate::kinematics::{read_pose, EquationOwner, Problem, Twist};
use crate::system::System;

/// Below this slip (surface units) a direction is taken as zero: the
/// coefficient is zero there anyway, so the friction load is too.
const SLIP_EPS: f64 = 1.0e-300;

/// What the applied loads are evaluated against besides the state: the
/// constraint multipliers joint friction reads, and the pose a static
/// solve measures its stiction displacement from.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Loads<'a> {
    /// The multipliers over [`Problem::rows`]; a slice of another length
    /// (the empty one) reads as all-zero.
    pub(crate) lambda: &'a [f64],
    /// The static-equilibrium stiction anchor: the pose the joint
    /// displacement is measured from. `None` in dynamics — the slip is
    /// then the joint's rate.
    pub(crate) anchor: Option<&'a [f64]>,
}

impl<'a> Loads<'a> {
    /// The multipliers of a dynamic instant.
    pub(crate) fn dynamic(lambda: &'a [f64]) -> Self {
        Loads {
            lambda,
            anchor: None,
        }
    }

    /// A static solve: the multipliers and the stiction anchor.
    pub(crate) fn static_at(lambda: &'a [f64], anchor: &'a [f64]) -> Self {
        Loads {
            lambda,
            anchor: Some(anchor),
        }
    }
}

/// The friction load of one element at an instant: what the commercial
/// tools plot as the joint's friction force and torque.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrictionLoad {
    /// The friction element.
    pub force_element: ForceId,
    /// The joint it rubs.
    pub joint: JointId,
    /// The friction load the reaction gives (N·m for a rotation, N for a
    /// translation): the load a coefficient of 1 would apply.
    pub normal_load: f64,
    /// The slip the coefficient was evaluated at: the joint's rate (rad/s
    /// revolute, m/s prismatic), the relative angular speed (spherical),
    /// or the surface slip speed at the pin (m/s, cylindrical) — in a
    /// static solve, the stiction displacement instead (rad or m).
    pub slip: f64,
    /// The regularized coefficient at that slip.
    pub coefficient: f64,
    /// The friction torque resisting rotation (N·m, a magnitude).
    pub torque: f64,
    /// The friction force resisting sliding (N, a magnitude).
    pub force: f64,
}

/// Whether any element's load depends on the multipliers: such a system's
/// instantaneous solves iterate.
pub(crate) fn reads_reactions(sys: &System) -> bool {
    sys.forces.values().any(ForceElement::reads_reactions)
}

/// Whether the system has friction at all (the fast path skips it).
pub(crate) fn has_friction(sys: &System) -> bool {
    sys.forces
        .values()
        .any(|f| matches!(f, ForceElement::JointFriction { .. }))
}

/// The smooth `STEP` ramp on `[0, 1]` (Adams' cubic Hermite): `0` and `1`
/// with zero slope at the ends.
fn smooth_step(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    x * x * (3.0 - 2.0 * x)
}

/// The regularized Stribeck coefficient at slip speed `s ≥ 0`: the ramp to
/// `mu_static` over `v_stiction`, the fall to `mu_dynamic` at
/// `v_friction`, constant beyond.
pub(crate) fn stribeck(mu_static: f64, mu_dynamic: f64, v_stiction: f64, v_friction: f64, s: f64) -> f64 {
    if s >= v_friction {
        return mu_dynamic;
    }
    if s <= v_stiction {
        return mu_static * smooth_step(s / v_stiction);
    }
    mu_static + (mu_dynamic - mu_static) * smooth_step((s - v_stiction) / (v_friction - v_stiction))
}

/// The friction of one element, resolved into the generalized loads the
/// joint's coordinates take.
struct Friction {
    load: FrictionLoad,
    /// Generalized load along the rotation coordinate (N·m), signed:
    /// revolute and cylindrical.
    along_rotation: f64,
    /// Generalized load along the translation coordinate (N), signed:
    /// prismatic and cylindrical.
    along_translation: f64,
    /// The friction couple on marker J's body (N·m, world frame):
    /// spherical, whose slip has no single coordinate.
    couple: DVec3,
}

/// The world quaternion of a marker frame at `params`.
fn marker_rotation(problem: &Problem, params: &[f64], m: MarkerId) -> DQuat {
    let marker = &problem.sys.markers[&m];
    let slot = problem.slot_of[&marker.body];
    read_pose(params, slot).1 * marker.rotation
}

/// The relative rotation of marker J in marker I's axes.
fn relative_rotation(problem: &Problem, params: &[f64], i: MarkerId, j: MarkerId) -> DQuat {
    marker_rotation(problem, params, i).conjugate() * marker_rotation(problem, params, j)
}

/// One friction element at `(params, vel)` with the multipliers `lambda`
/// (over [`Problem::rows`]) and, in a static solve, the stiction `anchor`;
/// `None` when it applies nothing (not a friction element, a suppressed
/// joint, or a static solve with stiction switched off).
fn evaluate(
    problem: &Problem,
    params: &[f64],
    vel: &[Twist],
    lambda: &[f64],
    anchor: Option<&[f64]>,
    element: &ForceElement,
) -> Option<Friction> {
    let ForceElement::JointFriction {
        id,
        joint,
        mu_static,
        mu_dynamic,
        stiction_velocity,
        friction_velocity,
        reaction_force,
        pin_radius,
        thrust_arm,
        bending_arm,
        preload,
        max_stiction_deformation,
        ..
    } = element
    else {
        return None;
    };
    let sys = problem.sys;
    let j = &sys.joints[joint];
    if j.suppressed {
        return None;
    }
    // A static solve with no stiction deformation leaves friction out
    // (Adams' "sliding only"); otherwise the displacement from the anchor
    // stands in for the slip.
    let creep = match anchor {
        Some(anchor) => {
            if *max_stiction_deformation <= 0.0 {
                return None;
            }
            Some((anchor, 1.0 / max_stiction_deformation))
        }
        None => None,
    };

    // The reaction this joint carries, in marker I's axes: the radial and
    // axial force and the bending and axial moment about the pin centre.
    let (force, torque) = problem.joint_wrench(params, *joint, lambda);
    let axis = problem.frame(params, vel, j.marker1).z.v;
    let f_axial = force.dot(axis);
    let f_radial = (force - f_axial * axis).length();
    let m_axial = torque.dot(axis);
    let m_bending = (torque - m_axial * axis).length();
    let over = |arm: f64, v: f64| if arm > 0.0 { v / arm } else { 0.0 };
    // Adams' "input forces to friction": the reaction force creates
    // friction only when the element says so.
    let gated = |v: f64| if *reaction_force { v } else { 0.0 };

    let mut out = Friction {
        load: FrictionLoad {
            force_element: *id,
            joint: *joint,
            normal_load: 0.0,
            slip: 0.0,
            coefficient: 0.0,
            torque: 0.0,
            force: 0.0,
        },
        along_rotation: 0.0,
        along_translation: 0.0,
        couple: DVec3::ZERO,
    };
    // The slip of a joint coordinate: its rate, or â at rest â its
    // displacement from the stiction anchor.
    let slip_of = |slot: DriverSlot| match creep {
        None => problem.coordinate(params, vel, *joint, slot).d,
        Some((anchor, _)) => {
            let now = problem.coordinate(params, vel, *joint, slot).v;
            let was = problem.coordinate(anchor, vel, *joint, slot).v;
            now - was
        }
    };
    // The coefficient at a slip measure: the Stribeck curve of the rate,
    // or the stiction ramp of the displacement. The stiction ramp
    // saturates at `mu_static` over `max_stiction_deformation` and â unlike
    // the velocity regularization â has stiffness at zero displacement,
    // which is what lets a static solve find the force that holds the
    // mechanism (`tanh` is the smooth form of Adams' stiction spring: the
    // deformation is the scale, not a hard cap).
    let mu = |s: f64| match creep {
        None => stribeck(*mu_static, *mu_dynamic, *stiction_velocity, *friction_velocity, s),
        Some((_, inverse)) => mu_static * (s * inverse).tanh(),
    };

    match j.kind {
        JointKind::Revolute => {
            // The pin rubs at its radius, the thrust face at its own, and
            // a bending moment presses the two pin ends apart.
            let n = pin_radius * gated(f_radial)
                + thrust_arm * gated(f_axial.abs())
                + 2.0 * pin_radius * over(*bending_arm, m_bending)
                + preload;
            let slip = slip_of(DriverSlot::Rotation);
            let coefficient = mu(slip.abs());
            out.along_rotation = -coefficient * n * direction(slip);
            out.load.normal_load = n;
            out.load.slip = slip;
            out.load.coefficient = coefficient;
            out.load.torque = out.along_rotation.abs();
        }
        JointKind::Prismatic => {
            // The guide carries the transverse force, the bending moment
            // over its overlap, and the moment about the sliding axis at
            // the pin radius.
            let n = gated(f_radial)
                + 2.0 * over(*bending_arm, m_bending)
                + over(*pin_radius, m_axial.abs())
                + preload;
            let slip = slip_of(DriverSlot::Translation);
            let coefficient = mu(slip.abs());
            out.along_translation = -coefficient * n * direction(slip);
            out.load.normal_load = n;
            out.load.slip = slip;
            out.load.coefficient = coefficient;
            out.load.force = out.along_translation.abs();
        }
        JointKind::Cylindrical => {
            // Sliding and turning rub the same surface: one friction cone,
            // shared by the slip direction at the pin radius.
            let n = gated(f_radial) + 2.0 * over(*bending_arm, m_bending) + preload;
            let (slide, turn) = (
                slip_of(DriverSlot::Translation),
                slip_of(DriverSlot::Rotation) * pin_radius,
            );
            let s = (slide * slide + turn * turn).sqrt();
            let coefficient = mu(s);
            if s > SLIP_EPS {
                out.along_translation = -coefficient * n * slide / s;
                out.along_rotation = -coefficient * n * pin_radius * turn / s;
            }
            out.load.normal_load = n;
            out.load.slip = s;
            out.load.coefficient = coefficient;
            out.load.torque = out.along_rotation.abs();
            out.load.force = out.along_translation.abs();
        }
        JointKind::Spherical => {
            // The ball rubs at its radius against the resultant force; the
            // slip has no single coordinate, so the couple follows the
            // relative angular velocity (or the relative rotation from the
            // stiction anchor).
            let n = pin_radius * gated(force.length()) + preload;
            let slip = match creep {
                None => {
                    let slot = |m: MarkerId| problem.slot_of[&sys.markers[&m].body];
                    vel[slot(j.marker2)].angular - vel[slot(j.marker1)].angular
                }
                Some((anchor, scale)) => {
                    let was = relative_rotation(problem, anchor, j.marker1, j.marker2);
                    let now = relative_rotation(problem, params, j.marker1, j.marker2);
                    let (axis, angle) = (was.conjugate() * now).to_axis_angle();
                    marker_rotation(problem, params, j.marker1) * (axis * angle * scale)
                }
            };
            let s = slip.length();
            let coefficient = mu(s);
            if s > SLIP_EPS {
                out.couple = -coefficient * n * slip / s;
            }
            out.load.normal_load = n;
            out.load.slip = s;
            out.load.coefficient = coefficient;
            out.load.torque = out.couple.length();
        }
        _ => return None,
    }
    Some(out)
}

/// The sign of the slip, and `0` at an exactly zero slip — where Coulomb
/// friction has no direction (`f64::signum` would answer `+1`).
fn direction(slip: f64) -> f64 {
    if slip > 0.0 {
        1.0
    } else if slip < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The multipliers of `loads` when they cover the constraint rows, else
/// all zeros (no reaction known yet — the friction load is then the
/// preload alone).
fn multipliers<'a>(loads: Loads<'a>, problem: &Problem, zeros: &'a mut Vec<f64>) -> &'a [f64] {
    if loads.lambda.len() == problem.rows.len() {
        loads.lambda
    } else {
        *zeros = vec![0.0; problem.rows.len()];
        zeros
    }
}

/// Add every friction element's load to the per-slot applied loads `q`
/// (force, torque about the body origin).
pub(crate) fn friction_forces(
    problem: &Problem,
    params: &[f64],
    vel: &[Twist],
    loads: Loads,
    q: &mut [Twist],
) {
    let mut zeros = Vec::new();
    let lambda = multipliers(loads, problem, &mut zeros);
    for element in problem.sys.forces.values() {
        let Some(f) = evaluate(problem, params, vel, lambda, loads.anchor, element) else {
            continue;
        };
        let joint = f.load.joint;
        for (slot, load) in [
            (DriverSlot::Rotation, f.along_rotation),
            (DriverSlot::Translation, f.along_translation),
        ] {
            if load == 0.0 {
                continue;
            }
            for (s, grad) in coordinate_gradient(problem, params, joint, slot) {
                q[s].linear += load * grad.linear;
                q[s].angular += load * grad.angular;
            }
        }
        if f.couple != DVec3::ZERO {
            let j = &problem.sys.joints[&joint];
            let slot = |m: MarkerId| problem.slot_of[&problem.sys.markers[&m].body];
            q[slot(j.marker2)].angular += f.couple;
            q[slot(j.marker1)].angular -= f.couple;
        }
    }
}

/// Every friction element's load at an instant, in force-element id order.
pub(crate) fn friction_report(
    problem: &Problem,
    params: &[f64],
    vel: &[Twist],
    loads: Loads,
) -> Vec<FrictionLoad> {
    let mut zeros = Vec::new();
    let lambda = multipliers(loads, problem, &mut zeros);
    problem
        .sys
        .forces
        .values()
        .filter_map(|e| evaluate(problem, params, vel, lambda, loads.anchor, e).map(|f| f.load))
        .collect()
}

/// The multiplier indices friction reads: the constraint rows of every
/// joint a load-dependent friction element rubs. `∂Q/∂λ` is zero outside
/// them, so the iteration matrix's `λ` block needs these columns only.
pub(crate) fn read_rows(problem: &Problem) -> Vec<usize> {
    let mut rows = Vec::new();
    for element in problem.sys.forces.values() {
        if !element.reads_reactions() {
            continue;
        }
        let Some(joint) = element.friction_joint() else {
            continue;
        };
        if problem.sys.joints[&joint].suppressed {
            continue;
        }
        rows.extend(
            problem
                .rows
                .iter()
                .enumerate()
                .filter(|(_, r)| r.owner == EquationOwner::Joint(joint))
                .map(|(k, _)| k),
        );
    }
    rows.sort_unstable();
    rows.dedup();
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_coefficient_curve_is_continuous_and_bounded() {
        let (mu_s, mu_d, v_s, v_d) = (0.3, 0.2, 0.01, 0.1);
        let at = |s| stribeck(mu_s, mu_d, v_s, v_d, s);
        assert_eq!(at(0.0), 0.0);
        assert!((at(v_s) - mu_s).abs() < 1e-15);
        assert!((at(v_d) - mu_d).abs() < 1e-15);
        assert_eq!(at(10.0), mu_d);
        // Monotone up to the peak, down to the plateau, never above μs.
        let mut previous = 0.0;
        for i in 0..=100 {
            let s = v_s * i as f64 / 100.0;
            let m = at(s);
            assert!(m >= previous - 1e-15 && m <= mu_s + 1e-15);
            previous = m;
        }
        for i in 0..=100 {
            let s = v_s + (v_d - v_s) * i as f64 / 100.0;
            let m = at(s);
            assert!(m <= mu_s + 1e-15 && m >= mu_d - 1e-15);
        }
        // Smooth: the slope vanishes at both knots, so the one-sided
        // difference quotients agree well below the curve's own scale
        // (`mu_static / v_stiction`).
        let d = 1e-8;
        let scale = mu_s / v_s;
        for knot in [v_s, v_d] {
            let left = (at(knot) - at(knot - d)) / d;
            let right = (at(knot + d) - at(knot)) / d;
            assert!(
                (left - right).abs() < 1e-3 * scale,
                "kink at {knot}: {left} vs {right}"
            );
        }
    }
}
