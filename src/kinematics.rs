//! Kinematic solving: the position initial condition, the velocity level and
//! the acceleration level (M1).
//!
//! The state is a flat parameter vector: per body, translation (3) +
//! quaternion (4) = 7 slots. The constraint set is the per-body unit-norm
//! equation plus, per joint, the joint equations (see [`joint_residuals`]),
//! plus one equation per driver (the prescribed joint value). The position
//! IC is a damped Newton-Raphson solve on the shared `rcad-gcs` numerics
//! core; the velocity and acceleration levels are linear solves against the
//! same constraint Jacobian (M1 second slice).
//!
//! Everything here is pure math over the [`System`] data model: no B-rep, no
//! time integration (that is M2).

use std::collections::BTreeMap;

use glam::{DMat3, DQuat, DVec3};
use rcad_gcs::numerics::{newton_solve, SolveResult as GcsSolveResult};

use crate::body::{BodyId, MarkerId};
use crate::error::{codes, MbdError};
use crate::joint::{DriverSlot, JointKind};
use crate::system::System;

/// Parameter slots per body in the flat state vector.
pub const BODY_PARAMS: usize = 7;

/// A kinematic state: per-body pose in a flat parameter vector.
///
/// Slot `i` occupies `params[7i .. 7i+7]` = `[tx, ty, tz, qx, qy, qz, qw]`
/// (the quaternion is stored `xyzw`, the glam [`DQuat::from_xyzw`] order).
#[derive(Debug, Clone, PartialEq)]
pub struct KinematicState {
    params: Vec<f64>,
    order: Vec<BodyId>,
    slot_of: BTreeMap<BodyId, usize>,
}

impl KinematicState {
    /// Seed the state from the system's authored body poses (the reference
    /// configuration the Newton solve starts from).
    pub fn from_system(sys: &System) -> Self {
        let mut order = Vec::with_capacity(sys.bodies.len());
        let mut slot_of = BTreeMap::new();
        let mut params = Vec::with_capacity(sys.bodies.len() * BODY_PARAMS);
        for (slot, (id, body)) in sys.bodies.iter().enumerate() {
            order.push(*id);
            slot_of.insert(*id, slot);
            params.extend_from_slice(&[
                body.translation.x,
                body.translation.y,
                body.translation.z,
                body.rotation.x,
                body.rotation.y,
                body.rotation.z,
                body.rotation.w,
            ]);
        }
        Self { params, order, slot_of }
    }

    /// The flat parameter vector.
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// The bodies in slot order.
    pub fn order(&self) -> &[BodyId] {
        &self.order
    }

    /// Read a body's pose (translation, unit quaternion).
    pub fn body_pose(&self, body: BodyId) -> Option<(DVec3, DQuat)> {
        let slot = *self.slot_of.get(&body)?;
        let base = slot * BODY_PARAMS;
        let p = self.params.get(base..base + BODY_PARAMS)?;
        Some((
            DVec3::new(p[0], p[1], p[2]),
            DQuat::from_xyzw(p[3], p[4], p[5], p[6]).normalize(),
        ))
    }
}

/// The number of constraint equations for a system at a driver time:
/// one unit-norm equation per body, the joint equations, and one equation
/// per (unsuppressed) driver.
pub fn equation_count(sys: &System) -> usize {
    let mut n = sys.bodies.len();
    for joint in sys.joints.values() {
        if !joint.suppressed {
            n += joint.kind.constraint_count();
        }
    }
    n += sys
        .drivers
        .iter()
        .filter(|d| sys.joints.get(&d.joint).is_some_and(|j| !j.suppressed))
        .count();
    n
}

impl JointKind {
    /// The number of independent constraint equations this joint kind
    /// imposes (the relative DOF it removes from the 6 between two bodies).
    pub fn constraint_count(self) -> usize {
        match self {
            JointKind::Revolute => 5,
            JointKind::Prismatic => 5,
            JointKind::Cylindrical => 4,
            JointKind::Spherical => 3,
            JointKind::Planar => 3,
            JointKind::Universal => 4,
            JointKind::Fixed => 6,
        }
    }
}

/// A marker's world frame at a parameter vector: `(world position, world
/// orientation)` of the marker's attached frame.
fn marker_frame(sys: &System, slot_of: &BTreeMap<BodyId, usize>, params: &[f64], marker: MarkerId) -> (DVec3, DQuat) {
    let m = &sys.markers[&marker];
    let base = slot_of[&m.body] * BODY_PARAMS;
    let t = DVec3::new(params[base], params[base + 1], params[base + 2]);
    let q = DQuat::from_xyzw(params[base + 3], params[base + 4], params[base + 5], params[base + 6])
        .normalize();
    (t + q * m.translation, q * m.rotation)
}

/// Write one joint's constraint residuals into `out`; returns the count.
///
/// The formulations (all in the marker frames, `dp = p2 - p1`,
/// `R_rel = R1^-1 R2`):
/// - `Revolute`: origin coincidence (3) + the second marker's Z axis has no
///   component perpendicular to the first's (2).
/// - `Prismatic`: relative orientation identity (3, the upper-triangular
///   off-diagonal rotation entries) + the offset is axial (2).
/// - `Cylindrical`: axis parallelism (2) + axial offset (2).
/// - `Spherical`: origin coincidence (3).
/// - `Planar`: axis parallelism (2) + zero axial offset (1).
/// - `Universal`: origin coincidence (3) + perpendicular axes (1).
/// - `Fixed`: origin coincidence (3) + relative orientation identity (3).
fn joint_residuals(
    kind: JointKind,
    p1: DVec3,
    r1: DQuat,
    p2: DVec3,
    r2: DQuat,
    out: &mut [f64],
) -> usize {
    let dp = p2 - p1;
    let r_rel = r1.inverse() * r2;
    let z_rel = r_rel * DVec3::Z;
    let x1 = r1 * DVec3::X;
    let y1 = r1 * DVec3::Y;
    let z1 = r1 * DVec3::Z;
    let m = DMat3::from_quat(r_rel);
    match kind {
        JointKind::Revolute => {
            out[0] = dp.x;
            out[1] = dp.y;
            out[2] = dp.z;
            out[3] = z_rel.x;
            out[4] = z_rel.y;
            5
        }
        JointKind::Prismatic => {
            out[0] = m.x_axis.y;
            out[1] = m.x_axis.z;
            out[2] = m.y_axis.z;
            out[3] = dp.dot(x1);
            out[4] = dp.dot(y1);
            5
        }
        JointKind::Cylindrical => {
            out[0] = z_rel.x;
            out[1] = z_rel.y;
            out[2] = dp.dot(x1);
            out[3] = dp.dot(y1);
            4
        }
        JointKind::Spherical => {
            out[0] = dp.x;
            out[1] = dp.y;
            out[2] = dp.z;
            3
        }
        JointKind::Planar => {
            out[0] = z_rel.x;
            out[1] = z_rel.y;
            out[2] = dp.dot(z1);
            3
        }
        JointKind::Universal => {
            out[0] = dp.x;
            out[1] = dp.y;
            out[2] = dp.z;
            out[3] = z1.dot(r2 * DVec3::Z);
            4
        }
        JointKind::Fixed => {
            out[0] = dp.x;
            out[1] = dp.y;
            out[2] = dp.z;
            out[3] = m.x_axis.y;
            out[4] = m.x_axis.z;
            out[5] = m.y_axis.z;
            6
        }
    }
}

/// One driver's residual: the prescribed joint value minus the value the
/// current state implies. Only the slots that are actual DOF of the joint
/// kind are accepted (the same contract as [`crate::System::validate`]).
fn driver_residual(
    kind: JointKind,
    slot: DriverSlot,
    value: f64,
    p1: DVec3,
    r1: DQuat,
    p2: DVec3,
    r2: DQuat,
) -> f64 {
    let r_rel = r1.inverse() * r2;
    let m = DMat3::from_quat(r_rel);
    let z1 = r1 * DVec3::Z;
    let dp = p2 - p1;
    match (kind, slot) {
        // Rotation angle about the common Z axis: atan2(R_rel[1][0], R_rel[0][0]).
        (JointKind::Revolute, DriverSlot::Rotation) => {
            m.x_axis.y.atan2(m.x_axis.x) - value
        }
        (JointKind::Cylindrical, DriverSlot::Rotation) => {
            m.x_axis.y.atan2(m.x_axis.x) - value
        }
        (JointKind::Prismatic, DriverSlot::Translation) => dp.dot(z1) - value,
        (JointKind::Cylindrical, DriverSlot::Translation) => dp.dot(z1) - value,
        // A slot that is not a DOF of the kind is rejected by validation; the
        // residual is never evaluated for it.
        _ => 0.0,
    }
}

/// The kinematic problem at a driver time: the system plus the slot layout,
/// exposing the residual callback the Newton core consumes.
struct KinematicProblem<'a> {
    sys: &'a System,
    slot_of: BTreeMap<BodyId, usize>,
    n_eq: usize,
    time: f64,
}

impl<'a> KinematicProblem<'a> {
    fn new(sys: &'a System, time: f64) -> Self {
        let mut slot_of = BTreeMap::new();
        for (slot, id) in sys.bodies.keys().enumerate() {
            slot_of.insert(*id, slot);
        }
        Self { sys, slot_of, n_eq: equation_count(sys), time }
    }

    fn eval(&self, params: &[f64], out: &mut [f64]) {
        let mut k = 0;
        // Per-body unit-norm equations.
        for slot in 0..self.sys.bodies.len() {
            let base = slot * BODY_PARAMS;
            let q = DQuat::from_xyzw(
                params[base + 3],
                params[base + 4],
                params[base + 5],
                params[base + 6],
            );
            out[k] = q.length_squared() - 1.0;
            k += 1;
        }
        // Joint equations.
        for joint in self.sys.joints.values() {
            if joint.suppressed {
                continue;
            }
            let (p1, r1) = marker_frame(self.sys, &self.slot_of, params, joint.marker1);
            let (p2, r2) = marker_frame(self.sys, &self.slot_of, params, joint.marker2);
            k += joint_residuals(joint.kind, p1, r1, p2, r2, &mut out[k..]);
        }
        // Driver equations.
        for driver in &self.sys.drivers {
            let Some(joint) = self.sys.joints.get(&driver.joint) else {
                continue;
            };
            if joint.suppressed {
                continue;
            }
            let (p1, r1) = marker_frame(self.sys, &self.slot_of, params, joint.marker1);
            let (p2, r2) = marker_frame(self.sys, &self.slot_of, params, joint.marker2);
            let value = driver.value_at(self.time);
            out[k] = driver_residual(joint.kind, driver.slot, value, p1, r1, p2, r2);
            k += 1;
        }
        debug_assert_eq!(k, self.n_eq, "residual count matches the layout");
    }

    /// The fixed mask: a grounded body holds all 7 of its slots.
    fn fixed_mask(&self) -> Vec<bool> {
        let mut fixed = vec![false; self.sys.bodies.len() * BODY_PARAMS];
        for (id, body) in self.sys.bodies.iter() {
            if body.grounded {
                let base = self.slot_of[id] * BODY_PARAMS;
                for slot in fixed.iter_mut().skip(base).take(BODY_PARAMS) {
                    *slot = true;
                }
            }
        }
        fixed
    }
}

/// Solve the position initial condition at a driver time: the per-body poses
/// satisfying every joint equation and every driver value, starting from the
/// system's authored poses.
///
/// The solve is a damped Newton-Raphson on the shared `rcad-gcs` numerics
/// core (central-difference Jacobian). Failure to converge is data
/// ([`codes::KINEMATICS_NOT_CONVERGED`]), never a panic.
pub fn position_ic(sys: &System, time: f64) -> Result<KinematicState, MbdError> {
    sys.validate()?;
    let problem = KinematicProblem::new(sys, time);
    let mut state = KinematicState::from_system(sys);
    let fixed = problem.fixed_mask();
    let GcsSolveResult { converged, residual, iterations } =
        newton_solve(&mut state.params, &fixed, problem.n_eq, |p, out| {
            problem.eval(p, out);
        });
    if !converged {
        return Err(MbdError::new(
            codes::KINEMATICS_NOT_CONVERGED,
            format!(
                "position IC did not converge: rms residual {residual:.3e} after {iterations} iterations"
            ),
            "check the joint layout for over-constraint, or seed the bodies closer to a valid configuration",
        ));
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Body;
    use crate::force::ForceElement;
    use crate::joint::{Driver, DriverProfile, Joint};
    use crate::{BodyId, ForceId, JointId, Marker, MarkerId};

    fn grounded_base() -> Body {
        Body {
            id: BodyId(1),
            name: "base".into(),
            mass_kg: 0.0,
            com: DVec3::ZERO,
            inertia_com: DMat3::ZERO,
            translation: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            grounded: true,
        }
    }

    /// A grounded base plus one link on a revolute hinge at the origin, with
    /// a constant-rate driver. The link's authored pose already satisfies
    /// the joint; the driver value is what the position IC must hold.
    fn hinge(driver_angle: f64) -> System {
        let mut sys = System::new("hinge");
        sys.add_body(grounded_base()).unwrap();
        sys.add_body(Body {
            id: BodyId(2),
            name: "link".into(),
            mass_kg: 1.0,
            com: DVec3::new(0.0, 0.0, -0.5),
            inertia_com: DMat3::from_diagonal(DVec3::new(0.1, 0.1, 0.01)),
            translation: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            grounded: false,
        })
        .unwrap();
        for (id, body) in [(1, BodyId(1)), (2, BodyId(2))] {
            sys.add_marker(Marker {
                id: MarkerId(id),
                name: format!("m{id}"),
                body,
                translation: DVec3::ZERO,
                rotation: DQuat::IDENTITY,
            })
            .unwrap();
        }
        sys.add_joint(Joint {
            id: JointId(1),
            name: "hinge".into(),
            kind: JointKind::Revolute,
            marker1: MarkerId(1),
            marker2: MarkerId(2),
            suppressed: false,
        })
        .unwrap();
        sys.add_driver(Driver {
            joint: JointId(1),
            slot: DriverSlot::Rotation,
            profile: DriverProfile::Constant {
                start: driver_angle,
                rate: 0.0,
            },
        });
        sys
    }

    /// The residual layout matches the equation count, and a valid
    /// configuration evaluates to zero residuals.
    #[test]
    fn residual_layout_is_consistent() {
        let sys = hinge(0.0);
        assert_eq!(equation_count(&sys), 2 + 5 + 1);
        let problem = KinematicProblem::new(&sys, 0.0);
        let mut out = vec![0.0; problem.n_eq];
        let state = KinematicState::from_system(&sys);
        problem.eval(state.params(), &mut out);
        for (i, r) in out.iter().enumerate() {
            assert!(r.abs() < 1e-9, "residual {i} = {r}");
        }
    }

    /// The position IC holds the driver angle: the link rotates about the
    /// hinge Z axis by exactly the prescribed value, starting from the
    /// authored (identity) pose.
    #[test]
    fn position_ic_holds_the_driver_angle() {
        for angle in [0.0, 0.5, 1.2, -2.0] {
            let sys = hinge(angle);
            let state = position_ic(&sys, 0.0).expect("converges");
            let (t, r) = state.body_pose(BodyId(2)).unwrap();
            assert!(t.length() < 1e-9, "origin coincidence held: {t:?}");
            let z = r * DVec3::Z;
            assert!(z.dot(DVec3::Z) > 1.0 - 1e-9, "axis stayed parallel");
            let theta = 2.0 * r.z.atan2(r.w);
            assert!(
                (theta - angle).abs() < 1e-9,
                "driver angle {angle} held, got {theta}"
            );
        }
    }

    /// The validation gate rejects a driver slot that is not a DOF of the
    /// joint before any solve runs.
    #[test]
    fn bad_driver_slot_is_rejected_before_solving() {
        let mut sys = hinge(0.0);
        sys.drivers[0].slot = DriverSlot::Translation;
        let err = position_ic(&sys, 0.0).unwrap_err();
        assert_eq!(err.code, codes::SLOT_UNSUPPORTED);
    }

    /// A suppressed joint drops its equations (and its driver's) from the
    /// layout: only the per-body unit-norm equations remain.
    #[test]
    fn suppressed_joint_leaves_no_equations() {
        let mut sys = hinge(0.0);
        assert_eq!(equation_count(&sys), 8, "2 unit + 5 joint + 1 driver");
        sys.joints.get_mut(&JointId(1)).unwrap().suppressed = true;
        assert_eq!(equation_count(&sys), 2, "only the two unit-norm equations remain");
    }

    /// The force vocabulary is untouched by the kinematic slice (the M0
    /// tables stay valid next to the new module).
    #[test]
    fn force_tables_still_validate() {
        let mut sys = hinge(0.0);
        sys.add_force(ForceElement::SpringDamper {
            id: ForceId(1),
            name: "return".into(),
            marker1: MarkerId(1),
            marker2: MarkerId(2),
            stiffness: 100.0,
            damping: 1.0,
            free_length: 0.0,
        })
        .unwrap();
        assert!(sys.validate().is_ok());
    }
}
