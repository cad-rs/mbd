//! Kinematic solving: the position initial condition, the velocity level and
//! the acceleration level (M1), and the time sweep over them.
//!
//! **Coordinates.** The position state is a flat parameter vector: per body,
//! translation (3) + quaternion (4) = 7 slots. The velocity and acceleration
//! levels live in the tangent space instead: per free body, the linear
//! velocity of the body-frame origin and the angular velocity, both in the
//! inertial (world) frame ([`Twist`]) — 6 coordinates, no quaternion-rate
//! redundancy. This is the crate-wide convention (roadmap §3).
//!
//! **Equations.** Per joint, the joint equations; per driver, the
//! prescribed joint coordinate. The position IC adds one unit-norm equation
//! per body (the quaternion manifold) and solves on the shared `rcad-gcs`
//! Newton core. The velocity and acceleration levels are linear solves
//! against the analytic tangent-space Jacobian, which comes from the same
//! residual code evaluated on second-order jets ([`crate::jet`]), so the
//! position residuals, the Jacobian and the quadratic term `γ` cannot drift
//! apart.
//!
//! **Redundancy.** Real CAD assemblies are over-constrained (a planar
//! four-bar built from 3D revolutes carries three redundant equations).
//! Every linear solve goes through a rank-revealing factorization:
//! dependent but consistent equations are reported as redundant and
//! skipped; dependent inconsistent ones are conflicts and fail as data.
//!
//! **Angles.** A rotation driver's residual is the measured angle minus the
//! prescribed one wrapped into `[−π, π)`, so a motor may run any number of
//! revolutions; the pose is the same modulo a turn.

use std::collections::BTreeMap;
use std::f64::consts::{PI, TAU};

use glam::{DQuat, DVec3};
use rcad_gcs::numerics::{newton_solve, SolveResult as GcsSolveResult};

use crate::body::{BodyId, MarkerId};
use crate::error::{codes, MbdError};
use crate::jet::{BodyMotion, FrameJet, Jet};
use crate::joint::{joint_coordinate, joint_rows, DriverSlot, JointId};
use crate::linalg::RowFactor;
use crate::system::System;
use crate::coupler::{Coupler, CouplerId};

/// Parameter slots per body in the flat state vector.
pub const BODY_PARAMS: usize = 7;

/// Tangent coordinates per free body (linear + angular velocity).
pub const BODY_TWIST: usize = 6;

/// A kinematic state: per-body pose in a flat parameter vector.
///
/// Slot `i` occupies `params[7i .. 7i+7]` = `[tx, ty, tz, qx, qy, qz, qw]`
/// (the quaternion is stored `xyzw`, the glam [`DQuat::from_xyzw`] order).
///
/// The state also tracks every rotation joint coordinate (revolute,
/// cylindrical, planar) **unwrapped across turns** — what couplers relate
/// and what [`KinematicState::joint_angle`] reports.
#[derive(Debug, Clone, PartialEq)]
pub struct KinematicState {
    params: Vec<f64>,
    order: Vec<BodyId>,
    slot_of: BTreeMap<BodyId, usize>,
    angles: BTreeMap<JointId, f64>,
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
            params.extend_from_slice(&pose_params(body.translation, body.rotation));
        }
        let rest = vec![Twist::default(); order.len()];
        let angles = rotation_joints(sys)
            .map(|j| {
                let raw = coordinate_jet(sys, &slot_of, &params, &rest, j.id, DriverSlot::Rotation);
                (j.id, raw.v)
            })
            .collect();
        Self { params, order, slot_of, angles }
    }

    /// A rotation joint's coordinate (rad), unwrapped across turns since the
    /// authored configuration (the authored value is the raw angle in
    /// `(−π, π]`). `None` for joints without a rotation coordinate.
    pub fn joint_angle(&self, joint: JointId) -> Option<f64> {
        self.angles.get(&joint).copied()
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
        Some(read_pose(&self.params, slot))
    }

    /// Overwrite a body's pose (the quaternion is normalized). Returns
    /// `false` when the body is not in the state.
    pub fn set_body_pose(&mut self, body: BodyId, translation: DVec3, rotation: DQuat) -> bool {
        let Some(&slot) = self.slot_of.get(&body) else {
            return false;
        };
        let base = slot * BODY_PARAMS;
        self.params[base..base + BODY_PARAMS]
            .copy_from_slice(&pose_params(translation, rotation.normalize()));
        true
    }

    /// A body-fixed point (body-frame coordinates) in the world frame.
    pub fn world_point(&self, body: BodyId, local: DVec3) -> Option<DVec3> {
        let (t, r) = self.body_pose(body)?;
        Some(t + r * local)
    }

    /// Take over `other`'s tracked rotation angles for the joints both
    /// states track (a merged system keeps the original joint ids).
    pub(crate) fn copy_angles_from(&mut self, other: &KinematicState) {
        for (joint, angle) in self.angles.iter_mut() {
            if let Some(a) = other.angles.get(joint) {
                *angle = *a;
            }
        }
    }

    fn normalize_quaternions(&mut self) {
        for slot in 0..self.order.len() {
            let (t, r) = read_pose(&self.params, slot);
            let base = slot * BODY_PARAMS;
            self.params[base..base + BODY_PARAMS].copy_from_slice(&pose_params(t, r));
        }
    }
}

fn pose_params(t: DVec3, r: DQuat) -> [f64; BODY_PARAMS] {
    [t.x, t.y, t.z, r.x, r.y, r.z, r.w]
}

fn read_pose(params: &[f64], slot: usize) -> (DVec3, DQuat) {
    let p = &params[slot * BODY_PARAMS..(slot + 1) * BODY_PARAMS];
    (
        DVec3::new(p[0], p[1], p[2]),
        DQuat::from_xyzw(p[3], p[4], p[5], p[6]).normalize(),
    )
}

/// The unsuppressed joints that carry a rotation coordinate.
fn rotation_joints(sys: &System) -> impl Iterator<Item = &crate::joint::Joint> {
    sys.joints
        .values()
        .filter(|j| !j.suppressed && j.kind.drives(DriverSlot::Rotation))
}

/// A marker's world frame as jets along `motion`.
fn marker_frame(
    sys: &System,
    slot_of: &BTreeMap<BodyId, usize>,
    params: &[f64],
    motion: &[Twist],
    marker: MarkerId,
) -> FrameJet {
    let m = &sys.markers[&marker];
    let slot = slot_of[&m.body];
    let (t, r) = read_pose(params, slot);
    let body = BodyMotion { t, r, v: motion[slot].linear, w: motion[slot].angular };
    body.frame(m.translation, m.rotation)
}

/// A joint coordinate with rate and acceleration at a solved frame (the
/// measure behind [`KinematicFrame::joint_coordinate`]): the jet along the
/// frame's velocities gives the rate and the velocity-quadratic part, the
/// jet along its accelerations the linear part `J·q̈`.
pub(crate) fn coordinate_motion(
    sys: &System,
    frame: &KinematicFrame,
    joint: JointId,
    slot: DriverSlot,
) -> Option<crate::measure::JointMotion> {
    let state = &frame.state;
    if !state.order.iter().eq(sys.bodies.keys()) {
        return None;
    }
    let vel: Vec<Twist> = state.order.iter().map(|id| frame.velocity[id]).collect();
    let acc: Vec<Twist> = state.order.iter().map(|id| frame.acceleration[id]).collect();
    let jv = coordinate_jet(sys, &state.slot_of, &state.params, &vel, joint, slot);
    let ja = coordinate_jet(sys, &state.slot_of, &state.params, &acc, joint, slot);
    let value = match slot {
        DriverSlot::Rotation => state.angles.get(&joint).copied().unwrap_or(jv.v),
        DriverSlot::Translation => jv.v,
    };
    Some(crate::measure::JointMotion {
        value,
        velocity: jv.d,
        acceleration: jv.dd + ja.d,
    })
}

/// A joint coordinate as a jet (raw: rotation in `(−π, π]`).
fn coordinate_jet(
    sys: &System,
    slot_of: &BTreeMap<BodyId, usize>,
    params: &[f64],
    motion: &[Twist],
    joint: JointId,
    slot: DriverSlot,
) -> Jet {
    let j = &sys.joints[&joint];
    let f1 = marker_frame(sys, slot_of, params, motion, j.marker1);
    let f2 = marker_frame(sys, slot_of, params, motion, j.marker2);
    joint_coordinate(j.kind, slot, &f1, &f2).unwrap_or_default()
}

/// A rigid-body velocity (or acceleration): the linear rate of the body
/// frame origin and the angular rate, both in the world frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Twist {
    pub linear: DVec3,
    pub angular: DVec3,
}

impl Twist {
    /// The rate of a body-fixed point at world arm `arm` (point minus body
    /// origin) for a velocity twist: `v + ω × arm`.
    pub fn point_velocity(&self, arm: DVec3) -> DVec3 {
        self.linear + self.angular.cross(arm)
    }
}

/// Which entity owns a constraint equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EquationOwner {
    Joint(JointId),
    Coupler(CouplerId),
    /// A driver, by its index in [`System::drivers`].
    Driver(usize),
}

/// One constraint equation: its owner and the row within the owner (joint
/// rows follow the formulation order of the joint kind; a driver has one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EquationRef {
    pub owner: EquationOwner,
    pub row: usize,
}

/// The rank analysis of the constraint equations at a state.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KinematicDiagnosis {
    /// Tangent coordinates: 6 per free body.
    pub coordinates: usize,
    /// Rank of the full constraint Jacobian (joints + couplers + drivers).
    pub rank: usize,
    /// Degrees of freedom left with the drivers applied (`coordinates −
    /// rank`): 0 is what a kinematic analysis needs.
    pub dof: usize,
    /// The mechanism's mobility, drivers removed (`coordinates −
    /// rank(joint + coupler rows)`): how many drivers it takes to
    /// prescribe it.
    pub mobility: usize,
    /// Dependent equations consistent with the rest: removable without
    /// changing the motion (their reactions are not unique).
    pub redundant: Vec<EquationRef>,
    /// Dependent equations inconsistent with the rest at this
    /// linearization: no configuration satisfies everything.
    pub conflicting: Vec<EquationRef>,
}

/// One solved instant: pose, velocity and acceleration of every body.
#[derive(Debug, Clone, PartialEq)]
pub struct KinematicFrame {
    pub time: f64,
    pub state: KinematicState,
    /// Per-body velocity (grounded bodies carry zero).
    pub velocity: BTreeMap<BodyId, Twist>,
    /// Per-body acceleration (grounded bodies carry zero).
    pub acceleration: BTreeMap<BodyId, Twist>,
    /// The redundant equations skipped by the linear solves.
    pub redundant: Vec<EquationRef>,
}

/// The number of equations the position IC solves for a system: one
/// unit-norm equation per body, the joint equations, one per coupler and
/// one per driver (suppressed joints drop out with their couplers and
/// drivers).
pub fn equation_count(sys: &System) -> usize {
    sys.bodies.len() + constraint_layout(sys).len()
}

/// Whether every joint a coupler references is active.
fn coupler_active(sys: &System, coupler: &Coupler) -> bool {
    coupler
        .terms
        .iter()
        .all(|t| sys.joints.get(&t.joint).is_some_and(|j| !j.suppressed))
}

/// The constraint equations: joints in id order, couplers in id order (the
/// mechanism), then drivers in authoring order (the prescription);
/// suppressed joints drop out with their couplers and drivers.
fn constraint_layout(sys: &System) -> Vec<EquationRef> {
    let mut rows = Vec::new();
    for joint in sys.joints.values().filter(|j| !j.suppressed) {
        for row in 0..joint.kind.constraint_count() {
            rows.push(EquationRef { owner: EquationOwner::Joint(joint.id), row });
        }
    }
    for coupler in sys.couplers.values().filter(|c| coupler_active(sys, c)) {
        rows.push(EquationRef { owner: EquationOwner::Coupler(coupler.id), row: 0 });
    }
    for (index, driver) in sys.drivers.iter().enumerate() {
        if sys.joints.get(&driver.joint).is_some_and(|j| !j.suppressed) {
            rows.push(EquationRef { owner: EquationOwner::Driver(index), row: 0 });
        }
    }
    rows
}

/// Wrap an angle into `[−π, π)`.
fn wrap_angle(a: f64) -> f64 {
    (a + PI).rem_euclid(TAU) - PI
}

/// The kinematic problem at a driver time: the system plus the slot and
/// equation layouts.
struct Problem<'a> {
    sys: &'a System,
    slot_of: BTreeMap<BodyId, usize>,
    /// Tangent column base per slot (`None` = grounded).
    col_of: Vec<Option<usize>>,
    n_cols: usize,
    rows: Vec<EquationRef>,
    /// Joint rows (contiguous at the front).
    n_joint_rows: usize,
    /// Joint + coupler rows (the mechanism, contiguous at the front).
    n_mechanism_rows: usize,
    time: f64,
    /// Unwrapping references for rotation coordinates (the seed's tracked
    /// angles).
    refs: BTreeMap<JointId, f64>,
    /// Rotation coordinates prescribed by a driver: the driver index (its
    /// value is the unwrapping reference).
    rotation_driver: BTreeMap<JointId, usize>,
    /// Authored values of every coordinate a coupler uses.
    authored: BTreeMap<(JointId, DriverSlot), f64>,
}

impl<'a> Problem<'a> {
    fn new(sys: &'a System, time: f64, refs: &BTreeMap<JointId, f64>) -> Self {
        let mut slot_of = BTreeMap::new();
        let mut col_of = Vec::with_capacity(sys.bodies.len());
        let mut n_cols = 0;
        for (slot, (id, body)) in sys.bodies.iter().enumerate() {
            slot_of.insert(*id, slot);
            if body.grounded {
                col_of.push(None);
            } else {
                col_of.push(Some(n_cols));
                n_cols += BODY_TWIST;
            }
        }
        let rows = constraint_layout(sys);
        let count = |f: fn(&EquationOwner) -> bool| rows.iter().filter(|r| f(&r.owner)).count();
        let n_joint_rows = count(|o| matches!(o, EquationOwner::Joint(_)));
        let n_mechanism_rows = n_joint_rows + count(|o| matches!(o, EquationOwner::Coupler(_)));
        let rotation_driver = sys
            .drivers
            .iter()
            .enumerate()
            .filter(|(_, d)| d.slot == DriverSlot::Rotation)
            .map(|(i, d)| (d.joint, i))
            .collect();
        let mut problem = Self {
            sys,
            slot_of,
            col_of,
            n_cols,
            rows,
            n_joint_rows,
            n_mechanism_rows,
            time,
            refs: refs.clone(),
            rotation_driver,
            authored: BTreeMap::new(),
        };
        let authored_state = KinematicState::from_system(sys);
        let rest = vec![Twist::default(); problem.n_bodies()];
        let mut authored = BTreeMap::new();
        for coupler in sys.couplers.values().filter(|c| coupler_active(sys, c)) {
            for term in &coupler.terms {
                let raw = problem.raw(&authored_state.params, &rest, term.joint, term.slot);
                authored.insert((term.joint, term.slot), raw.v);
            }
        }
        problem.authored = authored;
        problem
    }

    fn n_bodies(&self) -> usize {
        self.col_of.len()
    }

    fn frame(&self, params: &[f64], motion: &[Twist], marker: MarkerId) -> FrameJet {
        marker_frame(self.sys, &self.slot_of, params, motion, marker)
    }

    fn raw(&self, params: &[f64], motion: &[Twist], joint: JointId, slot: DriverSlot) -> Jet {
        coordinate_jet(self.sys, &self.slot_of, params, motion, joint, slot)
    }

    /// The reference a rotation coordinate unwraps against: its driver's
    /// prescribed value when driven, else the seed's tracked angle.
    fn angle_reference(&self, joint: JointId) -> f64 {
        if let Some(&i) = self.rotation_driver.get(&joint) {
            return self.sys.drivers[i].value_at(self.time);
        }
        self.refs
            .get(&joint)
            .or_else(|| self.authored.get(&(joint, DriverSlot::Rotation)))
            .copied()
            .unwrap_or(0.0)
    }

    /// A joint coordinate as a jet, rotation unwrapped against its
    /// reference.
    fn coordinate(&self, params: &[f64], motion: &[Twist], joint: JointId, slot: DriverSlot) -> Jet {
        let mut jet = self.raw(params, motion, joint, slot);
        if slot == DriverSlot::Rotation {
            let reference = self.angle_reference(joint);
            jet.v = reference + wrap_angle(jet.v - reference);
        }
        jet
    }

    /// The tracked rotation angles at `params` (after a converged solve).
    fn track_angles(&self, params: &[f64]) -> BTreeMap<JointId, f64> {
        let rest = vec![Twist::default(); self.n_bodies()];
        rotation_joints(self.sys)
            .map(|j| (j.id, self.coordinate(params, &rest, j.id, DriverSlot::Rotation).v))
            .collect()
    }

    /// Every constraint row as a jet along `motion` (zero accelerations):
    /// joint rows, coupler rows (`Σ cᵢ·(qᵢ − qᵢ_authored)`), then the drivers'
    /// measured joint coordinates (time terms excluded — the callers add
    /// them).
    fn jets(&self, params: &[f64], motion: &[Twist]) -> Vec<Jet> {
        let mut out = Vec::with_capacity(self.rows.len());
        let mut buf = [Jet::default(); 6];
        for joint in self.sys.joints.values().filter(|j| !j.suppressed) {
            let f1 = self.frame(params, motion, joint.marker1);
            let f2 = self.frame(params, motion, joint.marker2);
            let n = joint_rows(joint.kind, &f1, &f2, &mut buf);
            out.extend_from_slice(&buf[..n]);
        }
        for row in &self.rows[self.n_joint_rows..] {
            out.push(match row.owner {
                EquationOwner::Joint(_) => unreachable!("joint rows come first"),
                EquationOwner::Coupler(id) => {
                    let mut sum = Jet::default();
                    for term in &self.sys.couplers[&id].terms {
                        let q = self.coordinate(params, motion, term.joint, term.slot);
                        let q0 = self.authored[&(term.joint, term.slot)];
                        sum.v += term.coefficient * (q.v - q0);
                        sum.d += term.coefficient * q.d;
                        sum.dd += term.coefficient * q.dd;
                    }
                    sum
                }
                EquationOwner::Driver(index) => {
                    let driver = &self.sys.drivers[index];
                    self.raw(params, motion, driver.joint, driver.slot)
                }
            });
        }
        debug_assert_eq!(out.len(), self.rows.len());
        out
    }

    /// The constraint residuals at the position level (drivers: measured
    /// minus prescribed, angles wrapped).
    fn constraint_residuals(&self, params: &[f64], out: &mut [f64]) {
        let rest = vec![Twist::default(); self.n_bodies()];
        let jets = self.jets(params, &rest);
        for (k, (row, jet)) in self.rows.iter().zip(&jets).enumerate() {
            out[k] = match row.owner {
                EquationOwner::Joint(_) | EquationOwner::Coupler(_) => jet.v,
                EquationOwner::Driver(index) => {
                    let driver = &self.sys.drivers[index];
                    let r = jet.v - driver.value_at(self.time);
                    match driver.slot {
                        DriverSlot::Rotation => wrap_angle(r),
                        DriverSlot::Translation => r,
                    }
                }
            };
        }
    }

    /// The position-IC residual vector: unit-norm rows, then constraints.
    fn eval(&self, params: &[f64], out: &mut [f64]) {
        let n = self.n_bodies();
        for (slot, o) in out.iter_mut().enumerate().take(n) {
            let base = slot * BODY_PARAMS;
            let q = DQuat::from_xyzw(params[base + 3], params[base + 4], params[base + 5], params[base + 6]);
            *o = q.length_squared() - 1.0;
        }
        self.constraint_residuals(params, &mut out[n..]);
    }

    /// The fixed mask: a grounded body holds all 7 of its slots.
    fn fixed_mask(&self) -> Vec<bool> {
        let mut fixed = vec![false; self.n_bodies() * BODY_PARAMS];
        for (slot, col) in self.col_of.iter().enumerate() {
            if col.is_none() {
                fixed[slot * BODY_PARAMS..(slot + 1) * BODY_PARAMS].fill(true);
            }
        }
        fixed
    }

    /// The analytic tangent-space Jacobian (rows × 6·free bodies): column
    /// `c` is the rate of every row under a unit motion in coordinate `c`.
    fn jacobian(&self, params: &[f64]) -> Vec<Vec<f64>> {
        let mut j = vec![vec![0.0; self.n_cols]; self.rows.len()];
        let mut motion = vec![Twist::default(); self.n_bodies()];
        for slot in 0..self.n_bodies() {
            let Some(base) = self.col_of[slot] else { continue };
            for dir in 0..BODY_TWIST {
                let mut unit = Twist::default();
                if dir < 3 {
                    unit.linear[dir] = 1.0;
                } else {
                    unit.angular[dir - 3] = 1.0;
                }
                motion[slot] = unit;
                for (row, jet) in j.iter_mut().zip(self.jets(params, &motion)) {
                    row[base + dir] = jet.d;
                }
            }
            motion[slot] = Twist::default();
        }
        j
    }

    /// Tangent solution vector → per-slot twists (grounded: zero).
    fn twists(&self, x: &[f64]) -> Vec<Twist> {
        self.col_of
            .iter()
            .map(|col| match col {
                None => Twist::default(),
                Some(b) => Twist {
                    linear: DVec3::new(x[*b], x[b + 1], x[b + 2]),
                    angular: DVec3::new(x[b + 3], x[b + 4], x[b + 5]),
                },
            })
            .collect()
    }

    fn refs(&self, indices: &[usize]) -> Vec<EquationRef> {
        indices.iter().map(|&k| self.rows[k]).collect()
    }

    fn diagnose(&self, params: &[f64]) -> KinematicDiagnosis {
        let j = self.jacobian(params);
        let mut r = vec![0.0; self.rows.len()];
        self.constraint_residuals(params, &mut r);
        let step: Vec<f64> = r.iter().map(|v| -v).collect();
        let factor = RowFactor::new(&j, self.n_cols);
        let (_, conflicts) = factor.solve(&step);
        let mut redundant: Vec<usize> = factor
            .dependent()
            .iter()
            .copied()
            .filter(|k| !conflicts.contains(k))
            .collect();
        redundant.sort_unstable();
        let joint_rank = RowFactor::new(&j[..self.n_mechanism_rows], self.n_cols).rank();
        KinematicDiagnosis {
            coordinates: self.n_cols,
            rank: factor.rank(),
            dof: self.n_cols - factor.rank(),
            mobility: self.n_cols - joint_rank,
            redundant: self.refs(&redundant),
            conflicting: self.refs(&conflicts),
        }
    }

    /// Solve one linear level `J x = b`; conflicts are a singular
    /// configuration.
    fn solve_level(&self, factor: &RowFactor, b: &[f64], level: &str) -> Result<Vec<Twist>, MbdError> {
        let (x, conflicts) = factor.solve(b);
        if !conflicts.is_empty() {
            return Err(MbdError::new(
                codes::KINEMATIC_SINGULAR,
                format!(
                    "{level} equations inconsistent at t = {}: {}",
                    self.time,
                    describe(self.sys, &self.refs(&conflicts))
                ),
                "the mechanism is at a lock-up/toggle position, or drivers over-prescribe one degree of freedom",
            ));
        }
        Ok(self.twists(&x))
    }
}

/// Human-readable equation list for error details.
fn describe(sys: &System, refs: &[EquationRef]) -> String {
    let names: Vec<String> = refs
        .iter()
        .map(|r| match r.owner {
            EquationOwner::Joint(id) => {
                let name = sys.joints.get(&id).map(|j| j.name.as_str()).unwrap_or("?");
                format!("joint {} ({name}) row {}", id.0, r.row)
            }
            EquationOwner::Coupler(id) => {
                let name = sys.couplers.get(&id).map(|c| c.name.as_str()).unwrap_or("?");
                format!("coupler {} ({name})", id.0)
            }
            EquationOwner::Driver(i) => {
                let joint = sys.drivers.get(i).map(|d| d.joint.0).unwrap_or(0);
                format!("driver #{i} on joint {joint}")
            }
        })
        .collect();
    names.join(", ")
}

fn check_state(sys: &System, state: &KinematicState) -> Result<(), MbdError> {
    if !state.order.iter().eq(sys.bodies.keys()) {
        return Err(MbdError::new(
            codes::STATE_MISMATCH,
            format!(
                "state holds {} bodies, the system {}; the body ids must match one to one",
                state.order.len(),
                sys.bodies.len()
            ),
            "seed from a state solved for this same system (or KinematicState::from_system)",
        ));
    }
    Ok(())
}

/// Solve the position initial condition at a driver time: the per-body poses
/// satisfying every joint equation and every driver value, starting from the
/// system's authored poses.
///
/// The solve is a damped Newton-Raphson on the shared `rcad-gcs` numerics
/// core. Failure to converge is data ([`codes::KINEMATICS_NOT_CONVERGED`],
/// naming the conflicting equations when the linearization shows any),
/// never a panic. Free degrees of freedom are allowed (an assembly solve).
pub fn position_ic(sys: &System, time: f64) -> Result<KinematicState, MbdError> {
    sys.validate()?;
    solve_position(sys, time, KinematicState::from_system(sys))
}

/// [`position_ic`] from a caller-supplied seed (the previous instant of a
/// sweep, or a dragged configuration). Grounded bodies are reset to their
/// authored poses.
pub fn position_ic_from(sys: &System, time: f64, seed: &KinematicState) -> Result<KinematicState, MbdError> {
    sys.validate()?;
    check_state(sys, seed)?;
    let mut state = seed.clone();
    for (id, body) in &sys.bodies {
        if body.grounded {
            state.set_body_pose(*id, body.translation, body.rotation);
        }
    }
    solve_position(sys, time, state)
}

fn solve_position(sys: &System, time: f64, mut state: KinematicState) -> Result<KinematicState, MbdError> {
    let problem = Problem::new(sys, time, &state.angles);
    let fixed = problem.fixed_mask();
    let n_eq = problem.n_bodies() + problem.rows.len();
    let GcsSolveResult { converged, residual, iterations } =
        newton_solve(&mut state.params, &fixed, n_eq, |p, out| problem.eval(p, out));
    if !converged {
        let diagnosis = problem.diagnose(&state.params);
        let conflicts = if diagnosis.conflicting.is_empty() {
            String::new()
        } else {
            format!("; conflicting: {}", describe(sys, &diagnosis.conflicting))
        };
        return Err(MbdError::new(
            codes::KINEMATICS_NOT_CONVERGED,
            format!(
                "position IC at t = {time} did not converge: rms residual {residual:.3e} after {iterations} iterations{conflicts}"
            ),
            "check the joint layout for over-constraint, or seed the bodies closer to a valid configuration",
        ));
    }
    state.normalize_quaternions();
    state.angles = problem.track_angles(&state.params);
    Ok(state)
}

/// The rank analysis of the constraint equations at `state`: degrees of
/// freedom, mobility, redundant and conflicting equations.
pub fn diagnose(sys: &System, state: &KinematicState, time: f64) -> Result<KinematicDiagnosis, MbdError> {
    sys.validate()?;
    check_state(sys, state)?;
    Ok(Problem::new(sys, time, &state.angles).diagnose(&state.params))
}

/// Solve one instant on all three levels: position (from `seed`, or the
/// authored poses), then velocity `J·q̇ = ṡ` and acceleration `J·q̈ = s̈ − γ`
/// (`s` the driver profiles, `γ` the velocity-quadratic term).
///
/// The velocity and acceleration levels need a fully prescribed mechanism:
/// free degrees of freedom left after the drivers are
/// [`codes::UNDERDETERMINED_MOTION`]; inconsistent levels (a toggle
/// position) are [`codes::KINEMATIC_SINGULAR`]. Redundant equations are
/// skipped and listed in the frame.
pub fn solve_frame(sys: &System, time: f64, seed: Option<&KinematicState>) -> Result<KinematicFrame, MbdError> {
    let state = match seed {
        Some(seed) => position_ic_from(sys, time, seed)?,
        None => position_ic(sys, time)?,
    };
    let problem = Problem::new(sys, time, &state.angles);
    let j = problem.jacobian(&state.params);
    let factor = RowFactor::new(&j, problem.n_cols);
    if factor.rank() < problem.n_cols {
        let free = problem.n_cols - factor.rank();
        return Err(MbdError::new(
            codes::UNDERDETERMINED_MOTION,
            format!("the mechanism keeps {free} free degree(s) of freedom at t = {time} after its drivers"),
            "a kinematic analysis needs every degree of freedom prescribed: add drivers (or run dynamics, M2)",
        ));
    }
    let samples: Vec<_> = problem
        .rows
        .iter()
        .map(|r| match r.owner {
            EquationOwner::Driver(i) => Some(sys.drivers[i].sample(time)),
            EquationOwner::Joint(_) | EquationOwner::Coupler(_) => None,
        })
        .collect();
    let b_vel: Vec<f64> = samples.iter().map(|s| s.map_or(0.0, |s| s.velocity)).collect();
    let velocity = problem.solve_level(&factor, &b_vel, "velocity")?;
    let jets = problem.jets(&state.params, &velocity);
    let b_acc: Vec<f64> = samples
        .iter()
        .zip(&jets)
        .map(|(s, jet)| s.map_or(0.0, |s| s.acceleration) - jet.dd)
        .collect();
    let acceleration = problem.solve_level(&factor, &b_acc, "acceleration")?;
    let mut redundant: Vec<usize> = factor.dependent().to_vec();
    redundant.sort_unstable();
    Ok(KinematicFrame {
        time,
        velocity: state.order.iter().copied().zip(velocity).collect(),
        acceleration: state.order.iter().copied().zip(acceleration).collect(),
        redundant: problem.refs(&redundant),
        state,
    })
}

/// Run the kinematic analysis over a time grid (finite, non-decreasing).
/// Each instant is seeded from the previous one advanced by its velocity
/// and acceleration, so the solution stays on one assembly branch and a
/// motor's angle runs on continuously. An error names the instant it
/// stopped at.
pub fn kinematic_sweep(sys: &System, times: &[f64]) -> Result<Vec<KinematicFrame>, MbdError> {
    if times.iter().any(|t| !t.is_finite()) || times.windows(2).any(|w| w[1] < w[0]) {
        return Err(MbdError::new(
            codes::BAD_TIME_GRID,
            "the time grid must be finite and non-decreasing",
            "pass output times in ascending order",
        ));
    }
    let mut frames: Vec<KinematicFrame> = Vec::with_capacity(times.len());
    for &time in times {
        let seed = frames.last().map(|prev| predict(prev, time));
        let frame = solve_frame(sys, time, seed.as_ref())?;
        frames.push(frame);
    }
    Ok(frames)
}

/// Second-order Taylor predictor from a solved frame to `time`.
fn predict(prev: &KinematicFrame, time: f64) -> KinematicState {
    let dt = time - prev.time;
    let mut state = prev.state.clone();
    for id in prev.state.order.clone() {
        let (t, r) = prev.state.body_pose(id).expect("own body");
        let v = prev.velocity[&id];
        let a = prev.acceleration[&id];
        let t = t + v.linear * dt + 0.5 * a.linear * dt * dt;
        let turn = v.angular * dt + 0.5 * a.angular * dt * dt;
        let r = DQuat::from_scaled_axis(turn) * r;
        state.set_body_pose(id, t, r);
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Body;
    use crate::force::ForceElement;
    use crate::joint::{Driver, DriverProfile, Joint, JointKind};
    use crate::{BodyId, ForceId, JointId, Marker, MarkerId};
    use glam::DMat3;

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

    /// A grounded base plus one link on a joint at the origin, with a
    /// constant-rate driver on `slot`.
    fn one_joint(kind: JointKind, slot: DriverSlot, start: f64, rate: f64) -> System {
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
            kind,
            marker1: MarkerId(1),
            marker2: MarkerId(2),
            suppressed: false,
        })
        .unwrap();
        sys.add_driver(Driver {
            joint: JointId(1),
            slot,
            profile: DriverProfile::Constant { start, rate },
        });
        sys
    }

    fn hinge(driver_angle: f64) -> System {
        one_joint(JointKind::Revolute, DriverSlot::Rotation, driver_angle, 0.0)
    }

    fn hinge_angle(state: &KinematicState) -> f64 {
        let (_, r) = state.body_pose(BodyId(2)).unwrap();
        2.0 * r.z.atan2(r.w)
    }

    /// The residual layout matches the equation count, and a valid
    /// configuration evaluates to zero residuals.
    #[test]
    fn residual_layout_is_consistent() {
        let sys = hinge(0.0);
        assert_eq!(equation_count(&sys), 2 + 5 + 1);
        let problem = Problem::new(&sys, 0.0, &BTreeMap::new());
        let mut out = vec![0.0; equation_count(&sys)];
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
            let theta = hinge_angle(&state);
            assert!((theta - angle).abs() < 1e-9, "driver angle {angle} held, got {theta}");
        }
    }

    /// A driver angle beyond ±π (a motor past half a turn) converges to the
    /// same pose modulo a full turn.
    #[test]
    fn driver_angle_past_half_a_turn_converges() {
        for angle in [4.0, -4.0, 7.5, 20.0] {
            let state = position_ic(&hinge(angle), 0.0).expect("converges");
            let off = wrap_angle(hinge_angle(&state) - angle);
            assert!(off.abs() < 1e-9, "angle {angle}: off by {off}");
        }
    }

    /// A seeded solve starts from the seed (and resets grounded bodies);
    /// a seed from another system is rejected.
    #[test]
    fn seeded_position_ic() {
        let sys = hinge(1.0);
        let solved = position_ic(&sys, 0.0).unwrap();
        let mut moved = solved.clone();
        moved.set_body_pose(BodyId(1), DVec3::X, DQuat::IDENTITY);
        let again = position_ic_from(&sys, 0.0, &moved).unwrap();
        assert_eq!(again.body_pose(BodyId(1)).unwrap().0, DVec3::ZERO);
        assert!((hinge_angle(&again) - 1.0).abs() < 1e-9);
        let mut other = System::new("other");
        other.add_body(grounded_base()).unwrap();
        assert_eq!(
            position_ic_from(&other, 0.0, &solved).unwrap_err().code,
            codes::STATE_MISMATCH
        );
    }

    /// Planar rotation drives (the angle about the common normal); planar
    /// translation and universal rotation have no single slot and are
    /// rejected before any solve.
    #[test]
    fn drivable_slots() {
        let sys = one_joint(JointKind::Planar, DriverSlot::Rotation, 0.7, 0.0);
        let state = position_ic(&sys, 0.0).expect("planar rotation drives");
        assert!((hinge_angle(&state) - 0.7).abs() < 1e-9);
        for (kind, slot) in [
            (JointKind::Planar, DriverSlot::Translation),
            (JointKind::Universal, DriverSlot::Rotation),
            (JointKind::Revolute, DriverSlot::Translation),
        ] {
            let err = position_ic(&one_joint(kind, slot, 0.0, 0.0), 0.0).unwrap_err();
            assert_eq!(err.code, codes::SLOT_UNSUPPORTED, "{kind:?} {slot:?}");
        }
        let mut twice = hinge(0.0);
        twice.add_driver(twice.drivers[0].clone());
        assert_eq!(twice.validate().unwrap_err().code, codes::DUPLICATE_DRIVER);
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

    /// The force vocabulary is untouched by the kinematic slice.
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

    /// The jets' first and second derivatives agree with finite differences
    /// of the residual values along a constant-twist motion — for every joint
    /// kind and both driver coordinates, at a generic configuration.
    #[test]
    fn jets_match_finite_differences() {
        let q1 = DQuat::from_euler(glam::EulerRot::XYZ, 0.3, -0.7, 1.1);
        let q2 = DQuat::from_euler(glam::EulerRot::XYZ, -0.4, 0.2, 2.3);
        let m1 = (DVec3::new(0.1, -0.2, 0.3), DQuat::from_euler(glam::EulerRot::XYZ, 0.5, 0.1, -0.3));
        let m2 = (DVec3::new(-0.3, 0.05, 0.2), DQuat::from_euler(glam::EulerRot::XYZ, -0.2, 0.6, 0.4));
        let body1 = BodyMotion {
            t: DVec3::new(0.2, 0.1, -0.1),
            r: q1,
            v: DVec3::new(0.3, -0.5, 0.2),
            w: DVec3::new(1.1, 0.4, -0.7),
        };
        let body2 = BodyMotion {
            t: DVec3::new(-0.1, 0.4, 0.3),
            r: q2,
            v: DVec3::new(-0.2, 0.6, 0.1),
            w: DVec3::new(-0.3, 0.9, 1.4),
        };
        let at = |b: &BodyMotion, h: f64| BodyMotion {
            t: b.t + b.v * h,
            r: DQuat::from_scaled_axis(b.w * h) * b.r,
            v: DVec3::ZERO,
            w: DVec3::ZERO,
        };
        let rows = |b1: &BodyMotion, b2: &BodyMotion, kind: JointKind| {
            let f1 = b1.frame(m1.0, m1.1);
            let f2 = b2.frame(m2.0, m2.1);
            let mut buf = [Jet::default(); 6];
            let n = joint_rows(kind, &f1, &f2, &mut buf);
            let mut out = buf[..n].to_vec();
            for slot in [DriverSlot::Rotation, DriverSlot::Translation] {
                if let Some(j) = joint_coordinate(JointKind::Cylindrical, slot, &f1, &f2) {
                    out.push(j);
                }
            }
            out
        };
        let h = 1e-4;
        for kind in [
            JointKind::Revolute,
            JointKind::Prismatic,
            JointKind::Cylindrical,
            JointKind::Spherical,
            JointKind::Planar,
            JointKind::Universal,
            JointKind::Fixed,
            JointKind::InLine,
            JointKind::InPlane,
            JointKind::Orientation,
            JointKind::ParallelAxes,
            JointKind::Perpendicular,
        ] {
            let jets = rows(&body1, &body2, kind);
            let ahead = rows(&at(&body1, h), &at(&body2, h), kind);
            let here = rows(&at(&body1, 0.0), &at(&body2, 0.0), kind);
            let behind = rows(&at(&body1, -h), &at(&body2, -h), kind);
            for k in 0..jets.len() {
                assert!((here[k].v - jets[k].v).abs() < 1e-12);
                let d = (ahead[k].v - behind[k].v) / (2.0 * h);
                let dd = (ahead[k].v - 2.0 * here[k].v + behind[k].v) / (h * h);
                assert!((d - jets[k].d).abs() < 1e-6, "{kind:?} row {k}: d {d} vs {}", jets[k].d);
                assert!((dd - jets[k].dd).abs() < 1e-4, "{kind:?} row {k}: dd {dd} vs {}", jets[k].dd);
            }
        }
    }

    /// The hinge's velocity and acceleration levels reproduce the driver
    /// rate and acceleration, and a hinge without its driver is reported as
    /// underdetermined (mobility 1).
    #[test]
    fn hinge_velocity_and_acceleration_levels() {
        let mut sys = one_joint(JointKind::Revolute, DriverSlot::Rotation, 0.0, 0.0);
        sys.drivers[0].profile = DriverProfile::Harmonic {
            offset: 0.2,
            amplitude: 0.5,
            frequency_hz: 0.8,
            phase: 0.1,
        };
        let t = 0.37;
        let frame = solve_frame(&sys, t, None).unwrap();
        let s = sys.drivers[0].sample(t);
        let v = frame.velocity[&BodyId(2)];
        let a = frame.acceleration[&BodyId(2)];
        assert!((v.angular - DVec3::Z * s.velocity).length() < 1e-9, "{v:?}");
        assert!(v.linear.length() < 1e-9);
        assert!((a.angular - DVec3::Z * s.acceleration).length() < 1e-9, "{a:?}");
        assert!(frame.redundant.is_empty());

        let mut free = sys.clone();
        free.drivers.clear();
        let state = position_ic(&free, 0.0).unwrap();
        let diag = diagnose(&free, &state, 0.0).unwrap();
        assert_eq!((diag.coordinates, diag.rank, diag.dof, diag.mobility), (6, 5, 1, 1));
        assert_eq!(
            solve_frame(&free, 0.0, None).unwrap_err().code,
            codes::UNDERDETERMINED_MOTION
        );
    }
}
