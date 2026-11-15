//! Forward dynamics (M2): the equations of motion in maximal coordinates,
//! the dynamic initial condition, static equilibrium, and the energy and
//! momentum audits.
//!
//! **Equations of motion.** Per free body, in the tangent coordinates of
//! [`Twist`] (world-frame velocity of the body-frame origin and angular
//! velocity, roadmap §3):
//!
//! ```text
//! M(q)·v̇ + b(q, v) + Jᵀ·λ = Q(q, v, t)     six rows per free body
//! J·v̇ + γ(q, v)          = s̈(t)           one row per constraint equation
//! ```
//!
//! The body origin is generally not the center of mass, so with the world
//! arm `c = R·com` and the world inertia about the center of mass
//! `J_c = R·I_c·Rᵀ` the mass matrix about the origin is
//!
//! ```text
//! M = | m·1      −m·[c]×           |      b = | m·ω×(ω×c)                  |
//!     | m·[c]×   J_c − m·[c]×[c]×  |          | c×(m·ω×(ω×c)) + ω×(J_c·ω)  |
//! ```
//!
//! (`M·v` is the linear momentum and the angular momentum about the moving
//! origin; `b` the velocity-quadratic remainder). `Q` stacks the external
//! force and the external torque about the body origin; the power of a
//! generalized force is `F·v + τ·ω`, so `−Jᵀλ` is exactly the force and
//! torque (about the origin) the constraints apply to each body. `J`, `γ`
//! and `s̈` are the kinematic levels' Jacobian, quadratic term and driver
//! accelerations: dynamics and kinematics share one constraint formulation.
//!
//! **Linear solves.** The instantaneous problems are saddle-point systems
//! `[M Jᵀ; J 0]`, solved by the sparse saddle-point LDLᵀ (`sparse` module,
//! O(bodies) for chains and trees) with the dense rank-revealing QR as the
//! fallback: redundant constraint equations are consistent dependent rows —
//! their multipliers are not unique; the sparse path removes them
//! (multiplier 0, the Adams / OndselSolver convention), the dense path
//! reports the minimum-norm split, and either way the equations are listed;
//! inconsistent rows are a singular configuration and fail as data.

use std::collections::BTreeMap;

use glam::{DMat3, DQuat, DVec3};

use crate::body::{Body, BodyId, MarkerId};
use crate::error::{codes, MbdError};
use crate::force::{Argument, ForceElement, Function};
use crate::friction::{
    friction_forces, friction_report, has_friction, reads_reactions, FrictionLoad, Loads,
};
use crate::driver::DriverSlot;
use crate::joint::JointId;
use crate::kinematics::{
    check_state, describe, joint_slots, position_ic, position_ic_from, read_pose, EquationRef,
    KinematicState, Problem, Twist, BODY_TWIST,
};
use crate::linalg::{inf_norm, transpose, RowFactor};
use crate::sparse::{dense_saddle, triplets_from_dense, SaddleFactor, SparseRows, Triplets};
use crate::system::System;

/// Below this length (m) a spring's line of action is undefined.
const LENGTH_EPS: f64 = 1.0e-12;

/// Static equilibrium: Newton iteration cap.
const STATIC_MAX_ITER: usize = 200;

/// Static equilibrium: force-balance tolerance relative to the load scale.
const STATIC_FORCE_TOL: f64 = 1.0e-9;

/// Static equilibrium: constraint residual tolerance (m / rad).
const STATIC_CONSTRAINT_TOL: f64 = 1.0e-10;

/// Static equilibrium: central-difference step of the tangent stiffness.
const STIFFNESS_STEP: f64 = 1.0e-6;

/// Static equilibrium: largest rotation (rad) one Newton step may apply to a
/// body.
const MAX_STEP_ROTATION: f64 = 0.5;

/// Load-dependent friction: how many times an instantaneous solve may
/// re-solve for the fixed point of the multipliers.
const FRICTION_MAX_ITER: usize = 60;

/// Load-dependent friction: the relative multiplier change an
/// instantaneous solve settles for.
const FRICTION_TOL: f64 = 1.0e-12;

/// A body's mass properties about its frame origin, in world axes, at one
/// orientation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BodyInertia {
    mass: f64,
    /// World arm from the body origin to the center of mass (`R·com`).
    arm: DVec3,
    /// Inertia about the center of mass, world axes (`R·I_c·Rᵀ`).
    inertia: DMat3,
}

impl BodyInertia {
    fn at(body: &Body, rotation: DQuat) -> Self {
        let r = DMat3::from_quat(rotation);
        Self {
            mass: body.mass_kg,
            arm: rotation * body.com,
            inertia: r * body.inertia_com * r.transpose(),
        }
    }

    /// `M·x` for a tangent vector `x`; for a velocity this is the momentum
    /// (linear, and angular about the body origin).
    pub(crate) fn apply(&self, x: Twist) -> Twist {
        let linear = self.mass * (x.linear + x.angular.cross(self.arm));
        Twist {
            linear,
            angular: self.arm.cross(linear) + self.inertia * x.angular,
        }
    }

    /// The velocity-quadratic term `b(ω)`.
    pub(crate) fn bias(&self, w: DVec3) -> Twist {
        let linear = self.mass * w.cross(w.cross(self.arm));
        Twist {
            linear,
            angular: self.arm.cross(linear) + w.cross(self.inertia * w),
        }
    }

    /// The 6×6 mass matrix (rows and columns: linear xyz, angular xyz).
    fn matrix(&self) -> [[f64; BODY_TWIST]; BODY_TWIST] {
        let cols: [[f64; BODY_TWIST]; BODY_TWIST] =
            std::array::from_fn(|k| self.apply(Twist::unit(k)).to_array());
        std::array::from_fn(|row| std::array::from_fn(|col| cols[col][row]))
    }
}

/// The 6×6 mass matrix of `body` at world orientation `rotation`, about the
/// body-frame origin, in the [`Twist`] coordinates (linear xyz, angular
/// xyz): `T = ½·vᵀ·M·v` is the body's kinetic energy.
pub fn mass_matrix(body: &Body, rotation: DQuat) -> [[f64; BODY_TWIST]; BODY_TWIST] {
    BodyInertia::at(body, rotation).matrix()
}

/// One dynamic instant: a consistent pose and velocity with the
/// acceleration and constraint multipliers the equations of motion give.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicFrame {
    pub time: f64,
    pub state: KinematicState,
    /// Per-body velocity (grounded bodies carry zero).
    pub velocity: BTreeMap<BodyId, Twist>,
    /// Per-body acceleration (grounded bodies carry zero).
    pub acceleration: BTreeMap<BodyId, Twist>,
    /// The Lagrange multiplier of every constraint equation, in equation
    /// order (joints, couplers, drivers). `−Jᵀλ` is the force and torque
    /// the constraints apply to the bodies; a driver's multiplier is the
    /// actuator effort it takes to impose its motion.
    pub multipliers: Vec<(EquationRef, f64)>,
    /// Redundant equations (dependent but consistent): the multipliers of
    /// the equations they depend on are not unique — one valid split is
    /// reported (see the module notes; [`crate::reactions`] lists the
    /// indeterminate directions).
    pub redundant: Vec<EquationRef>,
}

/// A solved static equilibrium.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticEquilibrium {
    pub time: f64,
    pub state: KinematicState,
    /// The constraint multipliers holding the loads (see
    /// [`DynamicFrame::multipliers`]); a driver's is the effort that holds
    /// its joint at the prescribed value — the "unknown force" of
    /// Inventor / the static motor load of Creo.
    pub multipliers: Vec<(EquationRef, f64)>,
    /// Redundant equations (non-unique multipliers; the minimum-norm split
    /// is reported).
    pub redundant: Vec<EquationRef>,
    /// Newton iterations taken.
    pub iterations: usize,
    /// The joint friction holding the equilibrium, in force-element id
    /// order (empty without friction elements). At rest the friction is
    /// the stiction one: the load that holds the mechanism within
    /// `max_stiction_deformation` of the pose the solve started from (see
    /// [`crate::ForceElement::JointFriction`]), which is why it is
    /// reported with the equilibrium rather than recomputed from the state.
    pub friction: Vec<FrictionLoad>,
}

/// Energy audit of a state: kinetic, gravitational and elastic (spring)
/// potential. Potentials are zero at the world origin / free length.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Energy {
    pub kinetic: f64,
    pub gravity: f64,
    pub elastic: f64,
}

impl Energy {
    pub fn total(&self) -> f64 {
        self.kinetic + self.gravity + self.elastic
    }
}

/// Momentum audit: total linear momentum and angular momentum about the
/// world origin.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Momentum {
    pub linear: DVec3,
    pub angular: DVec3,
}

/// Per-slot twists from a body-keyed map (missing and grounded: zero).
pub(crate) fn slot_twists(sys: &System, map: &BTreeMap<BodyId, Twist>) -> Vec<Twist> {
    sys.bodies
        .iter()
        .map(|(id, body)| {
            if body.grounded {
                Twist::default()
            } else {
                map.get(id).copied().unwrap_or_default()
            }
        })
        .collect()
}

/// Free-body tangent vector from per-slot twists.
pub(crate) fn flatten(problem: &Problem, twists: &[Twist]) -> Vec<f64> {
    let mut x = vec![0.0; problem.n_cols];
    for (slot, col) in problem.col_of.iter().enumerate() {
        if let Some(base) = col {
            x[*base..base + BODY_TWIST].copy_from_slice(&twists[slot].to_array());
        }
    }
    x
}

/// Per-slot inertia (`None` for grounded slots).
pub(crate) fn inertias(sys: &System, params: &[f64]) -> Vec<Option<BodyInertia>> {
    sys.bodies
        .values()
        .enumerate()
        .map(|(slot, body)| {
            (!body.grounded).then(|| BodyInertia::at(body, read_pose(params, slot).1))
        })
        .collect()
}

/// The generalized applied forces `Q` per slot (force, torque about the body
/// origin): gravity and the force elements, at pose `params` and velocity
/// `vel`. Joint friction also reads the constraint multipliers of `loads`
/// (its load follows the joint's reaction); multipliers that do not cover
/// the constraint rows read as zero, which leaves friction at its preload.
pub(crate) fn applied_forces(
    problem: &Problem,
    params: &[f64],
    vel: &[Twist],
    loads: Loads,
) -> Result<Vec<Twist>, MbdError> {
    let sys = problem.sys;
    let mut q = vec![Twist::default(); problem.n_bodies()];
    if let Some(g) = sys.gravity {
        for (slot, body) in sys.bodies.values().enumerate() {
            let r = read_pose(params, slot).1;
            let f = body.mass_kg * g;
            q[slot].linear += f;
            q[slot].angular += (r * body.com).cross(f);
        }
    }
    let point_force = |q: &mut [Twist], marker, point: DVec3, f: DVec3| {
        let slot = problem.slot_of[&sys.markers[&marker].body];
        let origin = read_pose(params, slot).0;
        q[slot].linear += f;
        q[slot].angular += (point - origin).cross(f);
    };
    for element in sys.forces.values() {
        match element {
            ForceElement::SpringDamper {
                id,
                name,
                marker1,
                marker2,
                stiffness,
                damping,
                free_length,
                preload,
            } => {
                let p1 = problem.frame(params, vel, *marker1).p;
                let p2 = problem.frame(params, vel, *marker2).p;
                let d = p2.v - p1.v;
                let rate = p2.d - p1.d;
                let l = d.length();
                // Tension along marker 1 → marker 2 (pulls marker 1 toward 2).
                let tension = if l > LENGTH_EPS {
                    let u = d / l;
                    (stiffness * (l - free_length) + damping * u.dot(rate) - preload) * u
                } else if free_length.abs() <= LENGTH_EPS && *preload == 0.0 {
                    // A zero-length spring is isotropic: f = k·d + c·ḋ.
                    *stiffness * d + *damping * rate
                } else {
                    return Err(MbdError::new(
                        codes::DEGENERATE_FORCE,
                        format!(
                            "spring-damper {} ({name}) has coincident markers but free length {free_length}: its line of action is undefined",
                            id.0
                        ),
                        "separate the markers, or use a zero free length for a coincident (bushing-like) spring",
                    ));
                };
                point_force(&mut q, *marker1, p1.v, tension);
                point_force(&mut q, *marker2, p2.v, -tension);
            }
            ForceElement::AppliedForce { marker, force, .. } => {
                let p = problem.frame(params, vel, *marker).p.v;
                point_force(&mut q, *marker, p, *force);
            }
            ForceElement::AppliedTorque { marker, torque, .. } => {
                let slot = problem.slot_of[&sys.markers[marker].body];
                q[slot].angular += *torque;
            }
            ForceElement::Force {
                marker,
                reference,
                direction,
                magnitude,
                ..
            } => {
                let f = evaluate(problem, params, vel, magnitude)?;
                let dir = world_direction(problem, params, vel, *reference, *direction);
                let p = problem.frame(params, vel, *marker).p.v;
                point_force(&mut q, *marker, p, f * dir);
            }
            ForceElement::Torque {
                marker,
                reference,
                direction,
                magnitude,
                ..
            } => {
                let t = evaluate(problem, params, vel, magnitude)?;
                let dir = world_direction(problem, params, vel, *reference, *direction);
                let slot = problem.slot_of[&sys.markers[marker].body];
                q[slot].angular += t * dir;
            }
            ForceElement::LineForce {
                id,
                name,
                marker1,
                marker2,
                tension,
            } => {
                let t = evaluate(problem, params, vel, tension)?;
                let p1 = problem.frame(params, vel, *marker1).p.v;
                let p2 = problem.frame(params, vel, *marker2).p.v;
                let d = p2 - p1;
                let l = d.length();
                if l <= LENGTH_EPS {
                    if t != 0.0 {
                        return Err(MbdError::new(
                            codes::DEGENERATE_FORCE,
                            format!(
                                "line force {} ({name}) has coincident markers and tension {t}: its line of action is undefined",
                                id.0
                            ),
                            "separate the markers",
                        ));
                    }
                    continue;
                }
                let pull = t * d / l;
                point_force(&mut q, *marker1, p1, pull);
                point_force(&mut q, *marker2, p2, -pull);
            }
            ForceElement::JointSpringDamper {
                joint,
                slot,
                stiffness,
                damping,
                free_value,
                preload,
                ..
            } => {
                if sys.joints[joint].suppressed {
                    continue;
                }
                let c = problem.coordinate(params, vel, *joint, *slot);
                let f = -stiffness * (c.v - free_value) - damping * c.d + preload;
                for (s, grad) in coordinate_gradient(problem, params, *joint, *slot) {
                    q[s].linear += f * grad.linear;
                    q[s].angular += f * grad.angular;
                }
            }
            ForceElement::Bushing {
                marker1,
                marker2,
                stiffness,
                damping,
                torsion_stiffness,
                torsion_damping,
                ..
            } => {
                let b = bushing_state(problem, params, vel, *marker1, *marker2);
                let force_i = -(*stiffness * b.d + *damping * b.d_rate);
                let torque_i = -(*torsion_stiffness * b.theta + *torsion_damping * b.w_rel);
                let force = b.axes * force_i;
                let torque = b.axes * torque_i;
                // On J at its origin; the reaction on I at the same point.
                point_force(&mut q, *marker2, b.p_j, force);
                point_force(&mut q, *marker1, b.p_j, -force);
                let (sj, si) = (
                    problem.slot_of[&sys.markers[marker2].body],
                    problem.slot_of[&sys.markers[marker1].body],
                );
                q[sj].angular += torque;
                q[si].angular -= torque;
            }
            // Friction reads the multipliers, so it is applied after the
            // state-only elements ([`friction_forces`]).
            ForceElement::JointFriction { .. } => {}
        }
    }
    if has_friction(sys) {
        friction_forces(problem, params, vel, loads, &mut q);
    }
    Ok(q)
}

/// `∂q/∂(v, ω)` of a joint coordinate for each free body of the joint —
/// the direction along which a generalized force on the coordinate acts.
pub(crate) fn coordinate_gradient(
    problem: &Problem,
    params: &[f64],
    joint: JointId,
    slot: DriverSlot,
) -> Vec<(usize, Twist)> {
    let mut motion = vec![Twist::default(); problem.n_bodies()];
    let mut out = Vec::new();
    for s in joint_slots(problem.sys, &problem.slot_of, joint) {
        if problem.col_of[s].is_none() {
            continue;
        }
        let mut g = [0.0; BODY_TWIST];
        for (dir, d) in g.iter_mut().enumerate() {
            motion[s] = Twist::unit(dir);
            *d = problem.coordinate(params, &motion, joint, slot).d;
        }
        motion[s] = Twist::default();
        out.push((s, Twist::from_slice(&g)));
    }
    out
}

/// A bushing's deflection state, in marker I's axes.
struct BushingState {
    /// Marker I's axes (columns), world frame.
    axes: DMat3,
    /// Displacement of J's origin from I's, and its rate as seen from I.
    d: DVec3,
    d_rate: DVec3,
    /// Rotation vector of J relative to I, and the relative angular velocity.
    theta: DVec3,
    w_rel: DVec3,
    /// J's origin, world frame.
    p_j: DVec3,
}

fn bushing_state(
    problem: &Problem,
    params: &[f64],
    vel: &[Twist],
    m1: MarkerId,
    m2: MarkerId,
) -> BushingState {
    let sys = problem.sys;
    let (fi, fj) = (
        problem.frame(params, vel, m1),
        problem.frame(params, vel, m2),
    );
    let axes = DMat3::from_cols(fi.x.v, fi.y.v, fi.z.v);
    let orientation = |m: MarkerId| {
        let marker = &sys.markers[&m];
        let slot = problem.slot_of[&marker.body];
        (slot, read_pose(params, slot).1 * marker.rotation)
    };
    let ((si, ri), (sj, rj)) = (orientation(m1), orientation(m2));
    let mut rel = ri.inverse() * rj;
    if rel.w < 0.0 {
        rel = -rel;
    }
    let dw = fj.p.v - fi.p.v;
    let (wi, wj) = (vel[si].angular, vel[sj].angular);
    let back = axes.transpose();
    BushingState {
        axes,
        d: back * dw,
        d_rate: back * (fj.p.d - fi.p.d - wi.cross(dw)),
        theta: rel.to_scaled_axis(),
        w_rel: back * (wj - wi),
        p_j: fj.p.v,
    }
}

/// A unit direction given in a reference marker's frame (`None`: world),
/// in world coordinates.
fn world_direction(
    problem: &Problem,
    params: &[f64],
    vel: &[Twist],
    reference: Option<MarkerId>,
    direction: DVec3,
) -> DVec3 {
    let unit = direction.normalize();
    let Some(r) = reference else { return unit };
    let f = problem.frame(params, vel, r);
    unit.x * f.x.v + unit.y * f.y.v + unit.z * f.z.v
}

/// A force function at the state `(params, vel)` and the problem's time.
fn evaluate(
    problem: &Problem,
    params: &[f64],
    vel: &[Twist],
    function: &Function,
) -> Result<f64, MbdError> {
    let x = match function.argument {
        Argument::Time => problem.time,
        Argument::JointCoordinate { joint, slot } => problem.coordinate(params, vel, joint, slot).v,
        Argument::JointRate { joint, slot } => problem.coordinate(params, vel, joint, slot).d,
        Argument::Distance { marker1, marker2 } | Argument::DistanceRate { marker1, marker2 } => {
            let p1 = problem.frame(params, vel, marker1).p;
            let p2 = problem.frame(params, vel, marker2).p;
            let d = p2.v - p1.v;
            let l = d.length();
            if matches!(function.argument, Argument::Distance { .. }) {
                l
            } else if l > LENGTH_EPS {
                d.dot(p2.d - p1.d) / l
            } else {
                0.0
            }
        }
    };
    Ok(function.at(x))
}

/// The power of every applied force (gravity and the force elements) at a
/// state, `Σ Q·v` (W). With scleronomic constraints the constraint forces do
/// no work, so `dT/dt` equals it — the work-energy audit.
pub fn power(
    sys: &System,
    time: f64,
    state: &KinematicState,
    velocity: &BTreeMap<BodyId, Twist>,
) -> Result<f64, MbdError> {
    sys.validate()?;
    check_state(sys, state)?;
    let problem = Problem::new(sys, time, &state.angles);
    let vel = slot_twists(sys, velocity);
    // Friction's load follows the reactions, so the audit needs them.
    let lambda = if has_friction(sys) {
        accelerations(&problem, &state.params, &vel)?.lambda
    } else {
        Vec::new()
    };
    let q = flatten(
        &problem,
        &applied_forces(&problem, &state.params, &vel, Loads::dynamic(&lambda))?,
    );
    Ok(q.iter()
        .zip(flatten(&problem, &vel))
        .map(|(a, b)| a * b)
        .sum())
}

/// A saddle-point solution with the redundant constraint rows (ascending).
struct SaddleSolution {
    x: Vec<f64>,
    mu: Vec<f64>,
    redundant: Vec<usize>,
}

/// Solve the saddle-point system `[A Jᵀ; J 0]·[x; μ] = [top; bottom]` (`A`
/// as `n × n` triplets): the sparse LDLᵀ ([`SaddleFactor`]) first; the dense
/// rank-revealing QR when the sparse path declines (an inconsistent or
/// singular system — the dense path then names the offending equations).
fn solve_saddle(
    problem: &Problem,
    a: &[(usize, usize, f64)],
    j: &SparseRows,
    top: &[f64],
    bottom: &[f64],
    what: &str,
) -> Result<SaddleSolution, MbdError> {
    if let Some(factor) = SaddleFactor::new(problem.n_cols, a, j) {
        if let Some((x, mu)) = factor.solve(top, bottom) {
            return Ok(SaddleSolution {
                x,
                mu,
                redundant: factor.redundant().to_vec(),
            });
        }
    }
    let n = problem.n_cols;
    let rows = dense_saddle(n, a, j);
    let rhs: Vec<f64> = top.iter().chain(bottom).copied().collect();
    let (x, conflicts) = RowFactor::new(&rows, rows.len()).solve(&rhs);
    if !conflicts.is_empty() {
        let constraint: Vec<usize> = conflicts
            .iter()
            .filter(|&&k| k >= n)
            .map(|&k| k - n)
            .collect();
        let detail = if constraint.is_empty() {
            format!(
                "{what} at t = {}: the loads have a component along a direction nothing resists (no mass, stiffness or constraint)",
                problem.time
            )
        } else {
            format!(
                "{what} at t = {}: inconsistent equations: {}",
                problem.time,
                describe(problem.sys, &problem.refs(&constraint))
            )
        };
        return Err(MbdError::new(
            codes::DYNAMICS_SINGULAR,
            detail,
            "check for a lock-up/toggle position, drivers over-prescribing a degree of freedom, or a free body without inertia",
        ));
    }
    let mut x = x;
    let mu = x.split_off(n);
    Ok(SaddleSolution {
        x,
        mu,
        redundant: RowFactor::new(&j.to_dense(), n).dependent_sorted(),
    })
}

/// The block-diagonal mass matrix as triplets.
fn mass_triplets(problem: &Problem, inertia: &[Option<BodyInertia>]) -> Triplets {
    let mut t = Vec::with_capacity(problem.n_cols * BODY_TWIST);
    for (slot, col) in problem.col_of.iter().enumerate() {
        let (Some(b), Some(inertia)) = (*col, inertia[slot]) else {
            continue;
        };
        for (i, mrow) in inertia.matrix().iter().enumerate() {
            for (k, v) in mrow.iter().enumerate() {
                if *v != 0.0 {
                    t.push((b + i, b + k, *v));
                }
            }
        }
    }
    t
}

/// The forward-dynamics solve: per-slot accelerations, the multipliers and
/// the redundant rows.
pub(crate) struct Accelerations {
    pub(crate) acc: Vec<Twist>,
    pub(crate) lambda: Vec<f64>,
    /// Redundant constraint rows (ascending).
    pub(crate) redundant: Vec<usize>,
}

/// The instantaneous forward-dynamics solve at a consistent pose and
/// velocity (slot-indexed): acceleration and multipliers.
pub(crate) fn accelerations(
    problem: &Problem,
    params: &[f64],
    vel: &[Twist],
) -> Result<Accelerations, MbdError> {
    let inertia = inertias(problem.sys, params);
    let mass = mass_triplets(problem, &inertia);
    let j = problem.jacobian_sparse(params);
    let bottom = problem.acceleration_rhs(params, vel);
    let solve = |lambda: &[f64]| -> Result<SaddleSolution, MbdError> {
        let q = applied_forces(problem, params, vel, Loads::dynamic(lambda))?;
        let mut top = vec![0.0; problem.n_cols];
        for (slot, col) in problem.col_of.iter().enumerate() {
            let (Some(b), Some(i)) = (*col, inertia[slot]) else {
                continue;
            };
            let bias = i.bias(vel[slot].angular);
            let f = Twist {
                linear: q[slot].linear - bias.linear,
                angular: q[slot].angular - bias.angular,
            };
            top[b..b + BODY_TWIST].copy_from_slice(&f.to_array());
        }
        solve_saddle(problem, &mass, &j, &top, &bottom, "forward dynamics")
    };
    let done = |solution: SaddleSolution| Accelerations {
        acc: problem.twists(&solution.x),
        lambda: solution.mu,
        redundant: solution.redundant,
    };
    // Load-dependent friction makes `Q` depend on the multipliers: the
    // saddle point is then the fixed point of `λ ↦ solve(Q(λ))`, reached
    // by repeating the solve (the map contracts with the friction
    // coefficient times its arm, so a handful of solves suffice at
    // physical parameters).
    let mut lambda = Vec::new();
    if !reads_reactions(problem.sys) {
        return Ok(done(solve(&lambda)?));
    }
    lambda = vec![0.0; problem.rows.len()];
    let mut change = f64::INFINITY;
    for _ in 0..FRICTION_MAX_ITER {
        let solution = solve(&lambda)?;
        change = solution
            .mu
            .iter()
            .zip(&lambda)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        lambda.copy_from_slice(&solution.mu);
        if change <= FRICTION_TOL * inf_norm(&lambda).max(1.0) {
            return Ok(done(solution));
        }
    }
    Err(MbdError::new(
        codes::FRICTION_NOT_CONVERGED,
        format!(
            "the friction load and the reaction it reads did not settle at t = {} after {FRICTION_MAX_ITER} solves (last multiplier change {change:.3e})",
            problem.time,
        ),
        "lower the friction coefficient or the friction arms, or check that the joint can carry the load its friction creates",
    ))
}

/// Forward dynamics at one instant: given a pose and velocity satisfying the
/// constraints (a [`dynamic_ic`] or integrator state), the accelerations and
/// constraint multipliers the equations of motion give. A fully driven
/// mechanism gets its kinematic accelerations back and the multipliers are
/// the inverse-dynamics reactions and actuator efforts.
pub fn forward_dynamics(
    sys: &System,
    time: f64,
    state: &KinematicState,
    velocity: &BTreeMap<BodyId, Twist>,
) -> Result<DynamicFrame, MbdError> {
    sys.validate()?;
    check_state(sys, state)?;
    let problem = Problem::new(sys, time, &state.angles);
    dynamic_frame(&problem, state, slot_twists(sys, velocity))
}

/// Project authored body velocities onto the velocity constraints at a
/// solved pose: the velocity `v` closest to `authored` in the kinetic-energy
/// norm (`min ½(v − v₀)ᵀM(v − v₀)` subject to `J·v = ṡ`). Physically: the
/// constraints act as an impulse, so momentum along every free direction of
/// the mechanism is kept — a pendulum keeps the authored angular momentum
/// about its hinge axis.
pub fn velocity_ic(
    sys: &System,
    time: f64,
    state: &KinematicState,
    authored: &BTreeMap<BodyId, Twist>,
) -> Result<BTreeMap<BodyId, Twist>, MbdError> {
    sys.validate()?;
    check_state(sys, state)?;
    let problem = Problem::new(sys, time, &state.angles);
    let v0 = slot_twists(sys, authored);
    let v = project_velocity(&problem, &state.params, &v0, "velocity initial condition")?;
    Ok(state.by_body(v))
}

/// The kinetic-energy-norm projection of per-slot velocities onto
/// `J·v = ṡ` at pose `params` (see [`velocity_ic`]).
pub(crate) fn project_velocity(
    problem: &Problem,
    params: &[f64],
    v0: &[Twist],
    what: &str,
) -> Result<Vec<Twist>, MbdError> {
    let inertia = inertias(problem.sys, params);
    let momentum: Vec<Twist> = v0
        .iter()
        .zip(&inertia)
        .map(|(v, i)| i.map_or_else(Twist::default, |i| i.apply(*v)))
        .collect();
    let top = flatten(problem, &momentum);
    let (s_vel, _) = problem.driver_levels();
    let j = problem.jacobian_sparse(params);
    let solution = solve_saddle(
        problem,
        &mass_triplets(problem, &inertia),
        &j,
        &top,
        &s_vel,
        what,
    )?;
    Ok(problem.twists(&solution.x))
}

/// A consistent output frame from an integrator state: the velocity
/// projected onto `J·v = ṡ`, then the acceleration and multipliers solved
/// on the acceleration level there (the index-3 state itself is not
/// changed).
pub(crate) fn output_frame(
    problem: &Problem,
    state: &KinematicState,
    raw_velocity: &[Twist],
) -> Result<DynamicFrame, MbdError> {
    let vel = project_velocity(
        problem,
        &state.params,
        raw_velocity,
        "output velocity projection",
    )?;
    dynamic_frame(problem, state, vel)
}

/// The frame at a consistent pose and velocity: the forward-dynamics
/// accelerations and multipliers there.
fn dynamic_frame(
    problem: &Problem,
    state: &KinematicState,
    vel: Vec<Twist>,
) -> Result<DynamicFrame, MbdError> {
    let solved = accelerations(problem, &state.params, &vel)?;
    Ok(DynamicFrame {
        time: problem.time,
        velocity: state.by_body(vel),
        acceleration: state.by_body(solved.acc),
        multipliers: problem.rows.iter().copied().zip(solved.lambda).collect(),
        redundant: problem.refs(&solved.redundant),
        state: state.clone(),
    })
}

impl DynamicFrame {
    /// The joint friction at this instant, in force-element id order
    /// (empty without friction elements): the load each element applies,
    /// the friction load it read from the reactions, and the slip it saw.
    /// The friction is already part of this frame's accelerations and
    /// multipliers — this reports it the way the commercial tools plot a
    /// joint's friction force and torque.
    pub fn friction(&self, sys: &System) -> Result<Vec<FrictionLoad>, MbdError> {
        check_state(sys, &self.state)?;
        let problem = Problem::new(sys, self.time, &self.state.angles);
        let vel = slot_twists(sys, &self.velocity);
        let lambda: Vec<f64> = self.multipliers.iter().map(|(_, l)| *l).collect();
        Ok(friction_report(
            &problem,
            &self.state.params,
            &vel,
            Loads::dynamic(&lambda),
        ))
    }
}

/// The dynamic initial condition at `time`: the position IC from the
/// authored poses, the authored body velocities projected onto the
/// constraints ([`velocity_ic`]), and the forward-dynamics accelerations
/// and multipliers there.
pub fn dynamic_ic(sys: &System, time: f64) -> Result<DynamicFrame, MbdError> {
    let state = position_ic(sys, time)?;
    let authored: BTreeMap<BodyId, Twist> = sys
        .bodies
        .iter()
        .map(|(id, b)| {
            (
                *id,
                Twist {
                    linear: b.velocity,
                    angular: b.angular_velocity,
                },
            )
        })
        .collect();
    let velocity = velocity_ic(sys, time, &state, &authored)?;
    forward_dynamics(sys, time, &state, &velocity)
}

/// `Q − Jᵀλ` at rest (the static force imbalance), with the Jacobian.
fn static_imbalance(
    problem: &Problem,
    params: &[f64],
    lambda: &[f64],
    anchor: &[f64],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), MbdError> {
    let rest = vec![Twist::default(); problem.n_bodies()];
    let mut f = flatten(
        problem,
        &applied_forces(problem, params, &rest, Loads::static_at(lambda, anchor))?,
    );
    let j = problem.jacobian(params);
    for (jr, l) in j.iter().zip(lambda) {
        for (fi, ji) in f.iter_mut().zip(jr) {
            *fi -= ji * l;
        }
    }
    Ok((f, j))
}

/// Static equilibrium at `time` (drivers held at their values): the pose at
/// which the applied loads balance the constraint reactions, `Q(q) = Jᵀλ`,
/// `Φ(q, t) = 0`, with velocities zero (dampers carry nothing). Equivalent
/// to Adams static / Creo static analysis; the driver multipliers are the
/// holding efforts (Inventor "unknown force").
///
/// Newton iteration from the position IC (from `seed`, or the authored
/// poses) on the tangent stiffness by central differences, with a
/// backtracking line search. Like the commercial solvers it finds the
/// equilibrium nearest the start, stable or not. A load along a direction
/// nothing resists is [`codes::DYNAMICS_SINGULAR`]; no convergence is
/// [`codes::STATIC_NOT_CONVERGED`].
pub fn static_equilibrium(
    sys: &System,
    time: f64,
    seed: Option<&KinematicState>,
) -> Result<StaticEquilibrium, MbdError> {
    let mut state = match seed {
        Some(seed) => position_ic_from(sys, time, seed)?,
        None => position_ic(sys, time)?,
    };
    let mut problem = Problem::new(sys, time, &state.angles);
    let n = problem.n_cols;
    let m = problem.rows.len();
    // The pose friction measures its stiction displacement from: where the
    // mechanism stood when the solve started.
    let anchor = state.params.clone();
    let mut params = state.params.clone();
    let mut lambda = vec![0.0; m];
    let mut phi = vec![0.0; m];
    if reads_reactions(sys) {
        // Load-dependent friction has to read a reaction before it can
        // apply anything, and a zero one would leave the direction it
        // resists with no stiffness at all. The minimum-norm multipliers
        // of the load balance at the start pose are that first estimate.
        let rest = vec![Twist::default(); problem.n_bodies()];
        let q = flatten(
            &problem,
            &applied_forces(&problem, &params, &rest, Loads::static_at(&lambda, &anchor))?,
        );
        let j = problem.jacobian(&params);
        lambda = RowFactor::new(&transpose(&j, n), m).solve(&q).0;
    }
    let (f0, _) = static_imbalance(&problem, &params, &lambda, &anchor)?;
    let load_scale = inf_norm(&f0).max(1.0);
    let merit = |f: &[f64], phi: &[f64]| {
        let a = f.iter().map(|v| v * v).sum::<f64>() / (load_scale * load_scale);
        a + phi.iter().map(|v| v * v).sum::<f64>()
    };
    for iteration in 0..STATIC_MAX_ITER {
        let (f, j) = static_imbalance(&problem, &params, &lambda, &anchor)?;
        problem.constraint_residuals(&params, &mut phi);
        if inf_norm(&f) <= STATIC_FORCE_TOL * load_scale && inf_norm(&phi) <= STATIC_CONSTRAINT_TOL
        {
            state.params = params;
            state.normalize_quaternions();
            state.angles = problem.track_angles(&state.params);
            // Report the minimum-norm multipliers at the converged pose.
            let rest = vec![Twist::default(); problem.n_bodies()];
            let q = flatten(
                &problem,
                &applied_forces(
                    &problem,
                    &state.params,
                    &rest,
                    Loads::static_at(&lambda, &anchor),
                )?,
            );
            let (lambda, _) = RowFactor::new(&transpose(&j, n), m).solve(&q);
            let friction =
                friction_report(&problem, &state.params, &rest, Loads::static_at(&lambda, &anchor));
            return Ok(StaticEquilibrium {
                time,
                multipliers: problem.rows.iter().copied().zip(lambda).collect(),
                redundant: problem.refs(&RowFactor::new(&j, n).dependent_sorted()),
                iterations: iteration,
                friction,
                state,
            });
        }
        // Tangent stiffness K = −∂(Q − Jᵀλ)/∂q by central differences.
        let mut k = vec![vec![0.0; n]; n];
        let mut delta = vec![0.0; n];
        for c in 0..n {
            delta[c] = STIFFNESS_STEP;
            let (fp, _) = static_imbalance(&problem, &problem.retract(&params, &delta), &lambda, &anchor)?;
            delta[c] = -STIFFNESS_STEP;
            let (fm, _) = static_imbalance(&problem, &problem.retract(&params, &delta), &lambda, &anchor)?;
            delta[c] = 0.0;
            for (row, (p, q)) in k.iter_mut().zip(fp.iter().zip(&fm)) {
                row[c] = -(p - q) / (2.0 * STIFFNESS_STEP);
            }
        }
        let neg_phi: Vec<f64> = phi.iter().map(|v| -v).collect();
        let solution = solve_saddle(
            &problem,
            &triplets_from_dense(&k),
            &SparseRows::from_dense(&j, n),
            &f,
            &neg_phi,
            "static equilibrium",
        )?;
        let (mut step, mut dl) = (solution.x, solution.mu);
        // Limit the per-body rotation of one step.
        let mut biggest: f64 = 0.0;
        for b in problem.col_of.iter().flatten() {
            biggest = biggest.max(DVec3::new(step[b + 3], step[b + 4], step[b + 5]).length());
        }
        if biggest > MAX_STEP_ROTATION {
            let s = MAX_STEP_ROTATION / biggest;
            step.iter_mut().for_each(|v| *v *= s);
            dl.iter_mut().for_each(|v| *v *= s);
        }
        let current = merit(&f, &phi);
        let mut accepted = false;
        let mut scale = 1.0;
        for _ in 0..12 {
            let trial_step: Vec<f64> = step.iter().map(|v| v * scale).collect();
            let trial = problem.retract(&params, &trial_step);
            let trial_lambda: Vec<f64> =
                lambda.iter().zip(&dl).map(|(l, d)| l + scale * d).collect();
            let (tf, _) = static_imbalance(&problem, &trial, &trial_lambda, &anchor)?;
            let mut tphi = vec![0.0; m];
            problem.constraint_residuals(&trial, &mut tphi);
            if merit(&tf, &tphi) < current {
                params = trial;
                lambda = trial_lambda;
                accepted = true;
                // Re-anchor the angle unwrapping on the new pose, so a joint
                // spring may wind past half a turn on the way.
                let refs = problem.track_angles(&params);
                problem = Problem::new(sys, time, &refs);
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    let (f, _) = static_imbalance(&problem, &params, &lambda, &anchor)?;
    problem.constraint_residuals(&params, &mut phi);
    Err(MbdError::new(
        codes::STATIC_NOT_CONVERGED,
        format!(
            "static equilibrium at t = {time} did not converge: force imbalance {:.3e} (load scale {load_scale:.3e}), constraint residual {:.3e}",
            inf_norm(&f),
            inf_norm(&phi)
        ),
        "seed closer to the expected equilibrium, or check for a mechanism with no stable rest pose",
    ))
}

/// The energy of a state: kinetic `½·vᵀMv`, gravitational `−m·g·x_com`,
/// the springs' `½·k·(l − l₀)² − preload·(l − l₀)` (joint springs alike,
/// bushings `½·dᵀK·d + ½·θᵀK_t·θ`). Function forces are not
/// potentials (their work is audited through [`power`]).
pub fn energy(
    sys: &System,
    state: &KinematicState,
    velocity: &BTreeMap<BodyId, Twist>,
) -> Result<Energy, MbdError> {
    check_state(sys, state)?;
    let problem = Problem::new(sys, 0.0, &state.angles);
    let vel = slot_twists(sys, velocity);
    let params = &state.params;
    let mut e = Energy::default();
    for (slot, (body, i)) in sys
        .bodies
        .values()
        .zip(inertias(sys, params))
        .enumerate()
    {
        let (t, r) = read_pose(params, slot);
        if let Some(i) = i {
            let p = i.apply(vel[slot]);
            e.kinetic += 0.5 * (p.linear.dot(vel[slot].linear) + p.angular.dot(vel[slot].angular));
        }
        if let Some(g) = sys.gravity {
            e.gravity -= body.mass_kg * g.dot(t + r * body.com);
        }
    }
    let rest = vec![Twist::default(); problem.n_bodies()];
    for element in sys.forces.values() {
        match element {
            ForceElement::SpringDamper {
                marker1,
                marker2,
                stiffness,
                free_length,
                preload,
                ..
            } => {
                let p1 = problem.frame(params, &rest, *marker1).p.v;
                let p2 = problem.frame(params, &rest, *marker2).p.v;
                let stretch = (p2 - p1).length() - free_length;
                e.elastic += 0.5 * stiffness * stretch * stretch - preload * stretch;
            }
            ForceElement::JointSpringDamper {
                joint,
                slot,
                stiffness,
                free_value,
                preload,
                ..
            } => {
                if sys.joints[joint].suppressed {
                    continue;
                }
                // The tracked (unwrapped) angle for rotation.
                let q = match slot {
                    DriverSlot::Rotation => {
                        state.joint_angle(*joint).unwrap_or_default()
                    }
                    DriverSlot::Translation => {
                        problem.coordinate(params, &rest, *joint, *slot).v
                    }
                };
                let s = q - free_value;
                e.elastic += 0.5 * stiffness * s * s - preload * s;
            }
            ForceElement::Bushing {
                marker1,
                marker2,
                stiffness,
                torsion_stiffness,
                ..
            } => {
                let b = bushing_state(&problem, params, &rest, *marker1, *marker2);
                e.elastic += 0.5 * (b.d * *stiffness).dot(b.d)
                    + 0.5 * (b.theta * *torsion_stiffness).dot(b.theta);
            }
            _ => {}
        }
    }
    Ok(e)
}

/// The total linear momentum and the angular momentum about the world
/// origin.
pub fn momentum(
    sys: &System,
    state: &KinematicState,
    velocity: &BTreeMap<BodyId, Twist>,
) -> Result<Momentum, MbdError> {
    check_state(sys, state)?;
    let vel = slot_twists(sys, velocity);
    let mut out = Momentum::default();
    for (slot, i) in inertias(sys, &state.params).into_iter().enumerate() {
        let Some(i) = i else { continue };
        let origin = read_pose(&state.params, slot).0;
        let p = i.apply(vel[slot]);
        out.linear += p.linear;
        out.angular += origin.cross(p.linear) + p.angular;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(com: DVec3, inertia: DVec3) -> Body {
        Body {
            id: BodyId(1),
            name: "b".into(),
            mass_kg: 2.5,
            com,
            inertia_com: DMat3::from_diagonal(inertia),
            translation: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            grounded: false,
            velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
        }
    }

    /// The mass matrix is symmetric positive definite and `½vᵀMv` equals the
    /// kinetic energy computed at the center of mass.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn mass_matrix_is_the_kinetic_energy() {
        let b = body(DVec3::new(0.3, -0.2, 0.5), DVec3::new(0.4, 0.7, 0.9));
        let r = DQuat::from_euler(glam::EulerRot::XYZ, 0.4, -1.1, 0.8);
        let m = mass_matrix(&b, r);
        for i in 0..6 {
            for k in 0..6 {
                assert!((m[i][k] - m[k][i]).abs() < 1e-12, "symmetric");
            }
        }
        let v = DVec3::new(0.3, -1.2, 0.7);
        let w = DVec3::new(-0.8, 0.5, 1.9);
        let x = Twist { linear: v, angular: w }.to_array();
        let quad: f64 = (0..6)
            .map(|i| (0..6).map(|k| x[i] * m[i][k] * x[k]).sum::<f64>())
            .sum();
        let c = r * b.com;
        let jc = DMat3::from_quat(r) * b.inertia_com * DMat3::from_quat(r).transpose();
        let v_com = v + w.cross(c);
        let t = 0.5 * b.mass_kg * v_com.length_squared() + 0.5 * w.dot(jc * w);
        assert!((0.5 * quad - t).abs() < 1e-12, "{} vs {t}", 0.5 * quad);
        // Positive definite: Cholesky succeeds.
        let mut l = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for k in 0..=i {
                let s: f64 = (0..k).map(|p| l[i][p] * l[k][p]).sum();
                if i == k {
                    let d = m[i][i] - s;
                    assert!(d > 0.0, "positive definite");
                    l[i][i] = d.sqrt();
                } else {
                    l[i][k] = (m[i][k] - s) / l[k][k];
                }
            }
        }
    }

    /// The multipliers are physical: on a swinging compound pendulum the
    /// constraint load `−Jᵀλ` on the link is the hinge force `m·a_com − m·g`
    /// with no torque about the hinge point (the hinge is at the body
    /// origin, a principal axis along the hinge).
    #[test]
    fn hinge_multipliers_are_the_hinge_load() {
        use crate::body::{Marker, MarkerId};
        use crate::joint::{Joint, JointId, JointKind};
        let g = DVec3::new(0.0, -9.81, 0.0);
        let mut sys = System::new("pendulum");
        sys.gravity = Some(g);
        let mut base = body(DVec3::ZERO, DVec3::ZERO);
        base.grounded = true;
        base.mass_kg = 0.0;
        sys.add_body(base).unwrap();
        let mut link = body(DVec3::new(0.0, -0.6, 0.0), DVec3::new(0.04, 0.01, 0.03));
        link.id = BodyId(2);
        link.rotation = DQuat::from_rotation_z(1.1);
        link.angular_velocity = DVec3::Z * 2.3;
        sys.add_body(link.clone()).unwrap();
        for (id, b) in [(1, BodyId(1)), (2, BodyId(2))] {
            sys.add_marker(Marker {
                id: MarkerId(id),
                name: format!("m{id}"),
                body: b,
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
        let frame = dynamic_ic(&sys, 0.0).unwrap();
        let problem = Problem::new(&sys, 0.0, &frame.state.angles);
        let j = problem.jacobian(&frame.state.params);
        let mut load = [0.0; BODY_TWIST];
        for ((_, l), row) in frame.multipliers.iter().zip(&j) {
            for (c, v) in load.iter_mut().enumerate() {
                *v -= row[c] * l;
            }
        }
        let (v, a) = (frame.velocity[&BodyId(2)], frame.acceleration[&BodyId(2)]);
        let c = link.rotation * link.com;
        let a_com = a.linear + a.angular.cross(c) + v.angular.cross(v.angular.cross(c));
        let force = link.mass_kg * (a_com - g);
        assert!(
            (DVec3::new(load[0], load[1], load[2]) - force).length() < 1e-9,
            "{load:?} vs {force:?}"
        );
        assert!(
            DVec3::new(load[3], load[4], load[5]).length() < 1e-9,
            "no hinge torque: {load:?}"
        );
    }

    /// `b(ω)` is the rate of the momentum at constant velocity twist: the
    /// time derivative of `M(q(t))·v` along a motion with constant `v, ω`,
    /// plus the transport term `v_o × p` of the moving-origin angular
    /// momentum, is the bias.
    #[test]
    fn bias_is_the_momentum_rate() {
        let b = body(DVec3::new(0.3, -0.2, 0.5), DVec3::new(0.4, 0.7, 0.9));
        let r0 = DQuat::from_euler(glam::EulerRot::XYZ, 0.4, -1.1, 0.8);
        let v = DVec3::new(0.3, -1.2, 0.7);
        let w = DVec3::new(-0.8, 0.5, 1.9);
        let tw = Twist {
            linear: v,
            angular: w,
        };
        let at = |h: f64| BodyInertia::at(&b, DQuat::from_scaled_axis(w * h) * r0).apply(tw);
        let h = 1e-5;
        let (p, m) = (at(h), at(-h));
        let dp = (p.linear - m.linear) / (2.0 * h);
        let dh = (p.angular - m.angular) / (2.0 * h);
        let p0 = at(0.0).linear;
        let bias = BodyInertia::at(&b, r0).bias(w);
        assert!(
            (dp - bias.linear).length() < 1e-8,
            "{dp:?} vs {:?}",
            bias.linear
        );
        // Euler about the moving origin: τ_o = ḣ_o + v_o × p.
        let rate = dh + v.cross(p0);
        assert!(
            (rate - bias.angular).length() < 1e-8,
            "{rate:?} vs {:?}",
            bias.angular
        );
    }
}
