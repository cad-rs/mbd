//! Joint reactions (M2): the constraint multipliers recovered as the force
//! and torque each joint and driver transmits — the bearing loads and
//! actuator efforts commercial tools report.
//!
//! Every joint and driver equation depends only on the relative pose of its
//! two markers, so its share of `−Jᵀλ` is an equal and opposite wrench pair
//! on the two bodies. A [`Reaction`] reports it the OndselSolver / Adams
//! way: the force on marker I's body (the joint's `marker1`) and the torque
//! on it about marker I's origin, world frame — plus the J side, the
//! driver effort along its coordinate, and the statically indeterminate
//! directions.
//!
//! **Redundancy.** With redundant equations the multipliers are not unique:
//! any `λ + n` with `Jᵀn = 0` satisfies the equations of motion. The
//! reported reactions follow the multipliers' split (forward dynamics drops
//! redundant equations on its sparse path, multiplier 0; its dense path and
//! static equilibrium give the minimum-norm split), and each reaction lists
//! the wrench directions along which that split is arbitrary (the image of
//! the null space of `Jᵀ` on that joint). A planar loop of 3D revolutes,
//! for example, determines every in-plane load but not how the out-of-plane
//! forces and moments share between its joints — the same indeterminacy a
//! statics textbook assigns a hyperstatic structure.

use glam::DVec3;

use crate::body::MarkerId;
use crate::dynamics::{DynamicFrame, StaticEquilibrium};
use crate::error::{codes, MbdError};
use crate::kinematics::{check_state, EquationOwner, EquationRef, KinematicState, Problem};
use crate::linalg::{transpose, RowFactor};
use crate::system::System;

/// Relative size below which an indeterminate wrench direction is noise.
const INDETERMINATE_TOL: f64 = 1.0e-9;

/// The load one joint or driver transmits.
#[derive(Debug, Clone, PartialEq)]
pub struct Reaction {
    /// The joint (all of its equations) or the driver.
    pub owner: EquationOwner,
    /// Force on marker I's body, world frame (N).
    pub force_on_i: DVec3,
    /// Torque on marker I's body about marker I's origin, world frame (N·m).
    pub torque_on_i: DVec3,
    /// Force on marker J's body (`−force_on_i`).
    pub force_on_j: DVec3,
    /// Torque on marker J's body about marker J's origin.
    pub torque_on_j: DVec3,
    /// A driver's effort along its coordinate — the torque (rotation) or
    /// force (translation) the actuator applies to marker J's body relative
    /// to marker I's; `None` for joints.
    pub effort: Option<f64>,
    /// Statically indeterminate directions of the I-side wrench
    /// `[Fx, Fy, Fz, Tx, Ty, Tz]` (orthonormal): the reaction's components
    /// along them are not determined by the model. Empty: unique.
    pub indeterminate: Vec<[f64; 6]>,
}

impl Reaction {
    /// Whether the model determines this reaction completely.
    pub fn is_unique(&self) -> bool {
        self.indeterminate.is_empty()
    }

    /// The I-side wrench as `[Fx, Fy, Fz, Tx, Ty, Tz]`.
    pub fn wrench_on_i(&self) -> [f64; 6] {
        wrench_array(self.force_on_i, self.torque_on_i)
    }

    /// `w` with its components along the indeterminate directions removed
    /// — what two valid multiplier splits must agree on.
    pub fn determinate_part(&self, w: [f64; 6]) -> [f64; 6] {
        let mut out = w;
        for e in &self.indeterminate {
            remove_along(&mut out, e);
        }
        out
    }
}

/// `[Fx, Fy, Fz, Tx, Ty, Tz]`.
fn wrench_array(f: DVec3, t: DVec3) -> [f64; 6] {
    [f.x, f.y, f.z, t.x, t.y, t.z]
}

/// Remove from `w` its component along the unit vector `e`.
fn remove_along(w: &mut [f64; 6], e: &[f64; 6]) {
    let d: f64 = w.iter().zip(e).map(|(a, b)| a * b).sum();
    for (wi, ei) in w.iter_mut().zip(e) {
        *wi -= d * ei;
    }
}

/// A joint or driver with its constraint rows and markers.
struct Owner {
    owner: EquationOwner,
    rows: Vec<usize>,
    i: MarkerId,
    j: MarkerId,
}

/// The wrench on I (force, torque about marker I's origin) that the rows
/// of `owner` carry for multipliers `mu`.
fn wrench(
    problem: &Problem,
    state: &KinematicState,
    j: &[Vec<f64>],
    owner: &Owner,
    mu: &[f64],
) -> (DVec3, DVec3) {
    let sys = problem.sys;
    let side = |m: MarkerId| {
        let marker = &sys.markers[&m];
        let origin = state.body_pose(marker.body).expect("checked state").0;
        (problem.slot_of[&marker.body], origin)
    };
    let (slot_i, origin_i) = side(owner.i);
    let (slot_j, origin_j) = side(owner.j);
    let p_i = marker_point(sys, state, owner.i);
    let load = |slot: usize| -> Option<(DVec3, DVec3)> {
        let base = problem.col_of[slot]?;
        let (mut f, mut t) = (DVec3::ZERO, DVec3::ZERO);
        for &k in &owner.rows {
            let row = &j[k];
            f -= mu[k] * DVec3::new(row[base], row[base + 1], row[base + 2]);
            t -= mu[k] * DVec3::new(row[base + 3], row[base + 4], row[base + 5]);
        }
        Some((f, t))
    };
    if let Some((f, t)) = load(slot_i) {
        (f, t - (p_i - origin_i).cross(f))
    } else if let Some((f, t)) = load(slot_j) {
        // Newton's third law: equal and opposite, moments about p_i cancel.
        let t_at_pi = t - (p_i - origin_j).cross(f);
        (-f, -t_at_pi)
    } else {
        (DVec3::ZERO, DVec3::ZERO)
    }
}

/// A marker's world origin.
fn marker_point(sys: &System, state: &KinematicState, m: MarkerId) -> DVec3 {
    let marker = &sys.markers[&m];
    state
        .world_point(marker.body, marker.translation)
        .expect("checked state")
}

/// Orthonormal basis of `vectors` (Gram–Schmidt, dropping components below
/// `tol`).
fn orthonormal(vectors: &[[f64; 6]], tol: f64) -> Vec<[f64; 6]> {
    let mut basis: Vec<[f64; 6]> = Vec::new();
    for v in vectors {
        let mut w = *v;
        for _ in 0..2 {
            for e in &basis {
                remove_along(&mut w, e);
            }
        }
        let norm = w.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm > tol {
            basis.push(w.map(|a| a / norm));
        }
    }
    basis
}

/// The reactions of every active joint (id order) and driver (authoring
/// order) at a solved instant, from its multipliers (a [`DynamicFrame`] or
/// [`StaticEquilibrium`]).
pub fn reactions(
    sys: &System,
    time: f64,
    state: &KinematicState,
    multipliers: &[(EquationRef, f64)],
) -> Result<Vec<Reaction>, MbdError> {
    check_state(sys, state)?;
    let problem = Problem::new(sys, time, &state.angles);
    if multipliers.len() != problem.rows.len()
        || multipliers
            .iter()
            .zip(&problem.rows)
            .any(|((r, _), q)| r != q)
    {
        return Err(MbdError::new(
            codes::STATE_MISMATCH,
            format!(
                "{} multipliers do not match the system's {} constraint equations",
                multipliers.len(),
                problem.rows.len()
            ),
            "pass the multipliers solved for this same system and instant",
        ));
    }
    let j = problem.jacobian(&state.params);
    let lambda: Vec<f64> = multipliers.iter().map(|(_, l)| *l).collect();
    let n = problem.n_cols;
    let m = problem.rows.len();

    // Null space of Jᵀ, spanned by the projections of the dependent rows'
    // unit vectors: n_k = e_k − x_k, x_k the minimum-norm λ with
    // Jᵀx_k = Jᵀe_k.
    let dependent = RowFactor::new(&j, n).dependent().to_vec();
    let null: Vec<Vec<f64>> = if dependent.is_empty() {
        Vec::new()
    } else {
        let jt = transpose(&j, n);
        let factor = RowFactor::new(&jt, m);
        dependent
            .iter()
            .map(|&k| {
                let (x, _) = factor.solve(&jt.iter().map(|r| r[k]).collect::<Vec<_>>());
                let mut v: Vec<f64> = x.iter().map(|a| -a).collect();
                v[k] += 1.0;
                v
            })
            .collect()
    };

    // Joints and drivers with their rows (couplers act between joint
    // coordinates, not between two markers: no reaction).
    let mut owners: Vec<Owner> = Vec::new();
    for (k, row) in problem.rows.iter().enumerate() {
        if let Some(last) = owners.last_mut().filter(|o| o.owner == row.owner) {
            last.rows.push(k);
            continue;
        }
        let joint = match row.owner {
            EquationOwner::Joint(id) => id,
            EquationOwner::Driver(i) => sys.drivers[i].joint,
            EquationOwner::Coupler(_) => continue,
        };
        let joint = &sys.joints[&joint];
        owners.push(Owner {
            owner: row.owner,
            rows: vec![k],
            i: joint.marker1,
            j: joint.marker2,
        });
    }

    // Indeterminate wrench images, scaled against the largest one.
    let images: Vec<Vec<[f64; 6]>> = owners
        .iter()
        .map(|owner| {
            null.iter()
                .map(|nv| {
                    let (f, t) = wrench(&problem, state, &j, owner, nv);
                    wrench_array(f, t)
                })
                .collect()
        })
        .collect();
    let biggest = images
        .iter()
        .flatten()
        .flat_map(|w| w.iter())
        .fold(0.0_f64, |a, v| a.max(v.abs()));

    let mut out = Vec::with_capacity(owners.len());
    for (owner, image) in owners.iter().zip(&images) {
        let (force_on_i, torque_on_i) = wrench(&problem, state, &j, owner, &lambda);
        let p_i = marker_point(sys, state, owner.i);
        let p_j = marker_point(sys, state, owner.j);
        out.push(Reaction {
            owner: owner.owner,
            force_on_i,
            torque_on_i,
            force_on_j: -force_on_i,
            torque_on_j: -torque_on_i - (p_i - p_j).cross(force_on_i),
            effort: match owner.owner {
                EquationOwner::Driver(_) => Some(-lambda[owner.rows[0]]),
                _ => None,
            },
            indeterminate: orthonormal(image, INDETERMINATE_TOL * biggest.max(f64::MIN_POSITIVE)),
        });
    }
    Ok(out)
}

impl DynamicFrame {
    /// The joint and driver reactions of this frame ([`reactions`]).
    pub fn reactions(&self, sys: &System) -> Result<Vec<Reaction>, MbdError> {
        reactions(sys, self.time, &self.state, &self.multipliers)
    }
}

impl StaticEquilibrium {
    /// The joint and driver reactions holding the equilibrium
    /// ([`reactions`]).
    pub fn reactions(&self, sys: &System) -> Result<Vec<Reaction>, MbdError> {
        reactions(sys, self.time, &self.state, &self.multipliers)
    }
}
