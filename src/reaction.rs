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
//! reported reactions follow the multipliers' split (redundant equations
//! removed on the sparse path, minimum-norm on the dense one), and each
//! reaction lists
//! the wrench directions along which that split is arbitrary (the image of
//! the null space of `Jᵀ` on that joint). A planar loop of 3D revolutes,
//! for example, determines every in-plane load but not how the out-of-plane
//! forces and moments share between its joints — the same indeterminacy a
//! statics textbook assigns a hyperstatic structure.

use glam::DVec3;

use crate::dynamics::{DynamicFrame, StaticEquilibrium};
use crate::error::{codes, MbdError};
use crate::kinematics::{
    check_state, read_pose, EquationOwner, EquationRef, KinematicState, Problem,
};
use crate::linalg::RowFactor;
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
        let (f, t) = (self.force_on_i, self.torque_on_i);
        [f.x, f.y, f.z, t.x, t.y, t.z]
    }

    /// `w` with its components along the indeterminate directions removed
    /// — what two valid multiplier splits must agree on.
    pub fn determinate_part(&self, w: [f64; 6]) -> [f64; 6] {
        let mut out = w;
        for e in &self.indeterminate {
            let d: f64 = out.iter().zip(e).map(|(a, b)| a * b).sum();
            for (o, b) in out.iter_mut().zip(e) {
                *o -= d * b;
            }
        }
        out
    }
}

/// The two marker ids of a joint or driver.
fn markers(
    sys: &System,
    owner: EquationOwner,
) -> Option<(crate::body::MarkerId, crate::body::MarkerId)> {
    let joint = match owner {
        EquationOwner::Joint(id) => id,
        EquationOwner::Driver(i) => sys.drivers.get(i)?.joint,
        EquationOwner::Coupler(_) => return None,
    };
    let j = sys.joints.get(&joint)?;
    Some((j.marker1, j.marker2))
}

/// The wrench on I (force, torque about marker I's origin) that the rows
/// `rows` carry for multipliers `mu`.
fn wrench(
    problem: &Problem,
    params: &[f64],
    j: &[Vec<f64>],
    owner: EquationOwner,
    rows: &[usize],
    mu: &[f64],
) -> (DVec3, DVec3) {
    let sys = problem.sys;
    let Some((mi, mj)) = markers(sys, owner) else {
        return (DVec3::ZERO, DVec3::ZERO);
    };
    let point = |m: crate::body::MarkerId| {
        let marker = &sys.markers[&m];
        let slot = problem.slot_of[&marker.body];
        let (t, r) = read_pose(params, slot);
        (slot, t, t + r * marker.translation)
    };
    let (slot_i, origin_i, p_i) = point(mi);
    let (slot_j, origin_j, _) = point(mj);
    let load = |slot: usize| -> Option<(DVec3, DVec3)> {
        let base = problem.col_of[slot]?;
        let (mut f, mut t) = (DVec3::ZERO, DVec3::ZERO);
        for &k in rows {
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

/// Orthonormal basis of `vectors` (Gram–Schmidt, dropping components below
/// `tol`).
fn orthonormal(vectors: &[[f64; 6]], tol: f64) -> Vec<[f64; 6]> {
    let mut basis: Vec<[f64; 6]> = Vec::new();
    for v in vectors {
        let mut w = *v;
        for _ in 0..2 {
            for e in &basis {
                let d: f64 = w.iter().zip(e).map(|(a, b)| a * b).sum();
                for (wi, ei) in w.iter_mut().zip(e) {
                    *wi -= d * ei;
                }
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
    let params = &state.params;
    let j = problem.jacobian(params);
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
        let jt: Vec<Vec<f64>> = (0..n).map(|c| j.iter().map(|r| r[c]).collect()).collect();
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

    // Owners with their rows.
    let mut owners: Vec<(EquationOwner, Vec<usize>)> = Vec::new();
    for (k, row) in problem.rows.iter().enumerate() {
        if matches!(row.owner, EquationOwner::Coupler(_)) {
            continue;
        }
        match owners.last_mut() {
            Some((o, rows)) if *o == row.owner => rows.push(k),
            _ => owners.push((row.owner, vec![k])),
        }
    }

    // Indeterminate wrench images, scaled against the largest one.
    let images: Vec<Vec<[f64; 6]>> = owners
        .iter()
        .map(|(owner, rows)| {
            null.iter()
                .map(|nv| {
                    let (f, t) = wrench(&problem, params, &j, *owner, rows, nv);
                    [f.x, f.y, f.z, t.x, t.y, t.z]
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
    for ((owner, rows), image) in owners.iter().zip(&images) {
        let (force_on_i, torque_on_i) = wrench(&problem, params, &j, *owner, rows, &lambda);
        let (mi, mj) = markers(sys, *owner).expect("joint or driver");
        let at = |m: crate::body::MarkerId| {
            let marker = &sys.markers[&m];
            let (t, r) = read_pose(params, problem.slot_of[&marker.body]);
            t + r * marker.translation
        };
        let (p_i, p_j) = (at(mi), at(mj));
        out.push(Reaction {
            owner: *owner,
            force_on_i,
            torque_on_i,
            force_on_j: -force_on_i,
            torque_on_j: -torque_on_i - (p_i - p_j).cross(force_on_i),
            effort: match owner {
                EquationOwner::Driver(_) => Some(-lambda[rows[0]]),
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
