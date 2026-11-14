//! Joints (kinematic constraints between two markers) and their constraint
//! equations. The joint vocabulary mirrors the assembly domain's `PairType`.
//! Drivers live in [`crate::driver`] (re-exported here for the M0 paths).

use serde::{Deserialize, Serialize};

use crate::body::MarkerId;
use crate::jet::{FrameJet, Jet};

pub use crate::driver::{Driver, DriverProfile, DriverSample, DriverSlot};

/// Stable joint identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct JointId(pub u64);

/// The kinematic joint vocabulary (maximal coordinates: each kind removes
/// a fixed number of relative degrees of freedom between the two markers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointKind {
    /// Rotation about the common Z axis (1 DOF remains).
    Revolute,
    /// Translation along the common Z axis (1 DOF remains).
    Prismatic,
    /// Rotation + translation along the common Z axis (2 DOF remain).
    Cylindrical,
    /// Ball joint: rotations only (3 DOF remain).
    Spherical,
    /// Motion in the common XY plane (3 DOF remain).
    Planar,
    /// Two perpendicular rotation axes (2 DOF remain).
    Universal,
    /// No relative motion (6 removed).
    Fixed,
    // ── Joint primitives (Adams JPRIM) ─────────────────────────────────
    // Single geometric relations, so CAD mates map one to one; the lower
    // pairs above are combinations of them (revolute = spherical +
    // parallel axes, prismatic = in-line + orientation, cylindrical =
    // in-line + parallel axes, planar = in-plane + parallel axes).
    /// Marker 2's origin stays on marker 1's Z axis (2 removed) — the
    /// "point on line" mate.
    InLine,
    /// Marker 2's origin stays in marker 1's XY plane (1 removed) — the
    /// "point on plane" / coincident-face mate.
    InPlane,
    /// The relative orientation is locked, translation free (3 removed).
    Orientation,
    /// The Z axes stay parallel (2 removed) — the parallel mate.
    ParallelAxes,
    /// The Z axes stay perpendicular (1 removed) — the perpendicular mate.
    Perpendicular,
}

impl JointKind {
    /// The number of relative degrees of freedom this joint leaves.
    pub fn remaining_dof(self) -> usize {
        6 - self.constraint_count()
    }

    /// The number of independent constraint equations this joint kind
    /// imposes (the relative DOF it removes from the 6 between two bodies).
    pub fn constraint_count(self) -> usize {
        match self {
            JointKind::Revolute | JointKind::Prismatic => 5,
            JointKind::Cylindrical | JointKind::Universal => 4,
            JointKind::Spherical | JointKind::Planar | JointKind::Orientation => 3,
            JointKind::InLine | JointKind::ParallelAxes => 2,
            JointKind::InPlane | JointKind::Perpendicular => 1,
            JointKind::Fixed => 6,
        }
    }

    /// Whether a driver may prescribe `slot` on this kind. Rotation is the
    /// angle about the common Z axis; translation is the offset along it.
    /// Planar translation (two directions) and the universal joint's two
    /// rotations have no single slot and are not drivable.
    pub fn drives(self, slot: DriverSlot) -> bool {
        match slot {
            DriverSlot::Rotation => matches!(
                self,
                JointKind::Revolute | JointKind::Cylindrical | JointKind::Planar
            ),
            DriverSlot::Translation => {
                matches!(self, JointKind::Prismatic | JointKind::Cylindrical)
            }
        }
    }
}

/// A joint constrains marker 2's frame to marker 1's frame per the joint
/// kind's geometry: the markers' frames must coincide at the joint's
/// remaining degrees of freedom.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Joint {
    pub id: JointId,
    pub name: String,
    pub kind: JointKind,
    pub marker1: MarkerId,
    pub marker2: MarkerId,
    /// Whether the joint is suppressed (constraint not enforced).
    pub suppressed: bool,
}

/// Write one joint's constraint residual jets into `out`; returns the count
/// (= [`JointKind::constraint_count`]).
///
/// The formulations, over the world frames of the two markers (`dp = p2 −
/// p1`, axes `x1 y1 z1` / `x2 y2 z2`); every row is a dot product, so its
/// derivatives are exact:
/// - `Revolute`: origin coincidence (3) + `z2 ⟂ x1, y1` (2).
/// - `Prismatic`: relative orientation locked (`x2·y1 = x2·z1 = y2·z1 = 0`,
///   3) + the offset is axial (`dp ⟂ x1, y1`, 2).
/// - `Cylindrical`: `z2 ⟂ x1, y1` (2) + `dp ⟂ x1, y1` (2).
/// - `Spherical`: origin coincidence (3).
/// - `Planar`: `z2 ⟂ x1, y1` (2) + `dp ⟂ z1` (1).
/// - `Universal`: origin coincidence (3) + `z1 ⟂ z2` (1).
/// - `Fixed`: origin coincidence (3) + relative orientation locked (3).
/// - Primitives: `InLine` `dp ⟂ x1, y1`; `InPlane` `dp ⟂ z1`;
///   `Orientation` the prismatic's three orientation rows; `ParallelAxes`
///   `z2 ⟂ x1, y1`; `Perpendicular` `z1 ⟂ z2` — the same rows the lower
///   pairs use, so a pair and its primitive decomposition are identical
///   equation sets.
///
/// Known limitation (shared with the dot-product formulations of the
/// commercial solvers): the axis and orientation rows are also satisfied by
/// the 180°-flipped configurations, so a seed far from the authored pose may
/// converge to a flipped assembly.
pub(crate) fn joint_rows(kind: JointKind, f1: &FrameJet, f2: &FrameJet, out: &mut [Jet]) -> usize {
    let dp = f2.p - f1.p;
    let rows: &[Jet] = match kind {
        JointKind::Revolute => &[
            dp.x(),
            dp.y(),
            dp.z(),
            f1.x.dot(f2.z),
            f1.y.dot(f2.z),
        ],
        JointKind::Prismatic => &[
            f1.y.dot(f2.x),
            f1.z.dot(f2.x),
            f1.z.dot(f2.y),
            dp.dot(f1.x),
            dp.dot(f1.y),
        ],
        JointKind::Cylindrical => &[
            f1.x.dot(f2.z),
            f1.y.dot(f2.z),
            dp.dot(f1.x),
            dp.dot(f1.y),
        ],
        JointKind::Spherical => &[dp.x(), dp.y(), dp.z()],
        JointKind::Planar => &[f1.x.dot(f2.z), f1.y.dot(f2.z), dp.dot(f1.z)],
        JointKind::Universal => &[dp.x(), dp.y(), dp.z(), f1.z.dot(f2.z)],
        JointKind::Fixed => &[
            dp.x(),
            dp.y(),
            dp.z(),
            f1.y.dot(f2.x),
            f1.z.dot(f2.x),
            f1.z.dot(f2.y),
        ],
        JointKind::InLine => &[dp.dot(f1.x), dp.dot(f1.y)],
        JointKind::InPlane => &[dp.dot(f1.z)],
        JointKind::Orientation => &[f1.y.dot(f2.x), f1.z.dot(f2.x), f1.z.dot(f2.y)],
        JointKind::ParallelAxes => &[f1.x.dot(f2.z), f1.y.dot(f2.z)],
        JointKind::Perpendicular => &[f1.z.dot(f2.z)],
    };
    out[..rows.len()].copy_from_slice(rows);
    rows.len()
}

/// The joint coordinate a driver slot measures, as a jet: the rotation angle
/// `atan2(x2·y1, x2·x1)` (in `(−π, π]`; the caller unwraps it against the
/// prescribed value) or the axial offset `dp·z1`. `None` when the slot is not
/// drivable on the kind (rejected by validation before any solve).
pub(crate) fn joint_coordinate(kind: JointKind, slot: DriverSlot, f1: &FrameJet, f2: &FrameJet) -> Option<Jet> {
    if !kind.drives(slot) {
        return None;
    }
    Some(match slot {
        DriverSlot::Rotation => Jet::atan2(f1.y.dot(f2.x), f1.x.dot(f2.x)),
        DriverSlot::Translation => (f2.p - f1.p).dot(f1.z),
    })
}
