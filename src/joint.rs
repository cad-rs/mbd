//! Joints (kinematic constraints between two markers) and drivers (which
//! joint slots move over time). The joint vocabulary mirrors the assembly
//! domain's `PairType`.

use serde::{Deserialize, Serialize};

use crate::body::MarkerId;

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
}

impl JointKind {
    /// The number of relative degrees of freedom this joint leaves.
    pub fn remaining_dof(self) -> usize {
        match self {
            JointKind::Revolute | JointKind::Prismatic => 1,
            JointKind::Cylindrical | JointKind::Universal => 2,
            JointKind::Spherical | JointKind::Planar => 3,
            JointKind::Fixed => 0,
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

/// Which slot of a joint a driver moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverSlot {
    /// Joint rotation (rad).
    Rotation,
    /// Joint translation (m).
    Translation,
}

/// How the driven value evolves over time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "profile", rename_all = "snake_case")]
pub enum DriverProfile {
    /// `value(t) = start + rate * t`.
    Constant { start: f64, rate: f64 },
    /// Linear interpolation through `(time_s, value)` keyframes sorted by
    /// time, clamped at the ends.
    Keyframes(Vec<(f64, f64)>),
}

/// A prescribed motion on one slot of one joint (the MbD "motor"). A joint
/// slot with a driver loses that degree of freedom to prescription; joints
/// without any driver keep their free degrees of freedom for the dynamics
/// to determine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Driver {
    pub joint: JointId,
    pub slot: DriverSlot,
    pub profile: DriverProfile,
}

impl Driver {
    /// The driven value at `t` (s).
    pub fn value_at(&self, t: f64) -> f64 {
        match &self.profile {
            DriverProfile::Constant { start, rate } => start + rate * t,
            DriverProfile::Keyframes(keys) => {
                let Some(first) = keys.first() else { return 0.0 };
                if t <= first.0 {
                    return first.1;
                }
                for pair in keys.windows(2) {
                    let (t0, v0) = pair[0];
                    let (t1, v1) = pair[1];
                    if t <= t1 {
                        let span = t1 - t0;
                        if span <= 0.0 {
                            return v1;
                        }
                        return v0 + (v1 - v0) * ((t - t0) / span);
                    }
                }
                keys.last().map(|k| k.1).unwrap_or(0.0)
            }
        }
    }
}
