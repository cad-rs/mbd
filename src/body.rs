//! Rigid bodies and the markers (attached frames) joints and force elements
//! reference. SI units: kg, m, s.

use glam::{DMat3, DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// Stable body identity (the ordering key of the system's body table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BodyId(pub u64);

/// Stable marker identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MarkerId(pub u64);

/// A rigid body: mass properties plus its reference pose. The reference
/// pose is the pose the body is authored at; the solver's state starts from
/// it.
///
/// `inertia_com` is the inertia tensor about the center of mass, expressed
/// in the body's reference-frame axes (the kernel's `GProp` matrix of
/// inertia, converted to SI by the caller).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Body {
    pub id: BodyId,
    pub name: String,
    pub mass_kg: f64,
    /// Center of mass in the body's reference frame (m).
    pub com: DVec3,
    /// Inertia tensor about the center of mass, body-frame axes (kg·m²).
    pub inertia_com: DMat3,
    /// Reference pose: translation (m) and rotation of the body frame in the
    /// inertial frame.
    pub translation: DVec3,
    pub rotation: DQuat,
    /// A grounded body is fixed in the inertial frame (the assembly's base).
    pub grounded: bool,
}

/// A frame rigidly attached to a body — what joints and force elements
/// attach to. `translation` / `rotation` are relative to the body frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Marker {
    pub id: MarkerId,
    pub name: String,
    pub body: BodyId,
    pub translation: DVec3,
    pub rotation: DQuat,
}
