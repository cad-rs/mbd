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
    /// Initial linear velocity of the body-frame origin, world frame (m/s)
    /// — the Adams PART `VX VY VZ`. The dynamic initial condition projects
    /// the authored velocities onto the joint constraints (mass-weighted
    /// least squares), so they need not be consistent. Ignored when
    /// grounded and by the kinematic levels.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub velocity: DVec3,
    /// Initial angular velocity, world frame (rad/s) — the Adams PART
    /// `WX WY WZ`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub angular_velocity: DVec3,
}

fn is_zero(v: &DVec3) -> bool {
    *v == DVec3::ZERO
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
