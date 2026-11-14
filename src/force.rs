//! Force elements: what the dynamics (M2/M3) integrate besides the joint
//! reactions. M0 carries the vocabulary the force library (M3) fills in.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::body::MarkerId;

/// Stable force-element identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ForceId(pub u64);

/// A force element acting on the system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ForceElement {
    /// A linear spring-damper between two markers:
    /// `f = k (l - l0) + c dl/dt` along the marker-to-marker line.
    SpringDamper {
        id: ForceId,
        name: String,
        marker1: MarkerId,
        marker2: MarkerId,
        /// Stiffness (N/m).
        stiffness: f64,
        /// Damping (N·s/m).
        damping: f64,
        /// Free length (m).
        free_length: f64,
    },
    /// A constant force applied at a marker, in the inertial frame (N).
    AppliedForce {
        id: ForceId,
        name: String,
        marker: MarkerId,
        force: DVec3,
    },
    /// A constant torque applied to a marker's body, in the inertial frame (N·m).
    AppliedTorque {
        id: ForceId,
        name: String,
        marker: MarkerId,
        torque: DVec3,
    },
}

impl ForceElement {
    pub fn id(&self) -> ForceId {
        match self {
            ForceElement::SpringDamper { id, .. }
            | ForceElement::AppliedForce { id, .. }
            | ForceElement::AppliedTorque { id, .. } => *id,
        }
    }

    /// The markers the element attaches to (for validation).
    pub fn markers(&self) -> Vec<MarkerId> {
        match self {
            ForceElement::SpringDamper {
                marker1, marker2, ..
            } => vec![*marker1, *marker2],
            ForceElement::AppliedForce { marker, .. } | ForceElement::AppliedTorque { marker, .. } => {
                vec![*marker]
            }
        }
    }
}
