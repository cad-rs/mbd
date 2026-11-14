//! Force elements: what the dynamics integrate besides the joint reactions.
//!
//! The vocabulary mirrors the commercial force libraries (Adams SFORCE /
//! VFORCE / VTORQUE / SPRINGDAMPER, SolidWorks Motion force / spring /
//! damper, Inventor applied forces): fixed springs and dampers, and forces
//! whose magnitude is a [`Function`] — a curve (any [`DriverProfile`]:
//! constant / linear, data points, spline, harmonic, STEP5) of time or of a
//! state measure (a joint coordinate or rate, a marker distance or its
//! rate).

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::body::MarkerId;
use crate::driver::{DriverProfile, DriverSlot};
use crate::joint::JointId;

/// Stable force-element identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ForceId(pub u64);

/// What a force [`Function`]'s curve is evaluated at.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "of", rename_all = "snake_case")]
pub enum Argument {
    /// Simulation time (s).
    Time,
    /// A joint coordinate (rad, unwrapped across turns / m).
    JointCoordinate { joint: JointId, slot: DriverSlot },
    /// The rate of a joint coordinate (rad/s / m/s).
    JointRate { joint: JointId, slot: DriverSlot },
    /// The distance between two markers' origins (m).
    Distance {
        marker1: MarkerId,
        marker2: MarkerId,
    },
    /// The rate of that distance (m/s; positive when separating).
    DistanceRate {
        marker1: MarkerId,
        marker2: MarkerId,
    },
}

/// A scalar function: `curve(argument)`. The curve is read as a plain
/// function of its argument — `Constant { start, rate }` is the affine
/// `start + rate·x`, `Keyframes` / `Spline` are data-point curves,
/// `Harmonic` and `Step5` the usual time functions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub argument: Argument,
    pub curve: DriverProfile,
}

impl Function {
    /// A constant value.
    pub fn constant(value: f64) -> Self {
        Self::linear(Argument::Time, value, 0.0)
    }

    /// `offset + slope·argument`.
    pub fn linear(argument: Argument, offset: f64, slope: f64) -> Self {
        Self {
            argument,
            curve: DriverProfile::Constant {
                start: offset,
                rate: slope,
            },
        }
    }

    /// A curve of time.
    pub fn of_time(curve: DriverProfile) -> Self {
        Self {
            argument: Argument::Time,
            curve,
        }
    }

    /// The curve's value at argument value `x`.
    pub fn at(&self, x: f64) -> f64 {
        self.curve.sample(x).value
    }

    fn markers(&self) -> Vec<MarkerId> {
        match self.argument {
            Argument::Distance { marker1, marker2 }
            | Argument::DistanceRate { marker1, marker2 } => {
                vec![marker1, marker2]
            }
            _ => Vec::new(),
        }
    }

    fn joint(&self) -> Option<(JointId, DriverSlot)> {
        match self.argument {
            Argument::JointCoordinate { joint, slot } | Argument::JointRate { joint, slot } => {
                Some((joint, slot))
            }
            _ => None,
        }
    }
}

/// A force element acting on the system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ForceElement {
    /// A linear spring-damper between two markers, along the
    /// marker-to-marker line: tension `k·(l − l₀) + c·l̇ − preload`
    /// (positive tension pulls the markers together; a positive preload
    /// pushes them apart at the free length — the Adams `SPRINGDAMPER`
    /// convention).
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
        /// Force at the free length, pushing apart (N).
        #[serde(default, skip_serializing_if = "is_zero")]
        preload: f64,
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
    /// An action-only force at a marker's origin (the reaction goes to
    /// ground): `magnitude · direction`, the unit `direction` expressed in
    /// the `reference` marker's frame (`None`: the world frame). A
    /// reference on the loaded body makes a follower force.
    Force {
        id: ForceId,
        name: String,
        marker: MarkerId,
        #[serde(default)]
        reference: Option<MarkerId>,
        direction: DVec3,
        magnitude: Function,
    },
    /// An action-only torque on a marker's body: `magnitude · direction`,
    /// the unit `direction` in the `reference` marker's frame (`None`:
    /// world).
    Torque {
        id: ForceId,
        name: String,
        marker: MarkerId,
        #[serde(default)]
        reference: Option<MarkerId>,
        direction: DVec3,
        magnitude: Function,
    },
    /// An action-reaction force along the line between two markers (Adams
    /// translational `SFORCE`): positive `tension` pulls them together.
    LineForce {
        id: ForceId,
        name: String,
        marker1: MarkerId,
        marker2: MarkerId,
        tension: Function,
    },
    /// A spring-damper on a joint coordinate (Inventor joint spring, the
    /// SolidWorks torsional spring on a hinge): generalized force
    /// `−k·(q − q₀) − c·q̇ + preload` along the coordinate (torque for
    /// rotation, force for translation). Rotation is measured unwrapped
    /// across turns, so the spring winds any number of revolutions.
    JointSpringDamper {
        id: ForceId,
        name: String,
        joint: JointId,
        slot: DriverSlot,
        /// N·m/rad or N/m.
        stiffness: f64,
        /// N·m·s/rad or N·s/m.
        damping: f64,
        /// The coordinate at which the spring is relaxed (rad / m).
        free_value: f64,
        /// Generalized force at the free value, driving the coordinate up.
        #[serde(default, skip_serializing_if = "is_zero")]
        preload: f64,
    },
    /// A 6-DOF linear bushing (Adams `BUSHING`) between markers I
    /// (`marker1`) and J: with `d` the displacement of J's origin from I's
    /// and `θ` the rotation vector of J relative to I, both in I's axes,
    /// the force on J is `−K·d − C·ḋ` and the torque `−K_t·θ − C_t·ω_rel`
    /// (diagonal stiffness / damping per I axis; the equal and opposite
    /// load acts on I at the same point). A small-deflection element, as in
    /// the commercial tools: exact for rotation about one I axis.
    Bushing {
        id: ForceId,
        name: String,
        marker1: MarkerId,
        marker2: MarkerId,
        /// N/m per I axis.
        stiffness: DVec3,
        /// N·s/m per I axis.
        damping: DVec3,
        /// N·m/rad per I axis.
        torsion_stiffness: DVec3,
        /// N·m·s/rad per I axis.
        torsion_damping: DVec3,
    },
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

impl ForceElement {
    pub fn id(&self) -> ForceId {
        match self {
            ForceElement::SpringDamper { id, .. }
            | ForceElement::AppliedForce { id, .. }
            | ForceElement::AppliedTorque { id, .. }
            | ForceElement::Force { id, .. }
            | ForceElement::Torque { id, .. }
            | ForceElement::LineForce { id, .. }
            | ForceElement::JointSpringDamper { id, .. }
            | ForceElement::Bushing { id, .. } => *id,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            ForceElement::SpringDamper { name, .. }
            | ForceElement::AppliedForce { name, .. }
            | ForceElement::AppliedTorque { name, .. }
            | ForceElement::Force { name, .. }
            | ForceElement::Torque { name, .. }
            | ForceElement::LineForce { name, .. }
            | ForceElement::JointSpringDamper { name, .. }
            | ForceElement::Bushing { name, .. } => name,
        }
    }

    /// The markers the element attaches to or measures (for validation).
    pub fn markers(&self) -> Vec<MarkerId> {
        match self {
            ForceElement::SpringDamper {
                marker1, marker2, ..
            }
            | ForceElement::Bushing {
                marker1, marker2, ..
            } => vec![*marker1, *marker2],
            ForceElement::JointSpringDamper { .. } => Vec::new(),
            ForceElement::AppliedForce { marker, .. }
            | ForceElement::AppliedTorque { marker, .. } => {
                vec![*marker]
            }
            ForceElement::Force {
                marker,
                reference,
                magnitude,
                ..
            }
            | ForceElement::Torque {
                marker,
                reference,
                magnitude,
                ..
            } => {
                let mut m = vec![*marker];
                m.extend(*reference);
                m.extend(magnitude.markers());
                m
            }
            ForceElement::LineForce {
                marker1,
                marker2,
                tension,
                ..
            } => {
                let mut m = vec![*marker1, *marker2];
                m.extend(tension.markers());
                m
            }
        }
    }

    /// The joint coordinate a function measures, if any (for validation).
    pub fn measured_joint(&self) -> Option<(JointId, DriverSlot)> {
        if let ForceElement::JointSpringDamper { joint, slot, .. } = self {
            return Some((*joint, *slot));
        }
        self.function().and_then(Function::joint)
    }

    fn function(&self) -> Option<&Function> {
        match self {
            ForceElement::Force { magnitude, .. } | ForceElement::Torque { magnitude, .. } => {
                Some(magnitude)
            }
            ForceElement::LineForce { tension, .. } => Some(tension),
            _ => None,
        }
    }

    /// Whether the element's numbers are usable (finite parameters, a
    /// non-zero direction, a valid curve); `Err` carries the reason.
    pub fn check(&self) -> Result<(), &'static str> {
        match self {
            ForceElement::SpringDamper {
                stiffness,
                damping,
                free_length,
                preload,
                ..
            } => {
                if ![*stiffness, *damping, *free_length, *preload]
                    .iter()
                    .all(|v| v.is_finite())
                {
                    return Err("non-finite spring-damper parameters");
                }
            }
            ForceElement::AppliedForce { force: v, .. }
            | ForceElement::AppliedTorque { torque: v, .. } => {
                if !v.is_finite() {
                    return Err("non-finite applied load");
                }
            }
            ForceElement::Force { direction, .. } | ForceElement::Torque { direction, .. } => {
                if !direction.is_finite() || direction.length() < 1e-12 {
                    return Err("the direction must be a finite non-zero vector");
                }
            }
            ForceElement::LineForce { .. } => {}
            ForceElement::JointSpringDamper {
                stiffness,
                damping,
                free_value,
                preload,
                ..
            } => {
                if ![*stiffness, *damping, *free_value, *preload]
                    .iter()
                    .all(|v| v.is_finite())
                {
                    return Err("non-finite joint spring-damper parameters");
                }
            }
            ForceElement::Bushing {
                stiffness,
                damping,
                torsion_stiffness,
                torsion_damping,
                ..
            } => {
                if ![stiffness, damping, torsion_stiffness, torsion_damping]
                    .iter()
                    .all(|v| v.is_finite())
                {
                    return Err("non-finite bushing parameters");
                }
            }
        }
        if let Some(f) = self.function() {
            f.curve.check()?;
        }
        Ok(())
    }
}
