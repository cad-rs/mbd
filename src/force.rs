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
    /// Adams-style joint friction on a revolute, prismatic, cylindrical or
    /// spherical joint: the friction load is `μ(slip) · N`, with `N` the
    /// *friction load* the joint's own reaction produces on its bearing
    /// surfaces (M5-1 — the SolidWorks Motion / Inventor joint-friction
    /// tier). `N` is in the joint's generalized units (N·m for a rotation,
    /// N for a translation):
    ///
    /// - **revolute**: `pin_radius·F_radial + thrust_arm·F_axial +
    ///   2·pin_radius·M_bending/bending_arm + preload` (N·m);
    /// - **prismatic**: `F_transverse + 2·M_bending/bending_arm +
    ///   M_axial/pin_radius + preload` (N);
    /// - **cylindrical**: the normal load `F_transverse +
    ///   2·M_bending/bending_arm` with the preload, shared between the
    ///   sliding and turning directions by the slip direction at the pin
    ///   surface (one friction cone, not two independent ones);
    /// - **spherical**: `pin_radius·|F| + preload` (N·m), opposing the
    ///   relative angular velocity.
    ///
    /// `F_radial` / `F_transverse` is the reaction force perpendicular to
    /// the joint axis, `F_axial` the force along it, `M_bending` the
    /// reaction moment perpendicular to the axis and `M_axial` the moment
    /// about it, all from the joint's multipliers — so the friction load
    /// follows the bearing load, as in the commercial solvers. **An arm of
    /// zero switches its term off** (Adams' "input forces to friction"
    /// toggles); with every arm zero the friction is the constant
    /// `μ(slip)·preload`, the load-independent Coulomb friction MuJoCo's
    /// `frictionloss` models.
    ///
    /// The coefficient is the regularized Stribeck curve: a smooth ramp
    /// `0 → mu_static` over `stiction_velocity`, the Stribeck fall to
    /// `mu_dynamic` at `friction_velocity`, constant beyond (both
    /// transitions the `STEP` Hermite the drivers use). The slip is the
    /// joint coordinate's rate (rad/s revolute, m/s prismatic), the
    /// relative angular speed (spherical) or the surface slip speed at the
    /// pin (m/s, cylindrical).
    ///
    /// **Rest.** The regularized curve has no force at zero slip, so a
    /// dynamic run cannot stick for ever (the commercial compromise too).
    /// [`crate::static_equilibrium`] instead measures the slip as the
    /// joint's displacement from the pose it started at, saturating at
    /// `max_stiction_deformation` (Adams' maximum stiction deformation) as
    /// the scale of a smooth stiction spring `μ_s·tanh(Δ/Δ_max)·N`: the
    /// friction holds the mechanism within about that displacement
    /// whenever `μ_s·N` can carry the load, which is how a block stays put
    /// on a slope. Unlike the velocity regularization this ramp has
    /// stiffness at zero displacement, which the static Newton needs. `max_stiction_deformation = 0` leaves friction out of static
    /// equilibrium entirely (Adams' "sliding only" effect).
    JointFriction {
        id: ForceId,
        name: String,
        joint: JointId,
        /// Static friction coefficient (the peak of the curve).
        mu_static: f64,
        /// Dynamic (sliding) friction coefficient.
        mu_dynamic: f64,
        /// Slip at which the coefficient reaches `mu_static` (rad/s or
        /// m/s): Adams' stiction transition velocity. Must be positive —
        /// it is the regularization width.
        stiction_velocity: f64,
        /// Slip at and beyond which the coefficient is `mu_dynamic`
        /// (rad/s or m/s): Adams' friction transition velocity. At least
        /// `stiction_velocity`.
        friction_velocity: f64,
        /// Whether the joint's reaction *force* creates friction (Adams'
        /// "input forces to friction" toggles; the moments have their arms
        /// to switch them off, and a guide's transverse force has no arm).
        /// `false` leaves the preload as the only friction load — the
        /// load-independent Coulomb friction, the one MuJoCo's
        /// `frictionloss` also models.
        #[serde(default = "default_true")]
        reaction_force: bool,
        /// Pin radius (revolute / cylindrical), ball radius (spherical),
        /// or the radius at which a prismatic guide reacts the moment
        /// about the sliding axis (m). `0`: no radial-load friction.
        #[serde(default, skip_serializing_if = "is_zero")]
        pin_radius: f64,
        /// The radius of the thrust face that carries the axial load of a
        /// revolute or cylindrical joint (m). `0`: no axial-load friction.
        #[serde(default, skip_serializing_if = "is_zero")]
        thrust_arm: f64,
        /// The separation of the two bearing surfaces that react a
        /// bending moment — the pin-end spacing or the slider overlap (m).
        /// `0`: no bending-moment friction.
        #[serde(default, skip_serializing_if = "is_zero")]
        bending_arm: f64,
        /// The friction load present with no reaction load at all (N·m for
        /// a rotation, N for a translation): an interference fit, a
        /// clamped seal, a preloaded bearing.
        #[serde(default, skip_serializing_if = "is_zero")]
        preload: f64,
        /// How far the joint may creep before the static-equilibrium
        /// stiction saturates at `mu_static` (rad or m). `0`: friction is
        /// ignored by static equilibrium.
        #[serde(default = "default_stiction_deformation")]
        max_stiction_deformation: f64,
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

/// The default maximum stiction deformation (rad or m): small enough to
/// read as "it did not move", large enough for the static Newton to find
/// the stiction force.
fn default_stiction_deformation() -> f64 {
    1.0e-4
}

/// The reaction force creates friction unless the document says otherwise.
fn default_true() -> bool {
    true
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
            | ForceElement::JointFriction { id, .. }
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
            | ForceElement::JointFriction { name, .. }
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
            ForceElement::JointSpringDamper { .. } | ForceElement::JointFriction { .. } => {
                Vec::new()
            }
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

    /// The joint this element loads as a whole, rather than through one
    /// coordinate: the joint [`ForceElement::JointFriction`] reads the
    /// reaction of (for validation and for the solver's coupling graph).
    pub fn friction_joint(&self) -> Option<JointId> {
        match self {
            ForceElement::JointFriction { joint, .. } => Some(*joint),
            _ => None,
        }
    }

    /// Whether this element's load may depend on the constraint
    /// multipliers — joint friction that reads the reaction force or a
    /// moment. Such a system's equations of motion are `Q(q, q̇, λ)`,
    /// which the solvers iterate on; the answer is deliberately
    /// conservative (a revolute with no arms at all reads nothing, and an
    /// extra solve finds that out).
    pub(crate) fn reads_reactions(&self) -> bool {
        match self {
            ForceElement::JointFriction {
                reaction_force,
                pin_radius,
                thrust_arm,
                bending_arm,
                ..
            } => {
                *reaction_force
                    || *pin_radius != 0.0
                    || *thrust_arm != 0.0
                    || *bending_arm != 0.0
            }
            _ => false,
        }
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
            ForceElement::JointFriction {
                mu_static,
                mu_dynamic,
                stiction_velocity,
                friction_velocity,
                pin_radius,
                thrust_arm,
                bending_arm,
                preload,
                max_stiction_deformation,
                ..
            } => {
                if ![
                    *mu_static,
                    *mu_dynamic,
                    *stiction_velocity,
                    *friction_velocity,
                    *pin_radius,
                    *thrust_arm,
                    *bending_arm,
                    *preload,
                    *max_stiction_deformation,
                ]
                .iter()
                .all(|v| v.is_finite())
                {
                    return Err("non-finite joint friction parameters");
                }
                if *mu_dynamic < 0.0 || *mu_static < *mu_dynamic {
                    return Err("joint friction needs mu_static ≥ mu_dynamic ≥ 0");
                }
                if *stiction_velocity <= 0.0 || *friction_velocity < *stiction_velocity {
                    return Err(
                        "joint friction needs 0 < stiction_velocity ≤ friction_velocity",
                    );
                }
                if [
                    *pin_radius,
                    *thrust_arm,
                    *bending_arm,
                    *preload,
                    *max_stiction_deformation,
                ]
                .iter()
                .any(|v| *v < 0.0)
                {
                    return Err("joint friction arms, preload and stiction deformation cannot be negative");
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
